"""
Germany combined capacity factor (wind onshore + wind offshore + solar),
ONDJF winter months only, TCo1279-DART-1950C control run.

Pipeline (dark-doldrum-project-proposal-update.md section 5.1/5.2/5.4):
  1. Crop to the Germany bounding box (47-55N, 6-15E); split onshore/offshore
     by the ERA5-derived land mask already used for the wind reconstruction.
  2. Wind CF: v = sqrt(u100^2+v100^2) (100 m hub height for BOTH onshore and
     offshore per user decision -- no separate 150 m offshore extrapolation),
     simplified cubic power curve applied per 3-hourly sample, then spatially
     averaged over onshore / offshore cells separately.
  3. Solar CF: ssrd (J m-2, 3h accumulation, already exact-monthly-rescaled --
     see memory note on rescaling) -> W/m2 -> CF_PV = G/1000, spatially
     averaged over onshore (land) cells.
  4. Combine with Germany's 2024 installed-capacity weights (Bundesnetzagentur,
     cross-checked against Mockert et al. 2023 Table 1):
       w_solar=0.577 (99.3 GW), w_onshore=0.369 (63.55 GW), w_offshore=0.054 (9.2 GW)
  5. 46h -> 48h centred moving average and Mockert/Li event flags, computed
     separately per winter block (Oct-Feb) so the mean never bridges the
     missing March-September gap.

Usable winters: spin-up excludes calendar years 1950-1951 (proposal section
4.3), so a winter is only usable if BOTH its Oct/Nov/Dec year and its Jan/Feb
year are outside spin-up and present on disk. For 1950C (data 1950-1969) that
gives winters labelled 1953..1969 (17 winters, OND(Y-1)+JF(Y)), not simply
"1952-1969" -- the spin-up cut lands mid-winter for the label-1952 season.
"""
import datetime as dt
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import netCDF4 as nc
import numpy as np

N_WORKERS = 16  # months are read independently -- embarrassingly parallel;
                 # kept well under the node's 256 cores since this is a shared login node

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/scripts")
from core.domain import DomainMasks, BBOX_GERMANY
from core.capacity_factor import wind_speed, wind_cf, solar_cf, ONSHORE, OFFSHORE, SSRD_ACCUM_SECONDS_DART

SOLAR_DIR = "solar/data/dart_reconstructed/{year}"
SOLAR_FILE = "ssrd_reduced_3h_{year}{month:02d}-{year}{month:02d}.nc"
WIND_DIR = "wind/data/results/dart_1950c_100m"
WIND_FILE = "u100v100_{year}{month:02d}.nc"

SPINUP_YEARS = {1950, 1951}
DATA_YEARS = range(1950, 1970)  # 1950C: 1950-1969

WEIGHTS = dict(solar=0.577, onshore=0.369, offshore=0.054)

WINTER_MONTHS_OND = [10, 11, 12]  # from year Y-1
WINTER_MONTHS_JF = [1, 2]         # from year Y


def usable_winters():
    """[(label_year, [(y,m), ...OND then JF chronological order])]"""
    out = []
    for label in range(min(DATA_YEARS) + 3, max(DATA_YEARS) + 1):
        ond_year = label - 1
        months = [(ond_year, m) for m in WINTER_MONTHS_OND] + [(label, m) for m in WINTER_MONTHS_JF]
        years_needed = {ond_year, label}
        if years_needed & SPINUP_YEARS:
            continue
        if not all(y in DATA_YEARS for y, m in months):
            continue
        out.append((label, months))
    return out


def load_month(year, month, masks):
    solar_path = f"{SOLAR_DIR.format(year=year)}/{SOLAR_FILE.format(year=year, month=month)}"
    wind_path = f"{WIND_DIR}/{WIND_FILE.format(year=year, month=month)}"

    fs = nc.Dataset(solar_path)
    fw = nc.Dataset(wind_path)

    t_var = fs["time_counter"]
    times = nc.num2date(t_var[:], t_var.units, calendar=getattr(t_var, "calendar", "standard"),
                         only_use_cftime_datetimes=False)
    times = np.array([dt.datetime(*d.timetuple()[:6]) for d in times])

    ssrd = masks.read_window(fs["ssrd"])
    u100 = masks.read_window(fw["u100"])
    v100 = masks.read_window(fw["v100"])
    fs.close()
    fw.close()

    cf_solar_field = solar_cf(ssrd[:, masks.onshore_local], SSRD_ACCUM_SECONDS_DART)
    v = wind_speed(u100, v100)
    cf_on_field = wind_cf(v[:, masks.onshore_local], **ONSHORE)
    cf_off_field = wind_cf(v[:, masks.offshore_local], **OFFSHORE)

    return dict(
        time=times,
        cf_solar=cf_solar_field.mean(axis=1),
        cf_onshore=cf_on_field.mean(axis=1),
        cf_offshore=cf_off_field.mean(axis=1),
    )


_worker_masks = None


def _init_worker():
    global _worker_masks
    _worker_masks = DomainMasks(BBOX_GERMANY)


def _load_month_worker(ym):
    year, month = ym
    t0 = time.time()
    data = load_month(year, month, _worker_masks)
    return year, month, data, time.time() - t0


def rolling_mean_16(x):
    """48h centred-ish trailing mean over 3-hourly data (16 samples), full windows only."""
    if len(x) < 16:
        return np.full(len(x), np.nan)
    kernel = np.ones(16) / 16.0
    m = np.convolve(x, kernel, mode="valid")
    pad = len(x) - len(m)
    return np.concatenate([np.full(pad, np.nan), m])


def main():
    t0 = time.time()
    masks = DomainMasks(BBOX_GERMANY)
    print(f"Germany bbox: {masks.bbox.sum()} cells "
          f"({masks.onshore.sum()} onshore, {masks.offshore.sum()} offshore)")

    winters = usable_winters()
    print(f"Usable winters (1950C, spin-up excludes {sorted(SPINUP_YEARS)}): "
          f"{winters[0][0]}..{winters[-1][0]} ({len(winters)} winters)")

    all_ym = sorted({ym for _, months in winters for ym in months})
    print(f"Reading {len(all_ym)} month-files with {N_WORKERS} parallel workers "
          f"(wind files are chunked full-grid-width, so each month read decompresses "
          f"the whole global wind field -- this is the slow, unavoidable part; "
          f"parallelizing across months is the fix)...")

    results_by_ym = {}
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
        for year, month, data, dt_s in ex.map(_load_month_worker, all_ym):
            results_by_ym[(year, month)] = data
            n_done += 1
            print(f"  [{n_done}/{len(all_ym)}] {year}-{month:02d} done in {dt_s:.1f}s "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    all_time, all_solar, all_on, all_off, all_combined, all_winter_id = [], [], [], [], [], []

    for label, months in winters:
        chunks = [results_by_ym[ym] for ym in months]
        time_cat = np.concatenate([c["time"] for c in chunks])
        solar_cat = np.concatenate([c["cf_solar"] for c in chunks])
        on_cat = np.concatenate([c["cf_onshore"] for c in chunks])
        off_cat = np.concatenate([c["cf_offshore"] for c in chunks])
        combined_cat = (WEIGHTS["solar"] * solar_cat +
                         WEIGHTS["onshore"] * on_cat +
                         WEIGHTS["offshore"] * off_cat)

        all_time.append(time_cat)
        all_solar.append(solar_cat)
        all_on.append(on_cat)
        all_off.append(off_cat)
        all_combined.append(combined_cat)
        all_winter_id.append(np.full(len(time_cat), label))
        print(f"  winter {label}: {len(time_cat)} steps, "
              f"mean CF solar/on/off/combined = "
              f"{solar_cat.mean():.3f}/{on_cat.mean():.3f}/{off_cat.mean():.3f}/{combined_cat.mean():.3f}")

    time_all = np.concatenate(all_time)
    solar_all = np.concatenate(all_solar)
    on_all = np.concatenate(all_on)
    off_all = np.concatenate(all_off)
    combined_all = np.concatenate(all_combined)
    winter_id_all = np.concatenate(all_winter_id)

    roll48 = np.concatenate([rolling_mean_16(c) for c in all_combined])
    mockert_event = roll48 < 0.06
    wind_all = (WEIGHTS["onshore"] * on_all + WEIGHTS["offshore"] * off_all) / \
               (WEIGHTS["onshore"] + WEIGHTS["offshore"])
    li_event = (wind_all < 0.20) & (solar_all < 0.20)

    out_path = "dunkelflaute/data/germany/combined_cf_1950c_ondjf.npz"
    np.savez(out_path,
             time=time_all, winter_id=winter_id_all,
             cf_solar=solar_all, cf_onshore=on_all, cf_offshore=off_all,
             cf_combined=combined_all, cf_combined_48h=roll48,
             mockert_event=mockert_event, li_event=li_event,
             weights_solar=WEIGHTS["solar"], weights_onshore=WEIGHTS["onshore"],
             weights_offshore=WEIGHTS["offshore"])

    n_mockert_steps = int(np.nansum(mockert_event))
    n_li_steps = int(np.nansum(li_event))
    print()
    print(f"Total 3-hourly steps: {len(time_all)} across {len(winters)} winters")
    print(f"Mean CF: solar={solar_all.mean():.3f} onshore={on_all.mean():.3f} "
          f"offshore={off_all.mean():.3f} combined={combined_all.mean():.3f}")
    print(f"Mockert (48h roll < 0.06): {n_mockert_steps} steps "
          f"({100*n_mockert_steps/len(time_all):.2f}% of time)")
    print(f"Li (wind & solar CF both < 0.20, instantaneous): {n_li_steps} steps "
          f"({100*n_li_steps/len(time_all):.2f}% of time)")
    print(f"Saved: {out_path}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
