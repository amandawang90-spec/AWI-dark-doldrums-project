"""Evaluate the ERA5-trained models on an arbitrary list of DART (year,month) stamps
against DART's own monthly ssrd -- built to span a year boundary (e.g. Oct-Feb winter).

Raw model output (NO monthly rescaling): monthly-mean predicted ssrd per native cell
vs DART's monthly ssrd product (remapped grid, nearest-cell mapping, /10800 -> W/m2).

Efficiency: the tisr template is chunked (all times x 100k cells), so any time slice
decompresses the whole file (~10 min).  Read the whole year ONCE into RAM (77 GB) and
slice per month.  Then per month, stream the DART fields in blocks of time steps so
memory stays small.

Usage: python3 dart_winter_eval.py <run_dir> <stamp0> <stamp1> ... [--cell-limit N] [--out DIR]
       e.g. dart_winter_eval.py .../TCo1279-DART-1950C 195010 195011 195012 195101 195102
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import argparse
import os
import sys
import time

import joblib
import numpy as np
import xarray as xr

from reconstruct_ssrd_dart import (DAY_MU, DT_DART, MONTHLY_ANCHOR_DIVISOR, TEMPLATE_DIR,
                                   is_leap, log, nearest_regular_index)

MODELS = {"v1": "models/model_v1/model_kt.joblib", "v3area": "models/model_v3/area/model_kt_v3.joblib"}
LAT_MIN, ELEV_MAX = -60.0, 3000   # same mask v3 was trained/scored under, applied here for a fair comparison
FEATURES = ["T", "tcc", "hcc", "mcc", "lcc", "mu"]
REGIONS = {"Global": None, "Europe": (35, 71, -25, 40), "Germany": (47.3, 55.1, 5.9, 15.0),
           "Korea": (33, 43, 124, 131)}
TB = 8   # time steps per block


def area_weights(lat):
    """Exact per-cell area weights on this (only roughly equal-area) grid:
    area share of each 1-degree band divided by the share of cells in it."""
    edges = np.arange(-90, 91, 1.0)
    band = np.clip(np.digitize(lat, edges) - 1, 0, len(edges) - 2)
    cnt = np.bincount(band, minlength=len(edges) - 1).astype("float64")
    area = (np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))) / 2
    w_band = np.where(cnt > 0, area / np.maximum(cnt / cnt.sum(), 1e-12), 0.0)
    return w_band[band]


def region_masks(lat, lon):
    lon0 = np.where(lon > 180, lon - 360, lon)
    out = {}
    for r, box in REGIONS.items():
        if box is None:
            out[r] = np.ones(lat.shape, bool)
        else:
            a, b, c, d = box
            out[r] = (lat >= a) & (lat <= b) & (lon0 >= c) & (lon0 <= d)
    return out


def v3_valid_mask(lat, lon):
    """Nearest-map ERA5's static elevation onto DART's native cells; True where v3's
    training domain applies (matches the mask used to train/score v3: not south of
    60S, not above 3000 m)."""
    z = xr.open_dataset("data/static/era5_geopotential_surface.nc")["z"].squeeze().values / 9.80665
    era_lat = np.arange(90, -90.01, -0.25); era_lon = np.arange(0, 360, 0.25)
    ilat, ilon = nearest_regular_index(lat, lon, era_lat, era_lon)
    elev = z[ilat, ilon]
    return (lat >= LAT_MIN) & (elev <= ELEV_MAX)


def stats(actual, pred, w):
    if w.sum() == 0:
        return dict(mean_actual=np.nan, bias=np.nan, rmse=np.nan, r2=np.nan, within10=np.nan,
                    scale_med=np.nan, scale_p5=np.nan, scale_p95=np.nan, n=int(len(actual)))
    W = w.sum()
    ma = (w * actual).sum() / W
    d = pred - actual
    ss_res = (w * d * d).sum() / W
    ss_tot = (w * (actual - ma) ** 2).sum() / W
    scale = np.where(pred > 0.1, actual / np.maximum(pred, 1e-9), np.nan)
    return dict(mean_actual=ma, bias=(w * d).sum() / W, rmse=np.sqrt(ss_res),
                r2=1 - ss_res / ss_tot, within10=(w * (np.abs(d) <= 10)).sum() / W,
                scale_med=np.nanmedian(scale), scale_p5=np.nanpercentile(scale, 5),
                scale_p95=np.nanpercentile(scale, 95), n=int(len(actual)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("stamps", nargs="+", help="YYYYMM ... (may span a year boundary)")
    ap.add_argument("--cell-limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stamps = args.stamps
    tagm = f"{stamps[0]}-{stamps[-1]}"
    out_dir = args.out or f"data/dart_eval_{tagm}"
    os.makedirs(out_dir, exist_ok=True)
    d = f"{args.run_dir}/outdata/oifs"
    N = args.cell_limit or None
    sl = slice(0, N)

    models = {k: joblib.load(p)["model"] for k, p in MODELS.items()}
    first = xr.open_dataset(f"{d}/atm_reduced_3h_tcc_3h_{stamps[0]}-{stamps[0]}.nc")
    lat = first.lat.values[sl].astype("float32"); lon = first.lon.values[sl].astype("float32"); first.close()
    ncell = lat.size
    w = area_weights(lat.astype("float64")) if not N else np.ones(ncell)
    masks = region_masks(lat, lon)
    v3ok = v3_valid_mask(lat, lon)
    masks["Global_v3domain"] = masks["Global"] & v3ok   # matches exactly what v3 was trained/scored on
    log(f"{ncell:,} cells; models {list(models)}; v3 domain keeps {100*v3ok.mean():.1f}% of cells")

    # ---- read the tisr template ONCE; every stamp indexes it relative to ITS OWN year-start ----
    # (1950 and 1951 are both non-leap -> same 365-day template, just a different day-of-year zero point)
    templates = {}
    idx_by_stamp = {}
    for stamp in stamps:
        y = int(stamp[:4])
        tpl = f"{TEMPLATE_DIR}/tisr_template_{'366' if is_leap(y) else '365'}day.nc"
        templates.setdefault(tpl, [])
        year_start = np.datetime64(f"{y}-01-01T03:00:00", "s")
        tt = xr.open_dataset(f"{d}/atm_reduced_3h_tcc_3h_{stamp}-{stamp}.nc").time_counter.values
        ix = np.round((tt.astype("datetime64[s]") - year_start) / np.timedelta64(3, "h")).astype(int)
        idx_by_stamp[stamp] = (tpl, ix)
        templates[tpl].append(stamp)
    tisr_sel = {}  # stamp -> (ntime, ncell) array
    for tpl, sts in templates.items():
        ds = xr.open_dataset(tpl)
        nt_year = ds.sizes["time_counter"]
        all_idx = np.unique(np.concatenate([idx_by_stamp[s][1] for s in sts]))
        assert all_idx.min() >= 0 and all_idx.max() < nt_year
        buf = np.empty((len(all_idx), ncell), "float32")
        t0 = time.time()
        for s_ in range(0, ncell, 100_000):
            e_ = min(s_ + 100_000, ncell)
            buf[:, s_:e_] = ds.tisr.isel(cell=slice(s_, e_)).values[all_idx]
        ds.close()
        log(f"template {os.path.basename(tpl)} read in {(time.time()-t0)/60:.1f} min for {sts}")
        for s in sts:
            rows_t = np.searchsorted(all_idx, idx_by_stamp[s][1])
            tisr_sel[s] = buf[rows_t]

    rows = []
    for stamp in stamps:
        tm = time.time()
        tag = f"{stamp}-{stamp}"
        dsv = {v: xr.open_dataset(f"{d}/atm_reduced_3h_{v}_3h_{tag}.nc") for v in ("tcc", "hcc", "mcc", "lcc", "tsr")}
        monthly = xr.open_dataset(f"{d}/atm_remapped_1m_ssrd_1m_{tag}.nc")
        times = dsv["tcc"].time_counter.values
        ntime = len(times)
        acc = {k: np.zeros(ncell, "float64") for k in models}
        for t0_ in range(0, ntime, TB):
            t1_ = min(t0_ + TB, ntime)
            f = {v: dsv[v][v].isel(time_counter=slice(t0_, t1_), cell=sl).values.astype("float32") for v in dsv}
            tisr = tisr_sel[stamp][t0_:t1_]
            mu = tisr / (DT_DART * 1361.0)
            day = mu > DAY_MU
            X = np.column_stack([np.clip(f["tsr"][day] / tisr[day], 0, 1.5), f["tcc"][day], f["hcc"][day],
                                 f["mcc"][day], f["lcc"][day], mu[day]])
            tday = tisr[day]
            for k, mod in models.items():
                ssrd_day = np.clip(mod.predict(X), 0, 1.5) * tday
                full = np.zeros(day.shape, "float32"); full[day] = ssrd_day
                acc[k] += full.sum(axis=0, dtype="float64")
        pred = {k: (acc[k] / ntime / DT_DART).astype("float32") for k in models}
        # DART's own monthly ssrd -> native cells (nearest cell on the remapped grid)
        mlat, mlon = monthly.lat.values, monthly.lon.values
        ilat, ilon = nearest_regular_index(lat, lon, mlat, mlon)
        actual = (monthly.ssrd.squeeze().values / MONTHLY_ANCHOR_DIVISOR)[ilat, ilon].astype("float32")
        for v in list(dsv.values()) + [monthly]:
            v.close()
        np.savez_compressed(f"{out_dir}/monthly_{stamp}.npz", actual=actual, **{f"pred_{k}": p for k, p in pred.items()})
        for r, mk in masks.items():
            for k in models:
                s = stats(actual[mk].astype("float64"), pred[k][mk].astype("float64"), w[mk])
                rows.append(dict(stamp=stamp, region=r, model=k, **s))
        g = [x for x in rows if x["stamp"] == stamp and x["region"] == "Global"]
        log(f"{stamp}: {(time.time()-tm)/60:.1f} min | real {g[0]['mean_actual']:.1f} | " +
            " | ".join(f"{x['model']} bias {x['bias']:+.1f} rmse {x['rmse']:.1f} R2 {x['r2']:.3f}" for x in g))
    np.savez_compressed(f"{out_dir}/grid_{tagm}.npz", lat=lat, lon=lon, w=w)
    import csv
    with open(f"{out_dir}/metrics_{tagm}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    log(f"saved {out_dir}/metrics_{tagm}.csv")


if __name__ == "__main__":
    sys.exit(main())
