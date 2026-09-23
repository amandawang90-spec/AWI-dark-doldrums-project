"""Evaluate the ERA5-trained models on ONE DART year against DART's own monthly ssrd.

Raw model output (NO monthly rescaling): monthly-mean predicted ssrd per native cell
vs DART's monthly ssrd product (remapped grid, nearest-cell mapping, /10800 -> W/m2).

Efficiency: the tisr template is chunked (all times x 100k cells), so any time slice
decompresses the whole file (~10 min).  Read the whole year ONCE into RAM (77 GB) and
slice per month.  Then per month, stream the DART fields in blocks of time steps so
memory stays small.

Usage: python3 dart_year_eval.py <run_dir> <year> [--months 1-12] [--cell-limit N] [--out DIR]
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

MODELS = {"v1": "models/model_v1/model_kt.joblib", "v2": "models/model_v2/model_kt_v2.joblib"}
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


def stats(actual, pred, w):
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
    ap.add_argument("run_dir"); ap.add_argument("year", type=int)
    ap.add_argument("--months", default="1-12")
    ap.add_argument("--cell-limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-3h", default="", help="comma list of models whose 3-hourly ssrd is written, e.g. v1,v2")
    ap.add_argument("--recon-dir", default=None)
    args = ap.parse_args()
    y = args.year
    m0, m1 = (int(v) for v in args.months.split("-")) if "-" in args.months else (int(args.months),) * 2
    out_dir = args.out or f"data/dart_eval_{y}"
    os.makedirs(out_dir, exist_ok=True)
    write_models = [m for m in args.write_3h.split(",") if m]
    recon_dir = args.recon_dir or f"data/dart_recon_{y}"
    d = f"{args.run_dir}/outdata/oifs"
    N = args.cell_limit or None
    sl = slice(0, N)

    models = {k: joblib.load(p)["model"] for k, p in MODELS.items()}
    first = xr.open_dataset(f"{d}/atm_reduced_3h_tcc_3h_{y}{m0:02d}-{y}{m0:02d}.nc")
    lat = first.lat.values[sl].astype("float32"); lon = first.lon.values[sl].astype("float32"); first.close()
    ncell = lat.size
    w = area_weights(lat.astype("float64")) if not N else np.ones(ncell)
    masks = region_masks(lat, lon)
    log(f"{ncell:,} cells; models {list(models)}")

    # ---- read the tisr template ONCE, keeping only the time steps the requested months need ----
    year_start = np.datetime64(f"{y}-01-01T03:00:00", "s")
    tpl = f"{TEMPLATE_DIR}/tisr_template_{'366' if is_leap(y) else '365'}day.nc"
    ds = xr.open_dataset(tpl)
    nt_year = ds.sizes["time_counter"]
    idx_by_month = {}
    for mth in range(m0, m1 + 1):
        tg = f"{y}{mth:02d}-{y}{mth:02d}"
        tt = xr.open_dataset(f"{d}/atm_reduced_3h_tcc_3h_{tg}.nc").time_counter.values
        ix = np.round((tt.astype("datetime64[s]") - year_start) / np.timedelta64(3, "h")).astype(int)
        assert ix.min() >= 0 and ix.max() < nt_year
        idx_by_month[mth] = ix
    all_idx = np.unique(np.concatenate(list(idx_by_month.values())))
    tisr_sel = np.empty((len(all_idx), ncell), "float32")
    t0 = time.time()
    for s_ in range(0, ncell, 100_000):
        e_ = min(s_ + 100_000, ncell)
        tisr_sel[:, s_:e_] = ds.tisr.isel(cell=slice(s_, e_)).values[all_idx]
        if (s_ // 100_000) % 10 == 0:
            log(f"  template cells {e_:,}/{ncell:,}  ({(time.time()-t0)/60:.1f} min)")
    ds.close()
    log(f"template {os.path.basename(tpl)} read in {(time.time()-t0)/60:.1f} min  "
        f"[kept {len(all_idx)} of {nt_year} steps, {tisr_sel.nbytes/1e9:.0f} GB]")

    rows = []
    for mth in range(m0, m1 + 1):
        tm = time.time()
        stamp = f"{y}{mth:02d}"; tag = f"{stamp}-{stamp}"
        dsv = {v: xr.open_dataset(f"{d}/atm_reduced_3h_{v}_3h_{tag}.nc") for v in ("tcc", "hcc", "mcc", "lcc", "tsr")}
        monthly = xr.open_dataset(f"{d}/atm_remapped_1m_ssrd_1m_{tag}.nc")
        times = dsv["tcc"].time_counter.values
        rows_t = np.searchsorted(all_idx, idx_by_month[mth])
        ntime = len(times)
        acc = {k: np.zeros(ncell, "float64") for k in models}
        writers = {}
        if write_models:
            import netCDF4 as nc
            secs = ((times.astype("datetime64[s]") - np.datetime64(f"{y}-01-01T00:00:00", "s")) /
                    np.timedelta64(1, "s")).astype("int64")
            for k in write_models:
                os.makedirs(f"{recon_dir}/{k}", exist_ok=True)
                wds = nc.Dataset(f"{recon_dir}/{k}/ssrd_reduced_3h_{tag}.nc", "w", format="NETCDF4")
                wds.createDimension("time_counter", ntime); wds.createDimension("cell", ncell)
                tv = wds.createVariable("time_counter", "i8", ("time_counter",)); tv[:] = secs
                tv.units = f"seconds since {y}-01-01 00:00:00"; tv.calendar = "proleptic_gregorian"
                wds.createVariable("lat", "f4", ("cell",))[:] = lat
                wds.createVariable("lon", "f4", ("cell",))[:] = lon
                v = wds.createVariable("ssrd", "f4", ("time_counter", "cell"), zlib=True, complevel=1,
                                       chunksizes=(1, ncell))
                v.units = "J m-2"
                v.long_name = "Reconstructed surface solar radiation downwards, 3-hour accumulation (ML, raw)"
                v.comment = ("kt x tisr from cloud fractions, tsr and solar geometry. NOT rescaled to DART's monthly "
                             "ssrd. Same 3-hour accumulation convention as DART tsr/ssrd (monthly W/m2 = mean/10800).")
                wds.model = MODELS[k]; wds.features = ", ".join(FEATURES)
                writers[k] = wds
        for t0_ in range(0, ntime, TB):
            t1_ = min(t0_ + TB, ntime)
            f = {v: dsv[v][v].isel(time_counter=slice(t0_, t1_), cell=sl).values.astype("float32") for v in dsv}
            tisr = tisr_sel[rows_t[t0_:t1_]]
            mu = tisr / (DT_DART * 1361.0)
            day = mu > DAY_MU
            X = np.column_stack([np.clip(f["tsr"][day] / tisr[day], 0, 1.5), f["tcc"][day], f["hcc"][day],
                                 f["mcc"][day], f["lcc"][day], mu[day]])
            tday = tisr[day]
            for k, mod in models.items():
                ssrd_day = np.clip(mod.predict(X), 0, 1.5) * tday
                full = np.zeros(day.shape, "float32"); full[day] = ssrd_day
                acc[k] += full.sum(axis=0, dtype="float64")
                if k in writers:
                    writers[k].variables["ssrd"][t0_:t1_, :] = full
        pred = {k: (acc[k] / ntime / DT_DART).astype("float32") for k in models}
        for k, wds in writers.items():
            wds.close()
            if mth == m0:      # read the written file back once: it must reproduce the evaluated monthly mean
                chk = xr.open_dataset(f"{recon_dir}/{k}/ssrd_reduced_3h_{tag}.nc")
                back = np.zeros(ncell, "float64")
                for a in range(0, ntime, TB):
                    back += chk.ssrd.isel(time_counter=slice(a, min(a + TB, ntime))).values.sum(axis=0, dtype="float64")
                chk.close()
                dmax = np.abs(back / ntime / DT_DART - pred[k]).max()
                log(f"  read-back check {k}: max |file monthly mean - evaluated| = {dmax:.2e} W/m2")
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
    tagm = f"{m0:02d}-{m1:02d}"
    np.savez_compressed(f"{out_dir}/grid_{tagm}.npz", lat=lat, lon=lon, w=w)
    import csv
    with open(f"{out_dir}/metrics_{tagm}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    log(f"saved {out_dir}/metrics_{tagm}.csv")


if __name__ == "__main__":
    sys.exit(main())
