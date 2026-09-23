"""Honest evaluation of the v2 model against REAL ERA5 3-hourly ssrd.

Every year 2015-2025 is scored by a v2 fold model that did NOT train on it
(see jobs/train_v2.sbatch: 3 year-block folds).  v1 (models/model_v1/model_kt.joblib)
is scored on the same months for comparison; 2025-08 is excluded from v1's
numbers because v1 trained on it.

Regions: Global (0.25deg grid thinned x4 -> 1deg), Europe / Germany
(Europe box, full 0.25deg), Korea (Korea box, full 0.25deg).
Per month it stores, per model and region:
  - pointwise weighted sums (per cell, per 3-hourly daylight step) vs real ssrd
  - monthly-mean spatial stats (same statistic used for the DART evaluation)
  - the monthly-mean error field on the global grid (for maps)

Usage: python3 evaluate_v2_cv.py [--workers 16]
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import argparse
import os
import pickle
import sys
import time

import joblib
import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from reconstruct_ssrd import (DATA_DIR, DT, FINAL_FEATURES, MONTHLY_ANCHOR_DIVISOR,
                              build_features, log)
from solar_geometry import toa_irradiance_accumulated

FOLDS = {
    "fold1_test_2015-2018": range(2015, 2019),
    "fold2_test_2019-2022": range(2019, 2023),
    "fold3_test_2023-2025": range(2023, 2026),
}
REGIONS = {"Global": None, "Europe": (35, 71, -25, 40),
           "Germany": (47.3, 55.1, 5.9, 15.0), "Korea": (33, 43, 124, 131)}
V1_PATH = "models/model_v1/model_kt.joblib"
V1_TRAINED_ON = {"202508", "202603", "202608"}
OUT_DIR = "models/model_v2/evaluation"
GLOBAL_STRIDE = 4


def box_slices(latlo, lathi, lonlo, lonhi):
    """Slices on the ERA5 0.25deg grid (lat 90->-90, lon 0->359.75). A box that
    crosses 0 deg longitude comes back as two lon slices (west part first)."""
    lat_sl = slice(int(round((90 - lathi) / 0.25)), int(round((90 - latlo) / 0.25)) + 1)
    i0, i1 = int(round(lonlo / 0.25)), int(round(lonhi / 0.25))
    if i0 >= 0:
        return lat_sl, [slice(i0, i1 + 1)]
    return lat_sl, [slice(1440 + i0, 1440), slice(0, i1 + 1)]


def _load_one(stamp, lat_sl, lon_sl):
    """Same content as reconstruct_ssrd.load_month, for one lat/lon slice."""
    sub = dict(latitude=lat_sl, longitude=lon_sl)
    clouds = xr.open_dataset(f"{DATA_DIR}/era5_clouds_3h_{stamp}.nc").isel(**sub)
    tsr = xr.open_dataset(f"{DATA_DIR}/era5_tsr_3h_{stamp}.nc").isel(**sub)
    ssrd = xr.open_dataset(f"{DATA_DIR}/era5_ssrd_3h_{stamp}.nc").isel(**sub)
    monthly = xr.open_dataset(f"{DATA_DIR}/era5_monthly_ssrd_{stamp}.nc").isel(**sub)
    lat, lon = clouds.latitude.values, clouds.longitude.values
    tisr = toa_irradiance_accumulated(lat, lon, clouds.valid_time.values, dt_seconds=DT)
    m = {"tcc": clouds.tcc.values.astype("float32"), "hcc": clouds.hcc.values.astype("float32"),
         "mcc": clouds.mcc.values.astype("float32"), "lcc": clouds.lcc.values.astype("float32"),
         "tsr": tsr.tsr.values.astype("float32"), "ssrd": ssrd.ssrd.values.astype("float32"),
         "tisr": tisr,
         "monthly_ssrd_wm2": (monthly.ssrd.squeeze().values / MONTHLY_ANCHOR_DIVISOR).astype("float32"),
         "lat": lat, "lon": lon, "times": clouds.valid_time.values}
    for ds in (clouds, tsr, ssrd, monthly):
        ds.close()
    return m


def load_piece(stamp, lat_sl, lon_sls):
    parts = [_load_one(stamp, lat_sl, ls) for ls in lon_sls]
    m = parts[0]
    for extra in parts[1:]:
        for k in ("tcc", "hcc", "mcc", "lcc", "tsr", "ssrd", "tisr", "monthly_ssrd_wm2"):
            m[k] = np.concatenate([m[k], extra[k]], axis=-1)
        m["lon"] = np.concatenate([m["lon"], extra["lon"]])
    m["ntime"], m["shape2d"] = m["tcc"].shape[0], m["tcc"].shape[1:]
    return m


def region_mask(lat, lon, box):
    lon0 = np.where(lon > 180, lon - 360, lon)
    if box is None:
        return np.ones((len(lat), len(lon)), bool)
    latlo, lathi, lonlo, lonhi = box
    return ((lat[:, None] >= latlo) & (lat[:, None] <= lathi) &
            (lon0[None, :] >= lonlo) & (lon0[None, :] <= lonhi))


def score_piece(m, models, region_names):
    """Pointwise sums and monthly-spatial stats for each model on one loaded piece."""
    lat, lon, ntime = m["lat"], m["lon"], m["ntime"]
    nlat, nlon = m["shape2d"]
    feat, kt, tisr_day, day = build_features(m)
    X = np.column_stack([feat[c] for c in FINAL_FEATURES])
    idx = np.nonzero(day)[0]
    cell = idx % (nlat * nlon)
    w_row = np.cos(np.deg2rad(lat[cell // nlon])).astype("float64")
    act_day = m["ssrd"].ravel()[day].astype("float64")
    masks = {r: region_mask(lat, lon, REGIONS[r]) for r in region_names}
    w_cell = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, nlon))
    out = {}
    for name, model in models.items():
        p_day = np.clip(model.predict(X), 0, 1.5) * tisr_day
        p_day = p_day.astype("float64")
        point, spatial = {}, {}
        full = np.zeros(ntime * nlat * nlon, "float32")
        full[day] = p_day
        pred_month = full.reshape(ntime, nlat, nlon).mean(0) / DT
        act_month = m["monthly_ssrd_wm2"].astype("float64")
        for r in region_names:
            sel = masks[r].ravel()[cell]
            w, a, p = w_row[sel], act_day[sel], p_day[sel]
            point[r] = dict(n=int(sel.sum()), W=w.sum(), Sa=(w * a).sum(), Sp=(w * p).sum(),
                            Saa=(w * a * a).sum(), Spp=(w * p * p).sum(), Sap=(w * a * p).sum(),
                            Sdd=(w * (a - p) ** 2).sum())
            mk = masks[r]
            wc, am, pm = w_cell[mk], act_month[mk], pred_month[mk].astype("float64")
            ma = np.average(am, weights=wc)
            ss_res = np.average((am - pm) ** 2, weights=wc)
            ss_tot = np.average((am - ma) ** 2, weights=wc)
            spatial[r] = dict(n=int(mk.sum()), r2=1 - ss_res / ss_tot, rmse=float(np.sqrt(ss_res)),
                              bias=float(np.average(pm - am, weights=wc)), mean_actual=float(ma))
        out[name] = dict(point=point, spatial=spatial,
                         err=(pred_month - act_month).astype("float32"))
    return out


def eval_month(stamp, fold, v2_path):
    t0 = time.time()
    v2 = joblib.load(v2_path)["model"]
    models = {"v2": v2}
    if stamp not in V1_TRAINED_ON:
        models["v1"] = joblib.load(V1_PATH)["model"]
    res = {"stamp": stamp, "fold": fold}
    # Global on a 1-degree grid
    gl = slice(0, 721, GLOBAL_STRIDE)
    res["Global"] = score_piece(load_piece(stamp, gl, [slice(0, 1440, GLOBAL_STRIDE)]), models, ["Global"])
    res["Global_lat"], res["Global_lon"] = 90 - 0.25 * np.arange(0, 721, GLOBAL_STRIDE), 0.25 * np.arange(0, 1440, GLOBAL_STRIDE)
    # Europe box (contains Germany), full resolution
    res["Europe"] = score_piece(load_piece(stamp, *box_slices(35, 71, -25, 40)), models, ["Europe", "Germany"])
    # Korea box, full resolution
    res["Korea"] = score_piece(load_piece(stamp, *box_slices(33, 43, 124, 131)), models, ["Korea"])
    res["secs"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fold-root", default="models/model_v2/cv_folds")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = []
    for fold, years in FOLDS.items():
        path = f"{args.fold_root}/{fold}/model_kt_v2.joblib"
        for y in years:
            for mth in range(1, 13):
                jobs.append((f"{y}{mth:02d}", fold, path))
    if args.limit:
        jobs = jobs[:args.limit]
    log(f"evaluating {len(jobs)} months with {args.workers} workers ...")
    t0 = time.time()
    results = Parallel(n_jobs=args.workers, backend="loky")(delayed(eval_month)(*j) for j in jobs)
    log(f"done in {(time.time()-t0)/60:.1f} min; mean {np.mean([r['secs'] for r in results]):.0f}s/month")
    with open(f"{OUT_DIR}/raw_results.pkl", "wb") as f:
        pickle.dump(results, f)
    log(f"saved {OUT_DIR}/raw_results.pkl")


if __name__ == "__main__":
    sys.exit(main())
