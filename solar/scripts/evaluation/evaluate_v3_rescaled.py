"""Full-pipeline evaluation (ML model + exact constrained monthly rescale) vs REAL ERA5 3-hourly ssrd, Oct-Feb months 2015-2025.

Models: v1 (models/model_v1/model_kt.joblib), v2 (all-season CV folds), v3 plain / v3 area (ONDJF-only, masked training CV folds).
Every month is scored by fold models that never trained on that year.  Monthly target of the rescale = monthly sum of the real
3-hourly samples.  Scoring excludes cells south of 60S and above 3000 m (same masks as the v3 training), daylight steps only.
Per-cell sums are kept for the whole ONDJF set and for the DJF subset.

Usage: python3 evaluate_v3_rescaled.py [--workers 16] [--limit N]"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import argparse, sys, time
import joblib
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from reconstruct_ssrd import DAY_MU, DT, FINAL_FEATURES, SOLAR_CONSTANT, build_features, log
from evaluate_v2_cv import FOLDS, V1_PATH, GLOBAL_STRIDE, box_slices, load_piece
from rescale import constrained_rescale

OUT = "models/model_v2/evaluation/v3_eval_{tag}.npz"
KMAX = 1.1
LAT_MIN, ELEV_MAX = -60.0, 3000
PIECES = {"Global": (slice(0, 721, GLOBAL_STRIDE), [slice(0, 1440, GLOBAL_STRIDE)]),
          "Europe": box_slices(35, 71, -25, 40), "Korea": box_slices(33, 43, 124, 131)}
MODEL_PATHS = {"v2": "models/model_v2/cv_folds/{fold}/model_kt_v2.joblib",
               "v3plain": "models/model_v3/plain/cv_folds/{fold}/model_kt_v3.joblib",
               "v3area": "models/model_v3/area/cv_folds/{fold}/model_kt_v3.joblib"}
NAMES = ["v1", "v2", "v3plain", "v3area"]


def valid_mask(m, lat_sl, lon_sls):
    z = xr.open_dataset("data/static/era5_geopotential_surface.nc")["z"].squeeze().values / 9.80665
    elev = np.concatenate([z[lat_sl][:, ls] for ls in lon_sls], axis=1)
    assert elev.shape == tuple(m["shape2d"])
    return (m["lat"][:, None] >= LAT_MIN) & (elev <= ELEV_MAX)


def piece_stats(m, models, valid):
    ntime = m["ntime"]; nlat, nlon = m["shape2d"]; ncell = nlat * nlon
    feat, _, tisr_day, day = build_features(m)
    X = np.column_stack([feat[c] for c in FINAL_FEATURES])
    tisr = m["tisr"].reshape(ntime, ncell).astype("float64")
    real = m["ssrd"].reshape(ntime, ncell).astype("float64")
    daymask = (tisr / (DT * SOLAR_CONSTANT)) > DAY_MU
    use = daymask & valid.reshape(1, ncell)
    target = real.sum(axis=0)
    hours = (m["times"] - m["times"].astype("datetime64[D]")) / np.timedelta64(1, "h") - 0.5
    local = (hours[:, None] + np.tile(m["lon"], nlat)[None, :] / 15.0) % 24.0
    tw = np.maximum(np.cos(2 * np.pi * (local - 12.0) / 24.0), 0.0) + 1e-9
    # daily bucket per 3-hourly step (for the day-level comparison: monthly is forced
    # exact by the rescale for every model alike, so it cannot distinguish them; the
    # day is the coarsest level that still carries real, un-forced skill)
    day_of_month = ((m["times"] - m["times"].astype("datetime64[M]")) / np.timedelta64(1, "D")).astype(int)
    ndays = int(day_of_month.max()) + 1
    real_r2 = np.where(use, real, np.nan)                      # sum of real^2, for pointwise R2
    res = {"n": use.sum(axis=0).reshape(nlat, nlon).astype("float64"),
           "sr": np.where(use, real, 0.0).sum(axis=0).reshape(nlat, nlon),
           "srr": np.where(use, real * real, 0.0).sum(axis=0).reshape(nlat, nlon)}
    for name, model in models.items():
        p = np.zeros(ntime * ncell, "float64")
        p[day] = np.clip(model.predict(X), 0, 1.5) * tisr_day
        q, diag = constrained_rescale(p.reshape(ntime, ncell), tisr, target, kmax=KMAX, twilight_w=tw)
        d = np.where(use, q - real, 0.0)
        # daily sums (real and predicted), only for days with >=1 valid daylight step
        real_day = np.zeros((ndays, ncell)); pred_day = np.zeros((ndays, ncell)); nvalid_day = np.zeros((ndays, ncell))
        np.add.at(real_day, day_of_month, np.where(use, real, 0.0))
        np.add.at(pred_day, day_of_month, np.where(use, q, 0.0))
        np.add.at(nvalid_day, day_of_month, use.astype("float64"))
        dd = np.where(nvalid_day > 0, pred_day - real_day, 0.0)
        ndaysused = (nvalid_day > 0)
        res[name] = dict(sse=(d * d).sum(axis=0).reshape(nlat, nlon), sb=d.sum(axis=0).reshape(nlat, nlon),
                         diag=diag, dsse=(dd * dd).sum(axis=0).reshape(nlat, nlon),
                         dsb=dd.sum(axis=0).reshape(nlat, nlon))
    res["dn"] = ndaysused.sum(axis=0).reshape(nlat, nlon).astype("float64")
    res["dsr"] = real_day.sum(axis=0).reshape(nlat, nlon)
    res["dsrr"] = (real_day * real_day * ndaysused).sum(axis=0).reshape(nlat, nlon)
    return res


def one_month(stamp, fold):
    models = {"v1": joblib.load(V1_PATH)["model"]}
    for k, pth in MODEL_PATHS.items():
        models[k] = joblib.load(pth.format(fold=fold))["model"]
    out = {"stamp": stamp}
    for piece, (lat_sl, lon_sls) in PIECES.items():
        m = load_piece(stamp, lat_sl, lon_sls)
        out[piece] = piece_stats(m, models, valid_mask(m, lat_sl, lon_sls))
        out[piece + "_lat"], out[piece + "_lon"] = m["lat"], m["lon"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="ondjf")
    args = ap.parse_args()
    jobs = [(f"{y}{mth:02d}", fold) for fold, yrs in FOLDS.items() for y in yrs for mth in (10, 11, 12, 1, 2)]
    if args.limit: jobs = jobs[:args.limit]
    log(f"{len(jobs)} months, {args.workers} workers, models {NAMES}")
    t = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky")(delayed(one_month)(*j) for j in jobs)
    log(f"done in {(time.time()-t)/60:.1f} min")
    arrays = {}
    for grp, months in (("ondjf", (10, 11, 12, 1, 2)), ("djf", (12, 1, 2)),
                        ("m10", (10,)), ("m11", (11,)), ("m12", (12,)), ("m1", (1,)), ("m2", (2,))):
        rs = [r for r in res if int(r["stamp"][4:]) in months]
        arrays[f"{grp}_nmonths"] = len(rs)
        for piece in PIECES:
            arrays[f"{piece}_lat"], arrays[f"{piece}_lon"] = res[0][piece + "_lat"], res[0][piece + "_lon"]
            arrays[f"{grp}_{piece}_n"] = sum(r[piece]["n"] for r in rs); arrays[f"{grp}_{piece}_sr"] = sum(r[piece]["sr"] for r in rs)
            arrays[f"{grp}_{piece}_srr"] = sum(r[piece]["srr"] for r in rs)
            arrays[f"{grp}_{piece}_dn"] = sum(r[piece]["dn"] for r in rs); arrays[f"{grp}_{piece}_dsr"] = sum(r[piece]["dsr"] for r in rs)
            arrays[f"{grp}_{piece}_dsrr"] = sum(r[piece]["dsrr"] for r in rs)
            for mod in NAMES:
                for k in ("sse", "sb", "dsse", "dsb"):
                    arrays[f"{grp}_{piece}_{mod}_{k}"] = sum(r[piece][mod][k] for r in rs)
    dg = [r[p][m]["diag"] for r in res for p in PIECES for m in NAMES]
    nc = sum(x["n_cells"] for x in dg)
    log("CONSTRAINT CHECK (worst case over all months, pieces, models):")
    log(f"  max relative error of the monthly sum, ALL cells: {max(x['max_rel_sum_err'] for x in dg):.2e}")
    log(f"  ceiling raised: {sum(x['n_ceiling_raised'] for x in dg)} / polar-twilight: {sum(x['n_twilight'] for x in dg)} of {nc:,} cell-months")
    log(f"  max kt after rescale (other cells): {max(x['max_kt_normal'] for x in dg):.3f} (ceiling {KMAX}); night max {max(x['night_max_normal'] for x in dg):.1e}")
    np.savez_compressed(OUT.format(tag=args.tag), **arrays)
    log(f"saved {OUT.format(tag=args.tag)}")


if __name__ == "__main__":
    sys.exit(main())
