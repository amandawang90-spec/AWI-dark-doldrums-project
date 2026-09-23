"""Per-cell RMSE of the FULL pipeline (ML model + exact constrained monthly rescale) vs REAL ERA5 3-hourly ssrd.

Monthly target of the rescale = the monthly sum of the real 3-hourly samples (the exact analogue of DART,
whose monthly ssrd is exactly the mean of its 3-hourly values).  Model v2 = the CV fold model that never saw
the month; v1 = models/model_v1/model_kt.joblib.  RMSE over the daylight steps of each cell, all held-out months given.

Usage: python3 rmse_maps_rescaled.py --months 12,1,2 --tag rescaled_DJF [--workers 16] [--limit N]
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import argparse, sys, time
import joblib
import numpy as np
from joblib import Parallel, delayed
from reconstruct_ssrd import DAY_MU, DT, FINAL_FEATURES, SOLAR_CONSTANT, build_features, log
from evaluate_v2_cv import FOLDS, V1_PATH, GLOBAL_STRIDE, box_slices, load_piece
from rescale import constrained_rescale

OUT = "models/model_v2/evaluation/rmse_cells_{tag}.npz"
KMAX = 1.1
PIECES = {"Global": (slice(0, 721, GLOBAL_STRIDE), [slice(0, 1440, GLOBAL_STRIDE)]),
          "Europe": box_slices(35, 71, -25, 40), "Korea": box_slices(33, 43, 124, 131)}


def piece_stats(m, models):
    ntime = m["ntime"]; nlat, nlon = m["shape2d"]; ncell = nlat * nlon
    feat, _, tisr_day, day = build_features(m)
    X = np.column_stack([feat[c] for c in FINAL_FEATURES])
    tisr = m["tisr"].reshape(ntime, ncell).astype("float64")
    real = m["ssrd"].reshape(ntime, ncell).astype("float64")
    daymask = (tisr / (DT * SOLAR_CONSTANT)) > DAY_MU
    target = real.sum(axis=0)                                   # real monthly sum per cell (J/m2)
    hours = (m["times"] - m["times"].astype("datetime64[D]")) / np.timedelta64(1, "h") - 0.5   # window centre, UTC
    lon_cell = np.tile(m["lon"], nlat)
    local = (hours[:, None] + lon_cell[None, :] / 15.0) % 24.0
    twilight_w = np.maximum(np.cos(2 * np.pi * (local - 12.0) / 24.0), 0.0) + 1e-9
    res = {"n": daymask.sum(axis=0).reshape(nlat, nlon).astype("float64")}
    for name, model in models.items():
        p = np.zeros(ntime * ncell, "float64")
        p[day] = np.clip(model.predict(X), 0, 1.5) * tisr_day
        q, diag = constrained_rescale(p.reshape(ntime, ncell), tisr, target, kmax=KMAX, twilight_w=twilight_w)
        d = np.where(daymask, q - real, 0.0)
        res[name] = dict(sse=(d * d).sum(axis=0).reshape(nlat, nlon), sb=d.sum(axis=0).reshape(nlat, nlon), diag=diag)
    return res


def one_month(stamp, model_path):
    models = {"v1": joblib.load(V1_PATH)["model"], "v2": joblib.load(model_path)["model"]}
    out = {"stamp": stamp}
    for piece, (lat_sl, lon_sls) in PIECES.items():
        m = load_piece(stamp, lat_sl, lon_sls)
        out[piece] = piece_stats(m, models)
        out[piece + "_lat"], out[piece + "_lon"] = m["lat"], m["lon"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--months", default=""); ap.add_argument("--tag", default="rescaled_all")
    args = ap.parse_args()
    keep = {int(v) for v in args.months.split(",")} if args.months else None
    jobs = [(f"{y}{mth:02d}", f"models/model_v2/cv_folds/{fold}/model_kt_v2.joblib")
            for fold, yrs in FOLDS.items() for y in yrs for mth in range(1, 13)
            if f"{y}{mth:02d}" != "202508" and (keep is None or mth in keep)]
    if args.limit: jobs = jobs[:args.limit]
    log(f"{len(jobs)} months, {args.workers} workers, kmax={KMAX}")
    t = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky")(delayed(one_month)(*j) for j in jobs)
    log(f"done in {(time.time()-t)/60:.1f} min")
    arrays = {}
    for piece in PIECES:
        arrays[f"{piece}_lat"], arrays[f"{piece}_lon"] = res[0][piece + "_lat"], res[0][piece + "_lon"]
        arrays[f"{piece}_n"] = sum(r[piece]["n"] for r in res); arrays[f"{piece}_nmonths"] = len(res)
        for mod in ("v1", "v2"):
            for k in ("sse", "sb"):
                arrays[f"{piece}_{mod}_{k}"] = sum(r[piece][mod][k] for r in res)
    # verification of the constraint, worst case over every month / piece / model
    log("CONSTRAINT CHECK (worst case over all months, pieces, models):")
    dg = [r[p][m]["diag"] for r in res for p in PIECES for m in ("v1", "v2")]
    log(f"  max relative error of the monthly sum, ALL cells: {max(x['max_rel_sum_err'] for x in dg):.2e}")
    nc = sum(x['n_cells'] for x in dg)
    log(f"  cell-months where the ceiling had to be raised to keep the sum : {sum(x['n_ceiling_raised'] for x in dg)} of {nc:,}")
    log(f"  cell-months in polar twilight (analytic sun never up, real ssrd>0): {sum(x['n_twilight'] for x in dg)} of {nc:,}")
    log(f"  max kt after rescale (all other cells): {max(x['max_kt_normal'] for x in dg):.3f}   (ceiling {KMAX})")
    log(f"  max value at analytic night (all other cells): {max(x['night_max_normal'] for x in dg):.1e}")
    np.savez_compressed(OUT.format(tag=args.tag), **arrays)
    log(f"saved {OUT.format(tag=args.tag)}")


if __name__ == "__main__":
    sys.exit(main())
