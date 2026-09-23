"""Per-cell RMSE of the v1 / v2 models against REAL ERA5 3-hourly ssrd (held-out years).

Per grid cell, over all daylight 3-hourly steps of all held-out months:
    RMSE = sqrt( mean( (pred - real)^2 ) )  in W/m2   (also bias, and the RMSE of the monthly means)
Global at 1 degree, Europe and Korea boxes at full 0.25 degree.  v2 = the CV fold model that never
saw the month; v1 = models/model_v1/model_kt.joblib; 202508 skipped for both (v1 trained on it).

Usage: python3 rmse_maps.py [--workers 16] [--limit N]
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
from reconstruct_ssrd import DT, FINAL_FEATURES, build_features, log
from evaluate_v2_cv import FOLDS, V1_PATH, GLOBAL_STRIDE, box_slices, load_piece

OUT = "models/model_v2/evaluation/rmse_cells.npz"
PIECES = {"Global": (slice(0, 721, GLOBAL_STRIDE), [slice(0, 1440, GLOBAL_STRIDE)]),
          "Europe": box_slices(35, 71, -25, 40), "Korea": box_slices(33, 43, 124, 131)}


def cell_stats(m, models):
    ntime = m["ntime"]; nlat, nlon = m["shape2d"]; ncell = nlat * nlon
    feat, _, tisr_day, day = build_features(m)
    X = np.column_stack([feat[c] for c in FINAL_FEATURES])
    cell = np.nonzero(day)[0] % ncell
    act = m["ssrd"].ravel()[day].astype("float64")
    res = {"n": np.bincount(cell, minlength=ncell).reshape(nlat, nlon).astype("float64")}
    for name, model in models.items():
        p = (np.clip(model.predict(X), 0, 1.5) * tisr_day).astype("float64")
        d = p - act
        full = np.zeros(ntime * ncell, "float32"); full[day] = p
        em = full.reshape(ntime, nlat, nlon).mean(0).astype("float64") / DT - m["monthly_ssrd_wm2"]
        res[name] = dict(sse=np.bincount(cell, weights=d * d, minlength=ncell).reshape(nlat, nlon),
                         sb=np.bincount(cell, weights=d, minlength=ncell).reshape(nlat, nlon),
                         em2=em ** 2, em=em)
    return res


def one_month(stamp, model_path):
    models = {"v1": joblib.load(V1_PATH)["model"], "v2": joblib.load(model_path)["model"]}
    out = {"stamp": stamp}
    for piece, (lat_sl, lon_sls) in PIECES.items():
        m = load_piece(stamp, lat_sl, lon_sls)
        out[piece] = cell_stats(m, models)
        out[piece + "_lat"], out[piece + "_lon"] = m["lat"], m["lon"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--months", default="", help="calendar months to keep, e.g. 12,1,2")
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()
    jobs = [(f"{y}{mth:02d}", f"models/model_v2/cv_folds/{fold}/model_kt_v2.joblib")
            for fold, yrs in FOLDS.items() for y in yrs for mth in range(1, 13) if f"{y}{mth:02d}" != "202508" and (not args.months or mth in {int(v) for v in args.months.split(",")})]
    if args.limit: jobs = jobs[:args.limit]
    log(f"{len(jobs)} months, {args.workers} workers")
    t = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky")(delayed(one_month)(*j) for j in jobs)
    log(f"done in {(time.time()-t)/60:.1f} min")
    arrays = {}
    for piece in PIECES:
        arrays[f"{piece}_lat"], arrays[f"{piece}_lon"] = res[0][piece + "_lat"], res[0][piece + "_lon"]
        arrays[f"{piece}_n"] = sum(r[piece]["n"] for r in res)
        arrays[f"{piece}_nmonths"] = len(res)
        for mod in ("v1", "v2"):
            for k in ("sse", "sb", "em2", "em"):
                arrays[f"{piece}_{mod}_{k}"] = sum(r[piece][mod][k] for r in res)
    out = OUT.replace("rmse_cells.npz", f"rmse_cells_{args.tag}.npz")
    np.savez_compressed(out, **arrays)
    log(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
