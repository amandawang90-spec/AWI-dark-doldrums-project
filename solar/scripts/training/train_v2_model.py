"""Train the v2 model: same method as v1 (HistGradientBoostingRegressor on
kt = ssrd/tisr, features T,tcc,hcc,mcc,lcc,mu, same hyperparameters, no fal),
but on many more years of ERA5.  Only the amount of data changes vs v1.

Per-month feature extraction is cached (data/training_cache_stride8/<stamp>.npz)
so the job is resumable and a later refit on more years only extracts new months.

Usage: python3 train_v2_model.py --years 2016-2022 [--stride 8] [--workers 8]
                                 [--limit N] [--cache-only]
Output: models/model_v2/model_kt_v2.joblib + training_config.json
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

from reconstruct_ssrd import DT, FINAL_FEATURES, build_features, load_month, log

OUT_DIR = "models/model_v2"


def cache_month(stamp, stride, cache_dir):
    path = f"{cache_dir}/{stamp}.npz"
    if os.path.exists(path):
        return stamp, path, None
    t0 = time.time()
    m = load_month(stamp, stride)
    feat, kt, _, _ = build_features(m)
    X = np.column_stack([feat[c] for c in FINAL_FEATURES]).astype("float32")
    tmp = f"{cache_dir}/{stamp}.tmp.npz"
    np.savez(tmp, X=X, y=kt.astype("float32"))
    os.replace(tmp, path)
    return stamp, path, (len(kt), time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", required=True, help="e.g. 2016-2022")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--exclude", default="", help="years to hold out, e.g. 2015-2018 or 2019,2021")
    ap.add_argument("--limit", type=int, default=0, help="only first N months (testing)")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    y0, y1 = (int(v) for v in args.years.split("-"))
    stamps = [f"{y}{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13)]
    excluded = set()
    for part in filter(None, args.exclude.split(",")):
        a, _, b = part.partition("-")
        excluded |= set(range(int(a), int(b or a) + 1))
    stamps = [s for s in stamps if int(s[:4]) not in excluded]
    if args.limit:
        stamps = stamps[: args.limit]
    cache_dir = f"data/training_cache_stride{args.stride}"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    log(f"extracting features for {len(stamps)} months, stride={args.stride}, "
        f"workers={args.workers} (cached months are skipped)")
    t0 = time.time()
    results = Parallel(n_jobs=args.workers, backend="loky", verbose=0)(
        delayed(cache_month)(s, args.stride, cache_dir) for s in stamps)
    paths, failed = {}, []
    for i, r in enumerate(results, 1):
        stamp, path, info = r
        paths[stamp] = path
        if info:
            log(f"  [{i}/{len(stamps)}] {stamp}: {info[0]:,} rows in {info[1]:.0f}s")
    log(f"feature extraction done in {(time.time()-t0)/60:.1f} min")
    if args.cache_only:
        return 0

    log("loading cached arrays ...")
    Xs, ys = [], []
    for s in stamps:
        d = np.load(paths[s])
        Xs.append(d["X"]); ys.append(d["y"])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    del Xs, ys
    log(f"training set: {len(y):,} rows x {X.shape[1]} features {FINAL_FEATURES}")

    t0 = time.time()
    model = HistGradientBoostingRegressor(max_iter=args.max_iter, random_state=0).fit(X, y)
    log(f"fit done in {(time.time()-t0)/60:.1f} min; n_iter_={model.n_iter_}")

    out_model = f"{args.out}/model_kt_v2.joblib"
    joblib.dump({"model": model, "features": FINAL_FEATURES, "dt_era5": DT,
                 "train_months": stamps, "stride": args.stride}, out_model)
    cfg = {
        "description": "v2: same method as v1 (HistGradientBoostingRegressor, kt=ssrd/tisr), more data",
        "features": FINAL_FEATURES, "uses_fal": False,
        "train_years": args.years, "held_out_years": sorted(excluded), "n_months": len(stamps), "stride": args.stride,
        "n_rows": int(len(y)), "hyperparameters": {"max_iter": args.max_iter, "random_state": 0,
                                                    "other": "sklearn defaults, identical to v1"},
        "n_iter_used": int(model.n_iter_), "sklearn_version": sklearn.__version__,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "v1_for_comparison": "models/model_v1/model_kt.joblib (ERA5 Aug2025, Mar2026, Aug2026; stride 16)",
    }
    with open(f"{args.out}/training_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log(f"saved {out_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
