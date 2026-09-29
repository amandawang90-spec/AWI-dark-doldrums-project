"""v5: identical to v4 in every respect (HistGradientBoostingRegressor on kt=ssrd/tisr, features
T,tcc,hcc,mcc,lcc,mu, 250 iterations, true 3-hour-SUM ssrd/tsr with DT=10800, same v3-style exclusion
mask south of 60S/above 3000 m, area-weighted only, SAME year range 2015-2026) -- v4 vs v5 is a
controlled test of ONE variable: v4 trains on all 12 months, v5 trains on Sep-Mar only. Reuses v4's
existing training cache (it already covers every month needed).
3 year-block CV folds (2015-2018 / 2019-2022 / 2023-2026, matching v4's exact fold scheme) + a final model.

Usage: python3 train_v5_model.py [--rows 32000000] [--only fold1_test_2015-2018]
Output: models/model_v5/cv_folds/<fold>/model_kt_v5.joblib and models/model_v5/model_kt_v5.joblib"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))
_sys.path.insert(0, _os.path.join(_root, "scripts", "preprocessing"))

import argparse, glob, json, os, time
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from extract_all_v4 import CACHE
from reconstruct_ssrd_v4 import DT, FINAL_FEATURES

MONTHS = (9, 10, 11, 12, 1, 2, 3)
YEARS = range(2015, 2027)   # matches v4's exact range -- v4 and v5 must differ ONLY in month selection
FOLDS = {"fold1_test_2015-2018": range(2015, 2019), "fold2_test_2019-2022": range(2019, 2023),
         "fold3_test_2023-2026": range(2023, 2027), "final": range(0)}
LAT_MIN, ELEV_MAX = -60.0, 3000
ROOT = "models/model_v5"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32_000_000)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    all_cached = {os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz")}
    stamps_all = sorted(s for s in all_cached if int(s[:4]) in YEARS and int(s[4:6]) in MONTHS)
    log(f"{len(stamps_all)} Sep-Mar 2015-2025 months in cache")

    for name, test_years in FOLDS.items():
        if args.only and name != args.only:
            continue
        stamps = [s for s in stamps_all if int(s[:4]) not in test_years]
        parts = [np.load(f"{CACHE}/{s}.npz") for s in stamps]
        X = np.concatenate([p["X"] for p in parts]); y = np.concatenate([p["y"] for p in parts])
        lat = np.concatenate([p["lat"] for p in parts]); elev = np.concatenate([p["elev"] for p in parts])
        n0 = len(y)
        keep = (lat >= LAT_MIN) & (elev <= ELEV_MAX)
        X, y, lat = X[keep], y[keep], lat[keep]
        log(f"[{name}] {len(stamps)} months, {n0:,} rows -> {len(y):,} after excluding south of 60S and above 3000 m "
            f"(removed {100*(1-keep.mean()):.1f}%)")
        if len(y) > args.rows:
            sel = np.random.default_rng(0).choice(len(y), args.rows, replace=False)
            X, y, lat = X[sel], y[sel], lat[sel]
        log(f"[{name}] training rows: {len(y):,}")

        w = np.cos(np.deg2rad(lat)).astype("float64")
        t = time.time()
        model = HistGradientBoostingRegressor(max_iter=250, random_state=0).fit(X, y, sample_weight=w)
        out = ROOT + ("" if name == "final" else f"/cv_folds/{name}")
        os.makedirs(out, exist_ok=True)
        joblib.dump({"model": model, "features": FINAL_FEATURES, "dt_era5": DT, "train_months": stamps, "variant": "area"},
                    f"{out}/model_kt_v5.joblib")
        json.dump({"variant": "area", "months": "Sep-Mar (extended winter), 2015-2025 only", "train_years_excluded": sorted(test_years),
                   "n_months": len(stamps), "rows_used": int(len(y)), "mask": "lat >= -60 and elevation <= 3000 m",
                   "weights": "cos(lat)", "max_iter": 250, "n_iter_": int(model.n_iter_), "sklearn": sklearn.__version__,
                   "dt_seconds": DT, "note": "true 3-hour-sum ssrd/tsr (era5_3h_mean), DT=10800; reuses v4's cache "
                   "restricted to Sep-Mar 2015-2025"},
                  open(f"{out}/training_config.json", "w"), indent=1)
        log(f"[{name}] fit {((time.time()-t)/60):.1f} min, n_iter_={model.n_iter_} -> {out}")


if __name__ == "__main__":
    main()
