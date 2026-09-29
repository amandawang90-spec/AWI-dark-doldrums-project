"""v4: same method as v1/v2/v3 (HistGradientBoostingRegressor on kt=ssrd/tisr, features T,tcc,hcc,mcc,lcc,mu, 250 iterations),
trained on ALL 12 months of 2015-2026 (not ONDJF-only like v3), using the properly-aggregated 3-hour-SUM ssrd/tsr
(DT=10800, matching DART's own accumulation window exactly -- see reconstruct_ssrd_v4.py), excluding rows south of
60S and above 3000 m (same mask as v3). Area-weighted only (rows weighted by cos(lat)) -- no "plain" variant this time.
3 year-block CV folds (for honest evaluation) + a final model on everything.

Usage: python3 train_v4_model.py [--rows 32000000] [--only fold1_test_2015-2018]
Output: models/model_v4/cv_folds/<fold>/model_kt_v4.joblib and models/model_v4/model_kt_v4.joblib"""

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

FOLDS = {"fold1_test_2015-2018": range(2015, 2019), "fold2_test_2019-2022": range(2019, 2023),
         "fold3_test_2023-2026": range(2023, 2027), "final": range(0)}
LAT_MIN, ELEV_MAX = -60.0, 3000
ROOT = "models/model_v4"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32_000_000)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    stamps_all = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz"))
    log(f"{len(stamps_all)} months in cache")

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
                    f"{out}/model_kt_v4.joblib")
        json.dump({"variant": "area", "months": "all 12 (2015-2026)", "train_years_excluded": sorted(test_years),
                   "n_months": len(stamps), "rows_used": int(len(y)), "mask": "lat >= -60 and elevation <= 3000 m",
                   "weights": "cos(lat)", "max_iter": 250, "n_iter_": int(model.n_iter_), "sklearn": sklearn.__version__,
                   "dt_seconds": DT, "note": "true 3-hour-sum ssrd/tsr (era5_3h_mean), DT=10800 matches DART exactly, "
                   "unlike v1/v2/v3 which used the 1-hour-sample proxy with DT=3600"},
                  open(f"{out}/training_config.json", "w"), indent=1)
        log(f"[{name}] fit {((time.time()-t)/60):.1f} min, n_iter_={model.n_iter_} -> {out}")


if __name__ == "__main__":
    main()
