"""v3: same method as v1/v2 (HistGradientBoostingRegressor on kt=ssrd/tisr, features T,tcc,hcc,mcc,lcc,mu, 250 iterations),
trained ONLY on Oct-Feb (ONDJF) months of 2015-2025, excluding rows south of 60S and above 3000 m.
Two variants trained on IDENTICAL rows:  plain = unweighted (as v1/v2),  area = rows weighted by cos(lat).
3 year-block CV folds (for honest evaluation) + a final model on everything.

Usage: python3 train_v3_model.py [--rows 32000000] [--only-final]
Output: models/model_v3/<variant>/cv_folds/<fold>/model_kt_v3.joblib and models/model_v3/<variant>/model_kt_v3.joblib"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import argparse, json, os, time
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from extract_ondjf import CACHE, STAMPS
from reconstruct_ssrd import DT, FINAL_FEATURES, log

FOLDS = {"fold1_test_2015-2018": range(2015, 2019), "fold2_test_2019-2022": range(2019, 2023),
         "fold3_test_2023-2025": range(2023, 2026), "final": range(0)}
LAT_MIN, ELEV_MAX = -60.0, 3000
ROOT = "models/model_v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32_000_000)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    for name, test_years in FOLDS.items():
        if args.only and name != args.only:
            continue
        stamps = [s for s in STAMPS if int(s[:4]) not in test_years]
        parts = [np.load(f"{CACHE}/{s}.npz") for s in stamps]
        X = np.concatenate([p["X"] for p in parts]); y = np.concatenate([p["y"] for p in parts])
        lat = np.concatenate([p["lat"] for p in parts]); elev = np.concatenate([p["elev"] for p in parts])
        n0 = len(y)
        keep = (lat >= LAT_MIN) & (elev <= ELEV_MAX)
        X, y, lat = X[keep], y[keep], lat[keep]
        log(f"[{name}] {len(stamps)} ONDJF months, {n0:,} rows -> {len(y):,} after excluding south of 60S and above 3000 m "
            f"(removed {100*(1-keep.mean()):.1f}%)")
        if len(y) > args.rows:
            sel = np.random.default_rng(0).choice(len(y), args.rows, replace=False)
            X, y, lat = X[sel], y[sel], lat[sel]
        log(f"[{name}] training rows: {len(y):,}")
        for variant in ("plain", "area"):
            w = np.cos(np.deg2rad(lat)).astype("float64") if variant == "area" else None
            t = time.time()
            model = HistGradientBoostingRegressor(max_iter=250, random_state=0).fit(X, y, sample_weight=w)
            out = f"{ROOT}/{variant}" + ("" if name == "final" else f"/cv_folds/{name}")
            os.makedirs(out, exist_ok=True)
            joblib.dump({"model": model, "features": FINAL_FEATURES, "dt_era5": DT, "train_months": stamps, "variant": variant}, f"{out}/model_kt_v3.joblib")
            json.dump({"variant": variant, "months": "Oct-Feb (ONDJF)", "train_years_excluded": sorted(test_years), "n_months": len(stamps),
                       "rows_used": int(len(y)), "mask": "lat >= -60 and elevation <= 3000 m", "weights": "cos(lat)" if w is not None else "none",
                       "max_iter": 250, "n_iter_": int(model.n_iter_), "sklearn": sklearn.__version__,
                       "note": "v1/v2 hyperparameters; rows randomly subsampled (learning curve is flat beyond ~1M rows)"},
                      open(f"{out}/training_config.json", "w"), indent=1)
            log(f"[{name}] {variant}: fit {((time.time()-t)/60):.1f} min, n_iter_={model.n_iter_} -> {out}")


if __name__ == "__main__":
    main()
