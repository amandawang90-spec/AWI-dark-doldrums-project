"""Fit ONE production model on all three ERA5 months combined, and save it.

reconstruct_ssrd.py deliberately never does this -- its leave-one-month-out
loop exists to test the method honestly, holding out each month in turn. That
test is done and passed (pooled R2=0.986 after the monthly constraint). A
production model applied to DART should use every bit of ERA5 signal
available, not sacrifice a third of it to a held-out fold that already served
its purpose. This script does that single fit and persists it, so the DART
reconstruction doesn't refit from scratch for every month.

Usage: python3 train_final_model.py [--stride N]
Output: model_kt.joblib  (HistGradientBoostingRegressor, features in this
        exact order: T, tcc, hcc, mcc, lcc, mu -- see FINAL_FEATURES)
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))


import argparse
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from reconstruct_ssrd import DT, FINAL_FEATURES, MONTHS, build_features, load_month, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args()

    log(f"loading all {len(MONTHS)} ERA5 months, stride={args.stride} ...")
    Xall, yall = [], []
    for s in MONTHS:
        m = load_month(s, args.stride)
        feat, kt, _, _ = build_features(m)
        Xall.append(np.column_stack([feat[c] for c in FINAL_FEATURES]))
        yall.append(kt)
        log(f"  {s}: {len(kt):,} daylight samples")

    X = np.concatenate(Xall)
    y = np.concatenate(yall)
    log(f"fitting HistGradientBoostingRegressor on {len(y):,} rows, "
        f"features={FINAL_FEATURES} ...")
    model = HistGradientBoostingRegressor(max_iter=250, random_state=0).fit(X, y)

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": FINAL_FEATURES, "dt_era5": DT}, "models/model_v1/model_kt.joblib")
    log("saved models/model_v1/model_kt.joblib")


if __name__ == "__main__":
    sys.exit(main())
