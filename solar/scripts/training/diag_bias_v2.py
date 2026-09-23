
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import joblib, numpy as np
v1 = joblib.load("models/model_v1/model_kt.joblib")["model"]
v2 = joblib.load("models/model_v2/model_kt_v2.joblib")["model"]
stamps = [f"{y}{m:02d}" for y in (2018, 2021) for m in (1, 4, 7, 10)]
X = np.concatenate([np.load(f"data/training_cache_stride8/{s}.npz")["X"] for s in stamps])
y = np.concatenate([np.load(f"data/training_cache_stride8/{s}.npz")["y"] for s in stamps])
tisr_wm2 = X[:, 5] * 1361.0                      # mu * S0  -> W/m2-equivalent of tisr
print(f"{len(y):,} rows (training-style sampling: every 8th grid point, no area weighting)")
print(f"mean kt (real) = {y.mean():.4f}   mean 'real ssrd' = {(y*tisr_wm2).mean():.2f} W/m2")
for name, m in (("v1", v1), ("v2", v2)):
    p = np.clip(m.predict(X), 0, 1.5)
    print(f"{name}: mean kt pred = {p.mean():.4f}  (kt bias {p.mean()-y.mean():+.4f})   "
          f"energy bias = {((p-y)*tisr_wm2).mean():+.2f} W/m2   rmse_kt = {np.sqrt(((p-y)**2).mean()):.4f}")
