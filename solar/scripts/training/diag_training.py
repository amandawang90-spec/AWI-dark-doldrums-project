"""Why did v2 (much more data) not beat v1?  Three tests on ONE common held-out set:
  1. learning curve   (same params, 1M -> 32M rows)         -> is more data still helping?
  2. train vs held-out error of the biggest baseline         -> overfitting or information limit?
  3. area / energy row weights, and a bigger model           -> is the training objective or capacity the issue?
Held-out set: Jan/Apr/Jul/Oct of 2023 and 2024 (never seen by v1 or by the v2 fold-3 model).
Metrics are AREA-weighted (cos lat) on daylight rows, in W/m2 (= error in kt x tisr)."""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import json, time
import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from reconstruct_ssrd import FINAL_FEATURES, build_features, load_month, log

TRAIN = [f"{y}{m:02d}" for y in (2016, 2018, 2020, 2022) for m in (1, 4, 7, 10)]
TEST = [f"{y}{m:02d}" for y in (2023, 2024) for m in (1, 4, 7, 10)]
BOXES = {"Global": None, "Europe": (35, 71, -25, 40), "Germany": (47.3, 55.1, 5.9, 15.0), "Korea": (33, 43, 124, 131)}


def extract(stamp):
    m = load_month(stamp, 8)
    feat, kt, _, day = build_features(m)
    nlat, nlon = m["shape2d"]
    cell = np.nonzero(day)[0] % (nlat * nlon)
    X = np.column_stack([feat[c] for c in FINAL_FEATURES]).astype("float32")
    return X, kt.astype("float32"), m["lat"][cell // nlon].astype("float32"), m["lon"][cell % nlon].astype("float32")


def load(stamps):
    parts = Parallel(n_jobs=8, backend="loky")(delayed(extract)(s) for s in stamps)
    return [np.concatenate([p[i] for p in parts]) for i in range(4)]


def score(model, X, y, lat, lon):
    p = np.clip(model.predict(X), 0, 1.5)
    tis = X[:, 5] * 1361.0
    err = (p - y) * tis
    w = np.cos(np.deg2rad(lat)).astype("float64")
    lon0 = np.where(lon > 180, lon - 360, lon)
    out = {"kt_rmse": float(np.sqrt(np.mean((p - y) ** 2)))}
    for r, b in BOXES.items():
        k = np.ones(len(y), bool) if b is None else (lat >= b[0]) & (lat <= b[1]) & (lon0 >= b[2]) & (lon0 <= b[3])
        W = w[k].sum()
        out[f"{r}_bias"] = float((w[k] * err[k]).sum() / W)
        out[f"{r}_rmse"] = float(np.sqrt((w[k] * err[k] ** 2).sum() / W))
    return out


def fit(X, y, w=None, **kw):
    t = time.time()
    m = HistGradientBoostingRegressor(random_state=0, **kw).fit(X, y, sample_weight=w)
    return m, time.time() - t


def main():
    log("extracting train / test months (stride 8) ...")
    Xtr, ytr, latt, lont = load(TRAIN)
    Xte, yte, late, lone = load(TEST)
    log(f"train {len(ytr):,} rows, test {len(yte):,} rows")
    rng = np.random.default_rng(0)
    res = {}

    def run(name, model, secs=None):
        r = score(model, Xte, yte, late, lone); r["n_iter"] = int(model.n_iter_)
        if secs is not None: r["fit_min"] = round(secs / 60, 1)
        res[name] = r
        log(f"{name:34s} kt_rmse {r['kt_rmse']:.4f} | Global bias {r['Global_bias']:+5.1f} rmse {r['Global_rmse']:5.1f} | "
            f"Europe {r['Europe_bias']:+5.1f}/{r['Europe_rmse']:5.1f} | Germany {r['Germany_bias']:+5.1f}/{r['Germany_rmse']:5.1f} | "
            f"Korea {r['Korea_bias']:+5.1f}/{r['Korea_rmse']:5.1f} | iters {r['n_iter']}")

    run("REF v1 (3 months, stride16)", joblib.load("models/model_v1/model_kt.joblib")["model"])
    run("REF v2 fold3 (2015-2022, 190M rows)", joblib.load("models/model_v2/cv_folds/fold3_test_2023-2025/model_kt_v2.joblib")["model"])

    # 1. learning curve, identical parameters
    for n in (1_000_000, 4_000_000, 16_000_000, len(ytr)):
        sel = rng.choice(len(ytr), n, replace=False) if n < len(ytr) else slice(None)
        m, s = fit(Xtr[sel], ytr[sel], max_iter=250)
        run(f"LC baseline {n//1_000_000}M rows", m, s)
        if n == len(ytr):
            base = m
    # 2. overfitting check on the biggest baseline
    k = rng.choice(len(ytr), 4_000_000, replace=False)
    res["train_vs_test_kt_rmse"] = {"train": float(np.sqrt(np.mean((np.clip(base.predict(Xtr[k]), 0, 1.5) - ytr[k]) ** 2))),
                                    "test": res["LC baseline 32M rows"]["kt_rmse"]}
    log(f"kt RMSE  train {res['train_vs_test_kt_rmse']['train']:.4f}  vs  held-out {res['train_vs_test_kt_rmse']['test']:.4f}")
    # 3. objective and capacity
    wa = np.cos(np.deg2rad(latt)).astype("float64")
    m, s = fit(Xtr, ytr, wa, max_iter=250); run("WEIGHT area", m, s)
    m, s = fit(Xtr, ytr, wa * Xtr[:, 5], max_iter=250); run("WEIGHT area x sunlight", m, s)
    m, s = fit(Xtr, ytr, max_iter=800, max_leaf_nodes=127); run("CAPACITY 800 it, 127 leaves", m, s)
    m, s = fit(Xtr, ytr, wa * Xtr[:, 5], max_iter=800, max_leaf_nodes=127); run("CAPACITY + area x sunlight", m, s)
    json.dump(res, open("models/model_v2/diagnostics/diag_training.json", "w"), indent=1)
    log("saved diag_training.json")


if __name__ == "__main__":
    main()
