"""Per-cell RMSE/bias maps for v5, PRE-rescale (raw kt_pred x tisr) AND POST-rescale
(after the exact monthly constrained_rescale), side by side -- the map-based
evaluation step that exists for v1/v2 (rmse_maps.py / rmse_maps_rescaled.py) but
was never built for v4/v5. Answers "is the model itself any good, and WHERE does
it struggle" before any Germany/SMARD-specific comparison.

Every month scored by the v5 CV-fold model that EXCLUDED that year from training
(fold1 excludes 2015-2018, fold2 excludes 2019-2022, fold3 excludes 2023-2026) --
genuinely held-out, matching the fold convention already used for v4 in
evaluate_v4_vs_all.py and for the Germany check in
scratch/compute_model_heldout_validation.py.

Pieces: Global (0.25deg thinned x4 -> 1deg), Europe, Korea, Germany (all three
boxes at full 0.25deg). Months: Sep-Mar (v5's own training window), years 2015-2026,
calendar-year grouping (Sep-Dec of year Y, Jan-Mar of year Y) matching
compute_germany_solar_cf_v5_sepmar.py's convention.

Usage: python3 rmse_maps_v5_prerescale.py [--workers 16] [--limit N]
"""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import argparse
import os
import sys
import time

import joblib
import numpy as np
from joblib import Parallel, delayed

from reconstruct_ssrd import DAY_MU, FINAL_FEATURES, SOLAR_CONSTANT, log
from reconstruct_ssrd_compare import load_piece
from evaluate_v2_cv import box_slices
from rescale import constrained_rescale

MODEL_PATH = "models/model_v5/cv_folds/{fold}/model_kt_v5.joblib"
FOLD_FOR_YEAR = {}
for y in range(2015, 2019): FOLD_FOR_YEAR[y] = "fold1_test_2015-2018"
for y in range(2019, 2023): FOLD_FOR_YEAR[y] = "fold2_test_2019-2022"
for y in range(2023, 2027): FOLD_FOR_YEAR[y] = "fold3_test_2023-2026"

MONTHS = (9, 10, 11, 12, 1, 2, 3)
YEARS = range(2015, 2027)
KMAX = 1.1
GLOBAL_STRIDE = 4
PIECES = {"Global": (slice(0, 721, GLOBAL_STRIDE), [slice(0, 1440, GLOBAL_STRIDE)]),
          "Europe": box_slices(35, 71, -25, 40), "Korea": box_slices(33, 43, 124, 131),
          "Germany": box_slices(47.3, 55.1, 5.9, 15.0)}
OUT = "models/model_v5/evaluation/v5_prerescale_maps_sepmar.npz"


def piece_stats(m, model):
    """Returns per-cell n/sr/srr (ground truth) and, for BOTH pre- and post-rescale,
    per-cell sse/sb -- the map-building statistics rmse_maps.py computed for v1/v2,
    never built for v4/v5."""
    ntime, (nlat, nlon) = m["ntime"], m["shape2d"]
    ncell = nlat * nlon
    tisr = m["tisr"].reshape(ntime, ncell).astype("float64")
    tisr_flat = m["tisr"].ravel()
    mu = tisr_flat / (m["dt"] * SOLAR_CONSTANT)
    day = mu > DAY_MU
    real = m["ssrd"].reshape(ntime, ncell).astype("float64")
    use = day.reshape(ntime, ncell)
    target = real.sum(axis=0)

    hours = (m["times"] - m["times"].astype("datetime64[D]")) / np.timedelta64(1, "h") - 0.5
    local = (hours[:, None] + np.tile(m["lon"], nlat)[None, :] / 15.0) % 24.0
    tw = np.maximum(np.cos(2 * np.pi * (local - 12.0) / 24.0), 0.0) + 1e-9

    feat = {"tcc": m["tcc"].ravel()[day], "hcc": m["hcc"].ravel()[day],
            "mcc": m["mcc"].ravel()[day], "lcc": m["lcc"].ravel()[day],
            "mu": mu[day], "T": np.clip(m["tsr"].ravel()[day] / tisr_flat[day], 0, 1.5)}
    X = np.column_stack([feat[c] for c in FINAL_FEATURES])
    kt_pred = np.clip(model.predict(X), 0, 1.5)
    p = np.zeros(ntime * ncell, "float64")
    p[day.ravel()] = kt_pred * tisr_flat[day.ravel()]
    p = p.reshape(ntime, ncell)
    q, diag = constrained_rescale(p, tisr, target, kmax=KMAX, twilight_w=tw)

    dp = np.where(use, p - real, 0.0)
    dq = np.where(use, q - real, 0.0)
    return dict(n=use.sum(axis=0).reshape(nlat, nlon).astype("float64"),
                sr=np.where(use, real, 0.0).sum(axis=0).reshape(nlat, nlon),
                srr=np.where(use, real * real, 0.0).sum(axis=0).reshape(nlat, nlon),
                pre_sse=(dp * dp).sum(axis=0).reshape(nlat, nlon), pre_sb=dp.sum(axis=0).reshape(nlat, nlon),
                post_sse=(dq * dq).sum(axis=0).reshape(nlat, nlon), post_sb=dq.sum(axis=0).reshape(nlat, nlon),
                diag=diag)


def one_month(stamp, fold):
    model = joblib.load(MODEL_PATH.format(fold=fold))["model"]
    out = {"stamp": stamp}
    for piece, (lat_sl, lon_sls) in PIECES.items():
        parts = [load_piece(stamp, lat_sl, ls, "new") for ls in lon_sls]
        m = dict(parts[0])
        for extra in parts[1:]:
            for k in ("tcc", "hcc", "mcc", "lcc", "tsr", "ssrd", "tisr"):
                m[k] = np.concatenate([m[k], extra[k]], axis=-1)
            m["lon"] = np.concatenate([m["lon"], extra["lon"]])
        m["ntime"], m["shape2d"] = m["tcc"].shape[0], m["tcc"].shape[1:]
        out[piece] = piece_stats(m, model)
        out[piece + "_lat"], out[piece + "_lon"] = m["lat"], m["lon"]
    return out


def month_available(y, mth):
    stamp = f"{y}{mth:02d}"
    return (os.path.exists(f"data/era5/era5_clouds_3h_{stamp}.nc") and
            os.path.exists(f"data/era5_3h_mean/era5_tsr_3hmean_{stamp}.nc") and
            os.path.exists(f"data/era5_3h_mean/era5_ssrd_3hmean_{stamp}.nc"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    jobs = [(f"{y}{mth:02d}", FOLD_FOR_YEAR[y]) for y in YEARS for mth in MONTHS if month_available(y, mth)]
    if args.limit:
        jobs = jobs[:args.limit]
    log(f"{len(jobs)} months, {args.workers} workers, model v5 (held-out by fold)")
    t = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky")(delayed(one_month)(*j) for j in jobs)
    log(f"done in {(time.time()-t)/60:.1f} min")

    arrays = {"nmonths": len(res)}
    for piece in PIECES:
        arrays[f"{piece}_lat"], arrays[f"{piece}_lon"] = res[0][piece + "_lat"], res[0][piece + "_lon"]
        for k in ("n", "sr", "srr", "pre_sse", "pre_sb", "post_sse", "post_sb"):
            arrays[f"{piece}_{k}"] = sum(r[piece][k] for r in res)

    dg = [r[p]["diag"] for r in res for p in PIECES]
    nc = sum(x["n_cells"] for x in dg)
    log("CONSTRAINT CHECK (worst case over all months, pieces):")
    log(f"  max relative error of the monthly sum, ALL cells: {max(x['max_rel_sum_err'] for x in dg):.2e}")
    log(f"  ceiling raised: {sum(x['n_ceiling_raised'] for x in dg)} / polar-twilight: {sum(x['n_twilight'] for x in dg)} of {nc:,} cell-months")

    os.makedirs("models/model_v5/evaluation", exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    log(f"saved {OUT}")


if __name__ == "__main__":
    sys.exit(main())
