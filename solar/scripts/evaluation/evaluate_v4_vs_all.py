"""Fair four-way comparison: v1, v2, v3area (old convention: 1-hour ERA5 sample every
3h, DT=3600) vs v4 (new convention: true 3-hour accumulated sum, DT=10800), all
evaluated against the SAME ground truth -- the true 3-hour-sum ssrd (era5_3h_mean)
-- and the same monthly rescale target, on the SAME held-out years (2015-2018,
2019-2022, 2023-2025; the 2015-2025 range common to every model's fold scheme --
v4 also has 2026 in its own training/fold split, but 2026 is excluded here since
v1/v2/v3 have no fold covering it).

Every month scored by fold models that never trained on that year. v3plain is
skipped here (v3area is production; v3plain was already compared to v2 earlier).

Usage: python3 evaluate_v4_vs_all.py [--workers 16] [--limit N] [--tag ondjf]
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
import xarray as xr
from joblib import Parallel, delayed

from reconstruct_ssrd import DAY_MU, FINAL_FEATURES, SOLAR_CONSTANT, log
from reconstruct_ssrd_compare import load_piece
from evaluate_v2_cv import FOLDS, V1_PATH, GLOBAL_STRIDE, box_slices
from rescale import constrained_rescale

OUT = "models/model_v4/evaluation/v4_eval_{tag}.npz"
KMAX = 1.1
LAT_MIN, ELEV_MAX = -60.0, 3000
PIECES = {"Global": (slice(0, 721, GLOBAL_STRIDE), [slice(0, 1440, GLOBAL_STRIDE)]),
          "Europe": box_slices(35, 71, -25, 40), "Korea": box_slices(33, 43, 124, 131)}
MODEL_PATHS = {"v2": "models/model_v2/cv_folds/{fold}/model_kt_v2.joblib",
               "v3area": "models/model_v3/area/cv_folds/{fold}/model_kt_v3.joblib",
               "v4": "models/model_v4/cv_folds/{fold}/model_kt_v4.joblib"}
V4_FOLD_MAP = {"fold1_test_2015-2018": "fold1_test_2015-2018", "fold2_test_2019-2022": "fold2_test_2019-2022",
               "fold3_test_2023-2025": "fold3_test_2023-2026"}   # v4's fold3 also excludes 2026; fine for scoring 2023-2025
NAMES = ["v1", "v2", "v3area", "v4"]
CONVENTION = {"v1": "old", "v2": "old", "v3area": "old", "v4": "new"}


def build_features_generic(m, features):
    tisr = m["tisr"].ravel()
    mu = tisr / (m["dt"] * SOLAR_CONSTANT)
    day = mu > DAY_MU
    feat = {
        "tcc": m["tcc"].ravel()[day], "hcc": m["hcc"].ravel()[day],
        "mcc": m["mcc"].ravel()[day], "lcc": m["lcc"].ravel()[day],
        "mu": mu[day], "T": np.clip(m["tsr"].ravel()[day] / tisr[day], 0, 1.5),
    }
    X = np.column_stack([feat[c] for c in features])
    return X, tisr[day], day


def valid_mask(m, lat_sl, lon_sls):
    z = xr.open_dataset("data/static/era5_geopotential_surface.nc")["z"].squeeze().values / 9.80665
    elev = np.concatenate([z[lat_sl][:, ls] for ls in lon_sls], axis=1)
    assert elev.shape == tuple(m["shape2d"])
    return (m["lat"][:, None] >= LAT_MIN) & (elev <= ELEV_MAX)


def piece_stats(m_old, m_new, models, valid):
    """m_new is the ground truth + rescale target (true 3h sum). m_old supplies
    features for the old-convention models; predictions from both are aligned
    onto m_new's time axis (identical except for the single dropped 201501
    boundary stamp)."""
    ntime = m_new["ntime"]; nlat, nlon = m_new["shape2d"]; ncell = nlat * nlon
    common = np.intersect1d(m_old["times"], m_new["times"])
    old_keep = np.isin(m_old["times"], common)
    new_keep = np.isin(m_new["times"], common)

    tisr_new = m_new["tisr"].reshape(m_new["ntime"], ncell)[new_keep].astype("float64")
    real = m_new["ssrd"].reshape(m_new["ntime"], ncell)[new_keep].astype("float64")
    ntime = real.shape[0]
    daymask = (tisr_new / (m_new["dt"] * SOLAR_CONSTANT)) > DAY_MU
    use = daymask & valid.reshape(1, ncell)
    target = real.sum(axis=0)

    times = common
    hours = (times - times.astype("datetime64[D]")) / np.timedelta64(1, "h") - 0.5
    local = (hours[:, None] + np.tile(m_new["lon"], nlat)[None, :] / 15.0) % 24.0
    tw = np.maximum(np.cos(2 * np.pi * (local - 12.0) / 24.0), 0.0) + 1e-9

    day_of_month = ((times - times.astype("datetime64[M]")) / np.timedelta64(1, "D")).astype(int)
    ndays = int(day_of_month.max()) + 1
    res = {"n": use.sum(axis=0).reshape(nlat, nlon).astype("float64"),
           "sr": np.where(use, real, 0.0).sum(axis=0).reshape(nlat, nlon),
           "srr": np.where(use, real * real, 0.0).sum(axis=0).reshape(nlat, nlon)}

    for name, model in models.items():
        conv = CONVENTION[name]
        m_src = m_new if conv == "new" else m_old
        src_keep = new_keep if conv == "new" else old_keep
        # rebuild features restricted to the common time rows
        m_sub = dict(m_src)
        for k in ("tcc", "hcc", "mcc", "lcc", "tsr", "ssrd", "tisr"):
            m_sub[k] = m_src[k][src_keep]
        X, tisr_day, day = build_features_generic(m_sub, FINAL_FEATURES)
        p = np.zeros(ntime * ncell, "float64")
        p[day] = np.clip(model.predict(X), 0, 1.5) * tisr_day
        q, diag = constrained_rescale(p.reshape(ntime, ncell), tisr_new, target, kmax=KMAX, twilight_w=tw)
        d = np.where(use, q - real, 0.0)
        real_day = np.zeros((ndays, ncell)); pred_day = np.zeros((ndays, ncell)); nvalid_day = np.zeros((ndays, ncell))
        np.add.at(real_day, day_of_month, np.where(use, real, 0.0))
        np.add.at(pred_day, day_of_month, np.where(use, q, 0.0))
        np.add.at(nvalid_day, day_of_month, use.astype("float64"))
        dd = np.where(nvalid_day > 0, pred_day - real_day, 0.0)
        ndaysused = (nvalid_day > 0)
        res[name] = dict(sse=(d * d).sum(axis=0).reshape(nlat, nlon), sb=d.sum(axis=0).reshape(nlat, nlon),
                         diag=diag, dsse=(dd * dd).sum(axis=0).reshape(nlat, nlon),
                         dsb=dd.sum(axis=0).reshape(nlat, nlon))
    res["dn"] = ndaysused.sum(axis=0).reshape(nlat, nlon).astype("float64")
    res["dsr"] = real_day.sum(axis=0).reshape(nlat, nlon)
    res["dsrr"] = (real_day * real_day * ndaysused).sum(axis=0).reshape(nlat, nlon)
    return res


def one_month(stamp, fold):
    models = {"v1": joblib.load(V1_PATH)["model"]}
    for k, pth in MODEL_PATHS.items():
        f = V4_FOLD_MAP[fold] if k == "v4" else fold
        models[k] = joblib.load(pth.format(fold=f))["model"]
    out = {"stamp": stamp}
    for piece, (lat_sl, lon_sls) in PIECES.items():
        m_old_parts = [load_piece(stamp, lat_sl, ls, "old") for ls in lon_sls]
        m_new_parts = [load_piece(stamp, lat_sl, ls, "new") for ls in lon_sls]
        def concat_parts(parts):
            m = dict(parts[0])
            for extra in parts[1:]:
                for k in ("tcc", "hcc", "mcc", "lcc", "tsr", "ssrd", "tisr"):
                    m[k] = np.concatenate([m[k], extra[k]], axis=-1)
                m["lon"] = np.concatenate([m["lon"], extra["lon"]])
            m["ntime"], m["shape2d"] = m["tcc"].shape[0], m["tcc"].shape[1:]
            return m
        m_old, m_new = concat_parts(m_old_parts), concat_parts(m_new_parts)
        out[piece] = piece_stats(m_old, m_new, models, valid_mask(m_new, lat_sl, lon_sls))
        out[piece + "_lat"], out[piece + "_lon"] = m_new["lat"], m_new["lon"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="ondjf")
    args = ap.parse_args()
    jobs = [(f"{y}{mth:02d}", fold) for fold, yrs in FOLDS.items() for y in yrs for mth in (10, 11, 12, 1, 2)]
    if args.limit: jobs = jobs[:args.limit]
    log(f"{len(jobs)} months, {args.workers} workers, models {NAMES}")
    t = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky")(delayed(one_month)(*j) for j in jobs)
    log(f"done in {(time.time()-t)/60:.1f} min")
    arrays = {}
    for grp, months in (("ondjf", (10, 11, 12, 1, 2)), ("djf", (12, 1, 2))):
        rs = [r for r in res if int(r["stamp"][4:]) in months]
        arrays[f"{grp}_nmonths"] = len(rs)
        for piece in PIECES:
            arrays[f"{piece}_lat"], arrays[f"{piece}_lon"] = res[0][piece + "_lat"], res[0][piece + "_lon"]
            arrays[f"{grp}_{piece}_n"] = sum(r[piece]["n"] for r in rs); arrays[f"{grp}_{piece}_sr"] = sum(r[piece]["sr"] for r in rs)
            arrays[f"{grp}_{piece}_srr"] = sum(r[piece]["srr"] for r in rs)
            arrays[f"{grp}_{piece}_dn"] = sum(r[piece]["dn"] for r in rs); arrays[f"{grp}_{piece}_dsr"] = sum(r[piece]["dsr"] for r in rs)
            arrays[f"{grp}_{piece}_dsrr"] = sum(r[piece]["dsrr"] for r in rs)
            for mod in NAMES:
                for k in ("sse", "sb", "dsse", "dsb"):
                    arrays[f"{grp}_{piece}_{mod}_{k}"] = sum(r[piece][mod][k] for r in rs)
    dg = [r[p][m]["diag"] for r in res for p in PIECES for m in NAMES]
    nc = sum(x["n_cells"] for x in dg)
    log("CONSTRAINT CHECK (worst case over all months, pieces, models):")
    log(f"  max relative error of the monthly sum, ALL cells: {max(x['max_rel_sum_err'] for x in dg):.2e}")
    log(f"  ceiling raised: {sum(x['n_ceiling_raised'] for x in dg)} / polar-twilight: {sum(x['n_twilight'] for x in dg)} of {nc:,} cell-months")
    os.makedirs("models/model_v4/evaluation", exist_ok=True)
    np.savez_compressed(OUT.format(tag=args.tag), **arrays)
    log(f"saved {OUT.format(tag=args.tag)}")


if __name__ == "__main__":
    import os
    sys.exit(main())
