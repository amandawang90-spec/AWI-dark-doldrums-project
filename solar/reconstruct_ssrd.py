"""Reconstruct 3-hourly surface downward shortwave (ssrd) from tsr + cloud
fractions, fit and validated on ERA5, for later application to TCo1279-DART,
which carries tsr/tcc/hcc/mcc/lcc 3-hourly but ssrd only as a monthly mean.

This version replaces ERA5's downloaded tisr with the analytic one from
solar_geometry.py throughout -- training, prediction, and un-normalising --
because DART will never have a tisr field to read. Using the analytic version
here, verified against ERA5's real tisr in check_solar_geometry.py (correlation
> 0.9999, error < 1%), makes this a faithful rehearsal of the DART pipeline
rather than an easier version of it.

METHOD
------
Target: the surface clearness index  kt = ssrd / tisr_analytic
Predictors:
    T    = tsr / tisr_analytic           TOA transmission -- the dominant one
    mu   = tisr_analytic / (3600*1361)   mean cos(zenith) over the window
    tcc, hcc, mcc, lcc                   cloud fractions
Model: HistGradientBoostingRegressor (beat linear regression by ~45% in RMSE
in the prior comparison: 40.2 vs 70.4 W/m2 averaged over 3 held-out months).

VALIDATION -- three stages, each gating the next
-------------------------------------------------------------------------------
1. GEOMETRY   analytic tisr vs ERA5's real tisr (check_solar_geometry.py, run
              separately -- this script assumes it already passed).
2. FIT        leave-one-month-out: train on 2 months, predict the 3rd. A random
              split would let the model see both August months in training and
              be tested on nearby days of the same month -- overstates skill.
3. CONSTRAIN  the out-of-fold 3-hourly reconstruction is aggregated to a
              monthly mean and RESCALED, per grid cell, so that mean EXACTLY
              equals the independently downloaded era5_monthly_ssrd file:
                  ssrd_final(t) = ssrd_pred(t) * [monthly_actual / monthly_pred]
              Multiplicative, not additive, because night is exactly zero and
              must stay zero -- an additive correction would inject false
              radiation into the night hours to hit the target mean. This is
              the same role DART's own (real) monthly ssrd will play: the
              trusted magnitude, with the model supplying only the within-month
              shape. The final 3-hourly R2/RMSE against ERA5's real ssrd is
              measured on this constrained field, so what it reports is shape
              skill alone -- magnitude is correct by construction.

Data is thinned by STRIDE in lat/lon before fitting -- full global 3-hourly
resolution across 3 months is ~750M rows, too large to hold in memory here, and
this is a per-pixel instantaneous relationship, not a spatial pattern to learn.

Usage: python3 reconstruct_ssrd.py [--stride N] [--gbm-rows N]
"""

import argparse
import time

import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from solar_geometry import toa_irradiance_accumulated, SOLAR_CONSTANT

DATA_DIR = "data/era5"
MONTHS = ["202508", "202603", "202608"]

# TWO settings that MUST change together when this script is pointed at DART
# instead of ERA5 -- both verified empirically, not assumed from a metadata
# label (DART's tsr is labelled online_operation="instant"/cell_methods=
# "time: point", which is misleading: a joule-per-square-metre quantity cannot
# be a genuine instant, and cross-checking the reduced-grid field's monthly
# mean against DART's own trusted monthly tsr file confirms it is really a
# 3-HOUR accumulation, not the 1-hour one ERA5 uses despite both being sampled
# every 3 hours and both carrying the identical units string "J m-2":
#   ERA5 tsr (this script)   : 1-hour accumulation  -> DT = 3600
#   TCo1279-DART tsr          : 3-hour accumulation  -> DT = 10800
# Getting this wrong doesn't crash anything -- it silently scales the
# transmission predictor T = tsr/tisr by (wrong DT)/(right DT) = 3x, which
# looks like unusually high transmission, not an error.
DT = 3600.0                       # ERA5 3-hourly tsr/ssrd/tisr accumulate over 1 h

DAY_MU = 0.02                     # daylight threshold on mean cos(zenith)
FINAL_FEATURES = ["T", "tcc", "hcc", "mcc", "lcc", "mu"]

# The monthly anchor file's accumulated J/m2 must be divided by the length of
# its OWN base accumulation window to get W/m2 -- and that window differs by
# source, despite every file's `units` attribute reading the same "J m-2":
#   ERA5 monthly ssrd (this script)     : mean of a 1-DAY accumulation  -> /86400
#     (GRIB_stepType=avgad; verified: matches the 3-hourly file's own /3600
#      mean to 0.01 W/m2, era5_monthly_ssrd_202508 = era5_ssrd_3h_202508 = 179.66)
#   TCo1279-DART monthly ssrd            : mean of a 3-HOUR accumulation -> /10800
#     (cell_methods="time: mean (interval: 3 h)", identical in 1950C and 2080C)
# Get this wrong and the rescaling step is silently off by 86400/10800 = 8x
# while still reporting a perfect residual, since that check only confirms
# consistency with whatever divisor was supplied, not that the divisor is
# right. Change MONTHLY_ANCHOR_DIVISOR (not the /86400 inline) when the
# anchor source changes.
MONTHLY_ANCHOR_DIVISOR = 86400.0   # ERA5 monthly convention; DART would be 10800.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_month(stamp, stride):
    """Load one month's predictors + target, thinned; tisr is ANALYTIC, not read."""
    sub = dict(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
    clouds = xr.open_dataset(f"{DATA_DIR}/era5_clouds_3h_{stamp}.nc").isel(**sub)
    tsr = xr.open_dataset(f"{DATA_DIR}/era5_tsr_3h_{stamp}.nc").isel(**sub)
    ssrd = xr.open_dataset(f"{DATA_DIR}/era5_ssrd_3h_{stamp}.nc").isel(**sub)   # ssrd only used
    monthly = xr.open_dataset(f"{DATA_DIR}/era5_monthly_ssrd_{stamp}.nc").isel(**sub)

    lat, lon = clouds.latitude.values, clouds.longitude.values
    # dt_seconds MUST equal the window tsr/ssrd actually accumulate over (DT
    # above), not the geometry function's own default -- passed explicitly so
    # the two can never silently drift apart.
    tisr = toa_irradiance_accumulated(lat, lon, clouds.valid_time.values, dt_seconds=DT)

    out = {
        "tcc": clouds.tcc.values.astype("float32"), "hcc": clouds.hcc.values.astype("float32"),
        "mcc": clouds.mcc.values.astype("float32"), "lcc": clouds.lcc.values.astype("float32"),
        "tsr": tsr.tsr.values.astype("float32"), "ssrd": ssrd.ssrd.values.astype("float32"),
        "tisr": tisr,
        "monthly_ssrd_wm2": (monthly.ssrd.squeeze().values / MONTHLY_ANCHOR_DIVISOR).astype("float32"),
        "lat": lat, "lon": lon,
    }
    out["ntime"], out["shape2d"] = out["tcc"].shape[0], out["tcc"].shape[1:]
    out["weight2d"] = np.cos(np.deg2rad(lat))[:, None] * np.ones(out["shape2d"], dtype="float32")
    for ds in (clouds, tsr, ssrd, monthly):
        ds.close()
    return out


def build_features(m):
    tisr = m["tisr"].ravel()
    mu = tisr / (DT * SOLAR_CONSTANT)
    day = mu > DAY_MU
    feat = {
        "tcc": m["tcc"].ravel()[day], "hcc": m["hcc"].ravel()[day],
        "mcc": m["mcc"].ravel()[day], "lcc": m["lcc"].ravel()[day],
        "mu": mu[day], "T": np.clip(m["tsr"].ravel()[day] / tisr[day], 0, 1.5),
    }
    kt = np.clip(m["ssrd"].ravel()[day] / tisr[day], 0, 1.5)
    return feat, kt, tisr[day], day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--gbm-rows", type=int, default=3_000_000)
    args = ap.parse_args()

    log(f"loading 3 months, stride={args.stride}, tisr ANALYTIC ...")
    months = {s: load_month(s, args.stride) for s in MONTHS}
    feats, kts, tisrs, days = {}, {}, {}, {}
    for s, m in months.items():
        feats[s], kts[s], tisrs[s], days[s] = build_features(m)
        log(f"  {s}: grid {m['shape2d']}, {len(kts[s]):,} daylight samples")

    cols = FINAL_FEATURES
    rng = np.random.default_rng(0)
    oof_kt_pred = {}

    log("\n" + "=" * 78)
    log("LEAVE-ONE-MONTH-OUT FIT (gradient boosting), analytic tisr throughout")
    log("=" * 78)
    for test in MONTHS:
        train = [s for s in MONTHS if s != test]
        Xtr = np.column_stack([np.concatenate([feats[s][c] for s in train]) for c in cols])
        ytr = np.concatenate([kts[s] for s in train])
        if len(ytr) > args.gbm_rows:
            idx = rng.choice(len(ytr), args.gbm_rows, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        Xte = np.column_stack([feats[test][c] for c in cols])
        model = HistGradientBoostingRegressor(max_iter=250, random_state=0).fit(Xtr, ytr)
        oof_kt_pred[test] = np.clip(model.predict(Xte), 0, 1.5)
        log(f"   fit done, held out {test} ({len(Xte):,} samples)")

    log("\n" + "=" * 78)
    log("3-HOURLY SKILL BEFORE THE MONTHLY CONSTRAINT (raw model output)")
    log("=" * 78)
    for s in MONTHS:
        pred_wm2 = oof_kt_pred[s] * tisrs[s] / DT
        true_wm2 = kts[s] * tisrs[s] / DT
        err = pred_wm2 - true_wm2
        rmse = np.sqrt(np.mean(err ** 2))
        bias = np.mean(err)
        r2 = 1 - np.sum(err ** 2) / np.sum((true_wm2 - true_wm2.mean()) ** 2)
        log(f"   {s}: R2={r2:.3f}  RMSE={rmse:5.1f} W/m2  bias={bias:+5.2f} W/m2")

    log("\n" + "=" * 78)
    log("MONTHLY CONSTRAINT: rescale each cell so the monthly mean matches EXACTLY")
    log("=" * 78)
    final_pred_wm2, final_true_wm2, final_weight = {}, {}, {}
    for s in MONTHS:
        m = months[s]
        raw = np.zeros(m["tcc"].size, dtype="float64")
        raw[days[s]] = oof_kt_pred[s] * tisrs[s]                       # J/m2, night = 0
        raw = raw.reshape((m["ntime"],) + m["shape2d"])
        monthly_pred_wm2 = raw.mean(axis=0) / DT                        # W/m2 per cell

        actual_wm2 = m["monthly_ssrd_wm2"]
        scale = np.ones_like(monthly_pred_wm2)
        valid = monthly_pred_wm2 > 0.1
        scale[valid] = actual_wm2[valid] / monthly_pred_wm2[valid]

        rescaled = raw * scale[None, :, :]                              # still J/m2, night = 0
        check_monthly = rescaled.mean(axis=0) / DT
        w = m["weight2d"]
        resid = np.average(np.abs(check_monthly - actual_wm2), weights=w)
        log(f"   {s}: post-rescale monthly mismatch (should be ~0): {resid:.2e} W/m2"
            f"   scale factor range [{scale[valid].min():.2f}, {scale[valid].max():.2f}]")

        pred_flat = (rescaled.reshape(-1)[days[s]] / DT)
        true_flat = kts[s] * tisrs[s] / DT
        wflat = np.broadcast_to(w, m["tcc"].shape).reshape(-1)[days[s]]
        final_pred_wm2[s], final_true_wm2[s], final_weight[s] = pred_flat, true_flat, wflat

    log("\n" + "=" * 78)
    log("FINAL 3-HOURLY SKILL, AFTER THE MONTHLY CONSTRAINT")
    log("(this is the number that answers 'how good is the method')")
    log("=" * 78)
    for s in MONTHS:
        p, t, w = final_pred_wm2[s], final_true_wm2[s], final_weight[s]
        err = p - t
        rmse = np.sqrt(np.average(err ** 2, weights=w))
        bias = np.average(err, weights=w)
        wmean_t = np.average(t, weights=w)
        r2 = 1 - np.sum(w * err ** 2) / np.sum(w * (t - wmean_t) ** 2)
        log(f"   {s}: R2={r2:.3f}  RMSE={rmse:5.1f} W/m2  bias={bias:+5.2f} W/m2")

    allp = np.concatenate(list(final_pred_wm2.values()))
    allt = np.concatenate(list(final_true_wm2.values()))
    allw = np.concatenate(list(final_weight.values()))
    err = allp - allt
    rmse = np.sqrt(np.average(err ** 2, weights=allw))
    bias = np.average(err, weights=allw)
    wmean_t = np.average(allt, weights=allw)
    r2 = 1 - np.sum(allw * err ** 2) / np.sum(allw * (allt - wmean_t) ** 2)
    log(f"\n   POOLED, all 3 months: R2={r2:.3f}  RMSE={rmse:5.1f} W/m2  bias={bias:+5.2f} W/m2")
    log("\ndone")


if __name__ == "__main__":
    main()
