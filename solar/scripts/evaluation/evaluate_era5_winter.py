"""Evaluate the EXISTING production model (models/model_v1/model_kt.joblib, trained on
Aug'25/Mar'26/Aug'26 -- no winter data) against real ERA5 ground truth for
December 2025 and January 2026, at full resolution.

This is the direct analogue of applying the production model to DART's
winter months: the model has never seen deep-winter ERA5 conditions during
training, exactly as when it's applied to DART's January. If a problem
shows up here, it's inherent to the model, not an ERA5->DART transfer
artifact -- and it isolates winter specifically, unlike the original
Aug/Mar/Aug leave-one-out folds which never tested true winter at all.

Usage: python3 evaluate_era5_winter.py [YYYYMM ...]
       (defaults to 202512 202601 if none given)
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import sys

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from reconstruct_ssrd import DT, FINAL_FEATURES, build_features, load_month, log

WINTER_MONTHS = sys.argv[1:] if len(sys.argv) > 1 else ["202512", "202601"]
REGIONS = {
    "Global":  None,
    "Europe":  (35, 71, -25, 40),
    "Germany": (47.3, 55.1, 5.9, 15.0),
    "Korea":   (33, 43, 124, 131),
}
FIG_DIR = "figures/era5"


def wstats(lat, actual, predicted, mask):
    a, p = actual[mask], predicted[mask]
    w = np.cos(np.deg2rad(lat[mask])) if lat.ndim == a.ndim else np.ones_like(a)
    wmean_a = np.average(a, weights=w)
    ss_res = np.average((a - p) ** 2, weights=w)
    ss_tot = np.average((a - wmean_a) ** 2, weights=w)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res)
    bias = np.average(p - a, weights=w)
    corr = np.corrcoef(a, p)[0, 1]
    return dict(n=mask.sum(), r2=r2, rmse=rmse, bias=bias, corr=corr)


def main():
    log("loading models/model_v1/model_kt.joblib (production model, no winter in training) ...")
    saved = joblib.load("models/model_v1/model_kt.joblib")
    model, features = saved["model"], saved["features"]

    results_pointwise = {}
    results_monthly = {}

    for stamp in WINTER_MONTHS:
        log(f"=== {stamp}, full resolution (stride=1) ===")
        m = load_month(stamp, stride=1)
        lat, lon = m["lat"], m["lon"]
        lon0 = np.where(lon > 180, lon - 360, lon)
        ntime, (nlat, nlon) = m["ntime"], m["shape2d"]

        feat, kt_actual, tisr_day, day = build_features(m)
        X = np.column_stack([feat[c] for c in features])
        kt_pred = np.clip(model.predict(X), 0, 1.5)
        log(f"  predicted {len(X):,} daylight rows")
        del X, feat

        ssrd_pred_flat = np.zeros(ntime * nlat * nlon, dtype="float32")
        ssrd_pred_flat[day] = (kt_pred * tisr_day).astype("float32")
        ssrd_pred = ssrd_pred_flat.reshape(ntime, nlat, nlon)
        ssrd_actual = m["ssrd"]

        lat2d = np.repeat(lat[:, None], nlon, axis=1)
        lon2d0 = np.repeat(lon0[None, :], nlat, axis=0)

        pred_flat = ssrd_pred.reshape(ntime, -1)
        act_flat = ssrd_actual.reshape(ntime, -1)
        lat_flat = np.repeat(lat2d.ravel()[None, :], ntime, axis=0)
        day2d = day.reshape(ntime, -1)

        for rname, box in REGIONS.items():
            if box is None:
                rmask2d = np.ones((nlat, nlon), dtype=bool)
            else:
                latlo, lathi, lonlo, lonhi = box
                rmask2d = (lat2d >= latlo) & (lat2d <= lathi) & (lon2d0 >= lonlo) & (lon2d0 <= lonhi)
            rmask_flat = np.repeat(rmask2d.ravel()[None, :], ntime, axis=0)
            sel = day2d & rmask_flat
            s = wstats(lat_flat, act_flat, pred_flat, sel)
            results_pointwise.setdefault(rname, {})[stamp] = s
            log(f"    [pointwise] {rname:8s} n={s['n']:>12,} R2={s['r2']:+.3f} "
                f"RMSE={s['rmse']/3600:6.1f} W/m2 bias={s['bias']/3600:+6.1f} W/m2 corr={s['corr']:.3f}")

        monthly_pred_wm2 = ssrd_pred.mean(axis=0) / DT
        monthly_actual_wm2 = m["monthly_ssrd_wm2"]
        err = monthly_pred_wm2 - monthly_actual_wm2

        for rname, box in REGIONS.items():
            if box is None:
                rmask2d = np.ones((nlat, nlon), dtype=bool)
            else:
                latlo, lathi, lonlo, lonhi = box
                rmask2d = (lat2d >= latlo) & (lat2d <= lathi) & (lon2d0 >= lonlo) & (lon2d0 <= lonhi)
            s = wstats(lat2d, monthly_actual_wm2, monthly_pred_wm2, rmask2d)
            results_monthly.setdefault(rname, {})[stamp] = s
            log(f"    [monthly]   {rname:8s} n={s['n']:>12,} R2={s['r2']:+.3f} RMSE={s['rmse']:6.2f} bias={s['bias']:+6.2f}")

        fig = plt.figure(figsize=(13, 6.5))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="gray", zorder=3)
        pcm = ax.pcolormesh(lon, lat, err, transform=ccrs.PlateCarree(),
                             cmap="RdBu_r", vmin=-40, vmax=40)
        cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, extend="both")
        cb.set_label(f"ERA5 winter error: predicted - real monthly ssrd [W/m2] -- {stamp}")
        ax.set_title(f"Production model on real ERA5 winter ground truth -- {stamp} (never trained on winter)")
        plt.tight_layout()
        outpath = f"{FIG_DIR}/era5_error_map_{stamp}.png"
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
        log(f"  saved {outpath}")
        del ssrd_pred, ssrd_actual, m, pred_flat, act_flat

    log("\n" + "=" * 100)
    log("SUMMARY -- pointwise (real 3-hourly ground truth), W/m2")
    log("=" * 100)
    for rname in REGIONS:
        row = results_pointwise[rname]
        cells = "  ".join(f"{s}: R2={row[s]['r2']:+.3f} RMSE={row[s]['rmse']/3600:5.1f} bias={row[s]['bias']/3600:+5.1f}" for s in WINTER_MONTHS)
        log(f"{rname:8s}  {cells}")

    log("\n" + "=" * 100)
    log("SUMMARY -- monthly-mean spatial, W/m2")
    log("=" * 100)
    for rname in REGIONS:
        row = results_monthly[rname]
        cells = "  ".join(f"{s}: R2={row[s]['r2']:+.3f} RMSE={row[s]['rmse']:5.2f} bias={row[s]['bias']:+5.2f}" for s in WINTER_MONTHS)
        log(f"{rname:8s}  {cells}")


if __name__ == "__main__":
    main()
