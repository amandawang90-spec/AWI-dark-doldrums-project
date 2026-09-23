"""Evaluate the solar-reconstruction method against ERA5's own REAL 3-hourly
ssrd (genuine ground truth, unlike DART), broken down by region, to check
whether the Germany/Korea weaknesses found on DART already show up on ERA5
itself -- i.e. whether it's a feature-set limitation or an ERA5->DART
transfer artifact.

Methodology: proper leave-one-month-out (same as the original ERA5
validation) -- for each of the 3 months, fit on the OTHER two (stride=16,
matching the original training setup) and evaluate on the held-out month at
FULL resolution (stride=1, no thinning) so Germany/Korea aren't starved of
points the way training necessarily is.

Reports, per region (Global/Europe/Germany/Korea) and per month:
  - pointwise (per cell, per 3h timestep) R2/RMSE/bias/corr against REAL ssrd
  - monthly-mean spatial R2 (same statistic used for the DART evaluation,
    for direct comparability)
  - a monthly-mean absolute-error world map, same style as the DART maps

Usage: python3 evaluate_era5_regional.py
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.ensemble import HistGradientBoostingRegressor

from reconstruct_ssrd import DT, FINAL_FEATURES, MONTHS, build_features, load_month, log

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
    results_pointwise = {}
    results_monthly = {}

    for test_stamp in MONTHS:
        train_stamps = [s for s in MONTHS if s != test_stamp]
        log(f"=== held-out month {test_stamp}, training on {train_stamps} (stride=16) ===")

        Xtr, ytr = [], []
        for s in train_stamps:
            m = load_month(s, stride=16)
            feat, kt, _, _ = build_features(m)
            Xtr.append(np.column_stack([feat[c] for c in FINAL_FEATURES]))
            ytr.append(kt)
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        model = HistGradientBoostingRegressor(max_iter=250, random_state=0).fit(Xtr, ytr)
        log(f"  fold model fit on {len(ytr):,} rows")
        del Xtr, ytr

        log(f"  loading held-out {test_stamp} at FULL resolution (stride=1) ...")
        m = load_month(test_stamp, stride=1)
        lat, lon = m["lat"], m["lon"]
        lon0 = np.where(lon > 180, lon - 360, lon)
        ntime, (nlat, nlon) = m["ntime"], m["shape2d"]

        feat, kt_actual, tisr_day, day = build_features(m)
        X = np.column_stack([feat[c] for c in FINAL_FEATURES])
        kt_pred = np.clip(model.predict(X), 0, 1.5)
        log(f"  predicted {len(X):,} daylight rows")
        del X, feat

        ssrd_pred_flat = np.zeros(ntime * nlat * nlon, dtype="float32")
        ssrd_pred_flat[day] = (kt_pred * tisr_day).astype("float32")
        ssrd_pred = ssrd_pred_flat.reshape(ntime, nlat, nlon)
        ssrd_actual = m["ssrd"]  # REAL ERA5 3-hourly ssrd, J/m2 (1h accumulation basis)

        lat2d = np.repeat(lat[:, None], nlon, axis=1)
        lon2d0 = np.repeat(lon0[None, :], nlat, axis=0)

        # ---- pointwise (per cell, per timestep), REAL ground truth ----
        pred_flat = ssrd_pred.reshape(ntime, -1)
        act_flat = ssrd_actual.reshape(ntime, -1)
        lat_flat = np.repeat(lat2d.ravel()[None, :], ntime, axis=0)
        lon_flat = np.repeat(lon2d0.ravel()[None, :], ntime, axis=0)
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
            results_pointwise.setdefault(rname, {})[test_stamp] = s
            log(f"    [pointwise] {rname:8s} n={s['n']:>12,} R2={s['r2']:+.3f} RMSE={s['rmse']:6.1f} "
                f"bias={s['bias']:+6.1f} corr={s['corr']:.3f}")

        # ---- monthly-mean spatial (comparable to DART's evaluation) ----
        # ssrd/tsr each accumulate over 1h (DT=3600) even though sampled every 3h;
        # time-averaging those per-slot values and dividing by DT gives W/m2,
        # exactly matching reconstruct_ssrd.py's own convention.
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
            results_monthly.setdefault(rname, {})[test_stamp] = s
            log(f"    [monthly]   {rname:8s} n={s['n']:>12,} R2={s['r2']:+.3f} RMSE={s['rmse']:6.2f} bias={s['bias']:+6.2f}")

        # ---- map ----
        fig = plt.figure(figsize=(13, 6.5))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="gray", zorder=3)
        pcm = ax.pcolormesh(lon, lat, err, transform=ccrs.PlateCarree(),
                             cmap="RdBu_r", vmin=-40, vmax=40)
        cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, extend="both")
        cb.set_label(f"ERA5 held-out error: predicted - real monthly ssrd [W/m2] -- {test_stamp}")
        ax.set_title(f"ERA5 leave-one-month-out error -- {test_stamp} (held out, full resolution)")
        plt.tight_layout()
        outpath = f"{FIG_DIR}/era5_error_map_{test_stamp}.png"
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
        log(f"  saved {outpath}")
        del ssrd_pred, ssrd_actual, m, pred_flat, act_flat

    log("\n" + "=" * 100)
    log("SUMMARY -- pointwise (real 3-hourly ground truth), R2 / RMSE / bias")
    log("=" * 100)
    for rname in REGIONS:
        row = results_pointwise[rname]
        cells = "  ".join(f"{s:>8s}: R2={row[s]['r2']:+.3f} RMSE={row[s]['rmse']:5.1f} bias={row[s]['bias']:+5.1f}" for s in MONTHS)
        log(f"{rname:8s}  {cells}")

    log("\n" + "=" * 100)
    log("SUMMARY -- monthly-mean spatial (comparable to the DART evaluation), R2 / RMSE / bias")
    log("=" * 100)
    for rname in REGIONS:
        row = results_monthly[rname]
        cells = "  ".join(f"{s:>8s}: R2={row[s]['r2']:+.3f} RMSE={row[s]['rmse']:5.2f} bias={row[s]['bias']:+5.2f}" for s in MONTHS)
        log(f"{rname:8s}  {cells}")


if __name__ == "__main__":
    main()
