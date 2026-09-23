"""
Log law, regression, and power law, all reconstructing 100m wind from 10m wind,
evaluated on the CONTINUOUS 3-hourly winter time series (Oct2025-Feb2026) rather
than single instantaneous snapshots -- gives far more samples per region/regime,
especially in the low-wind (dark-doldrum) band.

Regression: fit per calendar month on 80% of 10x10-deg spatial blocks (same split
convention as compare_loglaw_vs_regression.py), scored on the held-out 20%, pooled
across ALL timesteps in that month.
Log law: only computed for months where a matching z0 file has finished downloading
(skipped otherwise, reported as such).
Power law: fixed alpha=0.2 land / 0.14 sea, no fitting needed.
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import os
import json
import numpy as np
import xarray as xr

MONTHS = [
    ("Oct2025", "202510"),
    ("Nov2025", "202511"),
    ("Dec2025", "202512"),
    ("Jan2026", "202601"),
    ("Feb2026", "202602"),
]
LSM_FILE = "data/era5/era5_lsm_20260830_0000.nc"
ALPHA_LAND, ALPHA_SEA = 0.20, 0.14
LOW_WIND_THRESH = 3.0
BLOCK_DEG = 10

lsm2d = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values
land2d, ocean2d = lsm2d >= 0.5, lsm2d < 0.5


def block_ids(lat, lon):
    lat_block = (lat // BLOCK_DEG).astype(int)
    lon_block = (lon // BLOCK_DEG).astype(int)
    return lat_block[:, None] * 1000 + lon_block[None, :]


def wscore(pred, true, w):
    err = pred - true
    n = w.sum()
    rmse = np.sqrt(np.sum(w * err ** 2) / n)
    bias = np.sum(w * err) / n
    wt = np.sum(w * true) / n
    wp = np.sum(w * pred) / n
    cov = np.sum(w * (true - wt) * (pred - wp)) / n
    r = cov / np.sqrt((np.sum(w * (true - wt) ** 2) / n) * (np.sum(w * (pred - wp) ** 2) / n))
    mape = np.sum(w * np.abs(err) / true) / n * 100
    return dict(rmse=float(rmse), bias=float(bias), r=float(r), mape=float(mape), n=int((w > 0).sum()))


results = {}

for label, ym in MONTHS:
    winds_path = f"data/era5/training_data/era5_winds_{ym}_3hourly.nc"
    z0_path = f"data/era5/training_data/era5_z0_{ym}_3hourly.nc"
    if not os.path.exists(winds_path):
        print(f"{label}: winds not downloaded yet, skipping entirely")
        continue

    print(f"=== {label} ===")
    ds = xr.open_dataset(winds_path)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    nt = ds.dims["valid_time"]

    u10 = ds["u10"].values
    v10 = ds["v10"].values
    u100 = ds["u100"].values
    v100 = ds["v100"].values
    spd10 = np.hypot(u10, v10)
    spd100 = np.hypot(u100, v100)

    w_lat = np.cos(np.deg2rad(lat))[:, None] * np.ones((lat.size, lon.size))
    w_full = np.broadcast_to(w_lat, (nt, lat.size, lon.size))
    land = np.broadcast_to(land2d, (nt, lat.size, lon.size))
    ocean = np.broadcast_to(ocean2d, (nt, lat.size, lon.size))

    valid = spd10 > 0.5
    low = valid & (spd10 < LOW_WIND_THRESH)

    bid2d = block_ids(lat, lon)
    rng = np.random.default_rng(42)
    ub = np.unique(bid2d)
    rng.shuffle(ub)
    test_blocks = set(ub[: int(0.2 * len(ub))])
    is_test2d = np.isin(bid2d, list(test_blocks))
    is_test = np.broadcast_to(is_test2d, (nt, lat.size, lon.size))
    is_train = ~is_test

    # ---- power law (no fit) ----
    ratio_pow = np.where(land, 10.0 ** ALPHA_LAND, 10.0 ** ALPHA_SEA)
    spd100_pow = spd10 * ratio_pow

    # ---- log law (only if z0 available) ----
    have_z0 = os.path.exists(z0_path)
    if have_z0:
        z0ds = xr.open_dataset(z0_path)
        z0 = z0ds["fsr"].values  # (time, lat, lon), same nt
        z0v = (z0 > 1e-5) & (z0 < 10)
        ratio_log = np.full_like(spd10, np.nan)
        ratio_log[z0v] = np.log(100.0 / z0[z0v]) / np.log(10.0 / z0[z0v])
        spd100_log = spd10 * ratio_log
    else:
        spd100_log = None
        print(f"  (z0 not downloaded yet for {label} -- log law skipped)")

    month_res = {}
    for rname, mask in [("onshore", land), ("offshore", ocean)]:
        # ---- regression fit on train, pooled over all timesteps this month ----
        tr = mask & is_train & valid
        te = mask & is_test & valid
        x, y, wt_ = spd10[tr], spd100[tr], w_full[tr]
        wx = np.average(x, weights=wt_)
        wy = np.average(y, weights=wt_)
        slope = np.average((x - wx) * (y - wy), weights=wt_) / np.average((x - wx) ** 2, weights=wt_)
        intercept = wy - slope * wx
        spd100_reg = slope * spd10 + intercept

        for method, pred in [("power_law", spd100_pow), ("regression", spd100_reg)] + (
            [("log_law", spd100_log)] if have_z0 else []
        ):
            for sname, smask in [("overall", te), ("low_wind", te & low)]:
                key = f"{rname}_{method}_{sname}"
                sc = wscore(pred[smask], spd100[smask], w_full[smask])
                month_res[key] = sc
                print(f"  {rname:<9}{method:<12}{sname:<10} n={sc['n']:<9} RMSE={sc['rmse']:.3f} "
                      f"bias={sc['bias']:+.3f} r={sc['r']:.3f} MAPE={sc['mape']:.1f}%")
        print(f"  ({rname} regression: speed100 = {slope:.3f}*speed10 + {intercept:.3f})")
    results[label] = month_res
    print()

with open("data/results/three_methods_winter_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved data/results/three_methods_winter_results.json")
