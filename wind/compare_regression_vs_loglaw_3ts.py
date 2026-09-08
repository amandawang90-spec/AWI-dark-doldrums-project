"""
Compare land/ocean-stratified linear regression vs. the log-law (using each
timestamp's own, time-matched forecast_surface_roughness), across all three
downloaded ERA5 timestamps: Aug 2026, Mar 2026, Aug 2025.

Reports both overall RMSE (dominated by moderate/high wind) and low-wind
(<3 m/s, dark-doldrum-relevant) RMSE/MAPE, since earlier testing showed the
two metrics rank the methods differently.
"""

import numpy as np
import xarray as xr

TIMESTAMPS = [
    ("Aug2026", "data/era5_winds_20260830_0000.nc", "data/era5_z0_20260830_0000.nc"),
    ("Mar2026", "data/era5_winds_20260301_0000.nc", "data/era5_z0_20260301_0000.nc"),
    ("Aug2025", "data/era5_winds_20250830_0000.nc", "data/era5_z0_20250830_0000.nc"),
]
LSM_FILE = "data/era5_lsm_20260830_0000.nc"  # static, same for all times


def load(fwind, fz0, lsm):
    w = xr.open_dataset(fwind).squeeze()
    z0 = xr.open_dataset(fz0).squeeze()["fsr"].values
    u10, v10 = w["u10"].values, w["v10"].values
    u100, v100 = w["u100"].values, w["v100"].values
    lat = w["latitude"].values
    spd10, spd100 = np.hypot(u10, v10), np.hypot(u100, v100)
    w_area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(spd10)
    return spd10, spd100, z0, w_area


def wfit(x, y, w):
    wx, wy = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - wx) * (y - wy), weights=w)
    varx = np.average((x - wx) ** 2, weights=w)
    slope = cov / varx
    intercept = wy - slope * wx
    return slope, intercept


lsm = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values
land, ocean = lsm >= 0.5, lsm < 0.5

print(f"{'time':<9}{'region':<10}{'method':<12}{'overall RMSE':<14}{'low-wind RMSE':<16}{'low-wind MAPE%'}")
rows = []
for label, fwind, fz0 in TIMESTAMPS:
    spd10, spd100, z0, w = load(fwind, fz0, lsm)
    valid = (spd10 > 0.5) & (z0 > 1e-5) & (z0 < 10)
    ratio = np.full_like(spd10, np.nan)
    ratio[valid] = np.log(100.0 / z0[valid]) / np.log(10.0 / z0[valid])
    spd100_log = spd10 * ratio

    for rname, mask in [("onshore", land), ("offshore", ocean)]:
        slope, intercept = wfit(spd10[mask], spd100[mask], w[mask])
        pred_reg = slope * spd10 + intercept

        low = mask & (spd10 < 3)
        low_log = low & valid

        rmse_reg = np.sqrt(np.average((pred_reg[mask] - spd100[mask]) ** 2, weights=w[mask]))
        m2 = mask & valid
        rmse_log = np.sqrt(np.average((spd100_log[m2] - spd100[m2]) ** 2, weights=w[m2]))

        rmse_reg_low = np.sqrt(np.average((pred_reg[low] - spd100[low]) ** 2, weights=w[low]))
        rmse_log_low = np.sqrt(np.average((spd100_log[low_log] - spd100[low_log]) ** 2, weights=w[low_log]))
        mape_reg_low = np.average(np.abs(pred_reg[low] - spd100[low]) / np.maximum(spd100[low], 0.5), weights=w[low]) * 100
        mape_log_low = np.average(np.abs(spd100_log[low_log] - spd100[low_log]) / np.maximum(spd100[low_log], 0.5), weights=w[low_log]) * 100

        print(f"{label:<9}{rname:<10}{'regression':<12}{rmse_reg:<14.3f}{rmse_reg_low:<16.3f}{mape_reg_low:.1f}")
        print(f"{label:<9}{rname:<10}{'log-law':<12}{rmse_log:<14.3f}{rmse_log_low:<16.3f}{mape_log_low:.1f}")
        rows.append((label, rname, rmse_reg, rmse_log, rmse_reg_low, rmse_log_low, mape_reg_low, mape_log_low))

print("\n=== summary: which method wins, by regime ===")
print(f"{'time':<9}{'region':<10}{'overall winner':<18}{'low-wind winner'}")
for label, rname, rmse_reg, rmse_log, rmse_reg_low, rmse_log_low, mape_reg_low, mape_log_low in rows:
    ov_winner = "regression" if rmse_reg < rmse_log else "log-law"
    lw_winner = "regression" if mape_reg_low < mape_log_low else "log-law"
    print(f"{label:<9}{rname:<10}{ov_winner:<18}{lw_winner}")
