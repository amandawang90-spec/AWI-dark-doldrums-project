"""
Power-law skill specifically in the low-wind (<3 m/s) regime -- the dark-doldrum
band already used for the log-law-vs-regression comparison in FINDINGS.md.
Reports RMSE and MAPE on wind SPEED (not components), onshore vs offshore, for
each of the three ERA5 test timestamps, low-wind subset vs. overall.
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import numpy as np
import xarray as xr

TIMESTAMPS = [
    ("Aug2025", "20250830_0000"),
    ("Mar2026", "20260301_0000"),
    ("Aug2026", "20260830_0000"),
    ("Oct2025", "20251015_0000"),
    ("Nov2025", "20251115_0000"),
    ("Dec2025", "20251215_0000"),
    ("Jan2026", "20260115_0000"),
    ("Feb2026", "20260215_0000"),
]
LSM_FILE = "data/era5/era5_lsm_20260830_0000.nc"
ALPHA_LAND, ALPHA_SEA = 0.20, 0.14
LOW_WIND_THRESH = 3.0

lsm = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values
land, ocean = lsm >= 0.5, lsm < 0.5


def scores(pred, true, w):
    err = pred - true
    rmse = np.sqrt(np.average(err ** 2, weights=w))
    mape = np.average(np.abs(err) / true, weights=w) * 100
    return rmse, mape


print(f"Power-law skill, low wind (<{LOW_WIND_THRESH} m/s at 10m) vs. overall, speed-based.\n")
print(f"{'time':<9}{'region':<10}{'subset':<12}{'n':<10}{'RMSE':<9}{'MAPE %'}")

for label, stamp in TIMESTAMPS:
    winds = xr.open_dataset(f"data/era5/era5_winds_{stamp}.nc").squeeze()
    u10, v10 = winds["u10"].values, winds["v10"].values
    u100, v100 = winds["u100"].values, winds["v100"].values
    lat = winds["latitude"].values
    spd10, spd100 = np.hypot(u10, v10), np.hypot(u100, v100)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(u10)

    ratio = np.where(land, 10.0 ** ALPHA_LAND, 10.0 ** ALPHA_SEA)
    spd100_pow = spd10 * ratio  # magnitude-only prediction, direction unchanged by scalar ratio

    valid = spd10 > 0.5
    low = valid & (spd10 < LOW_WIND_THRESH)

    for rname, mask in [("onshore", land), ("offshore", ocean)]:
        for sname, smask in [("low-wind", low & mask), ("overall", valid & mask)]:
            rmse, mape = scores(spd100_pow[smask], spd100[smask], w[smask])
            print(f"{label:<9}{rname:<10}{sname:<12}{smask.sum():<10}{rmse:<9.3f}{mape:.1f}")
    print()
