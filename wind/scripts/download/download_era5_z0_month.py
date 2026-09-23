"""
Download a full month of forecast_surface_roughness (z0) from ERA5 at 3-hourly
resolution (00,03,06,09,12,15,18,21 UTC), matching download_era5_winds_month.py's
cadence -- needed for a time-matched log-law reconstruction over the same period.

land_sea_mask is NOT re-downloaded here: it's static and every script in this
project already shares one canonical copy (data/era5/era5_lsm_20260830_0000.nc).

Usage: python download_era5_z0_month.py YYYY MM
  e.g. python download_era5_z0_month.py 2025 10

Output: data/era5/training_data/era5_z0_YYYYMM_3hourly.nc
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import calendar
import os
import sys
import cdsapi

YEAR = sys.argv[1]
MONTH = sys.argv[2].zfill(2)

n_days = calendar.monthrange(int(YEAR), int(MONTH))[1]
DAYS = [f"{d:02d}" for d in range(1, n_days + 1)]
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]

DATASET = "reanalysis-era5-single-levels"

os.makedirs("data/era5/training_data", exist_ok=True)

client = cdsapi.Client()

client.retrieve(
    DATASET,
    {
        "product_type": "reanalysis",
        "variable": ["forecast_surface_roughness"],
        "year": YEAR,
        "month": MONTH,
        "day": DAYS,
        "time": TIMES,
        "data_format": "netcdf",
    },
    f"data/era5/training_data/era5_z0_{YEAR}{MONTH}_3hourly.nc",
)

print(f"Downloaded z0 for {YEAR}-{MONTH}, {n_days} days x {len(TIMES)} times/day")
