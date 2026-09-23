"""
Download a full month of u100, v100, u10, v10 from ERA5 at 3-hourly resolution
(00,03,06,09,12,15,18,21 UTC) -- a continuous time series, unlike
download_era5_winds.py's single instantaneous snapshot. Matches the hackathon
TCo1279-DART dataset's own 3-hourly output cadence.

Usage: python download_era5_winds_month.py YYYY MM
  e.g. python download_era5_winds_month.py 2025 10

Output: data/era5/training_data/era5_winds_YYYYMM_3hourly.nc
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import os

import calendar
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
        "variable": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": YEAR,
        "month": MONTH,
        "day": DAYS,
        "time": TIMES,
        "data_format": "netcdf",
    },
    f"data/era5/training_data/era5_winds_{YEAR}{MONTH}_3hourly.nc",
)

print(f"Downloaded {YEAR}-{MONTH}, {n_days} days x {len(TIMES)} times/day = {n_days * len(TIMES)} timesteps")
