"""
Download forecast_surface_roughness (z0, for the log law) and land_sea_mask (lsm,
for onshore/offshore split), matching a u10/v10/u100/v100 timestamp already
downloaded via download_era5_winds.py.

Usage: python download_era5_z0_lsm.py [YYYYMMDD] [HHMM]
  (defaults to 2026-08-30 00:00 UTC if omitted)
"""

import sys
import cdsapi

DATE = sys.argv[1] if len(sys.argv) > 1 else "20260830"
TIME = sys.argv[2] if len(sys.argv) > 2 else "0000"
YEAR, MONTH, DAY = DATE[:4], DATE[4:6], DATE[6:8]
TIME_HHMM = f"{TIME[:2]}:{TIME[2:]}"
DATASET = "reanalysis-era5-single-levels"

client = cdsapi.Client()

client.retrieve(
    DATASET,
    {
        "product_type": "reanalysis",
        "variable": ["forecast_surface_roughness"],
        "year": YEAR,
        "month": MONTH,
        "day": DAY,
        "time": TIME_HHMM,
        "data_format": "netcdf",
    },
    f"data/era5_z0_{DATE}_{TIME}.nc",
)

client.retrieve(
    DATASET,
    {
        "product_type": "reanalysis",
        "variable": ["land_sea_mask"],
        "year": YEAR,
        "month": MONTH,
        "day": DAY,
        "time": TIME_HHMM,
        "data_format": "netcdf",
    },
    f"data/era5_lsm_{DATE}_{TIME}.nc",
)

print(f"Downloaded z0 and lsm for {YEAR}-{MONTH}-{DAY} {TIME_HHMM} UTC")
