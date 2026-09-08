"""
Download u100, v100, u10, v10 wind components from ECMWF's Climate Data Store (ERA5)
for a single timestamp.

Usage: python download_era5_winds.py [YYYYMMDD] [HHMM]
  (defaults to 2026-08-30 00:00 UTC if omitted)

Setup:
1. Create an account at https://cds.climate.copernicus.eu
2. Go to https://cds.climate.copernicus.eu/how-to-api and follow the instructions to
   create ~/.cdsapirc with your personal access token (URL + key).
3. pip install "cdsapi>=0.7.2"
4. Accept the ERA5 dataset license terms on the dataset page (required once, via the
   website) before your first request.
"""

import sys
import cdsapi

DATE = sys.argv[1] if len(sys.argv) > 1 else "20260830"
TIME = sys.argv[2] if len(sys.argv) > 2 else "0000"
YEAR, MONTH, DAY = DATE[:4], DATE[4:6], DATE[6:8]
TIME_HHMM = f"{TIME[:2]}:{TIME[2:]}"

# Data requested "today" isn't in the final ERA5 archive yet (ERA5 has a ~5 day
# publication lag). Use the ERA5T (near-real-time, preliminary) dataset for recent
# dates; switch to "reanalysis-era5-single-levels" for anything older than ~5 days,
# which will also silently return ERA5T-flagged data for the last few days if you
# request a date that's not final yet.
DATASET = "reanalysis-era5-single-levels"

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
        "day": DAY,
        "time": TIME_HHMM,
        "data_format": "netcdf",  # or "grib"
        # "area": [north, west, south, east],  # uncomment + fill in for a subregion; omit for global
    },
    f"data/era5_winds_{DATE}_{TIME}.nc",
)
