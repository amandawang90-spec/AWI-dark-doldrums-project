"""
Download ERA5 MONTHLY-MEAN forecast_surface_roughness (fsr = z0, in m) for every month of
a year range (default 2015-2025, Jan..Dec), global 0.25 deg, from the CDS dataset
"reanalysis-era5-single-levels-monthly-means" (product "monthly_averaged_reanalysis").
Unlike the 3-hourly files in data/era5/training_data (Oct-Feb only), this covers all 12 months.

Usage: python download_era5_fsr_monthly.py [FIRST_YEAR LAST_YEAR]
Output: data/era5/era5_fsr_monthly_<FIRST>_<LAST>.nc   (dimension valid_time = 12 * n_years)
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import sys
import cdsapi

FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
LAST = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
OUT = f"data/era5/era5_fsr_monthly_{FIRST}_{LAST}.nc"

client = cdsapi.Client()
client.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": ["forecast_surface_roughness"],
        "year": [str(y) for y in range(FIRST, LAST + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "data_format": "netcdf",
    },
    OUT,
)
print(f"Downloaded {OUT}")
