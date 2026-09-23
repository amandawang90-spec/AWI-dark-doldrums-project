"""
Download winds (u10/v10/u100/v100) and z0 (fsr) for 9 winters (Oct-Feb, 3-hourly),
2016-17 through 2024-25 -- extends the existing Oct2025-Feb2026 winter so the
pooled log-law/regression/power-law comparison can run over 10 winters total.

Runs sequentially (one request at a time) to avoid hammering the CDS API, which
has already shown frequent connection timeouts. Each cdsapi retrieve() call has
its own internal retry loop (up to 500 attempts), so this script can just move on
to the next request once one finishes -- no extra retry logic needed here beyond
skipping files that already exist (so this script is safe to re-run if
interrupted).

Progress is printed with flush=True so it can be tailed live.
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

# Usage: python download_all_winters.py [FIRST_START_YEAR LAST_START_YEAR]
# Default 2016 2024 -> winters 2016-17 .. 2024-25. E.g. "2006 2015" -> 2006-07 .. 2015-16.
FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
LAST = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
WINTERS = [(y, y + 1) for y in range(FIRST, LAST + 1)]
MONTHS_IN_WINTER = [(10, 0), (11, 0), (12, 0), (1, 1), (2, 1)]  # (month, year_offset)

DATASET = "reanalysis-era5-single-levels"
os.makedirs("data/era5/training_data", exist_ok=True)

client = cdsapi.Client()


def download_winds(year, month):
    out = f"data/era5/training_data/era5_winds_{year}{month:02d}_3hourly.nc"
    if os.path.exists(out):
        print(f"  [skip] {out} already exists", flush=True)
        return
    n_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]
    times = [f"{h:02d}:00" for h in range(0, 24, 3)]
    print(f"  downloading winds {year}-{month:02d} ...", flush=True)
    client.retrieve(
        DATASET,
        {
            "product_type": "reanalysis",
            "variable": [
                "100m_u_component_of_wind", "100m_v_component_of_wind",
                "10m_u_component_of_wind", "10m_v_component_of_wind",
            ],
            "year": str(year), "month": f"{month:02d}", "day": days, "time": times,
            "data_format": "netcdf",
        },
        out,
    )
    print(f"  done: {out}", flush=True)


def download_z0(year, month):
    out = f"data/era5/training_data/era5_z0_{year}{month:02d}_3hourly.nc"
    if os.path.exists(out):
        print(f"  [skip] {out} already exists", flush=True)
        return
    n_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]
    times = [f"{h:02d}:00" for h in range(0, 24, 3)]
    print(f"  downloading z0 {year}-{month:02d} ...", flush=True)
    client.retrieve(
        DATASET,
        {
            "product_type": "reanalysis",
            "variable": ["forecast_surface_roughness"],
            "year": str(year), "month": f"{month:02d}", "day": days, "time": times,
            "data_format": "netcdf",
        },
        out,
    )
    print(f"  done: {out}", flush=True)


for wstart, wend in WINTERS:
    print(f"=== Winter {wstart}-{wend} ===", flush=True)
    for month, yoff in MONTHS_IN_WINTER:
        year = wstart if yoff == 0 else wend
        download_winds(year, month)
        download_z0(year, month)

print("ALL WINTERS DONE", flush=True)
