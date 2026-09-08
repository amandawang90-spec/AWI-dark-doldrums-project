"""
Download 3-hourly cloud cover and top-of-atmosphere solar radiation from ERA5,
one file per month.

Variables:
  tsr   top_net_solar_radiation   (accumulated over the preceding hour)
  tcc   total_cloud_cover         (instantaneous)
  hcc   high_cloud_cover          (instantaneous)
  mcc   medium_cloud_cover        (instantaneous)
  lcc   low_cloud_cover           (instantaneous)

Accumulated and instantaneous fields go in SEPARATE requests on purpose: if they are
mixed, CDS returns a zip holding one .nc per stepType (this is what produced the
data_stream-oper_stepType-{accum,instant}.nc pairs in wind/data/solar_extract/).
One stepType per request gives one clean .nc file.

Note on tsr: it is an accumulation over the hour ENDING at the stamped time, not a
3-hour total. Sampling every 3 h gives 8 one-hour snapshots per day, not a closed
energy budget for the day.

Usage: python3 download_era5_clouds_3h.py YYYYMM [YYYYMM ...]
Output: data/era5_tsr_3h_<YYYYMM>.nc
        data/era5_clouds_3h_<YYYYMM>.nc
"""

import sys
import cdsapi

DATASET = "reanalysis-era5-single-levels"
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]          # 00,03,...,21 UTC
DAYS = [f"{d:02d}" for d in range(1, 32)]                  # CDS ignores days a month lacks

ACCUM_VARS = ["top_net_solar_radiation"]
INSTANT_VARS = [
    "total_cloud_cover",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
]

# "area": [north, west, south, east] -- set to restrict to a box; None = global.
AREA = None

months = sys.argv[1:]
if not months:
    sys.exit("Usage: python3 download_era5_clouds_3h.py YYYYMM [YYYYMM ...]")

client = cdsapi.Client()

for stamp in months:
    year, month = stamp[:4], stamp[4:6]
    for tag, variables in (("tsr", ACCUM_VARS), ("clouds", INSTANT_VARS)):
        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": year,
            "month": month,
            "day": DAYS,
            "time": TIMES,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if AREA is not None:
            request["area"] = AREA
        target = f"data/era5_{tag}_3h_{stamp}.nc"
        print(f"--> {target}  ({year}-{month}, {len(TIMES)} times/day)", flush=True)
        client.retrieve(DATASET, request, target)

print(f"Done: {len(months)} month(s), 3-hourly.")
