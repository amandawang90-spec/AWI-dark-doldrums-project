"""
Download ERA5 monthly-mean radiation and column water vapour, one file per month.

Variables:
  ssrd  surface_solar_radiation_downwards          (accumulated)
  ssrc  surface_net_solar_radiation_clear_sky      (accumulated)
  tsrc  top_net_solar_radiation_clear_sky          (accumulated)
  tcwv  total_column_water_vapour                  (instantaneous)

Dataset is reanalysis-era5-single-levels-MONTHLY-MEANS with product_type
"monthly_averaged_reanalysis".

UNITS OF THE ACCUMULATED FIELDS (ssrd, ssrc, tsrc): each value is the mean DAILY
accumulation for the month, in J m-2 per day -- not per hour, and not a monthly
total. Verified in check_monthly_units.py: the maxima (~2.9e7 J m-2) exceed the
hourly physical ceiling of 1361 W m-2 * 3600 s = 4.9e6 J m-2, and the area-weighted
global means divided by 86400 give 180 W m-2 (ssrd) and 286 W m-2 (tsrc), matching
climatology. So:
    W m-2      = value / 86400
    month total = value * days_in_month
tcwv is instantaneous and needs no such conversion (kg m-2).

Accumulated and instantaneous variables are requested separately so CDS returns one
plain .nc per request instead of a zip of per-stepType files (the same reason
download_era5_clouds_3h.py splits its requests).

Note: monthly means only appear once a month is complete and processed, so the most
recent month or two may not exist yet -- a missing month fails its request rather
than returning partial data.

Usage: python3 download_era5_monthly.py YYYYMM [YYYYMM ...]
Output: data/era5_monthly_rad_<YYYYMM>.nc   (ssrd, ssrc, tsrc)
        data/era5_monthly_tcwv_<YYYYMM>.nc  (tcwv)
"""

import sys
import cdsapi

DATASET = "reanalysis-era5-single-levels-monthly-means"

ACCUM_VARS = [
    "surface_solar_radiation_downwards",
    "surface_net_solar_radiation_clear_sky",
    "top_net_solar_radiation_clear_sky",
]
INSTANT_VARS = ["total_column_water_vapour"]

# "area": [north, west, south, east] -- set to restrict to a box; None = global.
AREA = None

stamps = sys.argv[1:]
if not stamps:
    sys.exit("Usage: python3 download_era5_monthly.py YYYYMM [YYYYMM ...]")

client = cdsapi.Client()

failed = []
for stamp in stamps:
    year, month = stamp[:4], stamp[4:6]
    for tag, variables in (("rad", ACCUM_VARS), ("tcwv", INSTANT_VARS)):
        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variables,
            "year": year,
            "month": month,
            "time": "00:00",          # required by the form; ignored for monthly means
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if AREA is not None:
            request["area"] = AREA
        target = f"data/era5_monthly_{tag}_{stamp}.nc"
        print(f"--> {target}  ({year}-{month})", flush=True)
        try:
            client.retrieve(DATASET, request, target)
        except Exception as exc:
            # Keep going: a month that is not published yet should not abort the rest.
            print(f"    FAILED {target}: {exc}", flush=True)
            failed.append(target)

print(f"Done: {len(stamps)} month(s) requested, {len(failed)} request(s) failed.")
for t in failed:
    print(f"  missing: {t}")
