"""
Settle the accumulation period of the monthly-mean radiation fields.

ERA5 monthly means of ACCUMULATED variables are ambiguous unless checked: they could
be the mean hourly accumulation or the mean daily accumulation. Two independent tests:

  1. Physical ceiling. An hourly accumulation cannot exceed the solar constant times
     3600 s = 4.90e6 J m-2. A daily one is capped at 1361 * 86400 = 1.18e8 J m-2.
  2. Area-weighted global mean converted to W m-2 under each assumption, compared
     with known climatology (global mean ssrd ~185 W m-2, tsrc ~290 W m-2).

Area weighting by cos(lat) is essential here: a plain mean over a regular lat/lon grid
over-counts polar points (see wind/FINDINGS.md section 5).
"""

import numpy as np
import xarray as xr

SOLAR_CONSTANT = 1361.0  # W m-2

ds = xr.open_dataset("data/era5_monthly_rad_202508.nc").squeeze()
lat = ds["latitude"].values
w = np.cos(np.deg2rad(lat))[:, None]

print(f"hourly ceiling : {SOLAR_CONSTANT * 3600:.3e} J m-2")
print(f"daily  ceiling : {SOLAR_CONSTANT * 86400:.3e} J m-2\n")

for v in ("ssrd", "ssrc", "tsrc"):
    a = ds[v].values
    vmax = np.nanmax(a)
    gmean = np.nansum(a * w) / np.nansum(w * np.ones_like(a))
    print(f"{v}:")
    print(f"    observed max          : {vmax:.3e} J m-2")
    print(f"    exceeds hourly ceiling: {'YES -> not hourly' if vmax > SOLAR_CONSTANT * 3600 else 'no'}")
    print(f"    area-weighted mean    : {gmean:.3e} J m-2")
    print(f"      if per hour  -> {gmean / 3600:9.1f} W m-2")
    print(f"      if per day   -> {gmean / 86400:9.1f} W m-2")
    print()
ds.close()
