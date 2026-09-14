"""Verify the analytic tisr (solar_geometry.py) against ERA5's real downloaded
tisr field, before it is trusted anywhere a truth field doesn't exist (DART).

tisr is pure astronomy -- if the analytic version doesn't reproduce ERA5's own
field to a tight tolerance, the geometry code is wrong and nothing built on top
of it (the reconstruction's predictors, or the un-normalising step back to
ssrd) can be trusted either.

Usage: python3 check_solar_geometry.py [--stride N] [YYYYMM ...]
"""

import argparse

import numpy as np
import xarray as xr

from solar_geometry import toa_irradiance_accumulated

DATA_DIR = "data/era5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("months", nargs="*", default=["202508", "202603", "202608"])
    args = ap.parse_args()

    for stamp in args.months:
        ds = xr.open_dataset(f"{DATA_DIR}/era5_ssrd_3h_{stamp}.nc").isel(
            latitude=slice(None, None, args.stride), longitude=slice(None, None, args.stride))
        lat, lon = ds.latitude.values, ds.longitude.values
        actual = ds.tisr.values                       # (nt, nlat, nlon), J/m2

        analytic = toa_irradiance_accumulated(lat, lon, ds.valid_time.values)

        diff = analytic - actual
        w = np.cos(np.deg2rad(lat))[None, :, None] * np.ones_like(actual)
        rmse = np.sqrt(np.average(diff ** 2, weights=w))
        bias = np.average(diff, weights=w)
        gmean_actual = np.average(actual, weights=w)
        gmean_analytic = np.average(analytic, weights=w)
        corr = np.corrcoef(actual.ravel(), analytic.ravel())[0, 1]

        print(f"{stamp}: grid {lat.size}x{lon.size}, {actual.shape[0]} steps")
        print(f"    area-wtd mean   actual={gmean_actual/3600:7.2f}  analytic={gmean_analytic/3600:7.2f} W/m2")
        print(f"    RMSE={rmse/3600:6.3f} W/m2   bias={bias/3600:+6.3f} W/m2   "
              f"correlation={corr:.6f}")
        print(f"    max actual={actual.max()/3600:.1f} W/m2   max analytic={analytic.max()/3600:.1f} W/m2")
        print()
        ds.close()


if __name__ == "__main__":
    main()
