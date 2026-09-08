"""
Verify the ERA5 monthly-mean downloads: variables, time stamp, grid, value ranges,
and whether a month carries the preliminary ERA5T flag (expver=5) rather than the
final ERA5 stream (expver=1).
"""

import glob
import os

import numpy as np
import xarray as xr

for path in sorted(glob.glob("data/era5_monthly_*.nc")):
    size = os.path.getsize(path) / 1e6
    ds = xr.open_dataset(path)

    tname = "valid_time" if "valid_time" in ds.coords else "time"
    t = np.atleast_1d(ds[tname].values)
    dvars = [v for v in ds.data_vars if v not in ("number", "expver")]

    # expver appears as a coord or a data var depending on the stream; 1 = final
    # ERA5, 5 = preliminary ERA5T (values can still change when finalised).
    expver = "not present (single stream)"
    for name in ("expver",):
        if name in ds.coords or name in ds.variables:
            expver = np.unique(np.asarray(ds[name].values).ravel()).tolist()

    print(f"{path}  [{size:6.1f} MB]")
    print(f"    vars  : {dvars}")
    print(f"    grid  : {tuple(ds[dvars[0]].shape)}")
    print(f"    time  : {[str(x)[:10] for x in t]}")
    print(f"    expver: {expver}")
    for v in dvars:
        a = ds[v].values
        print(f"    {v:5s}: min {np.nanmin(a):12.2f}  max {np.nanmax(a):12.2f}  "
              f"mean {np.nanmean(a):12.2f}  [{ds[v].attrs.get('units', '?')}]")
    print()
    ds.close()
