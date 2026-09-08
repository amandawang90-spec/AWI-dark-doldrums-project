"""
Verify the 3-hourly ERA5 download: variables present, number of time steps, the
3-hour spacing, and that the month is fully covered with no gaps.
"""

import glob
import os

import numpy as np
import xarray as xr

for path in sorted(glob.glob("data/era5_*_3h_*.nc")):
    size = os.path.getsize(path) / 1e9
    try:
        ds = xr.open_dataset(path)
    except Exception as exc:
        print(f"{path}: OPEN FAILED ({exc})\n")
        continue

    tname = "valid_time" if "valid_time" in ds.coords else "time"
    t = ds[tname].values
    dvars = [v for v in ds.data_vars if v not in ("number", "expver")]

    steps = np.diff(t).astype("timedelta64[h]").astype(int) if len(t) > 1 else []
    uniq = sorted(set(steps.tolist())) if len(steps) else []
    shape = tuple(ds[dvars[0]].shape) if dvars else ()

    print(f"{path}  [{size:.2f} GB]")
    print(f"    vars : {dvars}")
    print(f"    grid : {shape}")
    print(f"    time : {len(t)} steps, {str(t[0])[:16]} -> {str(t[-1])[:16]}")
    print(f"    step : {uniq} h  {'OK' if uniq == [3] else '<-- IRREGULAR'}")

    days = len(np.unique(t.astype("datetime64[D]")))
    expect = days * 8
    print(f"    days : {days} days x 8 = {expect} expected  "
          f"{'OK' if expect == len(t) else '<-- MISMATCH'}")

    for v in dvars:
        a = ds[v].values
        finite = np.isfinite(a)
        print(f"    {v:5s}: min {np.nanmin(a):12.4f}  max {np.nanmax(a):12.4f}  "
              f"nan {100 * (1 - finite.mean()):.2f}%")
    print()
    ds.close()
