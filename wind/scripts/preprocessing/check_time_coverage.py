"""
Report the actual time coordinate of every ERA5 file already downloaded, so a new
download can be matched to the real coverage rather than to the filename stamp.
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import glob
import os

import xarray as xr

for path in sorted(glob.glob("data/era5/era5_*.nc")) + sorted(
    glob.glob("data/dart_extract/*/*.nc")
):
    try:
        ds = xr.open_dataset(path)
    except Exception as exc:
        print(f"{path}: OPEN FAILED ({exc})")
        continue

    tname = next(
        (c for c in ("valid_time", "time", "forecast_reference_time") if c in ds.coords),
        None,
    )
    if tname is None:
        times = "(no time coord)"
    else:
        vals = ds[tname].values.ravel()
        times = f"{len(vals)} step(s): " + ", ".join(str(v)[:16] for v in vals[:4])
        if len(vals) > 4:
            times += f", ... {str(vals[-1])[:16]}"

    dvars = [v for v in ds.data_vars if v not in ("number", "expver")]
    size = os.path.getsize(path) / 1e6
    print(f"{path}  [{size:6.1f} MB]")
    print(f"    vars: {dvars}")
    print(f"    time: {times}")
    ds.close()
