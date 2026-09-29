"""Re-derive the old 1-hour-sample-every-3-hours proxy (v1/v2/v3's training/eval
convention, DT=3600) for the ONDJF months of 2015-2025, from data/era5_1h/ --
the original era5_ssrd_3h/era5_tsr_3h files for these months were deleted once
the hourly backfill was verified complete.

This is just a single-hour SELECT at each stamp (00,03,...,21), not a sum --
exactly what those files always held. Written small and pre-extracted (rather
than re-reading the ~6 GB full-resolution hourly files on every evaluation call)
because those files' on-disk chunking makes a strided/boxed read of the full
file slow even for a small region.

Output: data/era5_old_proxy/era5_{ssrd,tsr}_3h_<YYYYMM>.nc (same layout/name
pattern as the original deleted files, so reconstruct_ssrd.py's load_month
convention still applies unchanged if pointed at this directory).
"""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)

import os
import sys
import time

import netCDF4 as nc
import numpy as np
import xarray as xr

HOURLY_DIR = "data/era5_1h"
OUT_DIR = "data/era5_old_proxy"
STAMPS = [f"{y}{m:02d}" for y in range(2015, 2026) for m in (10, 11, 12, 1, 2)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_one(var, stamp, force=False):
    out_path = f"{OUT_DIR}/era5_{var}_3h_{stamp}.nc"
    if os.path.exists(out_path) and not force:
        print(f"--- skip (exists) {out_path}", flush=True)
        return
    src = f"{HOURLY_DIR}/era5_{var}_1h_{stamp}.nc"
    ds = xr.open_dataset(src)
    vt = ds.valid_time.values.astype("datetime64[s]")
    hour = ((vt - vt.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(int)
    keep = (hour % 3) == 0
    sub = ds.isel(valid_time=keep)
    lat, lon = sub.latitude.values, sub.longitude.values
    times = sub.valid_time.values
    data = sub[var].values.astype("float32")
    ds.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = out_path + ".tmp.nc"
    wds = nc.Dataset(tmp, "w", format="NETCDF4")
    wds.createDimension("valid_time", len(times))
    wds.createDimension("latitude", len(lat))
    wds.createDimension("longitude", len(lon))
    wds.createVariable("latitude", "f8", ("latitude",))[:] = lat
    wds.createVariable("longitude", "f8", ("longitude",))[:] = lon
    tv = wds.createVariable("valid_time", "i8", ("valid_time",))
    tv[:] = times.astype("datetime64[s]").astype("int64")
    tv.units = "seconds since 1970-01-01 00:00:00"
    v = wds.createVariable(var, "f4", ("valid_time", "latitude", "longitude"), zlib=True, complevel=4)
    v[:] = data
    v.units = "J m-2"
    wds.comment = ("Re-derived from data/era5_1h/ -- single hourly sample AT each 3-hourly "
                    "stamp, reproducing the original (now-deleted) era5_" + var + "_3h convention. "
                    "DT=3600, NOT a 3-hour sum -- see data/era5_3h_mean/ for the true 3-hour sum.")
    wds.close()
    os.replace(tmp, out_path)
    print(f"--> {out_path}  ({len(times)} stamps)", flush=True)


def main(argv):
    force = "--force" in argv
    for stamp in STAMPS:
        for var in ("ssrd", "tsr"):
            try:
                extract_one(var, stamp, force)
            except Exception as exc:
                print(f"    FAILED {stamp} {var}: {exc}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
