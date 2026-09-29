"""Aggregate the true-hourly ssrd/tsr backfill (data/era5_1h/) into genuine 3-hour
accumulations at the same 8 daily stamps (00,03,...,21) used everywhere else in
this pipeline (era5_clouds_3h, DART's own 3-hourly fields).

Each ERA5 hourly value at hour H is itself a 1-hour accumulation ending at H
(J/m2). Summing 3 consecutive hourly values reconstructs the true 3-hour
accumulation ending at the last of the three -- exactly DART's own convention
(DT=10800), unlike the old era5_ssrd_3h/era5_tsr_3h files, which held a single
1-hour sample every 3 hours (DT=3600) as a proxy for a 3-hour mean.

Month-boundary handling: the 00:00 stamp of day 1 needs the last 2 hours
(22:00, 23:00) of the PREVIOUS month. Read from that month's era5_1h file when
it exists. The one exception is January 2015 itself (December 2014 was never
hourly-backfilled) -- for that single month only, the very first stamp
(2015-01-01 00:00) is dropped rather than approximated.

Output: data/era5_3h_mean/era5_{ssrd,tsr}_3hmean_<YYYYMM>.nc
  variable named ssrd / tsr, same convention as era5_{ssrd,tsr}_3h_<YYYYMM>.nc
  (J/m2, genuine 3-hour accumulation -- divide by 10800 for W/m2, NOT 3600).

Usage:
    python3 aggregate_hourly_to_3h.py               # all months with era5_1h data
    python3 aggregate_hourly_to_3h.py 201501 201502  # just these months
    python3 aggregate_hourly_to_3h.py --list         # show the plan, do nothing
"""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)

import calendar
import glob
import os
import sys
import time

import netCDF4 as nc
import numpy as np

SRC_DIR = "data/era5_1h"
OUT_DIR = "data/era5_3h_mean"
FIELDS = ["ssrd", "tsr"]
STAMPS_PER_DAY = 8   # hours 0,3,...,21


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prev_stamp(stamp):
    y, m = int(stamp[:4]), int(stamp[4:6])
    return f"{y-1}12" if m == 1 else f"{y}{m-1:02d}"


def all_available_months():
    have = sorted({os.path.basename(f).split("_")[-1][:6]
                   for f in glob.glob(f"{SRC_DIR}/era5_ssrd_1h_*.nc")})
    return have


def load_hourly(var, stamp, tail=None):
    """tail=N reads only the last N timesteps (a direct partial netCDF read, not a
    slice of an already-loaded array) -- needed so grabbing the previous month's
    last 2 hours doesn't pull its entire ~6 GB array into memory just to discard it."""
    path = f"{SRC_DIR}/era5_{var}_1h_{stamp}.nc"
    ds = nc.Dataset(path)
    var_obj = ds.variables[var]
    data = (var_obj[-tail:] if tail else var_obj[:]).astype("float32")   # (ntime, nlat, nlon)
    lat = ds.variables["latitude"][:]
    lon = ds.variables["longitude"][:]
    ds.close()
    return data, lat, lon


def aggregate_one(var, stamp, force):
    out_path = f"{OUT_DIR}/era5_{var}_3hmean_{stamp}.nc"
    if os.path.exists(out_path) and not force:
        print(f"--- skip (exists) {out_path}", flush=True)
        return

    cur, lat, lon = load_hourly(var, stamp)
    y, m = int(stamp[:4]), int(stamp[4:6])
    ndays = calendar.monthrange(y, m)[1]
    assert cur.shape[0] == ndays * 24, f"{stamp}: expected {ndays*24} hours, got {cur.shape[0]}"

    pstamp = prev_stamp(stamp)
    prev_path = f"{SRC_DIR}/era5_{var}_1h_{pstamp}.nc"
    drop_first = False
    if os.path.exists(prev_path):
        prev2, _, _ = load_hourly(var, pstamp, tail=2)
        seq = np.concatenate([prev2, cur], axis=0)
    elif stamp == "201501":
        seq = np.concatenate([np.zeros((2,) + cur.shape[1:], dtype="float32"), cur], axis=0)
        drop_first = True
    else:
        raise RuntimeError(f"{stamp}: previous month {pstamp} hourly file missing "
                            f"(expected for the boundary hours) and stamp != 201501")

    # 3h_sum[d, j] = seq[24d+3j : 24d+3j+3].sum(axis=0), j = 0..7 (stamps 00,03,...,21)
    out = np.empty((ndays * STAMPS_PER_DAY,) + cur.shape[1:], dtype="float32")
    for d in range(ndays):
        for j in range(STAMPS_PER_DAY):
            k = 24 * d + 3 * j
            out[d * STAMPS_PER_DAY + j] = seq[k:k + 3].sum(axis=0)

    valid = np.ones(out.shape[0], dtype=bool)
    if drop_first:
        valid[0] = False

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = out_path + ".tmp.nc"
    wds = nc.Dataset(tmp, "w", format="NETCDF4")
    wds.createDimension("time", int(valid.sum()))
    wds.createDimension("latitude", len(lat))
    wds.createDimension("longitude", len(lon))
    wds.createVariable("latitude", "f8", ("latitude",))[:] = lat
    wds.createVariable("longitude", "f8", ("longitude",))[:] = lon
    hours = np.array([f"{y}-{m:02d}-{d+1:02d}T{3*j:02d}:00:00"
                       for d in range(ndays) for j in range(STAMPS_PER_DAY)])[valid]
    tv = wds.createVariable("valid_time", str, ("time",))
    tv[:] = np.array(hours, dtype=object)
    v = wds.createVariable(var, "f4", ("time", "latitude", "longitude"), zlib=True, complevel=4)
    v[:] = out[valid]
    v.units = "J m-2"
    v.long_name = f"{var}, true 3-hour accumulation ending at valid_time (summed from hourly)"
    wds.comment = ("Built by aggregate_hourly_to_3h.py from data/era5_1h/ hourly values. "
                    "DT=10800 (3-hour window), matching TCo1279-DART's own convention -- "
                    "NOT the old era5_" + var + "_3h files (DT=3600, 1-hour samples).")
    wds.close()
    os.replace(tmp, out_path)
    print(f"--> {out_path}  ({valid.sum()}/{out.shape[0]} stamps kept)", flush=True)


def main(argv):
    force = "--force" in argv
    dry = "--list" in argv
    months = [a for a in argv if not a.startswith("-")]
    if not months:
        months = all_available_months()

    if dry:
        for stamp in months:
            for var in FIELDS:
                out_path = f"{OUT_DIR}/era5_{var}_3hmean_{stamp}.nc"
                state = "present" if os.path.exists(out_path) else "MISSING"
                print(f"  {state:8s} {out_path}")
        return 0

    for stamp in months:
        for var in FIELDS:
            try:
                aggregate_one(var, stamp, force)
            except Exception as exc:
                print(f"    FAILED {stamp} {var}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
