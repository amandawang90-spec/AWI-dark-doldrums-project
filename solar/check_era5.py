"""Verify the ERA5 download: completeness, structure, and the unit conventions.

Replaces check_downloads.py, check_monthly.py and check_monthly_units.py.

Three passes:
  1. INVENTORY   which of the files download_era5.py should have produced exist.
  2. STRUCTURE   variables, grid, time coverage, 3-hour spacing, gaps, NaN
                 fraction, value ranges, and the ERA5T preliminary-stream flag
                 (expver 5 means values can still change; 1 is final).
  3. UNITS       settles the accumulation period of every J m-2 field by converting
                 the area-weighted global mean to W m-2 and comparing against
                 climatology. Exits non-zero if a convention does not hold, so a
                 future CDS change cannot silently corrupt the reconstruction.

Area weighting by cos(lat) is essential: a plain mean over a regular lat/lon grid
over-counts polar points (see wind/FINDINGS.md section 5).

Usage: python3 check_era5.py [YYYYMM ...]      (default: whatever is in DATA_DIR)
"""

import glob
import os
import re
import sys

import numpy as np
import xarray as xr

DATA_DIR = "data/era5"

SOLAR_CONSTANT = 1361.0                       # W m-2
ACCUM = ("ssrd", "tsr", "tisr")               # every J m-2 field fetched

# group -> expected variables, matching download_era5.py
EXPECTED = {
    "clouds_3h":     ["tcc", "hcc", "mcc", "lcc"],
    "tsr_3h":        ["tsr"],
    "ssrd_3h":       ["ssrd", "tisr"],
    "monthly_ssrd":  ["ssrd"],
    "monthly_fal":   ["fal"],
}
MONTHLY_GROUPS = {"monthly_ssrd", "monthly_fal"}
# expected W m-2 after the right divisor, for a global area-weighted mean
CLIMATOLOGY = {"ssrd": (120, 260), "tsr": (200, 380), "tisr": (250, 500)}


def area_mean(ds, v):
    a = ds[v].values.squeeze()
    w = np.cos(np.deg2rad(ds["latitude"].values))
    w = np.broadcast_to(w[:, None], a.shape)
    return float(np.nansum(a * w) / np.nansum(np.where(np.isfinite(a), w, 0.0)))


def main(argv):
    months = [a for a in argv if re.fullmatch(r"\d{6}", a)]
    paths = sorted(glob.glob(f"{DATA_DIR}/era5_*.nc"))
    if months:
        paths = [p for p in paths if any(m in p for m in months)]
    if not paths:
        print(f"no files in {DATA_DIR} -- run download_era5.py first")
        return 1

    stamps = sorted({re.search(r"_(\d{6})\.nc$", p).group(1) for p in paths})
    problems = []

    print("=" * 78, "\n1. INVENTORY\n")
    for stamp in stamps:
        for group in EXPECTED:
            target = f"{DATA_DIR}/era5_{group}_{stamp}.nc"
            if not os.path.exists(target):
                print(f"   MISSING  {target}")
                problems.append(f"missing {target}")
    print("   all expected files present" if not problems else "")

    print("=" * 78, "\n2. STRUCTURE\n")
    for path in paths:
        group = re.search(r"era5_(.+)_\d{6}\.nc$", path).group(1)
        size = os.path.getsize(path) / 1e6
        try:
            ds = xr.open_dataset(path)
        except Exception as exc:
            print(f"{path}: OPEN FAILED ({exc})\n")
            problems.append(f"unreadable {path}")
            continue

        tname = "valid_time" if "valid_time" in ds.coords else "time"
        t = np.atleast_1d(ds[tname].values)
        dvars = [v for v in ds.data_vars if v not in ("number", "expver")]

        print(f"{path}  [{size:.1f} MB]")
        print(f"    vars : {dvars}")
        print(f"    grid : {tuple(ds[dvars[0]].shape)}")
        print(f"    time : {len(t)} step(s), {str(t[0])[:16]} -> {str(t[-1])[:16]}")

        want = EXPECTED.get(group)
        if want and sorted(dvars) != sorted(want):
            print(f"    <-- EXPECTED {want}")
            problems.append(f"{path}: vars {dvars} != {want}")

        if len(t) > 1:                                    # sub-monthly file
            steps = sorted(set(np.diff(t).astype("timedelta64[h]").astype(int)))
            days = len(np.unique(t.astype("datetime64[D]")))
            ok_step = steps == [3]
            ok_count = len(t) == days * 8
            print(f"    step : {steps} h  {'OK' if ok_step else '<-- IRREGULAR'}")
            print(f"    days : {days} x 8 = {days * 8} expected  "
                  f"{'OK' if ok_count else '<-- MISMATCH'}")
            if not (ok_step and ok_count):
                problems.append(f"{path}: irregular time axis")

        for name in ("expver",):
            if name in ds.coords or name in ds.variables:
                ev = np.unique(np.asarray(ds[name].values).ravel()).tolist()
                print(f"    expver: {ev}  {'(preliminary ERA5T)' if 5 in ev else ''}")

        for v in dvars:
            a = ds[v].values
            nan = 100 * (1 - np.isfinite(a).mean())
            print(f"    {v:5s}: min {np.nanmin(a):12.4f}  max {np.nanmax(a):12.4f}"
                  f"  nan {nan:5.2f}%  [{ds[v].attrs.get('units', '?')}]")
        print()
        ds.close()

    print("=" * 78, "\n3. UNITS\n")
    print(f"   hourly ceiling {SOLAR_CONSTANT * 3600:.2e} J m-2,"
          f"  daily ceiling {SOLAR_CONSTANT * 86400:.2e} J m-2\n")
    for path in paths:
        group = re.search(r"era5_(.+)_\d{6}\.nc$", path).group(1)
        ds = xr.open_dataset(path).squeeze()
        acc = [v for v in ds.data_vars if v in ACCUM]
        if not acc:
            ds.close()
            continue
        divisor, label = (86400, "per day") if group in MONTHLY_GROUPS else (3600, "per hour")
        print(f"{path}   assumed {label} (/{divisor})")
        for v in acc:
            gm = area_mean(ds, v)
            wm2 = gm / divisor
            lo, hi = CLIMATOLOGY.get(v, (0, 1e9))
            ok = lo <= wm2 <= hi
            print(f"    {v:5s}: max {np.nanmax(ds[v].values):.3e}  "
                  f"area-wtd mean {gm:.4e} -> {wm2:7.1f} W m-2  "
                  f"{'OK' if ok else f'<-- OUTSIDE [{lo},{hi}]'}")
            if not ok:
                problems.append(f"{path}:{v} -> {wm2:.1f} W m-2, convention wrong?")
        print()
        ds.close()

    print("=" * 78)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
