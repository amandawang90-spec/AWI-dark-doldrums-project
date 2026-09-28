"""Backfill full hourly ssrd/tsr for 2015-2025 and merge with the existing
3-hourly samples, writing the result to a SEPARATE location so the original
era5_ssrd_3h_*.nc / era5_tsr_3h_*.nc files are untouched.

WHY
---
era5_ssrd_3h_<YYYYMM>.nc / era5_tsr_3h_<YYYYMM>.nc hold 8 timestamps/day
(00,03,...,21), each a genuine 1-hour ERA5 accumulation -- NOT a 3-hour mean.
The pipeline needs a true 3-hourly MEAN, which requires all 24 hourly values
per day. This script fetches only the 16 missing hours/day (skipping the 8
already on disk) and merges them with the existing samples into a complete
hourly series, written to data/era5_1h/ -- a later step aggregates that down
to 3-hourly means.

tisr is deliberately dropped from the hourly output: it's only needed at the
existing 3-hourly cadence, not hourly. It lives *inside* era5_ssrd_3h_<YYYYMM>.nc
alongside ssrd; per explicit confirmation, once a month's ssrd_1h AND tsr_1h
are both verified complete, era5_ssrd_3h_<YYYYMM>.nc and era5_tsr_3h_<YYYYMM>.nc
are deleted outright -- tisr is NOT extracted or preserved anywhere first.
This finalize step runs per-month as soon as that month's pair is done, not
held until the whole 2015-2025 run finishes.

Output (one file per field per month, in OUT_DIR):
    era5_ssrd_1h_<YYYYMM>.nc   ssrd, valid_time 00:00..23:00 (hourly)
    era5_tsr_1h_<YYYYMM>.nc    tsr,  valid_time 00:00..23:00 (hourly)

Deleted, in SRC_DIR, once the above are confirmed complete for that month:
    era5_ssrd_3h_<YYYYMM>.nc   (ssrd now redundant, tisr inside it is NOT kept)
    era5_tsr_3h_<YYYYMM>.nc    (tsr now redundant -- superseded by era5_tsr_1h)

Usage:
    python3 backfill_hourly_2015_2025.py               # all of 2015-2025
    python3 backfill_hourly_2015_2025.py 201503 202007  # just these months
    python3 backfill_hourly_2015_2025.py --list         # show the plan, fetch nothing
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)

import calendar
import os
import sys
import tempfile

SRC_DIR = "data/era5"
OUT_DIR = "data/era5_1h"

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
EXISTING_HOURS = {f"{h:02d}:00" for h in range(0, 24, 3)}          # already on disk
MISSING_HOURS = [h for h in ALL_HOURS if h not in EXISTING_HOURS]  # to fetch
DAYS = [f"{d:02d}" for d in range(1, 32)]  # CDS ignores days a month lacks

SL = "reanalysis-era5-single-levels"

# (out_tag, src_tag, cds_variable, netcdf_var_name)
FIELDS = [
    ("ssrd_1h", "ssrd_3h", "surface_solar_radiation_downwards", "ssrd"),
    ("tsr_1h", "tsr_3h", "top_net_solar_radiation", "tsr"),
]


def months_2015_2025():
    out = []
    for y in range(2015, 2026):
        for m in range(1, 13):
            out.append(f"{y}{m:02d}")
    return out


def build_request(variable, year, month, hours=MISSING_HOURS):
    return {
        "variable": [variable],
        "year": year,
        "month": month,
        "day": DAYS,
        "time": hours,
        "product_type": "reanalysis",
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def expected_hourly_count(stamp):
    year, month = int(stamp[:4]), int(stamp[4:6])
    ndays = calendar.monthrange(year, month)[1]
    return ndays * 24


def already_merged(out_file, stamp):
    if not os.path.exists(out_file):
        return False
    import xarray as xr
    try:
        with xr.open_dataset(out_file) as ds:
            return ds.sizes.get("valid_time", 0) == expected_hourly_count(stamp)
    except Exception:
        return False


def merge_one(out_tag, src_tag, variable, var_name, stamp):
    import xarray as xr

    src_file = f"{SRC_DIR}/era5_{src_tag}_{stamp}.nc"
    out_file = f"{OUT_DIR}/era5_{out_tag}_{stamp}.nc"
    has_src = os.path.exists(src_file)

    if already_merged(out_file, stamp):
        print(f"--- skip (already merged) {out_file}", flush=True)
        return "skipped"

    os.makedirs(OUT_DIR, exist_ok=True)
    fd, new_hours_file = tempfile.mkstemp(
        suffix=".nc", prefix=f"tmp_{out_tag}_{stamp}_", dir=OUT_DIR
    )
    os.close(fd)

    try:
        import cdsapi
        client = cdsapi.Client()

        # Eager (not dask): a full-globe hourly month is ~3GB per variable in
        # memory. Tried dask-backed lazy loading to cap peak memory, but
        # concat+sortby on out-of-order dask chunks fragments into ~224x more
        # chunks and the write stalls (18+ min, still not done, on one field).
        # Simpler fix: keep eager loading, just give the job enough memory
        # (--mem=24G/32G, MAX_WORKERS=3 -> worst case ~3 workers x ~2 copies
        # in memory during concat = comfortably under that; nodes have 257G).
        if has_src:
            print(f"--> requesting {out_tag} {stamp} (missing hours) ...", flush=True)
            client.retrieve(SL, build_request(variable, stamp[:4], stamp[4:6]), new_hours_file)
            with xr.open_dataset(src_file) as src_ds, xr.open_dataset(new_hours_file) as new_ds:
                src_var = src_ds[[var_name]]
                new_var = new_ds[[var_name]]
                merged = xr.concat([src_var, new_var], dim="valid_time").sortby("valid_time")
        else:
            # No 3-hourly original to merge with (e.g. a 2026 month never
            # downloaded before) -- fetch all 24 hours directly instead.
            print(f"--> requesting {out_tag} {stamp} (all hours, no 3-hourly source) ...", flush=True)
            client.retrieve(SL, build_request(variable, stamp[:4], stamp[4:6], hours=ALL_HOURS), new_hours_file)
            with xr.open_dataset(new_hours_file) as new_ds:
                merged = new_ds[[var_name]].sortby("valid_time").load()

        n = merged.sizes["valid_time"]
        expected = expected_hourly_count(stamp)
        if n != expected:
            raise RuntimeError(
                f"{out_file}: got {n} timesteps after merge, expected {expected} "
                f"(duplicate or missing hours -- check source data)"
            )
        fd2, tmp_out = tempfile.mkstemp(suffix=".nc", prefix=f"tmp_out_{out_tag}_{stamp}_", dir=OUT_DIR)
        os.close(fd2)
        merged.load().to_netcdf(tmp_out)
        os.replace(tmp_out, out_file)

        print(f"    done: {out_file}", flush=True)
        return "done"
    finally:
        if os.path.exists(new_hours_file):
            os.remove(new_hours_file)


def finalize_month(stamp):
    """Once both hourly outputs for `stamp` are verified complete, delete the
    now-redundant 3-hourly originals (ssrd_3h, including tisr inside it, and
    tsr_3h). No extraction/preservation of tisr -- confirmed to be discarded."""
    ssrd_1h_done = already_merged(f"{OUT_DIR}/era5_ssrd_1h_{stamp}.nc", stamp)
    tsr_1h_done = already_merged(f"{OUT_DIR}/era5_tsr_1h_{stamp}.nc", stamp)
    if not (ssrd_1h_done and tsr_1h_done):
        return

    for tag in ("ssrd_3h", "tsr_3h"):
        f = f"{SRC_DIR}/era5_{tag}_{stamp}.nc"
        if os.path.exists(f):
            os.remove(f)
            print(f"    removed {f} (superseded by era5_1h/)", flush=True)


def main(argv):
    dry = "--list" in argv
    months = [a for a in argv if not a.startswith("-")] or months_2015_2025()

    plan = [(out_tag, src_tag, variable, var_name, m)
            for m in months for out_tag, src_tag, variable, var_name in FIELDS]

    if dry:
        for out_tag, src_tag, variable, var_name, m in plan:
            out_file = f"{OUT_DIR}/era5_{out_tag}_{m}.nc"
            state = "present" if already_merged(out_file, m) else "MISSING"
            print(f"  {state:8s} {out_file:38s} {variable}")
        return 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Lower than download_era5.py's 5: each merge here briefly needs dask/netCDF
    # write buffers for a full-globe hourly month (~3GB/variable), so concurrent
    # merges are the memory-bound step, not the CDS queue wait.
    MAX_WORKERS = 3
    done, skipped, failed = [], [], []
    pending_fields = {m: {"ssrd_1h", "tsr_1h"} for m in months}
    lock = threading.Lock()

    def merge_and_maybe_finalize(out_tag, src_tag, variable, var_name, stamp):
        result = merge_one(out_tag, src_tag, variable, var_name, stamp)
        with lock:
            pending_fields[stamp].discard(out_tag)
            ready = not pending_fields[stamp]
        if ready:
            finalize_month(stamp)
        return result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(merge_and_maybe_finalize, out_tag, src_tag, variable, var_name, m): (out_tag, m)
            for out_tag, src_tag, variable, var_name, m in plan
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                result = fut.result()
                (done if result == "done" else skipped).append(key)
            except Exception as exc:
                print(f"    FAILED {key}: {exc}", flush=True)
                failed.append(key)

    print(f"\n{len(done)} merged, {len(skipped)} skipped, {len(failed)} failed.")
    for k in failed:
        print(f"  missing: {k}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
