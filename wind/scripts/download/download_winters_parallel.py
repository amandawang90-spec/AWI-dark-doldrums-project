"""
Parallel version of download_all_winters.py for large backfills.

download_all_winters.py deliberately runs one CDS request at a time. That's
fine for a handful of months, but for a 17-winter backfill (1979-1996, 170
files) sequential is too slow when a chunk of requests fail mid-transfer and
have to restart from scratch (observed repeatedly on 2026-09-23: CDS-side
processing finishes in the normal 10-50 min, but the file transfer itself
drops and cdsapi's internal retry restarts the whole request). Overlapping
several requests' CDS queue-wait time -- exactly the approach solar's own
download_era5.py already uses successfully -- hides most of that latency.

MAX_WORKERS=5 matches solar/scripts/download/download_era5.py's own choice,
documented there as "a considerate, non-abusive concurrency level" for CDS.

Usage: python download_winters_parallel.py FIRST_START_YEAR LAST_START_YEAR
  e.g. "1979 1995" -> winters 1979-80 .. 1995-96 (Oct(y)-Feb(y+1) each).
Safe to re-run / run alongside download_all_winters.py on a different year
range: both skip files that already exist, and this script's default range
here is chosen to not overlap the 1996-2005 range that job already covers.
"""
import os
_root = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_root, "data")):
    _root = os.path.dirname(_root)
os.chdir(_root)

import calendar
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi

FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 1979
LAST = int(sys.argv[2]) if len(sys.argv) > 2 else 1995
WINTERS = [(y, y + 1) for y in range(FIRST, LAST + 1)]
MONTHS_IN_WINTER = [(10, 0), (11, 0), (12, 0), (1, 1), (2, 1)]

DATASET = "reanalysis-era5-single-levels"
MAX_WORKERS = 5

OUT_DIR = "data/era5/training_data"
os.makedirs(OUT_DIR, exist_ok=True)


def build_request(kind, year, month):
    n_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]
    times = [f"{h:02d}:00" for h in range(0, 24, 3)]
    variables = (
        ["100m_u_component_of_wind", "100m_v_component_of_wind",
         "10m_u_component_of_wind", "10m_v_component_of_wind"]
        if kind == "winds" else ["forecast_surface_roughness"]
    )
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": str(year), "month": f"{month:02d}", "day": days, "time": times,
        "data_format": "netcdf",
    }


def target_path(kind, year, month):
    return f"{OUT_DIR}/era5_{kind}_{year}{month:02d}_3hourly.nc"


def fetch_one(item):
    kind, year, month = item
    out = target_path(kind, year, month)
    if os.path.exists(out):
        print(f"  [skip] {out} already exists", flush=True)
        return out
    print(f"  --> requesting {kind} {year}-{month:02d} ...", flush=True)
    client = cdsapi.Client()  # one client per thread
    client.retrieve(DATASET, build_request(kind, year, month), out)
    print(f"  done: {out}", flush=True)
    return out


def main():
    plan = []
    for wstart, wend in WINTERS:
        for month, yoff in MONTHS_IN_WINTER:
            year = wstart if yoff == 0 else wend
            plan.append(("winds", year, month))
            plan.append(("z0", year, month))

    print(f"Winters {WINTERS[0][0]}-{WINTERS[0][1]} .. {WINTERS[-1][0]}-{WINTERS[-1][1]} "
          f"({len(WINTERS)} winters, {len(plan)} target files, {MAX_WORKERS} parallel workers)",
          flush=True)

    done, failed = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, item): item for item in plan}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                done.append(fut.result())
            except Exception as exc:
                print(f"  FAILED {item}: {exc}", flush=True)
                failed.append(item)

    print(f"\n{len(done)} done/skipped, {len(failed)} failed.", flush=True)
    for item in failed:
        print(f"  missing: {item}", flush=True)
    print("ALL WINTERS DONE" if not failed else "COMPLETED WITH FAILURES", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
