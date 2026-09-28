"""
Full-year backfill of era5_winds_*_3hourly.nc / era5_z0_*_3hourly.nc for
2015-2026 -- same variables and file convention as download_winters_parallel.py,
just every month instead of winter months (Oct-Feb) only.

2015-2025 currently only has Jan/Feb/Oct/Nov/Dec (winter); this fills in
Mar-Sep. 2026 currently only has Jan/Feb; this requests through Dec, but CDS
will simply fail whatever hasn't been published yet -- that's expected, not
a bug (ERA5 lags real time by a few days to a few months for some fields).

Already-existing files are skipped (same as download_winters_parallel.py), so
this is safe to run alongside/after it and to resubmit if interrupted.

Usage: python download_all_months_2015_2026.py [FIRST_YEAR] [LAST_YEAR]
  defaults to 2015 2026
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

FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
LAST = int(sys.argv[2]) if len(sys.argv) > 2 else 2026

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
    for year in range(FIRST, LAST + 1):
        for month in range(1, 13):
            plan.append(("winds", year, month))
            plan.append(("z0", year, month))

    print(f"Years {FIRST}-{LAST}, all 12 months ({len(plan)} target files, "
          f"{MAX_WORKERS} parallel workers)", flush=True)

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
    print("ALL MONTHS DONE" if not failed else "COMPLETED WITH FAILURES", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
