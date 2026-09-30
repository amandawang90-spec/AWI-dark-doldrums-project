"""
Download native-hourly ERA5 ssrd + tisr for Germany, winter 2024-2025, so the
SMARD validation's solar leg can use a genuine 3-hour accumulation
(SSRD_ACCUM_SECONDS_ERA5 = 1*3600 today -- see core/capacity_factor.py --
because the main pipeline's era5_ssrd_3h_*.nc files only ever sampled the
1-hour accumulation ending at each 3-hourly mark, not the full 3-hour sum;
download_era5.py's own docstring documents this as an accepted limitation for
training the kt model, but it matters more directly here).

Restricted to the Germany download box (46.75-56.25N, 2.75-15.5E) rather than
global -- unlike the main solar pipeline, this is only for the SMARD
comparison, not for retraining the global model, so there's no reason to pull
the whole planet's worth of hourly data.

Output: smard_validation/data/era5_hourly/era5_ssrd_hourly_<YYYYMM>.nc (ssrd, tisr)

Requirements: same as solar/scripts/download/download_era5.py -- cdsapi >= 0.7.2,
credentials in ~/.cdsapirc.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
DATA_DIR = f"{ROOT}/data/era5_hourly"

SL = "reanalysis-era5-single-levels"
TIMES = [f"{h:02d}:00" for h in range(24)]   # every hour, not just every 3rd
DAYS = [f"{d:02d}" for d in range(1, 32)]
AREA = [56.25, 2.75, 46.75, 15.5]            # [north, west, south, east] -- Germany download box
VARIABLES = ["surface_solar_radiation_downwards", "toa_incident_solar_radiation"]

MONTHS = ["202410", "202411", "202412", "202501", "202502"]


def build_request(year, month):
    return {
        "variable": VARIABLES, "year": year, "month": month, "day": DAYS, "time": TIMES,
        "product_type": "reanalysis", "data_format": "netcdf", "download_format": "unarchived",
        "area": AREA,
    }


def main():
    force = "--force" in sys.argv
    dry = "--list" in sys.argv

    plan = [(f"{DATA_DIR}/era5_ssrd_hourly_{m}.nc", m) for m in MONTHS]

    if dry:
        for target, _ in plan:
            print(f"  {'present' if os.path.exists(target) else 'MISSING':8s} {target}")
        return 0

    os.makedirs(DATA_DIR, exist_ok=True)
    import cdsapi
    from concurrent.futures import ThreadPoolExecutor, as_completed

    to_fetch = [(t, m) for t, m in plan if force or not os.path.exists(t)]
    for t, _ in plan:
        if (t, _) not in to_fetch:
            print(f"--- skip (exists) {t}", flush=True)

    def fetch_one(item):
        target, stamp = item
        client = cdsapi.Client()
        print(f"--> {target}", flush=True)
        client.retrieve(SL, build_request(stamp[:4], stamp[4:6]), target)
        return target

    done, failed = [], []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_one, item): item[0] for item in to_fetch}
        for fut in as_completed(futures):
            target = futures[fut]
            try:
                fut.result()
                done.append(target)
            except Exception as exc:
                print(f"    FAILED {target}: {exc}", flush=True)
                failed.append(target)

    print(f"\n{len(done)} downloaded, {len(failed)} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
