"""
Download native-hourly ERA5 tsr for Germany, winter 2024-2025 -- companion to
download_era5_ssrd_hourly.py. tsr feeds the "T = tsr/tisr" feature the solar
reconstruction model uses, and shares ssrd's same 1-hour-accumulation-window
issue (see core/capacity_factor.py's DT_ERA5 comment: "ERA5: tsr/ssrd/tisr:
1h accumulation"), so fixing ssrd's window without also fixing tsr would
leave the reconstruction's own input feature inconsistent with its target.

tisr (the other half of the T ratio) needs no download -- it's computed
locally from solar geometry (toa_irradiance_accumulated()), so making it a
true 3-hour window is just a dt_seconds change, not new data.

Output: smard_validation/data/era5_hourly/era5_tsr_hourly_<YYYYMM>.nc (tsr)
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
DATA_DIR = f"{ROOT}/data/era5_hourly"

SL = "reanalysis-era5-single-levels"
TIMES = [f"{h:02d}:00" for h in range(24)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
AREA = [56.25, 2.75, 46.75, 15.5]            # same Germany download box as the ssrd script
VARIABLES = ["top_net_solar_radiation"]

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

    plan = [(f"{DATA_DIR}/era5_tsr_hourly_{m}.nc", m) for m in MONTHS]

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
