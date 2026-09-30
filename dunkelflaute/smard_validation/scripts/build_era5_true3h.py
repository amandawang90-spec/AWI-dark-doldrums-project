"""
Turn the native-hourly ERA5 ssrd/tsr (Germany, winter 2024-2025) into genuine
3-hour accumulations, matching TCo1279-DART's convention -- instead of the
main pipeline's era5_ssrd_3h_*.nc / era5_tsr_3h_*.nc, which only ever sampled
the 1-hour accumulation ending at each 3-hourly mark (SSRD_ACCUM_SECONDS_ERA5
= 1*3600 in core/capacity_factor.py).

ERA5's own convention is "accumulated over the period ENDING at the stamp",
so the genuine 3-hour value at stamp T is the SUM of the 3 consecutive hourly
values at T-2h, T-1h, T (each already a 1-hour-ending accumulation) -- a
backward rolling sum, then keep only T in {00,03,...,21} UTC.

Edge case: 2024-10-01 00:00 needs 2024-09-30 22:00 and 23:00, which weren't
downloaded (outside the winter window). Solar/TOA flux is ~0 at that hour in
Germany regardless, so this one timestep is summed from whatever's available
(1 hour instead of 3) rather than dropped -- negligible in practice, noted
here rather than silently glossed over.

Output: smard_validation/data/era5_true3h/era5_{ssrd,tsr}_true3h_<YYYYMM>.nc
-- same variable names and Germany grid as the originals, so they're a
drop-in replacement for load_month() in a pipeline variant.
"""
import os

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
IN_DIR = f"{ROOT}/data/era5_hourly"
OUT_DIR = f"{ROOT}/data/era5_true3h"

MONTHS = ["202410", "202411", "202412", "202501", "202502"]


def build(var, file_tag):
    files = [f"{IN_DIR}/era5_{file_tag}_hourly_{m}.nc" for m in MONTHS]
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal", coords="minimal")
    ds = ds.sortby("valid_time")

    rolled = ds.rolling(valid_time=3, min_periods=1).sum()
    keep = rolled.valid_time.dt.hour % 3 == 0
    true3h = rolled.sel(valid_time=rolled.valid_time[keep.values])

    os.makedirs(OUT_DIR, exist_ok=True)
    for m in MONTHS:
        year, month = int(m[:4]), int(m[4:6])
        month_slice = true3h.sel(valid_time=(
            (true3h.valid_time.dt.year == year) & (true3h.valid_time.dt.month == month)
        ))
        out_path = f"{OUT_DIR}/era5_{file_tag}_true3h_{m}.nc"
        month_slice.astype("float32").to_netcdf(out_path)
        n = month_slice.dims["valid_time"]
        print(f"  {out_path}  ({n} timesteps)")


def main():
    print("ssrd -> true 3h:")
    build("ssrd", "ssrd")
    print("tsr -> true 3h:")
    build("tsr", "tsr")


if __name__ == "__main__":
    main()
