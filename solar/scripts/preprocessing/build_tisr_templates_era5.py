"""Build two reusable analytic-tisr templates for ERA5: one ordinary year (365
days), one leap year (366 days), on ERA5's real 0.25 deg grid (721 x 1440) and
the real 3-hour accumulation window of the era5_3h_mean files (DT=10800).

Same idea and same reasoning as build_tisr_templates.py (DART): tisr depends only
on lat, lon, time-of-day and day-of-year, so one template per calendar type is
exact -- see that script for the full argument.

DIFFERENCES FROM THE DART TEMPLATES
-------------------------------------------------------------------------------
* Grid: regular (latitude, longitude), 90 -> -90 and 0 -> 359.75, read from an
  actual ERA5 file, so it is identical to what the reconstruction indexes into.
  Small enough (1.04M cells) that no chunking over cells or multiprocessing is
  needed -- this loops over days in a single process.
* Time axis: ERA5 3-hourly stamps are 00,03,...,21 UTC, each the END of a
  3-hour window. Template step i is the stamp ref_year-01-01T00:00 + 3h*i, so
  index = (stamp - Jan 1 00:00 of its own year) / 3h, 0 .. 2919 (2927 if leap).
  (DART's templates start at 03:00 instead, because DART stamps are 03..24.)
* Step 0 is the window ending Jan 1 00:00, i.e. Dec 31 21:00-24:00 of the
  PREVIOUS year. A template cannot know that year's leap status, so step 0 is
  computed as the last evening of the reference year itself (end stamp
  ref_year+1-01-01T00:00). The astronomy differs from the true previous-year
  value only through day-of-year 365 vs 366 -- negligible for the same reason
  the whole template argument holds.
* N_SUBSTEPS = 12, the default toa_irradiance_accumulated uses on the ERA5 side
  of the pipeline (reconstruct_ssrd_v4.py line 65), so a template lookup
  reproduces what is computed on the fly today. DART's templates use 4.

OUTPUT
------
data/era5_analytical_tisr/tisr_template_era5_365day.nc   (reference year 2001)
data/era5_analytical_tisr/tisr_template_era5_366day.nc   (reference year 2000)
tisr(time_counter, latitude, longitude), float32, J m-2, one-day chunks.

Usage: python3 build_tisr_templates_era5.py
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))


import os
import time

import netCDF4 as nc
import numpy as np
import xarray as xr

from solar_geometry import toa_irradiance_accumulated

OUT_DIR = "data/era5_analytical_tisr"
GRID_SOURCE = "data/era5/era5_clouds_3h_198701.nc"
DT_SECONDS = 10800.0     # true 3-hour accumulation (era5_3h_mean), same as DART
N_SUBSTEPS = 12          # matches the on-the-fly ERA5 calculation in reconstruct_ssrd_v4.py
STEPS_PER_DAY = 8

TEMPLATES = [(2001, 365, "365day"), (2000, 366, "366day")]   # (ref year, ndays, tag)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def template_stamps(ref_year, ndays):
    """End-of-window stamps: Jan 1 00:00 + 3h*i. Step 0 is moved to Jan 1 00:00 of
    the NEXT year so its window is Dec 31 of the reference year (see docstring)."""
    stamps = (np.datetime64(f"{ref_year}-01-01T00:00:00", "s")
              + np.arange(ndays * STEPS_PER_DAY).astype("timedelta64[h]") * 3)
    stamps[0] = np.datetime64(f"{ref_year + 1}-01-01T00:00:00", "s")
    return stamps


def build_one(ref_year, ndays, tag, lat, lon):
    target = f"{OUT_DIR}/tisr_template_era5_{tag}.nc"
    log(f"--- {tag} (reference year {ref_year}, {ndays} days) -> {target}")
    stamps = template_stamps(ref_year, ndays)
    nt = stamps.size

    ds = nc.Dataset(target, "w", format="NETCDF4")
    ds.createDimension("time_counter", nt)
    ds.createDimension("latitude", lat.size)
    ds.createDimension("longitude", lon.size)
    v_lat = ds.createVariable("latitude", "f8", ("latitude",))
    v_lon = ds.createVariable("longitude", "f8", ("longitude",))
    v_time = ds.createVariable("time_counter", "f8", ("time_counter",))
    v_time.units = f"seconds since {ref_year}-01-01 00:00:00"
    v_time.calendar = "gregorian"
    v_tisr = ds.createVariable("tisr", "f4", ("time_counter", "latitude", "longitude"),
                               zlib=True, complevel=4,
                               chunksizes=(STEPS_PER_DAY, lat.size, lon.size))
    v_tisr.units = "J m-2"
    v_tisr.long_name = ("TOA incident shortwave, analytic (Spencer 1971 solar geometry), "
                        "3-hour accumulation ending at the stamp")
    v_tisr.comment = (f"Reusable ERA5-grid template for any {ndays}-day year; index = "
                      f"(stamp - Jan 1 00:00) / 3h. Step 0 is the window ending Jan 1 00:00 "
                      f"(Dec 31 evening). N_SUBSTEPS={N_SUBSTEPS}, DT={DT_SECONDS:.0f} s. "
                      f"Reference year {ref_year} itself carries no physical meaning.")
    v_lat[:] = lat
    v_lon[:] = lon
    # time_counter holds the nominal Jan 1 + 3h*i offsets (step 0 = Jan 1 00:00),
    # NOT the shifted step-0 stamp used for the astronomy
    v_time[:] = np.arange(nt) * DT_SECONDS

    t0 = time.time()
    for d in range(ndays):
        sl = slice(d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY)
        v_tisr[sl] = toa_irradiance_accumulated(lat, lon, stamps[sl],
                                                dt_seconds=DT_SECONDS, n_substeps=N_SUBSTEPS)
        if (d + 1) % 20 == 0 or d + 1 == ndays:
            el = time.time() - t0
            log(f"    day {d + 1}/{ndays}  elapsed {el / 60:.1f} min  "
                f"ETA {el / (d + 1) * (ndays - d - 1) / 60:.1f} min")
    ds.close()
    log(f"    done: {os.path.getsize(target) / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} min")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"reading ERA5 grid from {GRID_SOURCE}")
    g = xr.open_dataset(GRID_SOURCE)
    lat, lon = g.latitude.values, g.longitude.values
    g.close()
    log(f"grid: {lat.size} x {lon.size} = {lat.size * lon.size:,} cells")
    for ref_year, ndays, tag in TEMPLATES:
        build_one(ref_year, ndays, tag, lat, lon)
    log("both ERA5 templates done")


if __name__ == "__main__":
    main()
