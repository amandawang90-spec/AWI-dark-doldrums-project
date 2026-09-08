"""
Inspect the surface roughness (sr) field in an IFS GRIB init file: is it a real,
spatially varying roughness climatology, or a flat placeholder constant?

Reports grid info, value range, uniqueness, and land/ocean percentiles, so the
field can be judged usable (or not) as z0 for log-law wind extrapolation.
"""

import sys
import warnings

import cfgrib
import numpy as np

warnings.filterwarnings("ignore")

gribfile = sys.argv[1] if len(sys.argv) > 1 else "/albedo/pool/oifs-43r3/TCO1279L137/ICMGGhf05INIT"
print(f"file: {gribfile}\n")

datasets = cfgrib.open_datasets(gribfile, backend_kwargs={"indexpath": ""})

for i, ds in enumerate(datasets):
    if "sr" not in ds.data_vars:
        continue

    sr = np.asarray(ds["sr"].values).ravel()
    attrs = ds["sr"].attrs
    print(f"--- dataset {i} ---")
    print("name:  ", attrs.get("GRIB_name"))
    print("units: ", attrs.get("GRIB_units"))
    print("grid:  ", attrs.get("GRIB_gridType"), "N =", attrs.get("GRIB_N"),
          "npoints =", attrs.get("GRIB_numberOfPoints"))
    print("shape: ", sr.shape)
    print(f"min/max/mean: {np.nanmin(sr):.6g} / {np.nanmax(sr):.6g} / {np.nanmean(sr):.6g}")

    n_unique = len(np.unique(sr[np.isfinite(sr)]))
    print("unique values:", n_unique)
    if n_unique <= 1:
        print("\n>> FLAT PLACEHOLDER - not usable as z0.\n")
        continue

    print("percentiles [1,25,50,75,99]:", np.percentile(sr[np.isfinite(sr)], [1, 25, 50, 75, 99]))

    if "lsm" in ds.data_vars:
        lsm = np.asarray(ds["lsm"].values).ravel()
        land, ocean = lsm >= 0.5, lsm < 0.5
        print(f"land fraction: {np.mean(land):.4f}")
        print("sr over land :", np.percentile(sr[land], [1, 25, 50, 75, 99]))
        print("sr over ocean:", np.percentile(sr[ocean], [1, 25, 50, 75, 99]))

    print("\n>> Spatially varying - usable as z0.\n")
