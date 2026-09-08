"""
Inspect the orography-related fields in an IFS GRIB init file.

IFS carries subgrid-orography descriptors on the gridpoint file:
  sdor  - standard deviation of subgrid orography (m)
  sdfor - standard deviation of filtered subgrid orography (m)
  isor  - anisotropy of subgrid orography
  anor  - angle of subgrid orography (rad)
  slor  - slope of subgrid orography
and the resolved orography itself as surface geopotential (z, m2/s2) -> height = z/g.

Reports grid info, value range and land/ocean percentiles for each, so it is clear
which fields are properly populated and usable.
"""

import sys
import warnings

import cfgrib
import numpy as np

warnings.filterwarnings("ignore")

G = 9.80665
OROG_FIELDS = ["sdor", "sdfor", "isor", "anor", "slor", "z", "orog"]

gribfile = sys.argv[1] if len(sys.argv) > 1 else "/albedo/pool/oifs-43r3/TCO1279L137/ICMGGhf05INIT"
print(f"file: {gribfile}\n")

datasets = cfgrib.open_datasets(gribfile, backend_kwargs={"indexpath": ""})

# land-sea mask, wherever it lives in this file, for land/ocean stratification
lsm = None
for ds in datasets:
    if "lsm" in ds.data_vars:
        lsm = np.asarray(ds["lsm"].values).ravel()
        break

found_any = False
for i, ds in enumerate(datasets):
    present = [f for f in OROG_FIELDS if f in ds.data_vars]
    if not present:
        continue
    found_any = True
    print(f"=== dataset {i}: {present} ===")

    for name in present:
        arr = np.asarray(ds[name].values).ravel()
        attrs = ds[name].attrs
        finite = np.isfinite(arr)
        n_unique = len(np.unique(arr[finite]))

        print(f"\n-- {name} --")
        print("  name: ", attrs.get("GRIB_name"))
        print("  units:", attrs.get("GRIB_units"))
        print("  grid: ", attrs.get("GRIB_gridType"), "N =", attrs.get("GRIB_N"),
              "npoints =", attrs.get("GRIB_numberOfPoints"))
        print(f"  min/max/mean: {np.nanmin(arr):.6g} / {np.nanmax(arr):.6g} / {np.nanmean(arr):.6g}")
        print(f"  unique values: {n_unique}")

        if n_unique <= 1:
            print("  >> FLAT PLACEHOLDER - not usable.")
            continue

        print("  percentiles [1,25,50,75,99]:",
              np.round(np.percentile(arr[finite], [1, 25, 50, 75, 99]), 4))

        if name == "z":
            h = arr / G
            print(f"  as height (z/g): min {np.nanmin(h):.1f} m, max {np.nanmax(h):.1f} m")

        if lsm is not None and lsm.shape == arr.shape:
            land, ocean = lsm >= 0.5, lsm < 0.5
            print("  over land :", np.round(np.percentile(arr[land], [25, 50, 75, 99]), 4))
            print("  over ocean:", np.round(np.percentile(arr[ocean], [25, 50, 75, 99]), 4))

        print("  >> spatially varying - usable.")
    print()

if not found_any:
    print("No orography fields found in this file.")
    print("Variables present per dataset:")
    for i, ds in enumerate(datasets):
        print(f"  dataset {i}: {list(ds.data_vars.keys())}")
