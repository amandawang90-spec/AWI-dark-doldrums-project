"""
Are the TCO1279 subgrid-orography fields the same thing as surface roughness?

Compares, on actual data:
  - ERA5 'fsr' (forecast surface roughness, paramId 244) -- the z0 actually used
    as the log-law roughness length in this project
  - TCO1279 'sdor'  (standard deviation of subgrid orography, paramId 160)
  - TCO1279 'sdfor' (std dev of filtered subgrid orography, paramId 74)

Both carry units of metres, but they describe different scales: z0 is an
aerodynamic roughness length (the height where the log wind profile goes to zero),
while sdor/sdfor measure terrain elevation spread inside a grid box.
"""

import warnings

import cfgrib
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

GRIBFILE = "/albedo/pool/oifs-43r3/TCO1279L137/ICMGGhf05INIT"
ERA5_Z0 = "data/era5_z0_20260830_0000.nc"
ERA5_LSM = "data/era5_lsm_20260830_0000.nc"

PCTS = [1, 25, 50, 75, 95, 99, 100]


def wpct(x, w, qs):
    """Area-weighted percentiles.

    ERA5 sits on a regular lat/lon grid, so raw point percentiles over-count polar
    latitudes badly (48% of land points are poleward of 60N/S, but only 21% of land
    area). TCO1279's reduced Gaussian grid is quasi-equal-area, so weights there are
    ~uniform and this reduces to the plain percentile.
    """
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cw = 100.0 * (cw - 0.5 * w) / cw[-1]
    return np.interp(qs, cw, x)


def describe(name, arr, mask, label, weights=None):
    good = mask & np.isfinite(arr)
    a = arr[good]
    w = np.ones_like(a) if weights is None else weights[good]
    p = wpct(a, w, PCTS)
    print(f"  {name:<28}{label:<9}" + "".join(f"{v:>12.4g}" for v in p))


def show_meta(ds, name):
    a = ds[name].attrs
    print(f"  {name:<8} paramId={str(a.get('GRIB_paramId')):<8}"
          f"units={str(a.get('GRIB_units')):<10}"
          f"cfName={str(a.get('GRIB_cfName')):<28}{a.get('GRIB_name')}")


print("GRIB metadata as recorded by ecCodes (authoritative parameter table)\n")
_dss = cfgrib.open_datasets(GRIBFILE, backend_kwargs={"indexpath": ""})
_ds = next(d for d in _dss if "sdor" in d.data_vars)
for _n in ["sr", "sdor", "sdfor", "isor", "slor"]:
    if _n in _ds.data_vars:
        show_meta(_ds, _n)

_era5 = xr.open_dataset(ERA5_Z0).squeeze()["fsr"]
print(f"  {'fsr':<8} (ERA5)  units={_era5.attrs.get('units')!s:<10}"
      f"{_era5.attrs.get('long_name', _era5.attrs.get('standard_name'))}")

print("\n" + "=" * 116 + "\n")
print("Percentiles of each field (metres), split land / ocean\n")
print(f"  {'field':<28}{'region':<9}" + "".join(f"{p:>12}" for p in
      ["1%", "25%", "50%", "75%", "95%", "99%", "max"]))
print("  " + "-" * 112)

# ---- ERA5 forecast surface roughness: the z0 we validated the log-law with ----
_z0ds = xr.open_dataset(ERA5_Z0).squeeze()
_lat = _z0ds["latitude"].values
fsr2d = _z0ds["fsr"].values
area_era5 = (np.cos(np.deg2rad(_lat))[:, None] * np.ones_like(fsr2d)).ravel()
fsr = fsr2d.ravel()
lsm_era5 = xr.open_dataset(ERA5_LSM).squeeze()["lsm"].values.ravel()
describe("ERA5 fsr (z0, param 244)", fsr, lsm_era5 >= 0.5, "land", area_era5)
describe("ERA5 fsr (z0, param 244)", fsr, lsm_era5 < 0.5, "ocean", area_era5)

# ---- TCO1279 subgrid orography ----
datasets = cfgrib.open_datasets(GRIBFILE, backend_kwargs={"indexpath": ""})
ds = next(d for d in datasets if "sdor" in d.data_vars)
lsm_ifs = np.asarray(ds["lsm"].values).ravel()

for name, pid in [("sdor", 160), ("sdfor", 74)]:
    arr = np.asarray(ds[name].values).ravel()
    describe(f"TCO1279 {name} (param {pid})", arr, lsm_ifs >= 0.5, "land")
    describe(f"TCO1279 {name} (param {pid})", arr, lsm_ifs < 0.5, "ocean")

# ---- the ratio that answers the question ----
_lm = (lsm_era5 >= 0.5) & np.isfinite(fsr)
fsr_land_med = wpct(fsr[_lm], area_era5[_lm], [50])[0]
sdor_land_med = np.nanmedian(np.asarray(ds["sdor"].values).ravel()[lsm_ifs >= 0.5])
print(f"\n  median over land: ERA5 fsr = {fsr_land_med:.4g} m, "
      f"TCO1279 sdor = {sdor_land_med:.4g} m  "
      f"-> sdor is ~{sdor_land_med / fsr_land_med:.0f}x larger")
