"""
Map the roughness situation, three panels:

  1. TCo1279 `sr`   -- the surface roughness field in our own init file
  2. TCo1279 `sdor` -- subgrid orography, a CONTROL: this field is populated, so if
                       it maps fine while `sr` does not, the flat `sr` map is the data
                       and not a bug in the reduced-Gaussian regridding
  3. ERA5 `fsr`     -- what a genuinely populated z0 field looks like, for comparison

TCo1279 lives on a reduced Gaussian grid (1D array of 6.6M points with lat/lon
coordinate arrays), so it is binned onto a regular lat/lon mesh purely for display.
"""

import warnings

import cfgrib
import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

warnings.filterwarnings("ignore")

GRIBFILE = "/albedo/pool/oifs-43r3/TCO1279L137/ICMGGhf05INIT"
ERA5_Z0 = "data/era5_z0_20260830_0000.nc"

DLON = DLAT = 0.25
lon_edges = np.arange(-180.0, 180.0 + DLON, DLON)
lat_edges = np.arange(-90.0, 90.0 + DLAT, DLAT)


def to_regular(lon, lat, values):
    """Bin scattered reduced-Gaussian points onto a regular lat/lon mesh (mean)."""
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    total, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges], weights=values)
    count, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
    out = np.full_like(total, np.nan)
    good = count > 0
    out[good] = total[good] / count[good]
    return out


print("loading TCo1279 fields ...")
datasets = cfgrib.open_datasets(GRIBFILE, backend_kwargs={"indexpath": ""})
ds = next(d for d in datasets if "sdor" in d.data_vars)

lat_ifs = np.asarray(ds["sdor"].coords["latitude"].values).ravel()
lon_ifs = np.asarray(ds["sdor"].coords["longitude"].values).ravel()
sr = np.asarray(ds["sr"].values).ravel()
sdor = np.asarray(ds["sdor"].values).ravel()

print(f"  sr:   min {sr.min():.6g}  max {sr.max():.6g}  unique {len(np.unique(sr))}")
print(f"  sdor: min {sdor.min():.6g}  max {sdor.max():.6g}  unique {len(np.unique(sdor))}")

sr_map = to_regular(lon_ifs, lat_ifs, sr)
sdor_map = to_regular(lon_ifs, lat_ifs, sdor)

print("loading ERA5 fsr ...")
z0ds = xr.open_dataset(ERA5_Z0).squeeze()
fsr = z0ds["fsr"].values
lat_e = z0ds["latitude"].values
lon_e = z0ds["longitude"].values
# ERA5 comes on 0..360 with latitude descending; roll to -180..180, ascending
roll = np.sum(lon_e >= 180.0)
fsr = np.roll(fsr, roll, axis=1)[::-1, :]

fig, axes = plt.subplots(3, 1, figsize=(13, 15))

# --- panel 1: TCo1279 sr ---
ax = axes[0]
pc = ax.pcolormesh(lon_edges, lat_edges, sr_map, cmap="viridis",
                   norm=LogNorm(vmin=1e-4, vmax=2.0), shading="flat")
fig.colorbar(pc, ax=ax, label="z0 (m), log scale")
ax.set_title(f"1. TCo1279 'sr' (surface roughness, param 173) from ICMGGhf05INIT\n"
             f"constant {sr.min():.4g} m everywhere -- {len(np.unique(sr))} unique value: "
             f"UNPOPULATED PLACEHOLDER")

# --- panel 2: TCo1279 sdor (control) ---
ax = axes[1]
pc = ax.pcolormesh(lon_edges, lat_edges, sdor_map, cmap="magma",
                   norm=Normalize(vmin=0, vmax=300), shading="flat")
fig.colorbar(pc, ax=ax, label="std dev of subgrid orography (m)")
ax.set_title("2. CONTROL -- TCo1279 'sdor' (param 160), same file, same grid, same "
             "regridding\nthis one has real structure, so panel 1 being flat is the "
             "data, not the plotting")

# --- panel 3: ERA5 fsr ---
ax = axes[2]
pc = ax.pcolormesh(lon_e - 180.0, lat_e[::-1], fsr, cmap="viridis",
                   norm=LogNorm(vmin=1e-4, vmax=2.0), shading="auto")
fig.colorbar(pc, ax=ax, label="z0 (m), log scale")
ax.set_title("3. ERA5 'fsr' (forecast surface roughness, param 244), 0.25 deg\n"
             "what a genuinely populated z0 field looks like -- same colour scale as panel 1")

for ax in axes:
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

plt.tight_layout()
out = "figures/z0_tco1279_vs_era5.png"
plt.savefig(out, dpi=110)
print(f"saved {out}")
