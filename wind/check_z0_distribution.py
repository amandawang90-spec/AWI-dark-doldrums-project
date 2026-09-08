"""
Sanity-check the ERA5 land z0 (fsr) distribution.

The earlier quick percentile check was computed on raw grid points of a regular
0.25-deg lat/lon grid, with no area weighting -- which massively over-counts polar
points (Antarctica, Greenland), where z0 is tiny. This redoes it area-weighted by
cos(lat), and breaks the field down into Davenport-style roughness classes so the
values can be judged against textbook expectations.
"""

import warnings

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

Z0_FILE = "data/era5_z0_20260830_0000.nc"
LSM_FILE = "data/era5_lsm_20260830_0000.nc"

# Davenport / Wieringa roughness classification (z0 in m)
CLASSES = [
    (0.0, 0.0002, "sea / ice, very smooth"),
    (0.0002, 0.005, "snow, mud flats, bare ice"),
    (0.005, 0.03, "open flat terrain, bare soil, short grass"),
    (0.03, 0.10, "grassland, low crops"),
    (0.10, 0.25, "high crops, scattered obstacles"),
    (0.25, 0.50, "parkland, bushes, many obstacles"),
    (0.50, 1.00, "forest edge, suburban"),
    (1.00, 2.00, "dense forest, city"),
    (2.00, np.inf, "high-rise city centre"),
]


def wpct(x, w, qs):
    """Weighted percentiles."""
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cw = 100.0 * (cw - 0.5 * w) / cw[-1]
    return np.interp(qs, cw, x)


z0ds = xr.open_dataset(Z0_FILE).squeeze()
fsr = z0ds["fsr"].values
lat = z0ds["latitude"].values
lsm = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values

area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(fsr)
land = (lsm >= 0.5) & np.isfinite(fsr)

x, w, la = fsr[land], area[land], np.broadcast_to(lat[:, None], fsr.shape)[land]

qs = [1, 25, 50, 75, 95, 99]
print("ERA5 land z0 (fsr), Aug 2026\n")
print(f"  {'':<22}" + "".join(f"{q:>10}%" for q in qs))
print(f"  {'raw grid points':<22}" + "".join(f"{v:>11.4g}" for v in np.percentile(x, qs)))
print(f"  {'area-weighted':<22}" + "".join(f"{v:>11.4g}" for v in wpct(x, w, qs)))

print(f"\n  land area poleward of 60 deg: {100 * w[np.abs(la) >= 60].sum() / w.sum():.1f}% of land area")
print(f"  but {100 * (np.abs(la) >= 60).sum() / len(la):.1f}% of land GRID POINTS")

print("\nDistribution by roughness class (share of land area):\n")
print(f"  {'z0 range (m)':<22}{'share':>9}   description")
print("  " + "-" * 74)
for lo, hi, desc in CLASSES:
    m = (x >= lo) & (x < hi)
    share = 100 * w[m].sum() / w.sum()
    if share < 0.05:
        continue
    rng = f"{lo:g} - {hi:g}" if np.isfinite(hi) else f">= {lo:g}"
    print(f"  {rng:<22}{share:>8.1f}%   {desc}")

print("\nMost common discrete values (share of land area):\n")
vals, counts = np.unique(np.round(x, 6), return_counts=True)
wsum = np.array([w[np.round(x, 6) == v].sum() for v in vals[np.argsort(-counts)][:8]])
top = vals[np.argsort(-counts)][:8]
for v, ws in sorted(zip(top, wsum), key=lambda t: -t[1]):
    print(f"  z0 = {v:<12.6g} {100 * ws / w.sum():>6.1f}% of land area")
