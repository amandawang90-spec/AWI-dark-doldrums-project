"""
Map the TCo1279-DART seasonal scaling factors for the two dark-doldrum resource
drivers: 10 m wind speed (ws) and total cloud cover (tcc), DJF and JJA.

The files carry no metadata. The t2 scaling field has a global median of ~0.8 with
polar maxima of 4-5, which identifies these as pattern-scaling coefficients: local
change per 1 K of global-mean warming. ws is then m/s per K; tcc is cloud fraction
per K, shown here in percentage points per K.

The German and Korean study domains from the proposal are outlined.
"""

import warnings

import cartopy.crs as ccrs
import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

DATA = "/work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART_scaled_changes"
COARSEN = 4  # 5136x2560 -> 1284x640 (~0.28 deg), plenty for a global panel

# name, file key, factor to display units, colorbar label, symmetric limit
FIELDS = [
    ("10 m wind speed", "ws", 1.0, "m/s per K of global warming", 0.3),
    ("Total cloud cover", "tcc", 100.0, "% points per K of global warming", 2.0),
]
SEASONS = ["DJF", "JJA"]

DOMAINS = {  # (lon0, lon1, lat0, lat1)
    "Germany": (6, 15, 47, 55),
    "South Korea": (124, 132, 33, 39),
}

# Diverging: blue arm (decrease) <- neutral gray -> red arm (increase), equal steps
CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f4b0ab", "#e34948", "#8c1d1b"],
    N=256,
)
INK = "#2b2b29"
MUTED = "#6b6b66"


def load(var, season):
    ds = xr.open_dataset(f"{DATA}/scaling_results_TCo1279_{var}_seasons.nc")
    da = ds[f"{var}_{season}_scaling"].coarsen(lat=COARSEN, lon=COARSEN, boundary="trim").mean()
    return da.lon.values, da.lat.values, da.values


fig, axes = plt.subplots(
    2, 2, figsize=(15, 7.0),
    gridspec_kw={"hspace": 0.12, "wspace": 0.04},
    subplot_kw={"projection": ccrs.Robinson(central_longitude=60)},
)

for row, (name, var, factor, unit, lim) in enumerate(FIELDS):
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for col, season in enumerate(SEASONS):
        ax = axes[row, col]
        lon, lat, val = load(var, season)
        print(f"{var} {season}: p1 {np.nanpercentile(val*factor, 1):.3g}  "
              f"p99 {np.nanpercentile(val*factor, 99):.3g}")
        pc = ax.pcolormesh(lon, lat, val * factor, cmap=CMAP, norm=norm,
                           shading="nearest", transform=ccrs.PlateCarree(),
                           rasterized=True)
        ax.coastlines(linewidth=0.4, color=INK)
        ax.set_global()
        for label, (x0, x1, y0, y1) in DOMAINS.items():
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=INK, linewidth=1.2,
                                   transform=ccrs.PlateCarree()))
        ax.set_title(f"{name}, {season}", fontsize=12, color=INK, loc="left")
    cb = fig.colorbar(pc, ax=axes[row, :].tolist(), orientation="vertical",
                      shrink=0.85, pad=0.02, extend="both")
    cb.set_label(unit, color=MUTED)
    cb.outline.set_visible(False)

fig.suptitle("TCo1279-DART scaling factors: local change per K of global-mean warming",
             fontsize=14, color=INK, x=0.06, y=0.97, ha="left")
fig.text(0.06, 0.03,
         "Red = increase, blue = decrease. Boxes: Germany (47-55N, 6-15E) and "
         "South Korea (33-39N, 124-132E). Colour scale clipped at the ~1st/99th percentile.",
         fontsize=9, color=MUTED)

out = "figures/scaling_ws_tcc_TCo1279.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
print(f"saved {out}")
