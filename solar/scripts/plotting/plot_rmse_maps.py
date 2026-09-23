"""RMSE maps (winter) from models/model_v2/evaluation/rmse_cells_<tag>.npz"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

tag = sys.argv[1] if len(sys.argv) > 1 else "DJF"
label = {"DJF": "Winter (Dec-Jan-Feb)", "all": "All months", "rescaled_DJF": "Winter (Dec-Jan-Feb), after exact monthly rescale"}.get(tag, tag)
d = np.load(f"models/model_v2/evaluation/rmse_cells_{tag}.npz")
nm = int(d["Global_nmonths"])
INK, SEC, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
seq = LinearSegmentedColormap.from_list("blue_seq", ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
div = LinearSegmentedColormap.from_list("bwr_gray", ["#2a78d6", "#f0efec", "#e34948"])


def fields(piece):
    n = d[f"{piece}_n"]; lat = d[f"{piece}_lat"]; lon = d[f"{piece}_lon"]
    lon0 = np.where(lon > 180, lon - 360, lon); o = np.argsort(lon0)
    r = {m: np.sqrt(d[f"{piece}_{m}_sse"] / np.maximum(n, 1)) / 3600.0 for m in ("v1", "v2")}
    for m in r:
        r[m] = np.where(n > 0, r[m], np.nan)[:, o]
    w = np.cos(np.deg2rad(lat))[:, None] * (n > 0)
    pooled = {m: np.sqrt((w * d[f"{piece}_{m}_sse"]).sum() / (w * n).sum()) / 3600.0 for m in ("v1", "v2")}
    return lat, lon0[o], r, pooled


def panel(ax, lon, lat, z, cmap, vmin, vmax, title, extent=None, borders=False):
    if extent: ax.set_extent(extent, crs=ccrs.PlateCarree())
    else: ax.set_global()
    pcm = ax.pcolormesh(lon, lat, z, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor=INK, zorder=3)
    if borders: ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor=SEC, zorder=3)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    return pcm


# ---- Figure 1: global ----
lat, lon, r, pooled = fields("Global")
fig = plt.figure(figsize=(15, 9.6), facecolor=SURF)
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.16, wspace=0.04, left=0.03, right=0.97, top=0.86, bottom=0.08)
P = ccrs.Robinson()
ax1, ax2, ax3 = fig.add_subplot(gs[0, 0], projection=P), fig.add_subplot(gs[0, 1], projection=P), fig.add_subplot(gs[1, :], projection=P)
vmax = 90
pc = panel(ax1, lon, lat, r["v1"], seq, 0, vmax, f"v1     area-weighted mean {pooled['v1']:.1f} W/m²")
panel(ax2, lon, lat, r["v2"], seq, 0, vmax, f"v2     area-weighted mean {pooled['v2']:.1f} W/m²")
pd = panel(ax3, lon, lat, r["v2"] - r["v1"], div, -25, 25, "Change in RMSE, v2 minus v1")
cb = fig.colorbar(pc, ax=[ax1, ax2], orientation="horizontal", pad=0.03, shrink=0.5, extend="max")
cb.set_label("RMSE of 3-hourly ssrd vs real ERA5  [W/m²]", color=SEC); cb.ax.tick_params(colors=MUTED)
cb2 = fig.colorbar(pd, ax=ax3, orientation="horizontal", pad=0.04, shrink=0.4, extend="both")
cb2.set_label("W/m²      blue = v2 closer to the truth      red = v2 further from it", color=SEC); cb2.ax.tick_params(colors=MUTED)
fig.text(0.03, 0.955, f"Where the full model misses real 3-hourly sunlight: {label}", fontsize=16, fontweight="bold", color=INK)
fig.text(0.03, 0.925, f"RMSE per grid cell over {nm} held-out months, 2015-2025 (every month scored by a model that never trained on it)", fontsize=10.5, color=SEC)
fig.savefig(f"figures/models/rmse_map_{tag}_global.png", dpi=130, facecolor=SURF); plt.close(fig)

# ---- Figure 2: Europe and Korea zoom ----
fig = plt.figure(figsize=(15, 9.2), facecolor=SURF)
gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.05, left=0.03, right=0.97, top=0.88, bottom=0.07, height_ratios=[1.15, 1])
for row, (piece, ext, vmx, dv) in enumerate((("Europe", [-25, 40, 35, 71], 70, 20), ("Korea", [124, 131, 33, 43], 90, 25))):
    lat, lon, r, pooled = fields(piece)
    axs = [fig.add_subplot(gs[row, c], projection=ccrs.PlateCarree()) for c in range(3)]
    pa = panel(axs[0], lon, lat, r["v1"], seq, 0, vmx, f"{piece}: v1    mean {pooled['v1']:.1f} W/m²", ext, True)
    panel(axs[1], lon, lat, r["v2"], seq, 0, vmx, f"{piece}: v2    mean {pooled['v2']:.1f} W/m²", ext, True)
    pb = panel(axs[2], lon, lat, r["v2"] - r["v1"], div, -dv, dv, f"{piece}: v2 minus v1", ext, True)
    c1 = fig.colorbar(pa, ax=axs[:2], orientation="horizontal", pad=0.07, shrink=0.6, extend="max", aspect=40)
    c1.set_label("RMSE [W/m²]", color=SEC); c1.ax.tick_params(colors=MUTED)
    c2 = fig.colorbar(pb, ax=axs[2], orientation="horizontal", pad=0.07, shrink=0.9, extend="both", aspect=25)
    c2.set_label("change [W/m²]  (blue = v2 better)", color=SEC); c2.ax.tick_params(colors=MUTED)
fig.text(0.03, 0.945, f"RMSE of 3-hourly ssrd at full 0.25° resolution, {label}", fontsize=16, fontweight="bold", color=INK)
fig.savefig(f"figures/models/rmse_map_{tag}_regions.png", dpi=130, facecolor=SURF); plt.close(fig)

print("pooled RMSE W/m2 (area-weighted):")
for piece in ("Global", "Europe", "Korea"):
    _, _, _, p = fields(piece); print(f"  {piece:7s} v1 {p['v1']:.1f}   v2 {p['v2']:.1f}")
# Germany from the Europe box
lat, lon, r, _ = fields("Europe")
n = d["Europe_n"]; latg = d["Europe_lat"]; lon0 = np.where(d["Europe_lon"] > 180, d["Europe_lon"] - 360, d["Europe_lon"])
mk = ((latg[:, None] >= 47.3) & (latg[:, None] <= 55.1) & (lon0[None, :] >= 5.9) & (lon0[None, :] <= 15.0))
w = np.cos(np.deg2rad(latg))[:, None] * mk
print("  Germany", " ".join(f"{m} {np.sqrt((w*d[f'Europe_{m}_sse']).sum()/(w*n).sum())/3600:.1f}" for m in ("v1", "v2")))
