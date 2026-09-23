"""v2 vs v3area RMSE maps, from the already-computed models/model_v2/evaluation/v3_eval_ondjf.npz
(no new compute -- v2's per-cell sums are already stored there alongside v3's)."""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

GRP = "ondjf"
d = np.load(f"models/model_v2/evaluation/v3_eval_{GRP}.npz")
INK, SEC, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
seq = LinearSegmentedColormap.from_list("blue_seq", ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
div = LinearSegmentedColormap.from_list("bwr_gray", ["#2a78d6", "#f0efec", "#e34948"])


def fld(piece, mod):
    n = d[f"{GRP}_{piece}_n"]; lat, lon = d[f"{piece}_lat"], d[f"{piece}_lon"]
    lon0 = np.where(lon > 180, lon - 360, lon); o = np.argsort(lon0)
    r = np.where(n > 0, np.sqrt(d[f"{GRP}_{piece}_{mod}_sse"] / np.maximum(n, 1)) / 3600, np.nan)
    return lat, lon0[o], r[:, o]


def pooled(piece, mod, box=None):
    n, sse = d[f"{GRP}_{piece}_n"], d[f"{GRP}_{piece}_{mod}_sse"]
    lat, lon = d[f"{piece}_lat"], d[f"{piece}_lon"]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    if box:
        lon0 = np.where(lon > 180, lon - 360, lon)
        w = w * ((lat[:, None] >= box[0]) & (lat[:, None] <= box[1]) & (lon0[None, :] >= box[2]) & (lon0[None, :] <= box[3]))
    return np.sqrt((w * sse).sum() / (w * n).sum()) / 3600


def panel(ax, lon, lat, z, cmap, lo, hi, title, ext=None, borders=False):
    ax.set_extent(ext, crs=ccrs.PlateCarree()) if ext else ax.set_global()
    pc = ax.pcolormesh(lon, lat, z, transform=ccrs.PlateCarree(), cmap=cmap, vmin=lo, vmax=hi)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor=INK, zorder=3)
    if borders: ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor=SEC, zorder=3)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    return pc


lat, lon, r2 = fld("Global", "v2"); _, _, r3 = fld("Global", "v3area")
fig = plt.figure(figsize=(15, 9.4), facecolor=SURF)
gs = GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.04, left=0.03, right=0.97, top=0.87, bottom=0.11)
axs = [fig.add_subplot(gs[i // 2, i % 2], projection=ccrs.Robinson()) for i in range(4)]
pc = panel(axs[0], lon, lat, r2, seq, 0, 90, f"v2 (all-season)   mean {pooled('Global','v2'):.1f} W/m²")
panel(axs[1], lon, lat, r3, seq, 0, 90, f"v3 area-weighted (ONDJF-only)   mean {pooled('Global','v3area'):.1f} W/m²")
axs[2].axis("off")
pd = panel(axs[3], lon, lat, r3 - r2, div, -30, 30, "Change: v3 minus v2")
cax1 = fig.add_axes([0.06, 0.055, 0.4, 0.018]); cax2 = fig.add_axes([0.58, 0.055, 0.32, 0.018])
cb = fig.colorbar(pc, cax=cax1, orientation="horizontal", extend="max"); cb.set_label("RMSE after exact monthly rescale [W/m²]", color=SEC)
cb2 = fig.colorbar(pd, cax=cax2, orientation="horizontal", extend="both"); cb2.set_label("change, v3 minus v2 [W/m²]  (blue = v3 closer to the truth)", color=SEC)
fig.text(0.03, 0.945, "v2 vs v3: where does the ONDJF+mask retrain actually help?", fontsize=16, fontweight="bold", color=INK)
fig.text(0.03, 0.915, "Oct-Feb held-out months, 2015-2025; south of 60°S and above 3000 m excluded from scoring", fontsize=10.5, color=SEC)
fig.savefig("figures/v2_vs_v3_map_global.png", dpi=125, facecolor=SURF); plt.close(fig)

fig = plt.figure(figsize=(15, 9.2), facecolor=SURF)
gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.05, left=0.03, right=0.97, top=0.89, bottom=0.07, height_ratios=[1.15, 1])
for row, (piece, ext, vmx, dv) in enumerate((("Europe", [-25, 40, 35, 71], 60, 20), ("Korea", [124, 131, 33, 43], 70, 25))):
    lat, lon, a = fld(piece, "v2"); _, _, b = fld(piece, "v3area")
    ax = [fig.add_subplot(gs[row, c], projection=ccrs.PlateCarree()) for c in range(3)]
    box = (47.3, 55.1, 5.9, 15.0) if piece == "Europe" else None
    pa = panel(ax[0], lon, lat, a, seq, 0, vmx, f"{piece}: v2   mean {pooled(piece,'v2'):.1f} W/m²", ext, True)
    panel(ax[1], lon, lat, b, seq, 0, vmx, f"{piece}: v3 area-wtd   mean {pooled(piece,'v3area'):.1f} W/m²", ext, True)
    pb = panel(ax[2], lon, lat, b - a, div, -dv, dv, f"{piece}: v3 minus v2", ext, True)
    c1 = fig.colorbar(pa, ax=ax[:2], orientation="horizontal", pad=0.07, shrink=0.6, extend="max", aspect=40); c1.set_label("RMSE [W/m²]", color=SEC)
    c2 = fig.colorbar(pb, ax=ax[2], orientation="horizontal", pad=0.07, shrink=0.9, extend="both", aspect=25); c2.set_label("change [W/m²] (blue = v3 better)", color=SEC)
fig.text(0.03, 0.945, "RMSE at full 0.25° resolution, after the exact monthly rescale, Oct-Feb", fontsize=15, fontweight="bold", color=INK)
fig.savefig("figures/v2_vs_v3_map_regions.png", dpi=125, facecolor=SURF); plt.close(fig)
print("saved figures/v2_vs_v3_map_global.png and figures/v2_vs_v3_map_regions.png")
for piece, box in (("Global", None), ("Europe", None), ("Europe", (47.3, 55.1, 5.9, 15.0)), ("Korea", None)):
    print(piece, box, "v2:", round(float(pooled(piece, "v2", box)), 1), "v3area:", round(float(pooled(piece, "v3area", box)), 1))
