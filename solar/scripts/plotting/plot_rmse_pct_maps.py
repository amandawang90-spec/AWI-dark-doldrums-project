"""Maps of RMSE as a percentage of the real monthly mean ssrd, per individual winter month
(Oct,Nov,Dec,Jan,Feb) and pooled DJF, for the given model (default v3area).
After the exact monthly rescale; south of 60S and above 3000 m excluded."""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

MOD = sys.argv[1] if len(sys.argv) > 1 else "v3area"
d = np.load("models/model_v2/evaluation/v3_eval_ondjf.npz")
INK, SEC, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
# sequential: light -> dark red-orange (percentage-of-signal is a "badness" magnitude, distinct hue from the amber value map and the blue RMSE maps)
pct_seq = LinearSegmentedColormap.from_list("pct_seq", ["#fef6f4", "#fbd4c9", "#f19d84", "#dc6a4a", "#a83a26", "#6e1f11"])
GROUPS = [("m10", "October"), ("m11", "November"), ("m12", "December"), ("m1", "January"),
          ("m2", "February"), ("ondjf", "Oct-Feb pooled (every month)")]


def field(grp, piece):
    n = d[f"{grp}_{piece}_n"]; sr = d[f"{grp}_{piece}_sr"]; sse = d[f"{grp}_{piece}_{MOD}_sse"]
    lat, lon = d[f"{piece}_lat"], d[f"{piece}_lon"]
    lon0 = np.where(lon > 180, lon - 360, lon); o = np.argsort(lon0)
    mean = np.where(n > 0, sr / np.maximum(n, 1), np.nan)
    rmse = np.where(n > 0, np.sqrt(sse / np.maximum(n, 1)), np.nan)
    pct = 100 * rmse / np.maximum(mean, 1e-9)
    return lat, lon0[o], pct[:, o]


def pooled_pct(grp, piece, box=None):
    n, sr, sse = d[f"{grp}_{piece}_n"], d[f"{grp}_{piece}_sr"], d[f"{grp}_{piece}_{MOD}_sse"]
    lat, lon = d[f"{piece}_lat"], d[f"{piece}_lon"]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    if box:
        lon0 = np.where(lon > 180, lon - 360, lon)
        w = w * ((lat[:, None] >= box[0]) & (lat[:, None] <= box[1]) & (lon0[None, :] >= box[2]) & (lon0[None, :] <= box[3]))
    W = (w * n).sum(); mean = (w * sr).sum() / W; rmse = np.sqrt((w * sse).sum() / W)
    return 100 * rmse / mean


def panel(ax, lon, lat, z, lo, hi, title):
    ax.set_global()
    pc = ax.pcolormesh(lon, lat, z, transform=ccrs.PlateCarree(), cmap=pct_seq, vmin=lo, vmax=hi)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=INK, zorder=3)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    return pc


fig = plt.figure(figsize=(17, 12.5), facecolor=SURF)
gs = GridSpec(3, 2, figure=fig, hspace=0.22, wspace=0.04, left=0.03, right=0.97, top=0.9, bottom=0.07)
axs = [fig.add_subplot(gs[i // 2, i % 2], projection=ccrs.Robinson()) for i in range(6)]
for ax, (grp, name) in zip(axs, GROUPS):
    lat, lon, z = field(grp, "Global")
    pc = panel(ax, lon, lat, z, 0, 60, f"{name}   (Global mean: {pooled_pct(grp,'Global'):.0f}%)")
cax = fig.add_axes([0.3, 0.025, 0.4, 0.013])
cb = fig.colorbar(pc, cax=cax, orientation="horizontal", extend="max")
cb.set_label("RMSE as % of the real monthly-mean ssrd (that cell, that month)", color=SEC)
fig.text(0.03, 0.965, f"How big is the error, relative to how much sun there actually is? ({MOD})", fontsize=17, fontweight="bold", color=INK)
fig.text(0.03, 0.945, "3-hourly RMSE after the exact monthly rescale, divided by the real winter monthly-mean ssrd, per cell", fontsize=11, color=SEC)
fig.savefig(f"figures/rmse_pct_map_{MOD}_global.png", dpi=130, facecolor=SURF)
print(f"saved figures/rmse_pct_map_{MOD}_global.png")

# regional table + regional maps
REG = {"Global": ("Global", None), "Europe": ("Europe", None), "Germany": ("Europe", (47.3, 55.1, 5.9, 15.0)), "Korea": ("Korea", None)}
print(f"\nRMSE as % of real monthly mean ssrd, model={MOD}:")
print("region   | " + " | ".join(f"{n:>10s}" for _, n in GROUPS))
for name, (piece, box) in REG.items():
    vals = [pooled_pct(grp, piece, box) for grp, _ in GROUPS]
    print(f"{name:8s} | " + " | ".join(f"{v:9.0f}%" for v in vals))

fig = plt.figure(figsize=(22, 8.6), facecolor=SURF)
gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.08, left=0.02, right=0.98, top=0.86, bottom=0.14, height_ratios=[1.15, 1])
for row, (piece, ext) in enumerate((("Europe", [-25, 40, 35, 71]), ("Korea", [124, 131, 33, 43]))):
    for col, (grp, name) in enumerate(GROUPS):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        lat, lon, z = field(grp, piece)
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        pcz = ax.pcolormesh(lon, lat, z, transform=ccrs.PlateCarree(), cmap=pct_seq, vmin=0, vmax=80)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=INK, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor=SEC, zorder=3)
        box = (47.3, 55.1, 5.9, 15.0) if piece == "Europe" else None
        ax.set_title(f"{piece}: {name}\n{pooled_pct(grp, piece):.0f}%", loc="left", fontsize=9.5, fontweight="bold", color=INK)
cax = fig.add_axes([0.35, 0.045, 0.3, 0.02])
cb = fig.colorbar(pcz, cax=cax, orientation="horizontal", extend="max")
cb.set_label("RMSE as % of real monthly-mean ssrd", color=SEC)
fig.text(0.02, 0.955, f"Europe and Korea, month by month ({MOD})", fontsize=16, fontweight="bold", color=INK)
fig.savefig(f"figures/rmse_pct_map_{MOD}_regions.png", dpi=125, facecolor=SURF)
print(f"saved figures/rmse_pct_map_{MOD}_regions.png")
