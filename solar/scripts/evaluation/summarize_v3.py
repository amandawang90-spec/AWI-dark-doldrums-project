"""Tables + RMSE maps for the v3 evaluation (models/model_v2/evaluation/v3_eval_ondjf.npz)."""

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

d = np.load("models/model_v2/evaluation/v3_eval_ondjf.npz")
NAMES = ["v1", "v2", "v3plain", "v3area"]
LABEL = {"v1": "v1 (3 months)", "v2": "v2 (all seasons, all rows)", "v3plain": "v3 plain (ONDJF, masked)", "v3area": "v3 area-weighted (ONDJF, masked)"}
INK, SEC, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
seq = LinearSegmentedColormap.from_list("blue_seq", ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
div = LinearSegmentedColormap.from_list("bwr_gray", ["#2a78d6", "#f0efec", "#e34948"])


def pooled(grp, piece, mod, box=None):
    n, sr, sse = d[f"{grp}_{piece}_n"], d[f"{grp}_{piece}_sr"], d[f"{grp}_{piece}_{mod}_sse"]
    lat, lon = d[f"{piece}_lat"], d[f"{piece}_lon"]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    if box:
        lon0 = np.where(lon > 180, lon - 360, lon)
        w = w * ((lat[:, None] >= box[0]) & (lat[:, None] <= box[1]) & (lon0[None, :] >= box[2]) & (lon0[None, :] <= box[3]))
    W = (w * n).sum()
    return np.sqrt((w * sse).sum() / W) / 3600, (w * sr).sum() / W / 3600, (w * d[f"{grp}_{piece}_{mod}_sb"]).sum() / W / 3600


REG = {"Global (masked)": ("Global", None), "Europe": ("Europe", None), "Germany": ("Europe", (47.3, 55.1, 5.9, 15.0)), "Korea": ("Korea", None)}
lines = []
for grp, title in (("ondjf", "Oct-Nov-Dec-Jan-Feb (ONDJF)"), ("djf", "Winter Dec-Jan-Feb (DJF)")):
    lines.append(f"\n### {title} -- {int(d[grp + '_nmonths'])} held-out months, after exact monthly rescale, RMSE W/m2 (% of mean sunlight)")
    lines.append("| region | mean sunlight | " + " | ".join(LABEL[m] for m in NAMES) + " |")
    lines.append("|---|---|" + "---|" * len(NAMES))
    for name, (piece, box) in REG.items():
        vals = [pooled(grp, piece, m, box) for m in NAMES]
        mean = vals[0][1]
        lines.append(f"| {name} | {mean:.0f} | " + " | ".join(f"{v[0]:.1f} ({100*v[0]/mean:.0f}%)" for v in vals) + " |")
print("\n".join(lines))
open("models/model_v2/evaluation/v3_summary.md", "w").write("\n".join(lines) + "\n")


def fld(grp, piece, mod):
    n = d[f"{grp}_{piece}_n"]; lat, lon = d[f"{piece}_lat"], d[f"{piece}_lon"]
    lon0 = np.where(lon > 180, lon - 360, lon); o = np.argsort(lon0)
    r = np.where(n > 0, np.sqrt(d[f"{grp}_{piece}_{mod}_sse"] / np.maximum(n, 1)) / 3600, np.nan)
    return lat, lon0[o], r[:, o]


def panel(ax, lon, lat, z, cmap, lo, hi, title, ext=None, borders=False):
    ax.set_extent(ext, crs=ccrs.PlateCarree()) if ext else ax.set_global()
    pc = ax.pcolormesh(lon, lat, z, transform=ccrs.PlateCarree(), cmap=cmap, vmin=lo, vmax=hi)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor=INK, zorder=3)
    if borders: ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor=SEC, zorder=3)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    return pc


for grp, title in (("ondjf", "Oct-Feb (ONDJF)"), ("djf", "Winter (Dec-Jan-Feb)")):
    lat, lon, r1 = fld(grp, "Global", "v1"); _, _, rp = fld(grp, "Global", "v3plain"); _, _, ra = fld(grp, "Global", "v3area")
    fig = plt.figure(figsize=(15, 9.4), facecolor=SURF)
    gs = GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.04, left=0.03, right=0.97, top=0.87, bottom=0.11)
    axs = [fig.add_subplot(gs[i // 2, i % 2], projection=ccrs.Robinson()) for i in range(4)]
    pc = panel(axs[0], lon, lat, r1, seq, 0, 80, f"v1   mean {pooled(grp,'Global','v1')[0]:.1f} W/m²")
    panel(axs[1], lon, lat, rp, seq, 0, 80, f"v3 plain   mean {pooled(grp,'Global','v3plain')[0]:.1f} W/m²")
    panel(axs[2], lon, lat, ra, seq, 0, 80, f"v3 area-weighted   mean {pooled(grp,'Global','v3area')[0]:.1f} W/m²")
    pd = panel(axs[3], lon, lat, ra - r1, div, -20, 20, "Change: v3 area-weighted minus v1")
    cb = fig.colorbar(pc, ax=axs[:3], orientation="horizontal", pad=0.03, shrink=0.35, extend="max", location="bottom", fraction=0.03)
    cb.set_label("RMSE of 3-hourly ssrd after exact monthly rescale [W/m²]", color=SEC)
    cb2 = fig.colorbar(pd, ax=axs[3], orientation="horizontal", pad=0.06, shrink=0.8, extend="both")
    cb2.set_label("W/m²   (blue = v3 closer to the truth)", color=SEC)
    fig.text(0.03, 0.945, f"RMSE after the exact monthly rescale, {title}: south of 60°S and above 3000 m excluded", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.03, 0.915, "white = excluded.  Every month scored by fold models that never trained on that year.", fontsize=10.5, color=SEC)
    fig.savefig(f"figures/models/v3_rmse_map_{grp}_global.png", dpi=125, facecolor=SURF); plt.close(fig)

    fig = plt.figure(figsize=(15, 9.2), facecolor=SURF)
    gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.05, left=0.03, right=0.97, top=0.89, bottom=0.07, height_ratios=[1.15, 1])
    for row, (piece, ext, vmx, dv) in enumerate((("Europe", [-25, 40, 35, 71], 60, 15), ("Korea", [124, 131, 33, 43], 70, 20))):
        lat, lon, a = fld(grp, piece, "v1"); _, _, b = fld(grp, piece, "v3area")
        ax = [fig.add_subplot(gs[row, c], projection=ccrs.PlateCarree()) for c in range(3)]
        pa = panel(ax[0], lon, lat, a, seq, 0, vmx, f"{piece}: v1   mean {pooled(grp,piece,'v1')[0]:.1f} W/m²", ext, True)
        panel(ax[1], lon, lat, b, seq, 0, vmx, f"{piece}: v3 area-weighted   mean {pooled(grp,piece,'v3area')[0]:.1f} W/m²", ext, True)
        pb = panel(ax[2], lon, lat, b - a, div, -dv, dv, f"{piece}: v3 minus v1", ext, True)
        c1 = fig.colorbar(pa, ax=ax[:2], orientation="horizontal", pad=0.07, shrink=0.6, extend="max", aspect=40); c1.set_label("RMSE [W/m²]", color=SEC)
        c2 = fig.colorbar(pb, ax=ax[2], orientation="horizontal", pad=0.07, shrink=0.9, extend="both", aspect=25); c2.set_label("change [W/m²] (blue = v3 better)", color=SEC)
    fig.text(0.03, 0.945, f"RMSE at full 0.25° resolution after the exact monthly rescale, {title}", fontsize=15, fontweight="bold", color=INK)
    fig.savefig(f"figures/models/v3_rmse_map_{grp}_regions.png", dpi=125, facecolor=SURF); plt.close(fig)
print("\nsaved figures/models/v3_rmse_map_{ondjf,djf}_{global,regions}.png and models/model_v2/evaluation/v3_summary.md")
