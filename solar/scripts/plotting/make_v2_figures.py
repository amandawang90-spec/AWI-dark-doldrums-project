"""Figures + diagnostics for the v1-vs-v2 ERA5 evaluation (raw_results.pkl)."""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

OUT = "models/model_v2/evaluation"
PIECE = {"Global": "Global", "Europe": "Europe", "Germany": "Europe", "Korea": "Korea"}
REG = list(PIECE)
F = ("n", "W", "Sa", "Sp", "Saa", "Spp", "Sap", "Sdd")
C1, C2 = "#2a78d6", "#eb6834"            # v1 blue, v2 orange (categorical slots 1,2)
INK, SEC, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

res = pickle.load(open(f"{OUT}/raw_results.pkl", "rb"))
for r in res:
    r["y"], r["m"] = int(r["stamp"][:4]), int(r["stamp"][4:])
res.sort(key=lambda r: r["stamp"])


def st(s):
    W = s["W"]; ma, mp = s["Sa"] / W, s["Sp"] / W
    va = s["Saa"] / W - ma ** 2
    return 1 - (s["Sdd"] / W) / va, (mp - ma) / 3600, np.sqrt(s["Sdd"] / W) / 3600


def sums(rs, mod, reg):
    t = {k: 0.0 for k in F}
    for r in rs:
        if mod in r[PIECE[reg]]:
            for k in F:
                t[k] += r[PIECE[reg]][mod]["point"][reg][k]
    return t


def series(mod, reg, idx):
    out = []
    for r in res:
        out.append(st(sums([r], mod, reg))[idx] if mod in r[PIECE[reg]] else np.nan)
    return np.array(out)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def grid_fig(title, sub, ylabel, fn, plot, ylims=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), facecolor=SURF)
    for ax, reg in zip(axes.ravel(), REG):
        style(ax); plot(ax, reg)
        ax.set_title(reg, loc="left", fontsize=12, color=INK, fontweight="bold")
        ax.set_ylabel(ylabel, color=SEC, fontsize=9)
    h1 = plt.Line2D([], [], color=C1, lw=2); h2 = plt.Line2D([], [], color=C2, lw=2)
    fig.legend([h1, h2], ["v1  (3 training months, stride 16)", "v2  (2015-2025 CV, stride 8)"],
               loc="lower center", ncol=2, frameon=False, fontsize=10.5, labelcolor=SEC, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.012, 0.972, title, fontsize=15, fontweight="bold", color=INK, va="center")
    fig.text(0.012, 0.935, sub, fontsize=10, color=SEC, va="center")
    fig.tight_layout(rect=(0, 0.045, 1, 0.91))
    fig.savefig(f"figures/models/{fn}", dpi=140, facecolor=SURF); plt.close(fig)


t = np.array([r["y"] + (r["m"] - 0.5) / 12 for r in res])
# ---- Fig 1: R2 over the 10 years ----
def p1(ax, reg):
    ax.plot(t, series("v1", reg, 0), color=C1, lw=1.6); ax.plot(t, series("v2", reg, 0), color=C2, lw=1.6)
    ax.set_xlim(2015, 2026)
grid_fig("How well each model tracks real 3-hourly ssrd, month by month, 2015-2025",
         "Pointwise R² per month (every cell, every 3-hourly daylight step). Every month is scored by a model that never trained on it.",
         "R²", "v2_eval_r2_timeseries.png", p1)
# ---- Fig 2: seasonal cycle ----
def p2(ax, reg):
    for mod, c in (("v1", C1), ("v2", C2)):
        y = [st(sums([r for r in res if r["m"] == m and r["stamp"] != "202508"], mod, reg))[0] for m in range(1, 13)]
        ax.plot(range(1, 13), y, color=c, lw=1.8, marker="o", ms=5, mec=SURF, mew=1.2)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MON)
grid_fig("Seasonal cycle of skill (pooled over all years)",
         "Pointwise R² by calendar month. August 2025 excluded for both, because v1 trained on it.",
         "R²", "v2_eval_r2_seasonal.png", p2)
# ---- Fig 3: bias over time ----
def p3(ax, reg):
    ax.axhline(0, color=AXIS, lw=1.2)
    ax.plot(t, series("v1", reg, 1), color=C1, lw=1.6); ax.plot(t, series("v2", reg, 1), color=C2, lw=1.6)
    ax.set_xlim(2015, 2026)
grid_fig("Systematic offset (bias) of each model, month by month",
         "Predicted minus real ssrd, in W/m². Above zero = the model over-predicts sunlight.",
         "bias  [W/m²]", "v2_eval_bias_timeseries.png", p3)

# ---- Fig 4: where is v2 better / worse than v1 (map) ----
lat, lon = res[0]["Global_lat"], res[0]["Global_lon"]
lon0 = np.where(lon > 180, lon - 360, lon); o = np.argsort(lon0)
cmap = LinearSegmentedColormap.from_list("bwr_gray", ["#2a78d6", "#f0efec", "#e34948"])
SEAS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
fig, axes = plt.subplots(2, 2, figsize=(15, 8.4), subplot_kw=dict(projection=ccrs.Robinson()), facecolor=SURF)
for ax, (sea, ms) in zip(axes.ravel(), SEAS.items()):
    rs = [r for r in res if r["m"] in ms and r["stamp"] != "202508"]
    d = np.mean([np.abs(r["Global"]["v2"]["err"]) - np.abs(r["Global"]["v1"]["err"]) for r in rs], axis=0)
    ax.set_global()
    pcm = ax.pcolormesh(lon0[o], lat, d[:, o], transform=ccrs.PlateCarree(), cmap=cmap, vmin=-15, vmax=15)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=INK, zorder=3)
    ax.set_title(sea, loc="left", fontsize=12, fontweight="bold", color=INK)
cb = fig.colorbar(pcm, ax=axes, orientation="horizontal", pad=0.04, shrink=0.55, extend="both")
cb.set_label("change in monthly-mean |error|, v2 minus v1  [W/m²]    blue = v2 closer to the truth      red = v2 further from it", color=SEC, fontsize=10)
cb.ax.tick_params(colors=MUTED)
fig.text(0.012, 0.965, "Where is v2 better or worse than v1?", fontsize=15, fontweight="bold", color=INK)
fig.savefig("figures/models/v2_eval_where_better_worse.png", dpi=130, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

# ---- diagnostics for the write-up ----
print("BIAS by year (W/m2)   v1 / v2")
for reg in REG:
    row = []
    for y in range(2015, 2026):
        rs = [r for r in res if r["y"] == y and r["stamp"] != "202508"]
        row.append(f"{y}:{st(sums(rs,'v1',reg))[1]:+.1f}/{st(sums(rs,'v2',reg))[1]:+.1f}")
    print(f"  {reg:8s}", " ".join(row))
print("BIAS by calendar month, Europe v1 / v2")
print("  ", " ".join(f"{MON[m-1]}:{st(sums([r for r in res if r['m']==m and r['stamp']!='202508'],'v1','Europe'))[1]:+.1f}/{st(sums([r for r in res if r['m']==m and r['stamp']!='202508'],'v2','Europe'))[1]:+.1f}" for m in range(1,13)))
print("BIAS by calendar month, Global v1 / v2")
print("  ", " ".join(f"{MON[m-1]}:{st(sums([r for r in res if r['m']==m and r['stamp']!='202508'],'v1','Global'))[1]:+.1f}/{st(sums([r for r in res if r['m']==m and r['stamp']!='202508'],'v2','Global'))[1]:+.1f}" for m in range(1,13)))
print("RMSE  W/m2 (all months) v1 -> v2")
for reg in REG:
    rs = [r for r in res if r["stamp"] != "202508"]
    print(f"  {reg:8s} {st(sums(rs,'v1',reg))[2]:.1f} -> {st(sums(rs,'v2',reg))[2]:.1f}")
