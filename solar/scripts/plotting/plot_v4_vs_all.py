"""Bar-chart comparison of v1/v2/v3area/v4: R2, RMSE, bias, at the 3-hourly pointwise level
and the daily-aggregated level, all scored against the SAME true 3-hour-sum ground truth
(era5_3h_mean) -- the corrected data v4 was trained on. v1/v2/v3area's inputs are re-derived
in their own old (1-hour-sample, DT=3600) training convention, but the comparison target and
rescale anchor are the new true 3-hour sum throughout, so this is a fair apples-to-apples
read of real-world skill, not each model's own self-consistent (and for v1/v2/v3, flattering)
convention. Reads models/model_v4/evaluation/v4_eval_<tag>.npz."""
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

TAG = sys.argv[1] if len(sys.argv) > 1 else "ondjf"
GRP = sys.argv[2] if len(sys.argv) > 2 else "ondjf"
d = np.load(f"models/model_v4/evaluation/v4_eval_{TAG}.npz")
DT = 10800.0   # true 3-hour-sum ground truth throughout this file
NAMES = ["v1", "v2", "v3area", "v4"]
LABEL = {"v1": "v1", "v2": "v2", "v3area": "v3 area-wtd", "v4": "v4 (true 3h)"}
COL = {"v1": "#2a78d6", "v2": "#eb6834", "v3area": "#eda100", "v4": "#1baf7a"}
INK, SEC, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
REG = {"Global": ("Global", None), "Europe": ("Europe", None), "Germany": ("Europe", (47.3, 55.1, 5.9, 15.0)), "Korea": ("Korea", None)}


def wsum(piece, key, box, lat_key="lat", lon_key="lon"):
    v = d[f"{GRP}_{piece}_{key}"]
    lat, lon = d[f"{piece}_{lat_key}"], d[f"{piece}_{lon_key}"]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    if box:
        lon0 = np.where(lon > 180, lon - 360, lon)
        w = w * ((lat[:, None] >= box[0]) & (lat[:, None] <= box[1]) & (lon0[None, :] >= box[2]) & (lon0[None, :] <= box[3]))
    return v, w


def stats(piece, box, level):
    n, w = wsum(piece, ("dn" if level == "daily" else "n"), box)
    sr, _ = wsum(piece, ("dsr" if level == "daily" else "sr"), box)
    W = (w * n).sum(); ma = (w * sr).sum() / W
    if level == "3h":
        srr, _ = wsum(piece, "srr", box)
        var = (w * srr).sum() / W - ma ** 2
    else:
        var = None
    out = {}
    sk = "dsse" if level == "daily" else "sse"; bk = "dsb" if level == "daily" else "sb"
    for m in NAMES:
        sse, _ = wsum(piece, f"{m}_{sk}", box); sb, _ = wsum(piece, f"{m}_{bk}", box)
        rmse = np.sqrt((w * sse).sum() / W) / DT
        bias = (w * sb).sum() / W / DT
        r2 = (1 - (w * sse).sum() / W / var) if var is not None else None
        out[m] = dict(rmse=rmse, bias=bias, r2=r2)
    return ma / DT, out


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def bars(ax, values, ylabel, fmt="{:.1f}", ref=None):
    xs = np.arange(len(REG)); bw = 0.19
    headroom = 0
    for i, m in enumerate(NAMES):
        v = [values[r][m] for r in REG]
        xpos = xs + (i - 1.5) * bw
        ax.bar(xpos, v, width=bw * 0.92, color=COL[m], label=LABEL[m])
        headroom = max(headroom, max(v))
        for x, val in zip(xpos, v):
            ax.text(x, val + 0.015 * headroom, fmt.format(val), ha="center", va="bottom",
                    fontsize=7.3, color=SEC, rotation=90)
    ax.set_ylim(top=headroom * 1.3)
    if ref is not None:
        ax.axhline(ref, color=AXIS, lw=1.1, ls=(0, (3, 2)))
    ax.set_xticks(xs); ax.set_xticklabels(REG.keys(), fontsize=10.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=SEC)
    style(ax)


fig, axes = plt.subplots(2, 3, figsize=(17, 9), facecolor=SURF)
level_titles = {"3h": "3-hourly (pointwise)", "daily": "Daily-aggregated"}
for row, level in enumerate(("3h", "daily")):
    r2v, rmsev, biasv = {}, {}, {}
    for name, (piece, box) in REG.items():
        _, s = stats(piece, box, level)
        for m in NAMES:
            r2v.setdefault(name, {})[m] = s[m]["r2"]
            rmsev.setdefault(name, {})[m] = s[m]["rmse"]
            biasv.setdefault(name, {})[m] = s[m]["bias"]
    if level == "3h":
        bars(axes[row, 0], r2v, "R²", "{:.3f}")
    else:
        axes[row, 0].axis("off")
        axes[row, 0].text(0.5, 0.5, "R² not meaningful\nat daily scale here\n(shown at 3-hourly only)",
                          ha="center", va="center", fontsize=10, color=MUTED, transform=axes[row, 0].transAxes)
    bars(axes[row, 1], rmsev, "RMSE  [W/m²]", "{:.1f}")
    bars(axes[row, 2], biasv, "bias  [W/m²]", "{:+.2f}", ref=0)
    axes[row, 0].text(-0.28, 0.5, level_titles[level], fontsize=13, fontweight="bold", color=INK, rotation=90,
                      va="center", ha="center", transform=axes[row, 0].transAxes)

h = [plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in NAMES]
fig.legend(h, [LABEL[m] for m in NAMES], loc="lower center", ncol=4, frameon=False, fontsize=11, labelcolor=SEC, bbox_to_anchor=(0.5, 0.0))
fig.text(0.02, 0.975, "v4 vs v1/v2/v3: does true 3-hour data actually fix the reconstruction?", fontsize=17, fontweight="bold", color=INK)
fig.text(0.02, 0.95, f"All models scored against the SAME corrected true-3-hour-sum ground truth, {GRP.upper()} 2015-2025 "
                     "(south of 60°S and above 3000 m excluded from scoring)", fontsize=10.5, color=SEC)
fig.tight_layout(rect=(0.02, 0.05, 1, 0.93))
out = f"figures/v4_vs_all_bars_{GRP}.png"
fig.savefig(out, dpi=140, facecolor=SURF); plt.close(fig)
print("saved", out)

for level in ("3h", "daily"):
    print(f"\n{level_titles[level]}:")
    for name, (piece, box) in REG.items():
        mean, s = stats(piece, box, level)
        row = " | ".join(f"{m}: R2={s[m]['r2'] and round(s[m]['r2'],3)} RMSE={s[m]['rmse']:.1f} bias={s[m]['bias']:+.2f}" for m in NAMES)
        print(f"  {name:8s} mean={mean:6.1f}  {row}")
