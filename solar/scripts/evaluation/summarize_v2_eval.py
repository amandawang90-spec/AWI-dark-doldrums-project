"""Turn models/model_v2/evaluation/raw_results.pkl into tables, CSV and maps.
v1-vs-v2 head-to-head excludes 202508 for BOTH (v1 trained on it)."""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

OUT = "models/model_v2/evaluation"
PIECE = {"Global": "Global", "Europe": "Europe", "Germany": "Europe", "Korea": "Korea"}
REGIONS = list(PIECE)
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
FIELDS = ("n", "W", "Sa", "Sp", "Saa", "Spp", "Sap", "Sdd")

res = pickle.load(open(f"{OUT}/raw_results.pkl", "rb"))
for r in res:
    r["year"], r["month"] = int(r["stamp"][:4]), int(r["stamp"][4:])


def sums(rs, model, region):
    tot = {k: 0.0 for k in FIELDS}
    for r in rs:
        if model in r[PIECE[region]]:
            p = r[PIECE[region]][model]["point"][region]
            for k in FIELDS:
                tot[k] += p[k]
    return tot


def stats(s):
    W = s["W"]
    if W == 0:
        return dict(r2=np.nan, rmse=np.nan, bias=np.nan, corr=np.nan)
    ma, mp = s["Sa"] / W, s["Sp"] / W
    va, vp = s["Saa"] / W - ma ** 2, s["Spp"] / W - mp ** 2
    cov = s["Sap"] / W - ma * mp
    return dict(r2=1 - (s["Sdd"] / W) / va, rmse=np.sqrt(s["Sdd"] / W) / 3600,
                bias=(mp - ma) / 3600, corr=cov / np.sqrt(va * vp))


def spatial_mean(rs, model, region):
    v = [r[PIECE[region]][model]["spatial"][region]["r2"] for r in rs if model in r[PIECE[region]]]
    return float(np.mean(v)) if v else np.nan


lines = []
def out(s=""):
    print(s); lines.append(s)

h2h = [r for r in res if r["stamp"] != "202508"]      # months where v1 is a fair opponent

out("## A. Pointwise vs REAL 3-hourly ssrd, all held-out months pooled")
out("(v1 vs v2 head-to-head on identical months, 202508 excluded from both)\n")
out("| region | model | R2 | RMSE W/m2 | bias W/m2 | corr |")
out("|---|---|---|---|---|---|")
for reg in REGIONS:
    for mod in ("v1", "v2"):
        s = stats(sums(h2h, mod, reg))
        out(f"| {reg} | {mod} | {s['r2']:.3f} | {s['rmse']:.1f} | {s['bias']:+.1f} | {s['corr']:.3f} |")

out("\n## B. Pointwise R2 by calendar month (pooled over all years) -- v1 -> v2\n")
out("| region | " + " | ".join(MON) + " |")
out("|---|" + "---|" * 12)
for reg in REGIONS:
    cells = []
    for mth in range(1, 13):
        rs = [r for r in h2h if r["month"] == mth]
        a, b = stats(sums(rs, "v1", reg))["r2"], stats(sums(rs, "v2", reg))["r2"]
        cells.append(f"{a:.3f}->{b:.3f}")
    out(f"| {reg} | " + " | ".join(cells) + " |")

out("\n## C. v2 pointwise R2 by held-out year (each year scored by a model that never saw it)\n")
years = sorted({r["year"] for r in res})
out("| region | " + " | ".join(str(y) for y in years) + " |")
out("|---|" + "---|" * len(years))
for reg in REGIONS:
    cells = [f"{stats(sums([r for r in res if r['year'] == y], 'v2', reg))['r2']:.3f}" for y in years]
    out(f"| {reg} | " + " | ".join(cells) + " |")

out("\n## D. Monthly-mean SPATIAL R2 by calendar month (mean over years) -- v1 -> v2\n")
out("| region | " + " | ".join(MON) + " |")
out("|---|" + "---|" * 12)
for reg in REGIONS:
    cells = []
    for mth in range(1, 13):
        rs = [r for r in h2h if r["month"] == mth]
        cells.append(f"{spatial_mean(rs, 'v1', reg):+.2f}->{spatial_mean(rs, 'v2', reg):+.2f}")
    out(f"| {reg} | " + " | ".join(cells) + " |")

out("\n## E. v2 over ALL 132 held-out months (incl. 202508), pointwise\n")
out("| region | R2 | RMSE W/m2 | bias W/m2 | corr |")
out("|---|---|---|---|---|")
for reg in REGIONS:
    s = stats(sums(res, "v2", reg))
    out(f"| {reg} | {s['r2']:.3f} | {s['rmse']:.1f} | {s['bias']:+.1f} | {s['corr']:.3f} |")

open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
with open(f"{OUT}/per_month_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stamp", "fold", "region", "model", "pointwise_r2", "pointwise_rmse_wm2", "pointwise_bias_wm2",
                "pointwise_corr", "spatial_r2", "spatial_rmse_wm2", "spatial_bias_wm2", "n_cells"])
    for r in res:
        for reg in REGIONS:
            for mod in r[PIECE[reg]]:
                s = stats({k: r[PIECE[reg]][mod]["point"][reg][k] for k in FIELDS})
                sp = r[PIECE[reg]][mod]["spatial"][reg]
                w.writerow([r["stamp"], r["fold"], reg, mod, s["r2"], s["rmse"], s["bias"], s["corr"],
                            sp["r2"], sp["rmse"], sp["bias"], sp["n"]])

# ---- seasonal global error maps (monthly-mean error, pooled over all years) ----
lat, lon = res[0]["Global_lat"], res[0]["Global_lon"]
lon0 = np.where(lon > 180, lon - 360, lon)
order = np.argsort(lon0)
for mod in ("v2", "v1"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), subplot_kw=dict(projection=ccrs.Robinson()))
    for ax, (sea, months) in zip(axes.ravel(), SEASONS.items()):
        rs = [r for r in (h2h if mod == "v1" else res) if r["month"] in months and mod in r["Global"]]
        err = np.mean([r["Global"][mod]["err"] for r in rs], axis=0)
        ax.set_global()
        pcm = ax.pcolormesh(lon0[order], lat, err[:, order], transform=ccrs.PlateCarree(),
                            cmap="RdBu_r", vmin=-40, vmax=40)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.set_title(f"{sea}  ({len(rs)} months)")
    fig.colorbar(pcm, ax=axes, orientation="horizontal", pad=0.04, shrink=0.5, extend="both",
                 label="monthly-mean error, predicted - real ERA5 ssrd [W/m2]")
    fig.suptitle(f"{mod}: mean error vs real ERA5, held-out years 2015-2025, by season"
                 + ("  (202508 excluded)" if mod == "v1" else ""))
    fig.savefig(f"figures/models/v2_cv_error_maps_{mod}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
print("\nsaved tables, CSV and figures/v2_cv_error_maps_{v1,v2}.png")
