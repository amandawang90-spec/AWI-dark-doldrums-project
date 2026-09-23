"""
Map of the mean absolute error (m/s) of the three 10 m -> 100 m methods over Germany,
against the real ERA5 100 m wind speed, for ten winters (Oct-Feb, 2015-16 .. 2024-25),
3-hourly, on the 0.25 deg ERA5 grid.

Row 1: every 3-hourly step of the ten winters.
Row 2: only "low-wind days": days whose Germany-wide daily-mean 10 m wind speed is in the
       lowest LOW_DAY_PERCENTILE % of all winter days (all eight 3-hourly steps of those days).

Methods: log law (ERA5 z0 of that step), regression (pooled fit on 20 winters, land/sea
coefficients chosen by the ERA5 land-sea mask), power law (alpha 0.20 land / 0.14 sea).

Usage: python map_error_germany.py [--first 2015] [--last 2024] [--pct 10]
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import json
import sys
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


FIRST, LAST = arg("--first", 2015), arg("--last", 2024)      # winter START years, inclusive
LOW_DAY_PERCENTILE = arg("--pct", 10)
BOX = dict(lat=(47.0, 55.5), lon=(5.5, 15.5))
LSM_FILE = "data/era5/era5_lsm_20260830_0000.nc"
FIT = json.load(open("data/results/regression_fit_20winters.json"))
ALPHA = {"land": 0.20, "sea": 0.14}
METHODS = [("log_law", "Log law"), ("regression", "Regression"), ("power_law", "Power law")]

# ---- Germany polygon (Natural Earth 110 m, bundled with geopandas) ----
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
germany = world[world["name"] == "Germany"].geometry.unary_union

# ---- load the Germany box for every winter month ----
months = []
for y in range(FIRST, LAST + 1):
    months += [(y, 10), (y, 11), (y, 12), (y + 1, 1), (y + 1, 2)]

lat = lon = None
S10, S100, Z0, TIME = [], [], [], []
for yr, m in months:
    tag = f"{yr}{m:02d}"
    print("reading", tag, flush=True)
    with xr.open_dataset(f"data/era5/training_data/era5_winds_{tag}_3hourly.nc") as ds:
        sub = ds[["u10", "v10", "u100", "v100"]].sel(latitude=slice(BOX["lat"][1], BOX["lat"][0]),
                                                      longitude=slice(BOX["lon"][0], BOX["lon"][1]))
        sub = sub.load()
    with xr.open_dataset(f"data/era5/training_data/era5_z0_{tag}_3hourly.nc") as zs:
        z = zs["fsr"].sel(latitude=slice(BOX["lat"][1], BOX["lat"][0]),
                          longitude=slice(BOX["lon"][0], BOX["lon"][1])).load()
    assert np.array_equal(sub["valid_time"].values, z["valid_time"].values), tag
    lat, lon = sub["latitude"].values, sub["longitude"].values
    S10.append(np.hypot(sub["u10"].values, sub["v10"].values))
    S100.append(np.hypot(sub["u100"].values, sub["v100"].values))
    Z0.append(z.values)
    TIME.append(sub["valid_time"].values)

s10, s100, z0 = np.concatenate(S10), np.concatenate(S100), np.concatenate(Z0)
time = np.concatenate(TIME)
print("steps:", s10.shape, flush=True)

# ---- masks on the small grid ----
lsm = xr.open_dataset(LSM_FILE).squeeze()["lsm"].sel(latitude=slice(BOX["lat"][1], BOX["lat"][0]),
                                                     longitude=slice(BOX["lon"][0], BOX["lon"][1])).values
in_de = np.array([[germany.contains(Point(lo, la)) for lo in lon] for la in lat])
land = lsm >= 0.5
print("Germany cells:", int(in_de.sum()), " of which land by ERA5 mask:", int((in_de & land).sum()), flush=True)

# ---- predictions ----
ok_z = (z0 > 1e-5) & (z0 < 10)
zs = np.where(ok_z, z0, 1.0)
log_law = s10 * np.log(100.0 / zs) / np.log(10.0 / zs)
log_law = np.where(ok_z, log_law, np.nan)
slope = np.where(land, FIT["onshore"]["slope"], FIT["offshore"]["slope"])
icpt = np.where(land, FIT["onshore"]["intercept"], FIT["offshore"]["intercept"])
regression = slope * s10 + icpt
power_law = s10 * 10.0 ** np.where(land, ALPHA["land"], ALPHA["sea"])
preds = {"log_law": log_law, "regression": regression, "power_law": power_law}

# ---- low-wind days: Germany-wide daily mean 10 m speed in the lowest pct of all winter days ----
day = time.astype("datetime64[D]")
days, inv = np.unique(day, return_inverse=True)
de_mean = np.nanmean(np.where(in_de, s10, np.nan), axis=(1, 2))          # per step
daily = np.array([de_mean[inv == i].mean() for i in range(len(days))])
thr = np.percentile(daily, LOW_DAY_PERCENTILE)
low_day = daily <= thr
step_low = low_day[inv]
print(f"{len(days)} winter days; low-wind day threshold (Germany mean 10 m speed) = {thr:.2f} m/s; "
      f"{int(low_day.sum())} low-wind days, {int(step_low.sum())} steps", flush=True)
print(f"mean 10 m speed: all steps {np.nanmean(np.where(in_de, s10, np.nan)):.2f} m/s, "
      f"low-wind days {np.nanmean(np.where(in_de, s10[step_low], np.nan)):.2f} m/s", flush=True)

# ---- mean absolute error per grid cell ----
def mae(p, sel):
    e = np.abs(p[sel] - s100[sel])
    return np.nanmean(e, axis=0)

maps = {}
for row, sel in (("winter", np.ones(len(time), bool)), ("low_days", step_low)):
    for key, _ in METHODS:
        m = mae(preds[key], sel)
        maps[(row, key)] = np.where(in_de, m, np.nan)

summary = {}
print(f"\n{'':<10}{'method':<12}{'Germany mean':>14}{'min':>8}{'max':>8}   worst cell (lat, lon)")
for row in ("winter", "low_days"):
    for key, name in METHODS:
        a = maps[(row, key)]
        i, j = np.unravel_index(np.nanargmax(a), a.shape)
        summary[f"{row}|{key}"] = dict(mean=float(np.nanmean(a)), min=float(np.nanmin(a)), max=float(np.nanmax(a)),
                                       worst_lat=float(lat[i]), worst_lon=float(lon[j]))
        print(f"{row:<10}{key:<12}{np.nanmean(a):>14.3f}{np.nanmin(a):>8.3f}{np.nanmax(a):>8.3f}   ({lat[i]:.2f}, {lon[j]:.2f})")
land_de = in_de & land
print(f"\nGermany cells: {int(in_de.sum())}; classed as sea by ERA5's land-sea mask (coastal): {int((in_de & ~land).sum())}")
print("Germany mean excluding those coastal cells:")
for row in ("winter", "low_days"):
    print("  " + row + ": " + ", ".join(f"{k} {np.nanmean(np.where(land_de, maps[(row, k)], np.nan)):.3f}" for k, _ in METHODS))
for row in ("winter", "low_days"):
    for key, _ in METHODS:
        a = np.where(land_de, maps[(row, key)], np.nan)
        order = np.argsort(np.nan_to_num(a, nan=-1).ravel())[::-1][:3]
        print(f"  top-3 land cells {row:<8}{key:<11}" + "  ".join(f"({lat[o // a.shape[1]]:.2f}N,{lon[o % a.shape[1]]:.2f}E)={a.ravel()[o]:.2f}" for o in order))
LATG, LONG = np.meshgrid(lat, lon, indexing="ij")
print("Regional means (land cells), all winter / low-wind days:")
for nm, msk in (("west  (<8E)", LONG < 8), ("east  (>12E)", LONG > 12), ("north (>52.5N)", LATG > 52.5), ("south (<49N)", LATG < 49)):
    mm = land_de & msk
    print(f"  {nm:<15}" + "  ".join(f"{k} {np.nanmean(maps[('winter', k)][mm]):.2f}/{np.nanmean(maps[('low_days', k)][mm]):.2f}" for k, _ in METHODS))
np.savez("data/results/map_error_germany_maps.npz", lat=lat, lon=lon, in_de=in_de, land=land,
         **{f"{r}__{k}": maps[(r, k)] for r in ("winter", "low_days") for k, _ in METHODS})
json.dump(dict(summary=summary, low_day_threshold=float(thr), n_days=int(len(days)), n_low_days=int(low_day.sum()),
               winters=f"{FIRST}-{(FIRST + 1) % 100:02d} .. {LAST}-{(LAST + 1) % 100:02d}"),
          open("data/results/map_error_germany_summary.json", "w"), indent=1)

# ---- figure ----
world_box = world.cx[BOX["lon"][0]:BOX["lon"][1], BOX["lat"][0]:BOX["lat"][1]]
ROWS = [("winter", f"All winter time steps\n{FIRST}-{(FIRST + 1) % 100:02d} to {LAST}-{(LAST + 1) % 100:02d}"),
        ("low_days", f"Low-wind days only\n(lowest {LOW_DAY_PERCENTILE}% of winter days:\nGermany mean 10 m wind < {thr:.1f} m/s)")]
fig, axes = plt.subplots(2, 3, figsize=(14, 11.4), constrained_layout=True)
for ri, (row, rtitle) in enumerate(ROWS):
    allv = np.concatenate([maps[(row, k)][in_de] for k, _ in METHODS])
    vmax = np.ceil(np.nanpercentile(allv, 98) * 20) / 20
    vmin = np.floor(np.nanmin(allv) * 20) / 20
    for ci, (key, name) in enumerate(METHODS):
        ax = axes[ri, ci]
        a = maps[(row, key)]
        pc = ax.pcolormesh(lon, lat, np.ma.masked_invalid(a), cmap="YlOrRd", vmin=vmin, vmax=vmax, shading="nearest")
        world_box.boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.6)
        gpd.GeoSeries([germany]).boundary.plot(ax=ax, color="#222222", linewidth=1.1)
        cy, cx = np.where(in_de & ~land)
        ax.scatter(lon[cx], lat[cy], s=9, c="black", marker="o", linewidths=0, zorder=5)
        ax.set_xlim(*BOX["lon"]); ax.set_ylim(*BOX["lat"])
        ax.set_aspect(1 / np.cos(np.deg2rad(51)))
        ax.set_xticks([6, 8, 10, 12, 14]); ax.set_xticklabels(["6°E", "8°E", "10°E", "12°E", "14°E"], fontsize=8)
        ax.set_yticks([48, 50, 52, 54]); ax.set_yticklabels(["48°N", "50°N", "52°N", "54°N"], fontsize=8)
        if ri == 0:
            ax.set_title(name, fontsize=13, fontweight="bold")
        ax.text(0.03, 0.03, f"Germany mean {np.nanmean(a):.2f} m/s", transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    cb = fig.colorbar(pc, ax=axes[ri, :], shrink=0.8, pad=0.015, aspect=30, extend="max")
    cb.set_label("Mean absolute error of the 100 m wind speed (m/s)\n(scale capped at the 98th percentile)")
    axes[ri, 0].set_ylabel(rtitle, fontsize=11, labelpad=10)
fig.suptitle("Where do the methods miss the real ERA5 100 m wind over Germany?", fontsize=15, fontweight="bold")
fig.supxlabel("Black dots: coastal cells that ERA5's land-sea mask classes as sea (the regression and power law then use their offshore settings). "
              "Grey lines: other countries.", fontsize=9, color="#444444")
out = f"figures/methods/comparison/map_abs_error_germany_{FIRST}_{LAST + 1}.png"
fig.savefig(out, dpi=140)
print("saved", out)
