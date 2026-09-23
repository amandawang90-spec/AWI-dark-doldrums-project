"""
Log law vs. regression vs. power law over 10 winters (Oct-Feb, 2016-17 .. 2025-26),
3-hourly ERA5, evaluated against the real ERA5 100 m wind.

Design (streams one month at a time, so memory stays bounded):
  pass 1: fit ONE regression per region on the training spatial blocks, pooled over
          every available month (weighted sums only).
  pass 2: score all three methods on the held-out spatial blocks and store additive
          sufficient statistics per (month, region, method, subset). Any grouping
          (per winter, per calendar month, everything pooled) is then an exact sum
          of those statistics.

Refuses to run until the download job has logged "ALL WINTERS DONE" (pass --force
to override).
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import calendar
import json
import os
import sys
import numpy as np
import xarray as xr

FORCE = "--force" in sys.argv
LOG = "logs/download_all_winters.log"
if not FORCE:
    done = os.path.exists(LOG) and "ALL WINTERS DONE" in open(LOG, errors="ignore").read()
    if not done:
        sys.exit("Download job has not finished (no 'ALL WINTERS DONE' in log). Not running.")

LSM_FILE = "data/era5/era5_lsm_20260830_0000.nc"
ALPHA_LAND, ALPHA_SEA = 0.20, 0.14
LOW_WIND = 3.0
BLOCK_DEG = 10

def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# Optional: --first 2006 --last 2025 --tag 20winters  (defaults reproduce the 10-winter run)
FIRST, LAST, TAG = int(_arg("--first", 2016)), int(_arg("--last", 2025)), _arg("--tag", "10winters")
OUT_FIT = f"data/results/regression_fit_{TAG}.json"
OUT_STATS = f"data/results/three_methods_{TAG}_stats.json"

# (label, year, month) for winters 2016-17 .. 2025-26
MONTHS = []
for y in range(FIRST, LAST + 1):
    for m, yr in [(10, y), (11, y), (12, y), (1, y + 1), (2, y + 1)]:
        MONTHS.append((f"{y}-{(y + 1) % 100:02d}", yr, m))


def paths(yr, m):
    return (f"data/era5/training_data/era5_winds_{yr}{m:02d}_3hourly.nc",
            f"data/era5/training_data/era5_z0_{yr}{m:02d}_3hourly.nc")


def complete(path, yr, m):
    """A month is usable only if it opens cleanly and has every 3-hourly step."""
    if not os.path.exists(path):
        return False
    try:
        with xr.open_dataset(path) as ds:
            return ds.sizes["valid_time"] == calendar.monthrange(yr, m)[1] * 8
    except Exception:
        return False


usable = [(lab, yr, m) for lab, yr, m in MONTHS
          if all(complete(p, yr, m) for p in paths(yr, m))]
skipped = [(lab, yr, m) for lab, yr, m in MONTHS if (lab, yr, m) not in usable]
print(f"usable months: {len(usable)}/{len(MONTHS)}; skipped: {[f'{y}-{m:02d}' for _, y, m in skipped]}", flush=True)

lsm2d = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values
REGIONS = {"onshore": lsm2d >= 0.5, "offshore": lsm2d < 0.5}


def load_month(yr, m):
    wp, zp = paths(yr, m)
    with xr.open_dataset(wp) as ds:
        lat, lon = ds["latitude"].values, ds["longitude"].values
        u10, v10 = ds["u10"].values, ds["v10"].values
        u100, v100 = ds["u100"].values, ds["v100"].values
    x, y = np.hypot(u10, v10), np.hypot(u100, v100)
    with xr.open_dataset(zp) as zs:
        z0 = zs["fsr"].values
    return lat, lon, x, y, z0


lat0 = xr.open_dataset(LSM_FILE)["latitude"].values
lon0 = xr.open_dataset(LSM_FILE)["longitude"].values
bid = ((lat0 // BLOCK_DEG).astype(int)[:, None] * 1000 + (lon0 // BLOCK_DEG).astype(int)[None, :])
ub = np.unique(bid)
np.random.default_rng(42).shuffle(ub)
is_test2d = np.isin(bid, ub[: int(0.2 * len(ub))])
w2d = np.cos(np.deg2rad(lat0))[:, None] * np.ones((lat0.size, lon0.size))

# ---------------- pass 1: pooled regression fit on training blocks ----------------
if os.path.exists(OUT_FIT) and "--refit" not in sys.argv:
    fit = json.load(open(OUT_FIT))
    print("loaded existing regression fit", flush=True)
else:
    acc = {r: np.zeros(5) for r in REGIONS}  # Sx, Sy, Sxy, Sxx, Sw
    for lab, yr, m in usable:
        print(f"[fit] {yr}-{m:02d}", flush=True)
        _, _, x, y, _ = load_month(yr, m)
        for r, reg in REGIONS.items():
            sel = reg & ~is_test2d
            xs, ys = x[:, sel], y[:, sel]
            ws = np.broadcast_to(w2d[sel], xs.shape)
            ok = xs > 0.5
            xs, ys, ws = xs[ok], ys[ok], ws[ok]
            acc[r] += [np.sum(ws * xs), np.sum(ws * ys), np.sum(ws * xs * ys), np.sum(ws * xs * xs), np.sum(ws)]
        del x, y
    fit = {}
    for r, (Sx, Sy, Sxy, Sxx, Sw) in acc.items():
        xb, yb = Sx / Sw, Sy / Sw
        slope = (Sxy / Sw - xb * yb) / (Sxx / Sw - xb ** 2)
        fit[r] = {"slope": float(slope), "intercept": float(yb - slope * xb)}
    json.dump(fit, open(OUT_FIT, "w"), indent=2)
print("regression fit:", fit, flush=True)


# ---------------- pass 2: additive scoring statistics ----------------
def stats(pred, true, w):
    e = pred - true
    return [float(v) for v in (
        w.size, np.sum(w), np.sum(w * e), np.sum(w * e * e), np.sum(w * true), np.sum(w * pred),
        np.sum(w * true * true), np.sum(w * pred * pred), np.sum(w * true * pred),
        np.sum(w * np.abs(e) / true))]


results = {}
for lab, yr, m in usable:
    print(f"[score] {yr}-{m:02d}", flush=True)
    _, _, x, y, z0 = load_month(yr, m)
    ok_z = (z0 > 1e-5) & (z0 < 10)
    for r, reg in REGIONS.items():
        sel = reg & is_test2d
        xs, ys, zs = x[:, sel], y[:, sel], z0[:, sel]
        ws = np.broadcast_to(w2d[sel], xs.shape)
        okz = ok_z[:, sel]
        land = r == "onshore"
        preds = {
            "power_law": xs * 10.0 ** (ALPHA_LAND if land else ALPHA_SEA),
            "regression": fit[r]["slope"] * xs + fit[r]["intercept"],
            "log_law": xs * np.log(100.0 / np.where(okz, zs, 1.0)) / np.log(10.0 / np.where(okz, zs, 1.0)),
        }
        base = xs > 0.5
        for method, p in preds.items():
            valid = base & (okz if method == "log_law" else True)
            for subset, sm in (("overall", valid), ("low_wind", valid & (xs < LOW_WIND))):
                results[f"{yr}-{m:02d}|{lab}|{r}|{method}|{subset}"] = stats(p[sm], ys[sm], ws[sm])
    del x, y, z0, ok_z

json.dump({"fit": fit, "skipped": [f"{y}-{mm:02d}" for _, y, mm in skipped], "stats": results},
          open(OUT_STATS, "w"))
print(f"saved {OUT_STATS} ({len(results)} records)", flush=True)
