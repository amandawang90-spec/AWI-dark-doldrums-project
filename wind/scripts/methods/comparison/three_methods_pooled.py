"""
TRUE pooled comparison: one regression fit trained on ALL of Oct2025-Feb2026
combined (not per-month), then log law, power law, and that single pooled
regression all evaluated together on the same pooled Oct-Feb test set.

Memory strategy: process one month's file at a time. For the regression fit,
only small weighted-sum accumulators (Sx, Sy, Sxy, Sxx, Sw) need to survive
across months. For scoring, only the (much smaller, ~20%) held-out spatial-block
test points need to be cached across months -- not the full grids -- so 5 months
of global 3-hourly data never sit in memory simultaneously.
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)


import os
import json
import numpy as np
import xarray as xr

MONTHS = [
    ("Oct2025", "202510"),
    ("Nov2025", "202511"),
    ("Dec2025", "202512"),
    ("Jan2026", "202601"),
    ("Feb2026", "202602"),
]
LSM_FILE = "data/era5/era5_lsm_20260830_0000.nc"
ALPHA_LAND, ALPHA_SEA = 0.20, 0.14
LOW_WIND_THRESH = 3.0
BLOCK_DEG = 10

lsm2d = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values
land2d, ocean2d = lsm2d >= 0.5, lsm2d < 0.5


def block_ids(lat, lon):
    lat_block = (lat // BLOCK_DEG).astype(int)
    lon_block = (lon // BLOCK_DEG).astype(int)
    return lat_block[:, None] * 1000 + lon_block[None, :]


# accumulators for the pooled regression fit: {region: [Sx, Sy, Sxy, Sxx, Sw]}
train_acc = {"onshore": np.zeros(5), "offshore": np.zeros(5)}

# cached test-subset arrays per region: spd10, spd100, power-law pred
test_cache = {"onshore": {"spd10": [], "spd100": [], "pow": [], "w": []},
              "offshore": {"spd10": [], "spd100": [], "pow": [], "w": []}}
# log-law test cache is separate since only some months have z0 so far
log_cache = {"onshore": {"spd10": [], "spd100": [], "log": [], "w": []},
             "offshore": {"spd10": [], "spd100": [], "log": [], "w": []}}

months_used = []
months_with_z0 = []

for label, ym in MONTHS:
    winds_path = f"data/era5/training_data/era5_winds_{ym}_3hourly.nc"
    z0_path = f"data/era5/training_data/era5_z0_{ym}_3hourly.nc"
    if not os.path.exists(winds_path):
        print(f"{label}: winds not downloaded, skipping")
        continue
    months_used.append(label)
    print(f"Processing {label}...")

    ds = xr.open_dataset(winds_path)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    nt = ds.dims["valid_time"]

    u10, v10 = ds["u10"].values, ds["v10"].values
    u100, v100 = ds["u100"].values, ds["v100"].values
    spd10 = np.hypot(u10, v10)
    spd100 = np.hypot(u100, v100)
    del u10, v10, u100, v100

    w_lat = np.cos(np.deg2rad(lat))[:, None] * np.ones((lat.size, lon.size))
    w_full = np.broadcast_to(w_lat, (nt, lat.size, lon.size))
    land = np.broadcast_to(land2d, (nt, lat.size, lon.size))
    ocean = np.broadcast_to(ocean2d, (nt, lat.size, lon.size))

    valid = spd10 > 0.5

    bid2d = block_ids(lat, lon)
    rng = np.random.default_rng(42)  # SAME seed every month -> same spatial blocks held out everywhere
    ub = np.unique(bid2d)
    rng.shuffle(ub)
    test_blocks = set(ub[: int(0.2 * len(ub))])
    is_test2d = np.isin(bid2d, list(test_blocks))
    is_test = np.broadcast_to(is_test2d, (nt, lat.size, lon.size))
    is_train = ~is_test

    ratio_pow = np.where(land, 10.0 ** ALPHA_LAND, 10.0 ** ALPHA_SEA)
    spd100_pow = spd10 * ratio_pow

    have_z0 = os.path.exists(z0_path)
    if have_z0:
        months_with_z0.append(label)
        z0ds = xr.open_dataset(z0_path)
        z0 = z0ds["fsr"].values
        z0v = (z0 > 1e-5) & (z0 < 10)
        ratio_log = np.full_like(spd10, np.nan)
        ratio_log[z0v] = np.log(100.0 / z0[z0v]) / np.log(10.0 / z0[z0v])
        spd100_log = spd10 * ratio_log
        del z0, z0v, ratio_log
    else:
        spd100_log = None
        print(f"  (z0 not available for {label} -- log law excluded from pool for this month)")

    for rname, mask in [("onshore", land), ("offshore", ocean)]:
        tr = mask & is_train & valid
        x, y, wt_ = spd10[tr], spd100[tr], w_full[tr]
        train_acc[rname] += np.array([
            np.sum(wt_ * x), np.sum(wt_ * y), np.sum(wt_ * x * y),
            np.sum(wt_ * x * x), np.sum(wt_),
        ])

        te = mask & is_test & valid
        test_cache[rname]["spd10"].append(spd10[te])
        test_cache[rname]["spd100"].append(spd100[te])
        test_cache[rname]["pow"].append(spd100_pow[te])
        test_cache[rname]["w"].append(w_full[te])

        if have_z0:
            log_cache[rname]["spd10"].append(spd10[te])
            log_cache[rname]["spd100"].append(spd100[te])
            log_cache[rname]["log"].append(spd100_log[te])
            log_cache[rname]["w"].append(w_full[te])

    del ds, spd10, spd100, spd100_pow, w_full, land, ocean, valid, is_test, is_train
    if have_z0:
        del spd100_log

print(f"\nMonths pooled: {months_used}")
print(f"Months with log law: {months_with_z0}\n")


def wscore(pred, true, w):
    err = pred - true
    n = w.sum()
    rmse = np.sqrt(np.sum(w * err ** 2) / n)
    bias = np.sum(w * err) / n
    wt = np.sum(w * true) / n
    wp = np.sum(w * pred) / n
    cov = np.sum(w * (true - wt) * (pred - wp)) / n
    r = cov / np.sqrt((np.sum(w * (true - wt) ** 2) / n) * (np.sum(w * (pred - wp) ** 2) / n))
    mape = np.sum(w * np.abs(err) / true) / n * 100
    return dict(rmse=float(rmse), bias=float(bias), r=float(r), mape=float(mape), n=int(w.size))


results = {}
print(f"{'region':<10}{'method':<12}{'subset':<10}{'n':<12}{'RMSE':<9}{'bias':<9}{'r':<8}MAPE")

for rname in ["onshore", "offshore"]:
    Sx, Sy, Sxy, Sxx, Sw = train_acc[rname]
    xbar, ybar = Sx / Sw, Sy / Sw
    slope = (Sxy / Sw - xbar * ybar) / (Sxx / Sw - xbar ** 2)
    intercept = ybar - slope * xbar
    print(f"  ({rname} pooled regression: speed100 = {slope:.4f}*speed10 + {intercept:.4f}, "
          f"fit on {Sw:.3e} weighted train points across {len(months_used)} months)")

    spd10_te = np.concatenate(test_cache[rname]["spd10"])
    spd100_te = np.concatenate(test_cache[rname]["spd100"])
    pow_te = np.concatenate(test_cache[rname]["pow"])
    w_te = np.concatenate(test_cache[rname]["w"])
    reg_te = slope * spd10_te + intercept

    low_te = spd10_te < LOW_WIND_THRESH

    for method, pred in [("power_law", pow_te), ("regression", reg_te)]:
        for sname, smask in [("overall", slice(None)), ("low_wind", low_te)]:
            sc = wscore(pred[smask], spd100_te[smask], w_te[smask])
            results[f"{rname}_{method}_{sname}"] = sc
            print(f"{rname:<10}{method:<12}{sname:<10}{sc['n']:<12}{sc['rmse']:<9.3f}"
                  f"{sc['bias']:<+9.3f}{sc['r']:<8.3f}{sc['mape']:.1f}%")

    if log_cache[rname]["spd10"]:
        spd10_lo = np.concatenate(log_cache[rname]["spd10"])
        spd100_lo = np.concatenate(log_cache[rname]["spd100"])
        log_pred = np.concatenate(log_cache[rname]["log"])
        w_lo = np.concatenate(log_cache[rname]["w"])
        low_lo = spd10_lo < LOW_WIND_THRESH
        for sname, smask in [("overall", slice(None)), ("low_wind", low_lo)]:
            sc = wscore(log_pred[smask], spd100_lo[smask], w_lo[smask])
            results[f"{rname}_log_law_{sname}"] = sc
            print(f"{rname:<10}{'log_law':<12}{sname:<10}{sc['n']:<12}{sc['rmse']:<9.3f}"
                  f"{sc['bias']:<+9.3f}{sc['r']:<8.3f}{sc['mape']:.1f}%  "
                  f"[only {months_with_z0} -- partial pool]")

results["_meta"] = {"months_pooled": months_used, "months_with_log_law": months_with_z0}
with open("data/results/three_methods_pooled_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved data/results/three_methods_pooled_results.json")
