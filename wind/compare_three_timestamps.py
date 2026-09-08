"""
Same block-bootstrap methodology throughout, applied consistently to all three
timestamps, then pairwise difference tests to check whether each method differs
significantly between timestamps, onshore and offshore:

(1) the speed10->speed100 regression (slope, r)
(2) the log-law (ERA5 fsr z0) reconstruction of speed100 (RMSE, r)
"""

import numpy as np
import xarray as xr

WIND_FILES = {
    "Aug2026": "data/era5_winds_20260830_0000.nc",
    "Mar2026": "data/era5_winds_20260301_0000.nc",
    "Aug2025": "data/era5_winds_20250830_0000.nc",
}
Z0_FILES = {
    "Aug2026": "data/era5_z0_20260830_0000.nc",
    "Mar2026": "data/era5_z0_20260301_0000.nc",
    "Aug2025": "data/era5_z0_20250830_0000.nc",
}
LSM_FILE = "data/era5_lsm_20260830_0000.nc"


def load(f):
    w = xr.open_dataset(f).squeeze()
    u10, v10 = w["u10"].values, w["v10"].values
    u100, v100 = w["u100"].values, w["v100"].values
    lat = w["latitude"].values
    return np.hypot(u10, v10), np.hypot(u100, v100), lat


def _block_ids(spd10, lat, block_deg):
    lat_block = (lat // block_deg).astype(int)
    lon_block = (np.arange(spd10.shape[1]) * 0.25 // block_deg).astype(int)
    bid = lat_block[:, None] * 1000 + lon_block[None, :]
    return np.broadcast_to(bid, spd10.shape)


def _resample_weights(bid, mask, w_area, rng):
    ubid = np.unique(bid[mask])
    samp = rng.choice(ubid, size=len(ubid), replace=True)
    counts = {}
    for b in samp:
        counts[b] = counts.get(b, 0) + 1
    wgt = np.zeros_like(w_area)
    for b, c in counts.items():
        wgt[(bid == b) & mask] = c
    return wgt


def block_bootstrap_regression(spd10, spd100, mask, lat, n_boot=500, block_deg=10, seed=0):
    bid = _block_ids(spd10, lat, block_deg)
    w_area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(spd10)
    rng = np.random.default_rng(seed)
    rs, slopes = [], []
    for _ in range(n_boot):
        wgt = _resample_weights(bid, mask, w_area, rng)
        sel = wgt > 0
        x, y, ww = spd10[sel], spd100[sel], (w_area * wgt)[sel]
        wx, wy = np.average(x, weights=ww), np.average(y, weights=ww)
        cov = np.average((x - wx) * (y - wy), weights=ww)
        varx = np.average((x - wx) ** 2, weights=ww)
        rs.append(cov / np.sqrt(varx * np.average((y - wy) ** 2, weights=ww)))
        slopes.append(cov / varx)
    return np.array(rs), np.array(slopes)


def block_bootstrap_loglaw(spd10, spd100, z0, mask, lat, n_boot=500, block_deg=10, seed=0):
    valid = (spd10 > 0.5) & (z0 > 1e-5) & (z0 < 10)
    m = mask & valid
    ratio = np.log(100.0 / z0) / np.log(10.0 / z0)
    spd100_log = spd10 * ratio

    bid = _block_ids(spd10, lat, block_deg)
    w_area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(spd10)
    rng = np.random.default_rng(seed)
    rmses, rs = [], []
    for _ in range(n_boot):
        wgt = _resample_weights(bid, m, w_area, rng)
        sel = wgt > 0
        pred, true, ww = spd100_log[sel], spd100[sel], (w_area * wgt)[sel]
        rmses.append(np.sqrt(np.average((pred - true) ** 2, weights=ww)))
        wt, wp = np.average(true, weights=ww), np.average(pred, weights=ww)
        cov = np.average((true - wt) * (pred - wp), weights=ww)
        rs.append(cov / np.sqrt(np.average((true - wt) ** 2, weights=ww) * np.average((pred - wp) ** 2, weights=ww)))
    return np.array(rmses), np.array(rs)


def summarize(label, rname, a, b, name_a, name_b):
    ci_a, ci_b = np.percentile(a, [2.5, 97.5]), np.percentile(b, [2.5, 97.5])
    print(f"{label:<10}{rname:<10}{name_a}={np.mean(a):.4f} [{ci_a[0]:.4f},{ci_a[1]:.4f}]  "
          f"{name_b}={np.mean(b):.4f} [{ci_b[0]:.4f},{ci_b[1]:.4f}]")


def pairwise_tests(boot, labels, metric_names):
    for rname in ["onshore", "offshore"]:
        print(f"\n{rname}:")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                vals_a, vals_b = boot[(a, rname)], boot[(b, rname)]
                parts = []
                for name, va, vb in zip(metric_names, vals_a, vals_b):
                    d = va - vb
                    ci = np.percentile(d, [2.5, 97.5])
                    sig = "YES" if (ci[0] > 0 or ci[1] < 0) else "no"
                    parts.append(f"diff {name}={np.mean(d):+.4f} sig={sig:<4}[{ci[0]:+.4f},{ci[1]:+.4f}]")
                print(f"  {a} vs {b}: " + "   ".join(parts))


lsm = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values
land, ocean = lsm >= 0.5, lsm < 0.5
labels = list(WIND_FILES.keys())

print("=== (1) regression (speed10 -> speed100): slope, r, per timestamp ===")
boot_reg = {}
for i, label in enumerate(labels):
    spd10, spd100, lat = load(WIND_FILES[label])
    for rname, mask in [("onshore", land), ("offshore", ocean)]:
        rs, slopes = block_bootstrap_regression(spd10, spd100, mask, lat, n_boot=500, seed=100 * i + hash(rname) % 100)
        boot_reg[(label, rname)] = (rs, slopes)
        summarize(label, rname, rs, slopes, "r", "slope")

print("\n--- pairwise difference tests: regression ---")
pairwise_tests(boot_reg, labels, ["r", "slope"])

print("\n=== (2) log-law (ERA5 fsr z0) reconstruction of speed100: RMSE, r, per timestamp ===")
boot_log = {}
for i, label in enumerate(labels):
    spd10, spd100, lat = load(WIND_FILES[label])
    z0 = xr.open_dataset(Z0_FILES[label]).squeeze()["fsr"].values
    for rname, mask in [("onshore", land), ("offshore", ocean)]:
        rmses, rs = block_bootstrap_loglaw(spd10, spd100, z0, mask, lat, n_boot=500, seed=200 * i + hash(rname) % 100)
        boot_log[(label, rname)] = (rmses, rs)
        summarize(label, rname, rmses, rs, "RMSE", "r")

print("\n--- pairwise difference tests: log-law ---")
pairwise_tests(boot_log, labels, ["RMSE", "r"])
