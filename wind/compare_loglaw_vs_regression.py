"""
Direct, apples-to-apples comparison: log-law vs. a fitted linear regression, for
reconstructing u100/v100 from u10/v10, onshore (land) vs offshore (ocean).

Fairness matters here: the log-law uses NO fitted parameters (z0 comes from ERA5's
own forecast_surface_roughness physics, not fit to this u10/u100 pair). A regression
fit and evaluated on the SAME points it was fit on would have an unfair advantage
(it's explicitly minimizing error on those points). So both methods are evaluated
out-of-sample: the globe is split into contiguous 10x10 degree spatial blocks, the
regression (u100 ~ a*u10 + b, v100 ~ a*v10 + b, fit separately per region) is fit on
80% of blocks, and BOTH methods are scored on the held-out 20% of blocks never used
for fitting.
"""

import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAMP = sys.argv[1] if len(sys.argv) > 1 else "20260830_0000"

winds = xr.open_dataset(f"data/era5_winds_{STAMP}.nc").squeeze()
z0ds = xr.open_dataset(f"data/era5_z0_{STAMP}.nc").squeeze()
lsmds = xr.open_dataset(f"data/era5_lsm_{STAMP}.nc").squeeze()

u10, v10 = winds["u10"].values, winds["v10"].values
u100, v100 = winds["u100"].values, winds["v100"].values
z0 = z0ds["fsr"].values
lsm = lsmds["lsm"].values
lat, lon = winds["latitude"].values, winds["longitude"].values

spd10, spd100 = np.hypot(u10, v10), np.hypot(u100, v100)
w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(u10)
land, ocean = lsm >= 0.5, lsm < 0.5

# ---- same spatial block split as validate_train_test.py ----
block_deg = 10
lat_block = (lat // block_deg).astype(int)
lon_block = (lon // block_deg).astype(int)
block_id = lat_block[:, None] * 1000 + lon_block[None, :]
block_id = np.broadcast_to(block_id, u10.shape)

rng = np.random.default_rng(42)
unique_blocks = np.unique(block_id)
rng.shuffle(unique_blocks)
n_test = int(0.2 * len(unique_blocks))
test_blocks = set(unique_blocks[:n_test])
is_test = np.isin(block_id, list(test_blocks))
is_train = ~is_test

# ---- log-law prediction (no fitting at all) ----
valid = (spd10 > 0.5) & (z0 > 1e-5) & (z0 < 10)
ratio_log = np.full_like(spd10, np.nan)
ratio_log[valid] = np.log(100.0 / z0[valid]) / np.log(10.0 / z0[valid])
u100_log, v100_log = u10 * ratio_log, v10 * ratio_log


def weighted_fit(x, y, w):
    x, y, w = x.ravel(), y.ravel(), w.ravel()
    wx, wy = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - wx) * (y - wy), weights=w)
    varx = np.average((x - wx) ** 2, weights=w)
    slope = cov / varx
    intercept = wy - slope * wx
    return slope, intercept


def scores(pred, true, w, mask):
    p, t, ww = pred[mask], true[mask], w[mask]
    err = p - t
    rmse = np.sqrt(np.average(err ** 2, weights=ww))
    bias = np.average(err, weights=ww)
    wt, wp = np.average(t, weights=ww), np.average(p, weights=ww)
    cov = np.average((t - wt) * (p - wp), weights=ww)
    r = cov / np.sqrt(np.average((t - wt) ** 2, weights=ww) * np.average((p - wp) ** 2, weights=ww))
    return rmse, bias, r


print("Regression fit on 80% of 10x10-deg spatial blocks (per region); both methods")
print("scored on the SAME held-out 20% of blocks, never used for fitting.\n")

print(f"{'region':<10}{'method':<14}{'var':<8}{'RMSE':<9}{'bias':<9}{'r':<8}")
results = {}
for name, mask in [("onshore", land), ("offshore", ocean)]:
    tr_mask = mask & is_train
    te_mask = mask & is_test & valid  # keep log-law's own validity mask so both see identical points

    a_u, b_u = weighted_fit(u10[tr_mask], u100[tr_mask], w[tr_mask])
    a_v, b_v = weighted_fit(v10[tr_mask], v100[tr_mask], w[tr_mask])
    u100_reg = a_u * u10 + b_u
    v100_reg = a_v * v10 + b_v
    spd100_reg = np.hypot(u100_reg, v100_reg)
    spd100_log = np.hypot(u100_log, v100_log)

    for method, up, vp, sp in [
        ("regression", u100_reg, v100_reg, spd100_reg),
        ("log-law", u100_log, v100_log, spd100_log),
    ]:
        for label, pred, true in [("u", up, u100), ("v", vp, v100), ("speed", sp, spd100)]:
            rmse, bias, r = scores(pred, true, w, te_mask)
            results[(name, method, label)] = (rmse, bias, r)
            print(f"{name:<10}{method:<14}{label:<8}{rmse:<9.3f}{bias:<9.3f}{r:<8.3f}")
    print(f"  (regression fit: u100={a_u:.3f}*u10+{b_u:.3f}, v100={a_v:.3f}*v10+{b_v:.3f})")
    print()

# ---- bar chart: RMSE, regression vs log-law, per region/variable ----
fig, ax = plt.subplots(figsize=(9, 5))
regions = ["onshore", "offshore"]
varlabels = ["u", "v", "speed"]
x = np.arange(len(regions) * len(varlabels))
width = 0.35

reg_rmse = [results[(rg, "regression", vl)][0] for rg in regions for vl in varlabels]
log_rmse = [results[(rg, "log-law", vl)][0] for rg in regions for vl in varlabels]

ax.bar(x - width / 2, reg_rmse, width, label="regression (fitted, out-of-sample)", color="#4c72b0")
ax.bar(x + width / 2, log_rmse, width, label="log-law (physics-based, unfitted)", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels([f"{rg}\n{vl}" for rg in regions for vl in varlabels])
ax.set_ylabel("out-of-sample RMSE (m/s)")
ax.set_title(f"Log-law vs. fitted regression: out-of-sample RMSE ({STAMP})")
ax.legend()
ax.axvline(len(varlabels) - 0.5, color="gray", lw=0.8, ls=":")
plt.tight_layout()
outfile = f"figures/loglaw_vs_regression_{STAMP}.png"
plt.savefig(outfile, dpi=130)
print(f"Saved plot: {outfile}")
