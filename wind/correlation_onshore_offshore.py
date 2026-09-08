"""
Direct correlation study: u10 vs u100, v10 vs v100 (and wind speed), split onshore
(land) vs offshore (ocean), using ERA5's land_sea_mask.

Correlation here is SPATIAL: across all grid points at this one timestamp, do
locations with strong 10 m wind also have strong 100 m wind (and vice versa)? This is
not a temporal correlation (would need a time series), and not a directional
correlation (u/v components are correlated independently, not u10-vs-v100).

Usage: python correlation_onshore_offshore.py [STAMP]
  (STAMP defaults to 20260830_0000; pass e.g. 20250830_0000 or 20260301_0000 to run
  on the other downloaded timestamps)
"""

import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAMP = sys.argv[1] if len(sys.argv) > 1 else "20260830_0000"

winds = xr.open_dataset(f"data/era5_winds_{STAMP}.nc").squeeze()
lsmds = xr.open_dataset(f"data/era5_lsm_{STAMP}.nc").squeeze()

u10, v10 = winds["u10"].values, winds["v10"].values
u100, v100 = winds["u100"].values, winds["v100"].values
lsm = lsmds["lsm"].values
lat = winds["latitude"].values

spd10, spd100 = np.hypot(u10, v10), np.hypot(u100, v100)
w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(u10)
land, ocean = lsm >= 0.5, lsm < 0.5


def weighted_pearson(x, y, w):
    x, y, w = x.ravel(), y.ravel(), w.ravel()
    wx, wy = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - wx) * (y - wy), weights=w)
    varx, vary = np.average((x - wx) ** 2, weights=w), np.average((y - wy) ** 2, weights=w)
    r = cov / np.sqrt(varx * vary)
    slope = cov / varx
    intercept = wy - slope * wx
    return r, slope, intercept


print(f"{'region':<10}{'var':<8}{'r':<8}{'r^2':<8}{'slope':<8}{'intercept':<10}")
stats = {}
for name, mask in [("onshore", land), ("offshore", ocean), ("global", np.ones_like(land, dtype=bool))]:
    for label, x, y in [("u", u10, u100), ("v", v10, v100), ("speed", spd10, spd100)]:
        r, slope, intercept = weighted_pearson(x[mask], y[mask], w[mask])
        stats[(name, label)] = (r, slope, intercept)
        print(f"{name:<10}{label:<8}{r:<8.3f}{r**2:<8.3f}{slope:<8.3f}{intercept:<10.3f}")
    print()

# ---- plots: 2 rows (onshore/offshore) x 3 cols (u, v, speed) ----
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
regions = [("onshore", land, "Oranges"), ("offshore", ocean, "Blues")]
varsets = [("u", u10, u100, "u (m/s)"), ("v", v10, v100, "v (m/s)"), ("speed", spd10, spd100, "wind speed (m/s)")]

for row, (rname, mask, cmap) in enumerate(regions):
    for col, (label, x, y, axlabel) in enumerate(varsets):
        ax = axs[row, col]
        hb = ax.hexbin(x[mask].ravel(), y[mask].ravel(), gridsize=70, mincnt=1, cmap=cmap, bins="log")
        lims = [min(x[mask].min(), y[mask].min()), max(x[mask].max(), y[mask].max())]
        ax.plot(lims, lims, "r--", lw=1, label="1:1")
        r, slope, intercept = stats[(rname, label)]
        xs = np.linspace(*lims, 10)
        ax.plot(xs, slope * xs + intercept, "k-", lw=1.3, label=f"fit (r={r:.3f})")
        ax.set_xlabel(f"10 m {axlabel}")
        ax.set_ylabel(f"100 m {axlabel}")
        ax.set_title(f"{rname}: {label}10 vs {label}100  (r={r:.3f}, r²={r**2:.3f})")
        ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(hb, ax=ax, label="log10(count)")

plt.suptitle(f"Onshore vs offshore correlation ({STAMP})")
plt.tight_layout()
outfile = f"figures/correlation_onshore_offshore_{STAMP}.png"
plt.savefig(outfile, dpi=130)
print(f"Saved plot: {outfile}")
