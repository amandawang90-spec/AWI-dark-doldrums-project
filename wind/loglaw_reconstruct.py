"""
Reconstruct u100/v100 from u10/v10 with the log-law (using each timestamp's own,
time-matched ERA5 forecast_surface_roughness as z0), then check accuracy against
the real ERA5 u100/v100 at every grid point, separated onshore (land) vs.
offshore (ocean) via the ERA5 land-sea mask: RMSE/bias/r per region, a
full-density scatter of predicted vs. real per region, and a spatial error map.
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIMESTAMPS = [
    ("Aug2025", "20250830_0000"),
    ("Mar2026", "20260301_0000"),
    ("Aug2026", "20260830_0000"),
]
LSM_FILE = "data/era5_lsm_20260830_0000.nc"  # static, same for all times

lsm = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values
land, ocean = lsm >= 0.5, lsm < 0.5


def score(pred, true, w):
    err = pred - true
    rmse = np.sqrt(np.average(err ** 2, weights=w))
    bias = np.average(err, weights=w)
    wt, wp = np.average(true, weights=w), np.average(pred, weights=w)
    cov = np.average((true - wt) * (pred - wp), weights=w)
    r = cov / np.sqrt(np.average((true - wt) ** 2, weights=w) * np.average((pred - wp) ** 2, weights=w))
    return rmse, bias, r


print("Log-law reconstruction (u10,v10 -> u100,v100) vs. real ERA5 u100/v100, onshore vs offshore.\n")
print(f"{'time':<9}{'region':<10}{'var':<8}{'RMSE':<9}{'bias':<9}{'r'}")

for label, stamp in TIMESTAMPS:
    winds = xr.open_dataset(f"data/era5_winds_{stamp}.nc").squeeze()
    z0 = xr.open_dataset(f"data/era5_z0_{stamp}.nc").squeeze()["fsr"].values

    u10, v10 = winds["u10"].values, winds["v10"].values
    u100, v100 = winds["u100"].values, winds["v100"].values
    lat, lon = winds["latitude"].values, winds["longitude"].values
    spd10 = np.hypot(u10, v10)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(u10)

    valid = (spd10 > 0.5) & (z0 > 1e-5) & (z0 < 10)
    ratio = np.full_like(spd10, np.nan)
    ratio[valid] = np.log(100.0 / z0[valid]) / np.log(10.0 / z0[valid])
    u100_log, v100_log = u10 * ratio, v10 * ratio

    err_u = np.where(valid, u100_log - u100, np.nan)
    err_v = np.where(valid, v100_log - v100, np.nan)

    region_scores = {}
    for rname, mask in [("onshore", land), ("offshore", ocean)]:
        m = mask & valid
        rmse_u, bias_u, r_u = score(u100_log[m], u100[m], w[m])
        rmse_v, bias_v, r_v = score(v100_log[m], v100[m], w[m])
        region_scores[rname] = (rmse_u, bias_u, r_u, rmse_v, bias_v, r_v)
        print(f"{label:<9}{rname:<10}{'u100':<8}{rmse_u:<9.3f}{bias_u:<9.3f}{r_u:.3f}")
        print(f"{label:<9}{rname:<10}{'v100':<8}{rmse_v:<9.3f}{bias_v:<9.3f}{r_v:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, (rname, mask) in enumerate([("onshore", land), ("offshore", ocean)]):
        m = mask & valid
        for row, (pred, true, name) in enumerate([(u100_log, u100, "u100"), (v100_log, v100, "v100")]):
            ax = axes[row, col]
            hb = ax.hexbin(true[m], pred[m], gridsize=150, cmap="viridis", mincnt=1, bins="log")
            lims = [min(true[m].min(), pred[m].min()), max(true[m].max(), pred[m].max())]
            ax.plot(lims, lims, "r--", lw=1)
            rmse, _, r = region_scores[rname][0:3] if row == 0 else region_scores[rname][3:6]
            ax.set_xlabel(f"real ERA5 {name} (m/s)")
            ax.set_ylabel(f"log-law {name} (m/s)")
            ax.set_title(f"{rname} {name}: RMSE={rmse:.3f}, r={r:.3f}")
            fig.colorbar(hb, ax=ax, label="log10(count)")
    fig.suptitle(f"Log-law vs. real ERA5, onshore vs offshore ({label}, {stamp})")
    plt.tight_layout()
    outfile = f"figures/loglaw_onshore_offshore_scatter_{stamp}.png"
    plt.savefig(outfile, dpi=130)
    plt.close(fig)
    print(f"Saved plot: {outfile}")

    vmax = np.nanpercentile(np.abs(np.concatenate([err_u[valid], err_v[valid]])), 99)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, err, name in [(axes2[0], err_u, "u100"), (axes2[1], err_v, "v100")]:
        pc = ax.pcolormesh(lon, lat, err, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.contour(lon, lat, lsm, levels=[0.5], colors="k", linewidths=0.4)
        ax.set_title(f"{name} error (log-law - ERA5), m/s\n(black contour = coastline)")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        fig2.colorbar(pc, ax=ax)
    fig2.suptitle(f"Spatial error, land-sea boundary shown ({label}, {stamp})")
    plt.tight_layout()
    outfile2 = f"figures/loglaw_onshore_offshore_map_{stamp}.png"
    plt.savefig(outfile2, dpi=130)
    plt.close(fig2)
    print(f"Saved plot: {outfile2}\n")
