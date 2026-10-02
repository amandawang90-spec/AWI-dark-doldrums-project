"""Turn v5_prerescale_maps_sepmar.npz into (1) printed per-piece pre/post-rescale
summary stats and (2) RMSE + bias maps, pre- and post-rescale side by side, for
Global/Europe/Korea/Germany. Also splits the Global piece into the region v3+
actually trained on (lat >= -60, elevation <= 3000m) vs. the excluded
extrapolation region, to put a real number on Key Finding 7's caveat for v5
specifically (that finding was stated qualitatively; this computes it).

Usage: python3 plot_rmse_maps_v5_prerescale.py
"""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = "models/model_v5/evaluation/v5_prerescale_maps_sepmar.npz"
OUT_DIR = "figures/model_v5_prerescale"
LAT_MIN, ELEV_MAX = -60.0, 3000.0

d = np.load(IN)
os.makedirs(OUT_DIR, exist_ok=True)

PIECES = ["Global", "Europe", "Korea", "Germany"]


def agg(piece, stage, mask=None):
    n = d[f"{piece}_n"].astype("float64")
    sr = d[f"{piece}_sr"]; srr = d[f"{piece}_srr"]
    sse = d[f"{piece}_{stage}_sse"]; sb = d[f"{piece}_{stage}_sb"]
    if mask is not None:
        n, sr, srr, sse, sb = n[mask], sr[mask], srr[mask], sse[mask], sb[mask]
    N = n.sum()
    mean_r = sr.sum() / N
    sst = srr.sum() - sr.sum() ** 2 / N
    rmse = np.sqrt(sse.sum() / N)
    bias = sb.sum() / N
    r2 = 1 - sse.sum() / sst
    return dict(n=N, mean=mean_r, rmse=rmse, bias=bias, r2=r2)


print("=== Per-piece, pooled over all cells and months ===")
for piece in PIECES:
    for stage in ("pre", "post"):
        s = agg(piece, stage)
        print(f"{piece:8s} {stage:5s}  n={s['n']:.3e}  mean={s['mean']:7.1f}  "
              f"RMSE={s['rmse']:6.1f}  bias={s['bias']:+6.1f}  R2={s['r2']:.4f}")

print("\n=== Global piece split by v3+ training mask (lat>=-60, elev<=3000m) ===")
z = xr.open_dataset("data/static/era5_geopotential_surface.nc")["z"].values.squeeze() / 9.80665
lat, lon = d["Global_lat"], d["Global_lon"]
elev = z[::4, ::4]  # matches GLOBAL_STRIDE=4 thinning
assert elev.shape == d["Global_n"].shape, (elev.shape, d["Global_n"].shape)
trained = (lat[:, None] >= LAT_MIN) & (elev <= ELEV_MAX)
excluded = ~trained
for name, mask in (("Trained region", trained), ("Excluded (extrapolated)", excluded)):
    for stage in ("pre", "post"):
        s = agg("Global", stage, mask)
        print(f"{name:24s} {stage:5s}  n={s['n']:.3e}  mean={s['mean']:7.1f}  "
              f"RMSE={s['rmse']:6.1f}  bias={s['bias']:+6.1f}  R2={s['r2']:.4f}")

# ---- maps ----
for piece in PIECES:
    plat, plon = d[f"{piece}_lat"], d[f"{piece}_lon"]
    n = d[f"{piece}_n"].astype("float64")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9 if piece == "Global" else 7),
                              constrained_layout=True)
    for col, stage in enumerate(("pre", "post")):
        sse = d[f"{piece}_{stage}_sse"]; sb = d[f"{piece}_{stage}_sb"]
        rmse = np.sqrt(np.divide(sse, n, out=np.full_like(sse, np.nan), where=n > 0))
        bias = np.divide(sb, n, out=np.full_like(sb, np.nan), where=n > 0)
        vmax_rmse = np.nanpercentile(rmse, 98)
        vmax_bias = np.nanpercentile(np.abs(bias), 98)
        im0 = axes[0, col].pcolormesh(plon, plat, rmse, cmap="viridis", vmin=0, vmax=vmax_rmse, shading="auto")
        axes[0, col].set_title(f"{piece} {stage}-rescale RMSE (W/m²)")
        fig.colorbar(im0, ax=axes[0, col], shrink=0.85)
        im1 = axes[1, col].pcolormesh(plon, plat, bias, cmap="RdBu_r", vmin=-vmax_bias, vmax=vmax_bias, shading="auto")
        axes[1, col].set_title(f"{piece} {stage}-rescale bias (W/m²)")
        fig.colorbar(im1, ax=axes[1, col], shrink=0.85)
    for ax in axes.ravel():
        ax.set_aspect("equal")
    fig.suptitle(f"v5, held-out (CV-fold), Sep–Mar 2015–2026 — {piece}")
    path = f"{OUT_DIR}/{piece.lower()}_rmse_bias_pre_post.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")
