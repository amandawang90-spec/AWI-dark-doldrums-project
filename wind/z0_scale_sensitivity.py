"""
Can ERA5's 0.25-deg (~25 km) z0 legitimately be applied to TCo1279 (~9 km) winds?

Two things are quantified here.

1. SENSITIVITY: how much does the log-law extrapolation ratio
        R = ln(100/z0) / ln(10/z0)
   actually care about z0? R is only logarithmically sensitive, so a large z0 error
   produces a much smaller wind error -- but "smaller" is not "negligible".

2. RESOLUTION / REPRESENTIVITY: z0 does not average linearly. Drag aggregates roughly
   as 1/ln^2(z_b/z0), so a coarse grid box holding a mix of surfaces has an effective
   z0 pulled toward its ROUGHER elements. Coarsening therefore both (a) loses variance
   and (b) biases z0 high in heterogeneous regions.

   We cannot see TCo1279's own 9 km z0 (its `sr` field is an unpopulated placeholder),
   so we measure the analogous degradation one step coarser in the direction we can
   observe: take ERA5 0.25 deg as "truth", coarsen to 0.75 and 1.0 deg, then ask what
   wind error results from using the coarse z0 in place of the true fine one. The
   25->100 km scale jump (4x) brackets the 25->9 km mismatch (2.8x) we care about.
"""

import warnings

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

Z0_FILE = "data/era5_z0_20260830_0000.nc"
LSM_FILE = "data/era5_lsm_20260830_0000.nc"
WIND_FILE = "data/era5_winds_20260830_0000.nc"

BLENDING_HEIGHT = 50.0  # m, standard choice for tile aggregation


def ratio(z0):
    """Log-law 10m -> 100m extrapolation factor."""
    return np.log(100.0 / z0) / np.log(10.0 / z0)


def wmean(x, w):
    return np.average(x, weights=w)


def wrmse(d, w):
    return np.sqrt(np.average(d ** 2, weights=w))


# ------------------------------------------------------------------ sensitivity
print("=" * 78)
print("1. How sensitive is the log-law to z0?")
print("=" * 78)
print(f"\n  {'z0 (m)':>10}{'R = u100/u10':>16}{'surface type':>34}")
print("  " + "-" * 60)
for z0, desc in [
    (0.0002, "calm open sea"),
    (0.0013, "ice sheet (ERA5 value)"),
    (0.03, "short grass / bare soil"),
    (0.10, "low crops"),
    (0.30, "parkland, bushes"),
    (1.00, "forest edge / suburban"),
    (2.00, "dense forest / city"),
]:
    print(f"  {z0:>10.4g}{ratio(z0):>16.3f}{desc:>34}")

print("\n  a 10x error in z0 (0.03 -> 0.3) changes R from "
      f"{ratio(0.03):.3f} to {ratio(0.30):.3f}  = {100 * (ratio(0.30) / ratio(0.03) - 1):.1f}% in wind")
print("  a 10x error in z0 (0.3 -> 3.0) changes R from "
      f"{ratio(0.30):.3f} to {ratio(3.00):.3f}  = {100 * (ratio(3.00) / ratio(0.30) - 1):.1f}% in wind")

# ------------------------------------------------------------------- load ERA5
z0ds = xr.open_dataset(Z0_FILE).squeeze()
lat = z0ds["latitude"].values
z0 = z0ds["fsr"].values
lsm = xr.open_dataset(LSM_FILE).squeeze()["lsm"].values
winds = xr.open_dataset(WIND_FILE).squeeze()
spd10 = np.hypot(winds["u10"].values, winds["v10"].values)

# trim to a shape that blocks evenly (721 -> 720 lat)
z0, lsm, spd10, lat = z0[:720], lsm[:720], spd10[:720], lat[:720]
area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(z0)

valid = (z0 > 1e-5) & (z0 < 10) & np.isfinite(z0)
land = (lsm >= 0.5) & valid


def coarsen(field, weights, factor, mode):
    """Coarsen by `factor`x`factor` blocks, then broadcast back to the fine grid.

    mode='linear' : plain area-weighted mean of z0 (the naive thing)
    mode='drag'   : aggregate drag coefficients 1/ln^2(z_b/z0), then invert
                    (the physically correct blending-height aggregation)
    """
    ny, nx = field.shape
    f, w = field.copy(), weights.copy()
    bad = ~np.isfinite(f) | (f <= 0)
    f[bad], w[bad] = 1e-4, 0.0

    def blocks(a):
        return a.reshape(ny // factor, factor, nx // factor, factor)

    if mode == "linear":
        num = (blocks(f * w)).sum(axis=(1, 3))
        den = (blocks(w)).sum(axis=(1, 3))
        coarse = num / np.maximum(den, 1e-12)
    else:
        cd = 1.0 / np.log(BLENDING_HEIGHT / f) ** 2
        num = (blocks(cd * w)).sum(axis=(1, 3))
        den = (blocks(w)).sum(axis=(1, 3))
        cd_eff = num / np.maximum(den, 1e-12)
        coarse = BLENDING_HEIGHT * np.exp(-1.0 / np.sqrt(np.maximum(cd_eff, 1e-12)))

    return np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)


print("\n" + "=" * 78)
print("2. What happens to z0 -- and to the wind -- when the grid is coarsened?")
print("=" * 78)

print(f"\n  ERA5 native 0.25 deg (~25 km), land only:")
print(f"    area-weighted mean z0 : {wmean(z0[land], area[land]):.4f} m")
print(f"    std dev of z0         : {np.sqrt(wmean((z0[land] - wmean(z0[land], area[land])) ** 2, area[land])):.4f} m")

print(f"\n  {'coarsened to':<16}{'aggregation':<14}{'mean z0':>10}{'std z0':>10}"
      f"{'  |  ':<5}{'R bias':>9}{'wind RMSE':>12}{'wind bias':>11}")
print("  " + "-" * 92)

R_fine = ratio(np.clip(z0, 1e-5, 9.9))
u100_fine = spd10 * R_fine

for factor, label in [(3, "0.75 deg (~83km)"), (4, "1.0 deg (~111km)")]:
    for mode, mlabel in [("linear", "linear mean"), ("drag", "drag/blending")]:
        z0c = coarsen(z0, area * valid, factor, mode)
        z0c = np.clip(z0c, 1e-5, 9.9)

        R_coarse = ratio(z0c)
        u100_coarse = spd10 * R_coarse

        m = land
        mz = wmean(z0c[m], area[m])
        sz = np.sqrt(wmean((z0c[m] - mz) ** 2, area[m]))
        dR = wmean(R_coarse[m] - R_fine[m], area[m])
        rmse = wrmse(u100_coarse[m] - u100_fine[m], area[m])
        bias = wmean(u100_coarse[m] - u100_fine[m], area[m])

        print(f"  {label:<16}{mlabel:<14}{mz:>10.4f}{sz:>10.4f}"
              f"{'  |  ':<5}{dR:>+9.4f}{rmse:>12.4f}{bias:>+11.4f}")

print("\n  (wind RMSE/bias are in m/s at 100 m, land only, area-weighted;")
print("   they isolate the z0 representivity error -- same u10 used throughout)")

# -------------------------------------------------- fetch / equilibrium scale
print("\n" + "=" * 78)
print("3. The horizontal scale the log law implicitly assumes")
print("=" * 78)
print("""
  An internal boundary layer grows at roughly 1:100 downwind of a roughness change,
  so a wind at height z is in equilibrium with its surface only after a fetch ~100z:

      10 m wind   ->  ~1 km of upwind fetch
      100 m wind  ->  ~10 km of upwind fetch

  ERA5 at ~25 km resolves grid boxes LARGER than the 100 m equilibrium fetch, so a
  local log law is a defensible grid-box approximation.
  TCo1279 at ~9 km has grid boxes at or BELOW that fetch -- neighbouring boxes are
  no longer independent, and the 100 m wind over one box is partly set by the surface
  of its upwind neighbours. A purely local log law is weaker there on principle,
  independent of which z0 field is supplied.""")
