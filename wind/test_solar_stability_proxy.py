"""
Test whether tsr (top net solar radiation) and cloud cover -- fields you already have
from TCo1279-DART as a solar proxy -- can double as a stability proxy that explains
the log-law's residual structure (the systematic, non-random error diagnosed earlier
as missing Monin-Obukhov stability correction).

Physical hypothesis: tsr=0 (nighttime) + low cloud cover => strong radiative cooling
=> stable stratification => log law under-predicts real shear (or decouples).
tsr>0 (daytime) => convective heating => near-neutral/unstable => log law should be
closer to correct.

If residual correlates with tsr/cloud cover as expected, that's evidence these fields
can be used to build an empirical stability correction on top of the plain log law,
even without direct heat-flux/temperature data.
"""

import numpy as np
import xarray as xr

STAMPS = ["20260830_0000", "20260301_0000", "20250830_0000"]

for stamp in STAMPS:
    winds = xr.open_dataset(f"data/era5_winds_{stamp}.nc").squeeze()
    z0ds = xr.open_dataset(f"data/era5_z0_{stamp}.nc").squeeze()
    lsmds = xr.open_dataset("data/era5_lsm_20260830_0000.nc").squeeze()
    tsr = xr.open_dataset(f"data/solar_extract/{stamp}/data_stream-oper_stepType-accum.nc").squeeze()["tsr"].values
    tcc = xr.open_dataset(f"data/solar_extract/{stamp}/data_stream-oper_stepType-instant.nc").squeeze()["tcc"].values

    u10, v10 = winds["u10"].values, winds["v10"].values
    u100, v100 = winds["u100"].values, winds["v100"].values
    z0 = z0ds["fsr"].values
    lsm = lsmds["lsm"].values
    lat = winds["latitude"].values

    spd10, spd100 = np.hypot(u10, v10), np.hypot(u100, v100)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(u10)
    land = lsm >= 0.5

    valid = (spd10 > 0.5) & (z0 > 1e-5) & (z0 < 10)
    ratio = np.full_like(spd10, np.nan)
    ratio[valid] = np.log(100.0 / z0[valid]) / np.log(10.0 / z0[valid])
    spd100_log = spd10 * ratio
    resid = spd100_log - spd100  # log-law residual: positive = over-prediction

    # split into "night" (tsr==0, no solar heating) vs "day" (tsr>0), onshore only
    m_land = land & valid
    night = m_land & (tsr <= 0)
    day = m_land & (tsr > 0)

    night_clear = night & (tcc < 0.3)   # clear-sky night: strongest radiative cooling expected
    night_cloudy = night & (tcc >= 0.7)  # cloudy night: cooling suppressed, closer to neutral

    def stats(mask, label):
        if mask.sum() < 100:
            print(f"    {label:<20} n/a (too few points)")
            return
        r, ww = resid[mask], w[mask]
        bias = np.average(r, weights=ww)
        rmse = np.sqrt(np.average(r**2, weights=ww))
        print(f"    {label:<20} n={mask.sum():<8} mean_resid={bias:+.3f} m/s   RMSE={rmse:.3f} m/s")

    print(f"\n=== {stamp} (onshore only) ===")
    stats(day, "daytime (tsr>0)")
    stats(night, "nighttime (tsr=0)")
    stats(night_clear, "  night, clear sky (tcc<0.3)")
    stats(night_cloudy, "  night, cloudy (tcc>=0.7)")
