"""v4 data loader: identical method to reconstruct_ssrd.py (kt = ssrd/tisr_analytic,
features T,tcc,hcc,mcc,lcc,mu), but pointed at the properly-aggregated 3-hour-SUM
ssrd/tsr in data/era5_3h_mean/ instead of the old era5_{ssrd,tsr}_3h_* files, which
held a single 1-hour ERA5 sample every 3 hours as a proxy for a genuine 3-hour value.

The one required change that comes with that: DT=10800 (a real 3-hour window),
not 3600 -- this must match the actual accumulation window of ssrd/tsr, exactly
as DT_DART=10800 does for TCo1279-DART. Every downstream quantity that depends on
DT (mu, T's implicit scaling via tisr, the W/m2 conversion) is only correct if
this matches; getting it wrong doesn't crash anything, it silently rescales T by
(wrong DT)/(right DT) -- see reconstruct_ssrd.py's own warning about this exact
failure mode.

tisr is analytic throughout (solar_geometry.py), as in reconstruct_ssrd.py --
never read from a file. Clouds are unchanged (era5_clouds_3h_*, instantaneous
samples, no aggregation needed). Monthly ssrd anchor is unchanged (ERA5's own
monthly product, /86400).
"""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import numpy as np
import xarray as xr

from solar_geometry import toa_irradiance_accumulated, SOLAR_CONSTANT

CLOUDS_DIR = "data/era5"
MEAN3H_DIR = "data/era5_3h_mean"

DT = 10800.0                      # true 3-hour accumulation window -- matches DART's DT_DART exactly
DAY_MU = 0.02
FINAL_FEATURES = ["T", "tcc", "hcc", "mcc", "lcc", "mu"]
MONTHLY_ANCHOR_DIVISOR = 86400.0  # ERA5 monthly ssrd convention, unchanged


def load_month(stamp, stride):
    """Load one month's predictors + target, thinned; tisr is ANALYTIC, dt_seconds=DT."""
    sub = dict(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
    clouds = xr.open_dataset(f"{CLOUDS_DIR}/era5_clouds_3h_{stamp}.nc").isel(**sub)
    tsr = xr.open_dataset(f"{MEAN3H_DIR}/era5_tsr_3hmean_{stamp}.nc").isel(**sub)
    ssrd = xr.open_dataset(f"{MEAN3H_DIR}/era5_ssrd_3hmean_{stamp}.nc").isel(**sub)
    monthly = xr.open_dataset(f"{CLOUDS_DIR}/era5_monthly_ssrd_{stamp}.nc").isel(**sub)

    # era5_3h_mean files store their "time" dimension without a coordinate --
    # valid_time is a plain string data variable there (aggregate_hourly_to_3h.py
    # writes it that way) -- so decode it explicitly rather than relying on xarray
    # to have parsed it as a datetime index.
    ct = clouds.valid_time.values.astype("datetime64[s]")
    tt = tsr.valid_time.values.astype("datetime64[s]")
    st = ssrd.valid_time.values.astype("datetime64[s]")

    # clouds and the 3h-mean files can differ by 1 timestep at month boundaries
    # (the aggregated file drops the very first stamp only for 201501); align on
    # the intersection of valid_time so every row has all fields.
    common = np.intersect1d(np.intersect1d(ct, tt), st)
    clouds = clouds.isel(valid_time=np.isin(ct, common))
    tsr = tsr.isel(time=np.isin(tt, common))
    ssrd = ssrd.isel(time=np.isin(st, common))

    lat, lon = clouds.latitude.values, clouds.longitude.values
    tisr = toa_irradiance_accumulated(lat, lon, common, dt_seconds=DT)

    out = {
        "tcc": clouds.tcc.values.astype("float32"), "hcc": clouds.hcc.values.astype("float32"),
        "mcc": clouds.mcc.values.astype("float32"), "lcc": clouds.lcc.values.astype("float32"),
        "tsr": tsr.tsr.values.astype("float32"), "ssrd": ssrd.ssrd.values.astype("float32"),
        "tisr": tisr,
        "monthly_ssrd_wm2": (monthly.ssrd.squeeze().values / MONTHLY_ANCHOR_DIVISOR).astype("float32"),
        "lat": lat, "lon": lon,
    }
    out["ntime"], out["shape2d"] = out["tcc"].shape[0], out["tcc"].shape[1:]
    out["weight2d"] = np.cos(np.deg2rad(lat))[:, None] * np.ones(out["shape2d"], dtype="float32")
    for ds in (clouds, tsr, ssrd, monthly):
        ds.close()
    return out


def build_features(m):
    tisr = m["tisr"].ravel()
    mu = tisr / (DT * SOLAR_CONSTANT)
    day = mu > DAY_MU
    feat = {
        "tcc": m["tcc"].ravel()[day], "hcc": m["hcc"].ravel()[day],
        "mcc": m["mcc"].ravel()[day], "lcc": m["lcc"].ravel()[day],
        "mu": mu[day], "T": np.clip(m["tsr"].ravel()[day] / tisr[day], 0, 1.5),
    }
    kt = np.clip(m["ssrd"].ravel()[day] / tisr[day], 0, 1.5)
    return feat, kt, tisr[day], day
