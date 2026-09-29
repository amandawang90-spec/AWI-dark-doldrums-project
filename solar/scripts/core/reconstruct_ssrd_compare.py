"""Piece (arbitrary lat/lon box) loaders for comparing v1/v2/v3 (old convention:
a single 1-hour ERA5 sample every 3 hours, DT=3600) against v4 (new convention:
a genuine 3-hour accumulated sum, DT=10800) on the SAME held-out months.

The old era5_ssrd_3h/era5_tsr_3h files for 2015-2025 no longer exist (superseded
by the hourly backfill), but the old convention is trivially reproducible from
data/era5_1h/ by taking the single hourly value AT each 3-hourly stamp (00,03,
...,21) instead of summing three of them -- exactly what those files held.
The new convention is read directly from the already-built data/era5_3h_mean/.

Both share the same clouds (era5_clouds_3h, unchanged) and are aligned to
clouds' valid_time, since the aggregated files can drop a boundary stamp.
"""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import numpy as np
import xarray as xr

from solar_geometry import toa_irradiance_accumulated

CLOUDS_DIR = "data/era5"
HOURLY_DIR = "data/era5_1h"
MEAN3H_DIR = "data/era5_3h_mean"
OLD_PROXY_DIR = "data/era5_old_proxy"

DT_OLD = 3600.0
DT_NEW = 10800.0
MONTHLY_ANCHOR_DIVISOR = 86400.0


def _load_clouds(stamp, lat_sl, lon_sl):
    return xr.open_dataset(f"{CLOUDS_DIR}/era5_clouds_3h_{stamp}.nc").isel(latitude=lat_sl, longitude=lon_sl)


def _load_monthly(stamp, lat_sl, lon_sl):
    return xr.open_dataset(f"{CLOUDS_DIR}/era5_monthly_ssrd_{stamp}.nc").isel(latitude=lat_sl, longitude=lon_sl)


def _tsr_ssrd_old(stamp, lat_sl, lon_sl, want_times):
    """Single hourly sample AT each stamp in want_times -- the pre-2026 proxy convention.
    Read from the small pre-extracted data/era5_old_proxy/ files (built once by
    extract_old_proxy_ondjf.py), not the ~6 GB era5_1h files directly -- those files'
    on-disk chunking makes a strided/boxed read of the full file slow even for a
    small region."""
    tsr_h = xr.open_dataset(f"{OLD_PROXY_DIR}/era5_tsr_3h_{stamp}.nc").isel(latitude=lat_sl, longitude=lon_sl)
    ssrd_h = xr.open_dataset(f"{OLD_PROXY_DIR}/era5_ssrd_3h_{stamp}.nc").isel(latitude=lat_sl, longitude=lon_sl)
    th = tsr_h.valid_time.values.astype("datetime64[s]")
    sh = ssrd_h.valid_time.values.astype("datetime64[s]")
    tsr = tsr_h.isel(valid_time=np.isin(th, want_times)).tsr.values.astype("float32")
    ssrd = ssrd_h.isel(valid_time=np.isin(sh, want_times)).ssrd.values.astype("float32")
    tsr_h.close(); ssrd_h.close()
    return tsr, ssrd


def _tsr_ssrd_new(stamp, lat_sl, lon_sl, want_times):
    """True 3-hour accumulated sum, from the already-aggregated era5_3h_mean files."""
    tsr_n = xr.open_dataset(f"{MEAN3H_DIR}/era5_tsr_3hmean_{stamp}.nc").isel(latitude=lat_sl, longitude=lon_sl)
    ssrd_n = xr.open_dataset(f"{MEAN3H_DIR}/era5_ssrd_3hmean_{stamp}.nc").isel(latitude=lat_sl, longitude=lon_sl)
    tn = tsr_n.valid_time.values.astype("datetime64[s]")
    sn = ssrd_n.valid_time.values.astype("datetime64[s]")
    tsr = tsr_n.isel(time=np.isin(tn, want_times)).tsr.values.astype("float32")
    ssrd = ssrd_n.isel(time=np.isin(sn, want_times)).ssrd.values.astype("float32")
    tsr_n.close(); ssrd_n.close()
    return tsr, ssrd


def load_piece(stamp, lat_sl, lon_sl, convention):
    """convention: 'old' (DT=3600, v1/v2/v3) or 'new' (DT=10800, v4).
    Returns a dict shaped like reconstruct_ssrd.load_month's output, at whichever
    DT the requested convention uses, aligned to the common valid_time."""
    clouds = _load_clouds(stamp, lat_sl, lon_sl)
    monthly = _load_monthly(stamp, lat_sl, lon_sl)
    ct = clouds.valid_time.values.astype("datetime64[s]")

    if convention == "old":
        tsr, ssrd = _tsr_ssrd_old(stamp, lat_sl, lon_sl, ct)
        dt = DT_OLD
    elif convention == "new":
        tsr, ssrd = _tsr_ssrd_new(stamp, lat_sl, lon_sl, ct)
        dt = DT_NEW
    else:
        raise ValueError(convention)

    # tsr/ssrd may have fewer valid stamps than clouds at a boundary month (old:
    # always full for a non-first month; new: 201501 only) -- align clouds down
    # to whatever the field actually returned.
    if convention == "new":
        tn = xr.open_dataset(f"{MEAN3H_DIR}/era5_tsr_3hmean_{stamp}.nc").valid_time.values.astype("datetime64[s]")
        have = np.intersect1d(ct, tn)
    else:
        have = ct
    clouds = clouds.isel(valid_time=np.isin(ct, have))
    lat, lon = clouds.latitude.values, clouds.longitude.values
    tisr = toa_irradiance_accumulated(lat, lon, have, dt_seconds=dt)

    m = {"tcc": clouds.tcc.values.astype("float32"), "hcc": clouds.hcc.values.astype("float32"),
         "mcc": clouds.mcc.values.astype("float32"), "lcc": clouds.lcc.values.astype("float32"),
         "tsr": tsr, "ssrd": ssrd, "tisr": tisr, "dt": dt,
         "monthly_ssrd_wm2": (monthly.ssrd.squeeze().values / MONTHLY_ANCHOR_DIVISOR).astype("float32"),
         "lat": lat, "lon": lon, "times": have}
    m["ntime"], m["shape2d"] = m["tcc"].shape[0], m["tcc"].shape[1:]
    clouds.close(); monthly.close()
    return m
