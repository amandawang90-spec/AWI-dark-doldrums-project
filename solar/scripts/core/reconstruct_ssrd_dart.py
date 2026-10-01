"""Apply the ERA5-fitted model to a real TCo1279-DART month: reconstruct
3-hourly ssrd on DART's own native reduced grid, using tisr from the two
lookup templates (never recomputed), then sanity-check the monthly mean
against DART's own real monthly ssrd.

WHAT THIS DOES AND DOESN'T DO YET
-------------------------------------------------------------------------------
Does: read DART's real tcc/hcc/mcc/lcc/tsr (reduced grid, 3-hourly), look up
tisr for those exact timestamps from tisr_template_dart_{365,366}day.nc, apply the
saved model, and report how the reconstructed monthly-mean GLOBAL AREA-WEIGHTED
MEAN compares to DART's own real monthly ssrd (remapped grid) for the same
month -- a coarse but meaningful first check.

Does NOT yet do: the exact per-cell monthly rescaling reconstruct_ssrd.py does
for ERA5. That needs DART's monthly ssrd (5136x2560, remapped) regridded onto
the reduced native grid (6,599,680 cells) first, a real step of its own --
deferred until this coarser check confirms the loader and lookup are correct.

TISR LOOKUP
-----------
DART's own reduced-grid files use the identical window-end convention as the
templates (confirmed: first stamp of any DART month is day 1, 03:00 UTC), so
matching is a direct index computed from each real timestamp's offset from
Jan 1 03:00 of ITS OWN calendar year -- no special-casing needed. Which
template (365 vs 366 day) is chosen by that same year's real leap-year status.

Usage: python3 reconstruct_ssrd_dart.py <run_dir> <YYYYMM>
Example: python3 reconstruct_ssrd_dart.py \\
             /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C 195001
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))


import calendar
import sys
import time

import joblib
import numpy as np
import xarray as xr

DT_DART = 10800.0          # DART's real tsr/ssrd accumulation window (3 h) -- see
                           # reconstruct_ssrd.py's DT comment for how this was confirmed
MONTHLY_ANCHOR_DIVISOR = 10800.0   # DART monthly ssrd: mean of 3h accumulations
DAY_MU = 0.02
TEMPLATE_DIR = "data/dart_analytical_tisr"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_leap(year):
    return calendar.isleap(year)


def nearest_regular_index(lat_native, lon_native, lat_reg, lon_reg):
    """Map each native (irregular) cell to its nearest cell on DART's remapped
    regular grid -- confirmed uniformly spaced (0.0703125 deg lat, 0.0701 deg
    lon), so this is direct arithmetic, not a search. Computed once and reused:
    the mapping only depends on the two (fixed) grids, never on the data.

    This is deliberately NOT a physical remap of a field -- it is how the
    MONTHLY RESCALING FACTOR (smooth, regional-scale by construction) gets
    applied per native cell, using DART's own trusted monthly ssrd as the
    anchor. Nearest-neighbour is the right tool for carrying a smooth
    correction onto a much finer grid; it would be the wrong tool for
    resampling the ssrd field itself, which is not being done here.
    """
    dlat = lat_reg[1] - lat_reg[0]
    dlon = lon_reg[1] - lon_reg[0]
    lon_native_pos = np.mod(lon_native, 360.0)          # match reg grid's 0-360 convention
    ilat = np.clip(np.round((lat_native - lat_reg[0]) / dlat).astype(int), 0, len(lat_reg) - 1)
    ilon = np.clip(np.round((lon_native_pos - lon_reg[0]) / dlon).astype(int), 0, len(lon_reg) - 1)
    return ilat, ilon


def lookup_tisr(lat, lon, times):
    """Pull tisr for these exact DART cells/timestamps from the matching
    template, by direct index -- never recomputed.
    """
    year = times[0].astype("datetime64[Y]").astype(int) + 1970
    template = f"{TEMPLATE_DIR}/tisr_template_dart_{'366' if is_leap(year) else '365'}day.nc"
    ds = xr.open_dataset(template)

    # index = offset from Jan 1 03:00 of THIS year, in 3h steps -- matches the
    # template's own convention exactly, confirmed against DART's real stamps
    year_start = np.datetime64(f"{year}-01-01T03:00:00", "s")
    idx = np.round((times.astype("datetime64[s]") - year_start) / np.timedelta64(3, "h")).astype(int)
    assert idx.min() >= 0 and idx.max() < ds.sizes["time_counter"], \
        f"timestamp outside template range: idx [{idx.min()},{idx.max()}] vs {ds.sizes['time_counter']}"

    tisr = ds.tisr.isel(time_counter=idx).values.astype("float32")   # (nt, n_cells)
    ds.close()
    return tisr, template


def load_dart_month(run_dir, stamp):
    d = f"{run_dir}/outdata/oifs"
    tag = f"{stamp}-{stamp}"
    tcc = xr.open_dataset(f"{d}/atm_reduced_3h_tcc_3h_{tag}.nc")
    hcc = xr.open_dataset(f"{d}/atm_reduced_3h_hcc_3h_{tag}.nc")
    mcc = xr.open_dataset(f"{d}/atm_reduced_3h_mcc_3h_{tag}.nc")
    lcc = xr.open_dataset(f"{d}/atm_reduced_3h_lcc_3h_{tag}.nc")
    tsr = xr.open_dataset(f"{d}/atm_reduced_3h_tsr_3h_{tag}.nc")
    monthly = xr.open_dataset(f"{d}/atm_remapped_1m_ssrd_1m_{tag}.nc")

    lat = tcc.lat.values.astype("float32")
    lon = tcc.lon.values.astype("float32")
    times = tcc.time_counter.values

    log(f"  looking up tisr for {len(times)} timestamps ...")
    tisr, template_used = lookup_tisr(lat, lon, times)

    out = {
        "tcc": tcc.tcc.values.astype("float32"), "hcc": hcc.hcc.values.astype("float32"),
        "mcc": mcc.mcc.values.astype("float32"), "lcc": lcc.lcc.values.astype("float32"),
        "tsr": tsr.tsr.values.astype("float32"), "tisr": tisr,
        "lat": lat, "lon": lon, "n_cells": lat.size, "ntime": len(times),
        "monthly_ssrd_wm2": (monthly.ssrd.squeeze().values / MONTHLY_ANCHOR_DIVISOR),
        "monthly_lat": monthly.lat.values, "monthly_lon": monthly.lon.values,
        "template_used": template_used,
    }
    for ds in (tcc, hcc, mcc, lcc, tsr, monthly):
        ds.close()
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("Usage:")[1].strip())
    run_dir, stamp = sys.argv[1], sys.argv[2]

    log(f"loading models/model_v1/model_kt.joblib ...")
    saved = joblib.load("models/model_v1/model_kt.joblib")
    model, features, dt_era5 = saved["model"], saved["features"], saved["dt_era5"]
    log(f"  features (order matters): {features}, trained with ERA5 DT={dt_era5}s")

    log(f"loading DART month {stamp} from {run_dir} ...")
    m = load_dart_month(run_dir, stamp)
    log(f"  {m['n_cells']:,} cells, {m['ntime']} timesteps, tisr from {m['template_used']}")

    tisr = m["tisr"].ravel()
    mu = tisr / (DT_DART * 1361.0)
    day = mu > DAY_MU
    feat = {
        "tcc": m["tcc"].ravel()[day], "hcc": m["hcc"].ravel()[day],
        "mcc": m["mcc"].ravel()[day], "lcc": m["lcc"].ravel()[day],
        "mu": mu[day], "T": np.clip(m["tsr"].ravel()[day] / tisr[day], 0, 1.5),
    }
    X = np.column_stack([feat[c] for c in features])
    log(f"  predicting kt for {len(X):,} daylight (cell, time) rows ...")
    kt_pred = np.clip(model.predict(X), 0, 1.5)

    ssrd_pred = np.zeros(m["n_cells"] * m["ntime"], dtype="float64")
    ssrd_pred[day] = kt_pred * tisr[day]                     # J/m2, night = 0
    ssrd_pred = ssrd_pred.reshape(m["ntime"], m["n_cells"])
    monthly_pred_wm2 = ssrd_pred.mean(axis=0) / DT_DART       # W/m2 per native cell

    w_native = np.cos(np.deg2rad(m["lat"]))
    gmean_pred = np.average(monthly_pred_wm2, weights=w_native)
    w_remap = np.cos(np.deg2rad(m["monthly_lat"]))[:, None] * np.ones_like(m["monthly_ssrd_wm2"])
    gmean_actual = np.average(m["monthly_ssrd_wm2"], weights=w_remap)

    log("\n" + "=" * 78)
    log(f"COARSE CHECK (gate) -- {stamp}, global area-weighted mean, before rescaling")
    log("=" * 78)
    log(f"  reconstructed : {gmean_pred:7.1f} W/m2")
    log(f"  DART's actual : {gmean_actual:7.1f} W/m2")
    log(f"  ratio         : {gmean_pred/gmean_actual:.3f}  "
        f"(expect close to 1 -- if wildly off, stop before trusting the exact step below)")

    # ---------- EXACT per-cell rescaling ----------
    # DART's own monthly ssrd lives on the remapped regular grid; the
    # reconstruction lives on the native reduced grid. Map each native cell to
    # its nearest regular-grid box (fixed geometry, computed once) and use
    # THAT box's monthly value as the trusted anchor for that cell -- same
    # multiplicative-only logic as reconstruct_ssrd.py used for ERA5, and for
    # the same reason: night is exactly zero and must stay that way.
    log("\n" + "=" * 78)
    log("EXACT PER-CELL RESCALE (nearest-neighbour anchor from DART's monthly ssrd)")
    log("=" * 78)
    ilat, ilon = nearest_regular_index(m["lat"], m["lon"], m["monthly_lat"], m["monthly_lon"])
    actual_native_wm2 = m["monthly_ssrd_wm2"][ilat, ilon]           # (n_cells,)

    scale = np.ones(m["n_cells"], dtype="float64")
    valid = monthly_pred_wm2 > 0.1
    scale[valid] = actual_native_wm2[valid] / monthly_pred_wm2[valid]
    ssrd_final = ssrd_pred * scale[None, :]                          # still J/m2, night = 0

    check_monthly = ssrd_final.mean(axis=0) / DT_DART
    resid = np.average(np.abs(check_monthly - actual_native_wm2), weights=w_native)
    log(f"  post-rescale monthly mismatch (should be ~0): {resid:.2e} W/m2")
    log(f"  scale factor range: [{scale[valid].min():.2f}, {scale[valid].max():.2f}]")

    # ---------- write the reconstructed field ----------
    out_dir = "data/reconstructed_ssrd"
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/scale_factor_{stamp}.npz",
                        lat=m["lat"], lon=m["lon"], scale=scale.astype("float32"))
    log(f"  wrote {out_dir}/scale_factor_{stamp}.npz (diagnostic, not the reconstruction itself)")
    out_path = f"{out_dir}/ssrd_reduced_3h_{stamp}-{stamp}.nc"
    import netCDF4 as nc
    ds = nc.Dataset(out_path, "w", format="NETCDF4")
    ds.createDimension("cell", m["n_cells"])
    ds.createDimension("time_counter", m["ntime"])
    ds.createVariable("lat", "f4", ("cell",), zlib=True, complevel=4)[:] = m["lat"]
    ds.createVariable("lon", "f4", ("cell",), zlib=True, complevel=4)[:] = m["lon"]
    v_ssrd = ds.createVariable("ssrd", "f4", ("time_counter", "cell"), zlib=True, complevel=4,
                               chunksizes=(m["ntime"], min(100_000, m["n_cells"])))
    v_ssrd[:] = ssrd_final.astype("float32")
    v_ssrd.units = "J m-2"
    v_ssrd.long_name = "Reconstructed surface solar radiation downwards (ML, monthly-rescaled)"
    v_ssrd.comment = (f"tisr from {m['template_used']}; monthly-rescaled to DART's own "
                      f"atm_remapped_1m_ssrd_1m_{stamp}-{stamp}.nc via nearest-neighbour anchor")
    ds.close()
    log(f"\n  wrote {out_path}")
    log("\ndone")


if __name__ == "__main__":
    main()
