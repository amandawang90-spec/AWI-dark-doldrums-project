"""Numerical unit/convention audit of every ERA5 month used for v2 training,
against the v1 training months (reference). Metadata labels are NOT trusted;
each check is a physical/numerical test that a unit or window change would break:

  tisr_ratio     sum(real tisr in file) / sum(analytic tisr, DT=3600)   -> 1.000 if J/m2
                 per 1-h window and time stamps aligned (the definitive check)
  monthly_ratio  monthly ssrd/86400  vs  mean(3-hourly ssrd)/3600      -> 1.00
  gmean_ssrd     area-weighted global mean ssrd in W/m2                -> ~170-200
  cloud range    all four fractions within [0,1], no NaN
  kt / T         mean clearness index and TOA transmission, and the share of
                 rows clipped at 1.5 (would jump if a window/unit factor of 3 crept in)
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import csv, sys
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from reconstruct_ssrd import DATA_DIR, DAY_MU, DT, SOLAR_CONSTANT, log
from solar_geometry import toa_irradiance_accumulated

STRIDE = 16
REF = ["202508", "202603", "202608"]          # v1 training months


def audit(stamp):
    sub = dict(latitude=slice(None, None, STRIDE), longitude=slice(None, None, STRIDE))
    cl = xr.open_dataset(f"{DATA_DIR}/era5_clouds_3h_{stamp}.nc").isel(**sub)
    ts = xr.open_dataset(f"{DATA_DIR}/era5_tsr_3h_{stamp}.nc").isel(**sub)
    ss = xr.open_dataset(f"{DATA_DIR}/era5_ssrd_3h_{stamp}.nc").isel(**sub)
    mo = xr.open_dataset(f"{DATA_DIR}/era5_monthly_ssrd_{stamp}.nc").isel(**sub)
    lat, lon = cl.latitude.values, cl.longitude.values
    t_c, t_t, t_s = cl.valid_time.values, ts.valid_time.values, ss.valid_time.values
    same_time = bool(np.array_equal(t_c, t_t) and np.array_equal(t_c, t_s))
    cloud = np.stack([cl[v].values for v in ("tcc", "hcc", "mcc", "lcc")]).astype("float64")
    tsr, ssrd, tisr_real = ts.tsr.values.astype("float64"), ss.ssrd.values.astype("float64"), ss.tisr.values.astype("float64")
    monthly = mo.ssrd.squeeze().values.astype("float64")
    tisr = toa_irradiance_accumulated(lat, lon, t_c, dt_seconds=DT).astype("float64")
    w = np.cos(np.deg2rad(lat))[None, :, None] * np.ones((1, 1, len(lon)))
    wm = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    mu = tisr / (DT * SOLAR_CONSTANT)
    day = mu > DAY_MU
    kt = np.clip(ssrd[day] / tisr[day], 0, 1.5)
    T = np.clip(tsr[day] / tisr[day], 0, 1.5)
    g3 = np.average(ssrd.mean(0), weights=wm) / 3600.0
    gm = np.average(monthly, weights=wm) / 86400.0
    r = dict(stamp=stamp, same_time_axes=same_time, nan=int(np.isnan(cloud).sum() + np.isnan(tsr).sum() + np.isnan(ssrd).sum()),
             cloud_min=cloud.min(), cloud_max=cloud.max(),
             tisr_ratio=tisr_real.sum() / tisr.sum(),
             tisr_corr=np.corrcoef(tisr_real.ravel(), tisr.ravel())[0, 1],
             gmean_ssrd=g3, monthly_ratio=gm / g3,
             kt_mean=kt.mean(), T_mean=T.mean(), kt_clip=(kt >= 1.5).mean(), T_clip=(T >= 1.5).mean(),
             ssrd_max_wm2=ssrd.max() / 3600, tsr_max_wm2=tsr.max() / 3600, tcc_mean=cloud[0][day].mean())
    for d in (cl, ts, ss, mo):
        d.close()
    return r


if __name__ == "__main__":
    stamps = REF + [f"{y}{m:02d}" for y in range(2015, 2026) for m in range(1, 13)]
    log(f"auditing {len(stamps)} months ...")
    rows = Parallel(n_jobs=16, backend="loky")(delayed(audit)(s) for s in stamps)
    with open("models/model_v2/unit_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ref = [r for r in rows if r["stamp"] in REF]
    print("\nREFERENCE (v1 training months):")
    for r in ref:
        print(f"  {r['stamp']}: tisr_ratio={r['tisr_ratio']:.4f} monthly_ratio={r['monthly_ratio']:.4f} "
              f"gmean={r['gmean_ssrd']:.1f} kt={r['kt_mean']:.3f} T={r['T_mean']:.3f} tcc={r['tcc_mean']:.3f}")
    new = [r for r in rows if r["stamp"] not in REF]
    def rng(k): v = np.array([r[k] for r in new]); return f"{v.min():.4f} .. {v.max():.4f}"
    print("\nNEW months 2015-2025 (132): range of each check")
    for k in ("tisr_ratio", "tisr_corr", "monthly_ratio", "gmean_ssrd", "kt_mean", "T_mean", "tcc_mean",
              "kt_clip", "T_clip", "cloud_min", "cloud_max", "ssrd_max_wm2", "tsr_max_wm2"):
        print(f"  {k:13s} {rng(k)}")
    bad = [r["stamp"] for r in new if (not r["same_time_axes"]) or r["nan"] > 0 or not (0.985 < r["tisr_ratio"] < 1.015)
           or not (0.985 < r["monthly_ratio"] < 1.015) or not (165 < r["gmean_ssrd"] < 205)
           or r["cloud_min"] < 0 or r["cloud_max"] > 1.0001 or r["kt_clip"] > 0.01 or r["tisr_corr"] < 0.999]
    print("\nFLAGGED months:", bad if bad else "none")
