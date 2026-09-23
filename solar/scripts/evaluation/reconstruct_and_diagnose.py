"""Full one-month DART reconstruction PLUS the pre-rescale diagnostic arrays,
in a single pass (avoids reloading/repredicting twice, which is what running
reconstruct_ssrd_dart.py and compute_error_map.py separately would cost).

Writes, for a given DART run+month:
  data/reconstructed_ssrd/ssrd_reduced_3h_<stamp>-<stamp>.nc   (the reconstruction)
  data/reconstructed_ssrd/scale_factor_<stamp>.npz             (per-cell rescale factor)
  data/reconstructed_ssrd/error_map_<stamp>.npz                (pre-rescale diagnostic:
                                                                  actual/predicted/abs_error/scale)

Usage: python3 reconstruct_and_diagnose.py <run_dir> <stamp>
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import os
import sys

import joblib
import netCDF4 as nc
import numpy as np

from reconstruct_ssrd_dart import (DAY_MU, DT_DART, load_dart_month, log,
                                    nearest_regular_index)


def main():
    run_dir, stamp = sys.argv[1], sys.argv[2]

    log("loading models/model_v1/model_kt.joblib ...")
    saved = joblib.load("models/model_v1/model_kt.joblib")
    model, features = saved["model"], saved["features"]
    log(f"  features (order matters): {features}")

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
    log(f"  predicting kt for {len(X):,} daylight rows ...")
    kt_pred = np.clip(model.predict(X), 0, 1.5)
    del X, feat

    ssrd_pred = np.zeros(m["n_cells"] * m["ntime"], dtype="float32")
    ssrd_pred[day] = (kt_pred * tisr[day]).astype("float32")
    ssrd_pred = ssrd_pred.reshape(m["ntime"], m["n_cells"])
    del kt_pred, day, mu
    monthly_pred_wm2 = ssrd_pred.mean(axis=0) / DT_DART

    ilat, ilon = nearest_regular_index(m["lat"], m["lon"], m["monthly_lat"], m["monthly_lon"])
    actual_native_wm2 = m["monthly_ssrd_wm2"][ilat, ilon]

    w_native = np.cos(np.deg2rad(m["lat"]))
    gmean_pred = np.average(monthly_pred_wm2, weights=w_native)
    gmean_actual = np.average(actual_native_wm2, weights=w_native)
    log(f"  COARSE CHECK: reconstructed={gmean_pred:.1f} W/m2  DART real={gmean_actual:.1f} W/m2  "
        f"ratio={gmean_pred/gmean_actual:.3f}")

    abs_error_wm2 = monthly_pred_wm2 - actual_native_wm2
    scale = np.ones(m["n_cells"], dtype="float64")
    valid = monthly_pred_wm2 > 0.1
    scale[valid] = actual_native_wm2[valid] / monthly_pred_wm2[valid]

    out_dir = "data/reconstructed_ssrd"
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(f"{out_dir}/error_map_{stamp}.npz",
                         lat=m["lat"].astype("float32"), lon=m["lon"].astype("float32"),
                         actual_wm2=actual_native_wm2.astype("float32"),
                         predicted_wm2=monthly_pred_wm2.astype("float32"),
                         abs_error_wm2=abs_error_wm2.astype("float32"),
                         scale=scale.astype("float32"))
    log(f"  wrote {out_dir}/error_map_{stamp}.npz")
    log(f"  abs error W/m2: mean={np.mean(abs_error_wm2):.2f} median={np.median(abs_error_wm2):.2f} "
        f"p5={np.percentile(abs_error_wm2,5):.2f} p95={np.percentile(abs_error_wm2,95):.2f} "
        f"min={abs_error_wm2.min():.2f} max={abs_error_wm2.max():.2f}")

    np.savez_compressed(f"{out_dir}/scale_factor_{stamp}.npz",
                         lat=m["lat"], lon=m["lon"], scale=scale.astype("float32"))
    log(f"  wrote {out_dir}/scale_factor_{stamp}.npz")

    # ---------- exact per-cell rescale + write the reconstructed 3-hourly field ----------
    ssrd_final = ssrd_pred * scale[None, :].astype("float32")
    check_monthly = ssrd_final.mean(axis=0) / DT_DART
    resid = np.average(np.abs(check_monthly - actual_native_wm2), weights=w_native)
    log(f"  post-rescale monthly mismatch (should be ~0): {resid:.2e} W/m2")
    log(f"  scale factor range: [{scale[valid].min():.2f}, {scale[valid].max():.2f}]")

    out_path = f"{out_dir}/ssrd_reduced_3h_{stamp}-{stamp}.nc"
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
    log(f"  wrote {out_path}")
    log("done")


if __name__ == "__main__":
    main()
