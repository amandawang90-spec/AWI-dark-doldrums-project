"""Training rows for ALL 12 months, 2015-2026 (not ONDJF-only like v3), stride 8,
with latitude and elevation per row so the masks (south of 60S, above 3000 m) can
be applied at training time. Uses the properly-aggregated 3-hour-SUM ssrd/tsr
(data/era5_3h_mean/, DT=10800) via reconstruct_ssrd_v4.py, not the old 1-hour-
sample proxy that v1/v2/v3 were trained on.

Cache: data/training_cache_v4_stride8/<stamp>.npz  (X float32 [T,tcc,hcc,mcc,lcc,mu], y=kt, lat, elev int16)."""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import glob
import os, sys
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from reconstruct_ssrd_v4 import FINAL_FEATURES, build_features, load_month, MEAN3H_DIR

STRIDE = 8
CACHE = "data/training_cache_v4_stride8"


def available_stamps():
    """Every month that has BOTH ssrd_3hmean and tsr_3hmean on disk."""
    ssrd = {os.path.basename(f).split("_")[-1][:6] for f in glob.glob(f"{MEAN3H_DIR}/era5_ssrd_3hmean_*.nc")}
    tsr = {os.path.basename(f).split("_")[-1][:6] for f in glob.glob(f"{MEAN3H_DIR}/era5_tsr_3hmean_*.nc")}
    return sorted(ssrd & tsr)


def orography():
    z = xr.open_dataset("data/static/era5_geopotential_surface.nc")["z"].squeeze().values / 9.80665
    return z.astype("float32")


def extract(stamp):
    path = f"{CACHE}/{stamp}.npz"
    if os.path.exists(path):
        return stamp, None
    m = load_month(stamp, STRIDE)
    feat, kt, _, day = build_features(m)
    nlat, nlon = m["shape2d"]
    cell = np.nonzero(day)[0] % (nlat * nlon)
    ilat, ilon = cell // nlon, cell % nlon
    elev = orography()[ilat * STRIDE, ilon * STRIDE]
    np.savez(path + ".tmp.npz", X=np.column_stack([feat[c] for c in FINAL_FEATURES]).astype("float32"), y=kt.astype("float32"),
             lat=m["lat"][ilat].astype("float32"), elev=np.round(elev).astype("int16"))
    os.replace(path + ".tmp.npz", path)
    return stamp, len(kt)


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    stamps = available_stamps()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if limit:
        stamps = stamps[:limit]
    print(f"{len(stamps)} months available", flush=True)
    for s, n in Parallel(n_jobs=int(os.environ.get("NJOBS", "16")), backend="loky")(delayed(extract)(s) for s in stamps):
        if n: print(f"  {s}: {n:,} rows", flush=True)
