"""Training rows for the Oct-Nov-Dec-Jan-Feb (ONDJF) months of 2015-2025, stride 8, with latitude and elevation per row
so that the masks (south of 60S, above 3000 m) can be applied at training time.
Cache: data/training_cache_ondjf_stride8/<stamp>.npz  (X float32 [T,tcc,hcc,mcc,lcc,mu], y=kt, lat, elev int16)."""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import os, sys
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from reconstruct_ssrd import FINAL_FEATURES, build_features, load_month, log

STRIDE = 8
STAMPS = [f"{y}{m:02d}" for y in range(2015, 2026) for m in (10, 11, 12, 1, 2)]
CACHE = "data/training_cache_ondjf_stride8"


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
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    stamps = STAMPS[:limit] if limit else STAMPS
    log(f"{len(stamps)} months")
    for s, n in Parallel(n_jobs=int(os.environ.get("NJOBS", "16")), backend="loky")(delayed(extract)(s) for s in stamps):
        if n: log(f"  {s}: {n:,} rows")
