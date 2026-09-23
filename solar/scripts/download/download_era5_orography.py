"""One-time download of ERA5 surface geopotential (time-invariant) -> orography height = z / 9.80665 m."""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import cdsapi
cdsapi.Client().retrieve(
    "reanalysis-era5-single-levels",
    {"product_type": "reanalysis", "variable": ["geopotential"], "year": "2020", "month": "01", "day": "01",
     "time": "00:00", "data_format": "netcdf"},
    "data/static/era5_geopotential_surface.nc")
