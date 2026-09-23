"""Does the mean of ERA5's REAL 3-hourly ssrd equal ERA5's own monthly ssrd product, cell by cell?
(no model involved)"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import numpy as np, xarray as xr
from evaluate_v2_cv import box_slices
D = "data/era5"
PIECES = {"Global": (slice(0, 721, 4), [slice(0, 1440, 4)]), "Europe": box_slices(35, 71, -25, 40), "Korea": box_slices(33, 43, 124, 131)}
BOX = {"Europe": None, "Germany": (47.3, 55.1, 5.9, 15.0)}
def load(stamp, lat_sl, lon_sls):
    a, b, lat, lon = [], [], None, []
    for ls in lon_sls:
        s = xr.open_dataset(f"{D}/era5_ssrd_3h_{stamp}.nc").isel(latitude=lat_sl, longitude=ls)
        m = xr.open_dataset(f"{D}/era5_monthly_ssrd_{stamp}.nc").isel(latitude=lat_sl, longitude=ls)
        a.append(s.ssrd.values.astype("float64").mean(0) / 3600.0); b.append(m.ssrd.squeeze().values.astype("float64") / 86400.0)
        lat = s.latitude.values; lon.append(s.longitude.values); s.close(); m.close()
    return np.concatenate(a, -1), np.concatenate(b, -1), lat, np.concatenate(lon)
rows = []
for stamp in [f"2022{m:02d}" for m in range(1, 13)]:
    for piece, (la, lo) in PIECES.items():
        s3, mo, lat, lon = load(stamp, la, lo)
        lon0 = np.where(lon > 180, lon - 360, lon); w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
        regs = {piece: np.ones(s3.shape, bool)}
        if piece == "Europe":
            regs["Germany"] = (lat[:, None] >= 47.3) & (lat[:, None] <= 55.1) & (lon0[None, :] >= 5.9) & (lon0[None, :] <= 15.0)
        for r, mk in regs.items():
            d = s3[mk] - mo[mk]; ww = w[mk]
            rows.append((stamp, r, np.average(mo[mk], weights=ww), np.average(d, weights=ww), np.sqrt(np.average(d ** 2, weights=ww)),
                         np.percentile(np.abs(d / np.maximum(mo[mk], 1)), 95) * 100))
print("mean of REAL 3-hourly ssrd  minus  ERA5 monthly product   (W/m2, area-weighted; 2022)")
print(f"{'region':8s} {'month':6s} {'monthly mean':>12s} {'bias':>7s} {'%':>6s} {'rmse/cell':>9s} {'%':>6s} {'p95 |err|%':>10s}")
for stamp in ("202201", "202204", "202207", "202210", "202212"):
    for r in ("Global", "Europe", "Germany", "Korea"):
        x = [t for t in rows if t[0] == stamp and t[1] == r][0]
        print(f"{r:8s} {stamp:6s} {x[2]:12.1f} {x[3]:+7.2f} {100*x[3]/x[2]:+5.1f}% {x[4]:9.2f} {100*x[4]/x[2]:5.1f}% {x[5]:9.1f}%")
print("\nall 12 months of 2022, RMS over months of the per-cell RMSE, and mean bias:")
for r in ("Global", "Europe", "Germany", "Korea"):
    xs = [t for t in rows if t[1] == r]
    print(f"  {r:8s} mean bias {np.mean([t[3]/t[2] for t in xs])*100:+5.2f}%   per-cell RMSE {np.sqrt(np.mean([(t[4]/t[2])**2 for t in xs]))*100:5.2f}% of the monthly mean")
