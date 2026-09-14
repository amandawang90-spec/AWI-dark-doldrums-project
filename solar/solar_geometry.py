"""Analytic top-of-atmosphere incident shortwave radiation (tisr), computed from
latitude, longitude and time alone -- no model or reanalysis field required.

This is the piece that makes the reconstruction transferable to TCo1279-DART:
DART has no tisr field, and never can, because tisr is pure astronomy, not
atmospheric physics. It depends only on where a point is and what time it is.
Every predictor built on tisr (TOA transmission tsr/tisr, mean cos-zenith
tisr/(dt*S0)) can therefore be formed for DART exactly as for ERA5, using DART's
own cell lat/lon and timestamps and nothing else.

Standard Spencer (1971) Fourier-series approximations for the orbital
eccentricity correction, solar declination and the equation of time -- accurate
to a few hundredths of a degree / seconds of time, the standard reference
formulas used in solar-engineering irradiance models (e.g. Iqbal 1983, "An
Introduction to Solar Radiation").

ERA5's tisr is an ACCUMULATION in J/m2 over the hour ending at the stamp, not an
instantaneous value. toa_irradiance_accumulated reproduces that convention by
averaging instantaneous irradiance over N sub-steps within the window (midpoint
rule) rather than evaluating a closed-form integral -- simpler to get right, and
its accuracy is checked directly against ERA5's own tisr field before it is
trusted anywhere a truth field doesn't exist (see check_solar_geometry.py).
"""

import numpy as np

SOLAR_CONSTANT = 1361.0        # W m-2


def _doy_frac(dt64):
    """Fractional day-of-year (1.0 = Jan 1st 00:00 UTC) for an array of datetime64."""
    dt64 = np.asarray(dt64, dtype="datetime64[s]")
    year_start = dt64.astype("datetime64[Y]").astype("datetime64[s]")
    return (dt64 - year_start) / np.timedelta64(1, "D") + 1.0


def _spencer_terms(doy_frac):
    """Eccentricity correction E0, declination (rad), equation of time (min)."""
    gamma = 2 * np.pi * (doy_frac - 1) / 365.25

    E0 = (1.000110 + 0.034221 * np.cos(gamma) + 0.001280 * np.sin(gamma)
          + 0.000719 * np.cos(2 * gamma) + 0.000077 * np.sin(2 * gamma))

    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.001480 * np.sin(3 * gamma))

    eot_min = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                        - 0.014615 * np.cos(2 * gamma) - 0.040890 * np.sin(2 * gamma))
    return E0, decl, eot_min


def cos_zenith_grid(lat, lon, dt64):
    """Instantaneous cosine of the solar zenith angle on a (lat, lon) grid, at
    each of nt instants, clipped to zero below the horizon.

    lat : (nlat,) degrees.   lon : (nlon,) degrees east.   dt64 : (nt,) UTC.
    Returns cosz (nt, nlat, nlon) and E0 (nt,), the eccentricity correction
    (returned separately since it multiplies the solar constant, not cosz).
    """
    doy = _doy_frac(dt64)                                  # (nt,)
    E0, decl, eot_min = _spencer_terms(doy)                 # each (nt,)

    dt64 = np.asarray(dt64, dtype="datetime64[s]")
    midnight = dt64.astype("datetime64[D]").astype("datetime64[s]")
    utc_min = (dt64 - midnight) / np.timedelta64(1, "m")    # (nt,)

    solar_time_min = utc_min[:, None] + 4.0 * lon[None, :] + eot_min[:, None]  # (nt, nlon)
    H = np.deg2rad(solar_time_min / 4.0 - 180.0)                              # (nt, nlon)

    lat_r = np.deg2rad(lat)
    sin_lat, cos_lat = np.sin(lat_r)[None, :, None], np.cos(lat_r)[None, :, None]  # (1,nlat,1)
    sin_decl, cos_decl = np.sin(decl)[:, None, None], np.cos(decl)[:, None, None]  # (nt,1,1)
    cosH = np.cos(H)[:, None, :]                                                   # (nt,1,nlon)

    cosz = sin_lat * sin_decl + cos_lat * cos_decl * cosH    # (nt, nlat, nlon)
    return np.clip(cosz, 0.0, None), E0


def cos_zenith_cells(lat, lon, dt64):
    """Same as cos_zenith_grid, but for a flat list of PAIRED (lat_i, lon_i)
    cells rather than an (nlat, nlon) outer product -- DART's native reduced
    grid is a flat cell list, not a regular lat/lon grid, so lon at cell i is
    NOT independent of lat at cell i.

    lat, lon : (n_cells,) degrees, paired elementwise.   dt64 : (nt,) UTC.
    Returns cosz (nt, n_cells) and E0 (nt,).
    """
    doy = _doy_frac(dt64)
    E0, decl, eot_min = _spencer_terms(doy)

    dt64 = np.asarray(dt64, dtype="datetime64[s]")
    midnight = dt64.astype("datetime64[D]").astype("datetime64[s]")
    utc_min = (dt64 - midnight) / np.timedelta64(1, "m")          # (nt,)

    # The (nt, n_cells) arrays dominate cost -- cos() on float32 measured at
    # ~4x the throughput of float64 on this node (317M vs 75M elements/s), and
    # the physics here doesn't need double precision: final output is float32
    # anyway and the whole method already carries a few-percent tolerance.
    solar_time_min = (utc_min[:, None] + 4.0 * lon[None, :] + eot_min[:, None]).astype("float32")
    H = np.deg2rad(solar_time_min / 4.0 - 180.0)                              # (nt, n_cells) f32

    lat_r = np.deg2rad(lat.astype("float32"))                      # (n_cells,)
    sin_lat, cos_lat = np.sin(lat_r)[None, :], np.cos(lat_r)[None, :]         # (1, n_cells)
    sin_decl = np.sin(decl)[:, None].astype("float32")             # (nt, 1)
    cos_decl = np.cos(decl)[:, None].astype("float32")

    cosz = sin_lat * sin_decl + cos_lat * cos_decl * np.cos(H)     # (nt, n_cells) f32
    return np.clip(cosz, 0.0, None), E0


def toa_irradiance_accumulated_cells(lat, lon, valid_time_end, dt_seconds=10800.0, n_substeps=8):
    """Cell-list version of toa_irradiance_accumulated, for DART's native grid.

    lat, lon : (n_cells,) paired.   valid_time_end : (nt,) datetime64.
    Returns (nt, n_cells) float32, J m-2, accumulated over dt_seconds ending at
    each stamp. Caller is expected to chunk over cells for large grids -- this
    function holds one (nt, n_cells_chunk) array per sub-step in memory, so
    passing the full 6.6M-cell DART grid at once for a whole month is what
    caused an unchunked ~40+ GB, multi-minute blow-up; chunk cells externally
    (see build_tisr_templates.py) to keep memory bounded regardless of grid size.
    """
    valid_time_end = np.asarray(valid_time_end, dtype="datetime64[s]")
    nt = valid_time_end.size
    assert dt_seconds % n_substeps == 0, "n_substeps must divide dt_seconds evenly"
    step = dt_seconds / n_substeps
    offsets = (((np.arange(n_substeps) + 0.5) * step - dt_seconds)
               .astype("int64").astype("timedelta64[s]"))

    acc = np.zeros((nt, lat.size), dtype="float32")
    for k in range(n_substeps):
        cosz, E0 = cos_zenith_cells(lat, lon, valid_time_end + offsets[k])
        acc += cosz * E0[:, None].astype("float32")   # avoid float64 upcast on the big array

    mean_irradiance = SOLAR_CONSTANT * acc / n_substeps
    return (mean_irradiance * dt_seconds).astype("float32")


def toa_irradiance_accumulated(lat, lon, valid_time_end, dt_seconds=3600.0, n_substeps=12):
    """ERA5-convention tisr: J/m2 accumulated over the dt_seconds window ENDING
    at each timestamp in valid_time_end, matching ERA5's own tisr field exactly
    in convention (see check_solar_geometry.py for the numeric agreement).

    lat : (nlat,)   lon : (nlon,)   valid_time_end : (nt,) datetime64
    Returns (nt, nlat, nlon) float32, in J m-2.
    """
    valid_time_end = np.asarray(valid_time_end, dtype="datetime64[s]")
    nt = valid_time_end.size
    assert dt_seconds % n_substeps == 0, "n_substeps must divide dt_seconds evenly"
    step = dt_seconds / n_substeps
    offsets = (((np.arange(n_substeps) + 0.5) * step - dt_seconds)
               .astype("int64").astype("timedelta64[s]"))     # midpoint of each sub-interval

    acc = np.zeros((nt, lat.size, lon.size), dtype="float64")
    for k in range(n_substeps):
        cosz, E0 = cos_zenith_grid(lat, lon, valid_time_end + offsets[k])
        acc += cosz * E0[:, None, None]

    mean_irradiance = SOLAR_CONSTANT * acc / n_substeps        # W m-2
    return (mean_irradiance * dt_seconds).astype("float32")    # J m-2
