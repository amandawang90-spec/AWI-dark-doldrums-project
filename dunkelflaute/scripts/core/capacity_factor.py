"""
Wind and solar capacity-factor formulas, per dark-doldrum-project-proposal-update.md
section 5.1 / 5.2, with hub height fixed at 100 m for BOTH onshore and offshore
(user decision: reuse the existing 100 m log-law wind reconstruction directly,
no 150 m offshore extrapolation).
"""
import numpy as np

# Simplified turbine power curve parameters (proposal Table, section 5.1).
# Offshore hub height uses the same 100 m wind field as onshore; only the
# cut-in/rated/cut-out speeds differ between onshore and offshore turbines.
ONSHORE = dict(v_in=3.0, v_rated=12.0, v_out=25.0)
OFFSHORE = dict(v_in=3.0, v_rated=13.0, v_out=25.0)

STC_IRRADIANCE = 1000.0  # W m-2, standard test conditions

# ssrd is sampled every 3 hours in both sources, but the two accumulate over
# DIFFERENT windows despite identical "J m-2" units and identical file naming
# (*_3h) -- see solar/scripts/download/download_era5.py's own units note.
# Passing the wrong one silently divides/multiplies CF by 3x, not a crash.
SSRD_ACCUM_SECONDS_DART = 3 * 3600.0   # DART: genuine 3-hour accumulation
SSRD_ACCUM_SECONDS_ERA5 = 1 * 3600.0   # ERA5: accumulated over the 1h ENDING at the stamp


def wind_speed(u, v):
    return np.sqrt(u * u + v * v)


def wind_cf(v, v_in, v_rated, v_out):
    """Capacity factor from the simplified cubic power curve. v in m/s."""
    cf = np.zeros_like(v, dtype=np.float32)
    ramp = (v >= v_in) & (v < v_rated)
    cf[ramp] = (v[ramp] ** 3 - v_in ** 3) / (v_rated ** 3 - v_in ** 3)
    rated = (v >= v_rated) & (v <= v_out)
    cf[rated] = 1.0
    return cf


def solar_cf(ssrd_Jm2, accum_seconds):
    """CF_PV(t) = G_est(t) / 1000, with G_est the mean irradiance (W/m2) over
    accum_seconds -- caller must pass SSRD_ACCUM_SECONDS_DART or _ERA5
    explicitly; there is no default, so a source mismatch cannot pass silently."""
    g = ssrd_Jm2 / accum_seconds
    return np.clip(g / STC_IRRADIANCE, 0.0, 1.0)
