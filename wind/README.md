# Wind capacity factor reconstruction

Reconstructs 100 m wind (u100/v100) from the 10 m wind ERA5/DART actually
carries, and compares three independent extrapolation methods.

## The three methods (`scripts/methods/`)

- **`loglaw/`** — log-law extrapolation using ERA5's own time-matched
  forecast_surface_roughness (z0), no fitted parameters.
- **`powerlaw/`** — power-law extrapolation (IEC/Moemken et al. 2018:
  alpha=0.2 land, 0.14 water), no roughness field needed.
- **`comparison/`** — everything that pits the methods against each other,
  including the only place a **regression** fit is used (there is no
  standalone regression script by design — it only exists as a comparison
  baseline): `compare_loglaw_vs_regression.py`,
  `compare_regression_vs_loglaw_3ts.py`, `compare_three_timestamps.py`,
  `three_methods_10winters.py`, `three_methods_pooled.py`,
  `three_methods_winter.py`, `analyze_winters.py`,
  `correlation_onshore_offshore.py`, `map_error_germany.py`.

## Other script folders

- **`scripts/download/`** — pull ERA5 winds/z0/fsr/lsm fields, single
  timestamps or full winters.
- **`scripts/preprocessing/`** — QC and sanity checks: orography vs.
  roughness (`check_orography.py`, `compare_z0_vs_orography.py`), z0
  distribution and scale sensitivity, time coverage, roughness maps.
- **`scripts/analysis/`** — exploratory extensions beyond the three
  methods (`test_solar_stability_proxy.py`).

Every script chdirs to this `wind/` root on import (see the `_root` snippet
at the top of each file), so they can be run from anywhere.

## Data / figure layout

- `data/era5/` — raw ERA5 downloads (single timestamps + `training_data/`
  winter time series). `data/dart_extract/` — TCo1279-DART field extracts.
  `data/results/` — derived json/npz output from the method-comparison
  scripts.
- `figures/era5/` — raw-field diagnostics (e.g. z0 vs. TCo1279).
  `figures/methods/{loglaw,powerlaw,comparison}/` — per-method figures.
- `logs/` — run logs. `reports/` — `FINDINGS.md`, the write-up of results.
