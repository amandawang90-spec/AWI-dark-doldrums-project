# AWI dark doldrums project

Analysis code for "dark doldrum" events — periods of simultaneously low wind and
low solar resource — using ERA5 as a testbed for methods that will be applied to
TCo1279-DART.

The work is in two strands:

- **[`wind/`](wind/)** — reconstructing 100 m winds (`u100`/`v100`) from 10 m winds
  (`u10`/`v10`), since TCo1279-DART does not carry a 100 m level. Log-law and
  regression approaches are validated against ERA5, which does have real
  `u100`/`v100`.
- **[`solar/`](solar/)** — reconstructing 3-hourly surface downward shortwave
  (`ssrd`) from the fields TCo1279-DART does carry 3-hourly (`tsr` plus the four
  cloud fractions), since the runs write `ssrd` only as a monthly mean. ERA5
  supplies the training target. The reconstructed irradiance then drives a solar
  capacity factor.

## Findings so far

[`wind/FINDINGS.md`](wind/FINDINGS.md) is the working record of results. In short:

- The log law alone reproduces ERA5 `u100`/`v100` with RMSE ~1.19 m/s onshore and
  ~0.60 m/s offshore (r = 0.97 / 0.997), stable across three test months.
- Regression beats the log law on overall onshore RMSE, but the two metrics
  disagree in the low-wind (<3 m/s) dark-doldrum regime: the regression's fitted
  intercept is a large fraction of a 1–2 m/s true value. Neither method is precise
  there (MAPE 15–39%) — a limitation to state rather than hide.
- **TCo1279 has no usable roughness field.** `sr` (param 173) is a flat placeholder
  at 0.0001 m across all 6,599,680 points. Orography (`sdor`/`sdfor`) is populated
  but is terrain relief, not aerodynamic roughness, and is exactly zero over ocean.
- Borrowing ERA5's `fsr` is defensible: the log-law ratio is only logarithmically
  sensitive to z0, and the 25→9 km resolution mismatch costs ~0.2–0.25 m/s.
  Fetch and ignored displacement height are the larger caveats.

## Layout

```
wind/
  download_era5_winds.py       u10/v10/u100/v100 for a timestamp
  download_era5_z0_lsm.py      forecast surface roughness + land-sea mask
  loglaw_reconstruct.py        log-law reconstruction vs. real ERA5 u100/v100
  compare_loglaw_vs_regression.py
  compare_three_timestamps.py  seasonal stability across the three test dates
  z0_scale_sensitivity.py      cost of coarsening z0 (the ERA5→TCo1279 question)
  plot_z0_maps.py              TCo1279 sr vs. sdor vs. ERA5 fsr
  check_*.py                   data sanity checks (roughness, coverage, orography)
  figures/                     generated plots
  FINDINGS.md

solar/
  download_era5.py             every ERA5 field the reconstruction needs, in one
                               pass; skips what already exists (--list, --force)
  check_era5.py                completeness, structure and unit-convention checks
```

Many scripts have a matching `*.sh` wrapper that loads the environment module
before running them — every script in `solar/`, and most of the check/plot scripts
in `wind/`. Where a wrapper exists, run it rather than the `.py` directly.

## Running

On the AWI HPC (`albedo`), `python` is not on `PATH`. Every wrapper starts with:

```bash
module load analysis-toolbox/python-04.2026
```

which provides Python 3.12, xarray 2026.4.0, and cfgrib/ecCodes. Then, from the
`wind/` or `solar/` directory, either run a wrapper:

```bash
./plot_z0_maps.sh
```

or load the module yourself and call a script that has no wrapper:

```bash
module load analysis-toolbox/python-04.2026
python3 loglaw_reconstruct.py
```

Downloads use the [CDS API](https://cds.climate.copernicus.eu/how-to-api) and need
a personal access token in `~/.cdsapirc`. No credentials are stored in this repo.

## Data

Input NetCDF is **not tracked** — the ERA5 downloads run to several GB. Recreate
`wind/data/` and `solar/data/` by running the `download_era5_*` scripts before the
analysis scripts.

## A note on ERA5's grid

ERA5 test data here is 0.25° regular lat/lon (721×1440). Raw percentiles over that
grid over-count polar points — 48% of land points sit poleward of 60°N/S but only
21% of land *area* does. All scores in this repo are area-weighted by cos(lat).

## License

MIT — see [LICENSE](LICENSE).
