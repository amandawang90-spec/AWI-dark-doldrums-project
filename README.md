# AWI dark doldrums project

A "Dunkelflaute" (dark doldrum) is a period when wind and solar power output
are both low at once — the case that matters most for grid resilience, since
the two resources are normally expected to compensate for each other. This
repo reconstructs the fields needed to detect Dunkelflaute events in
**TCo1279-DART** climate simulations (1950C control run and 2080C
high-emission future), validates that reconstruction against **ERA5**
reanalysis and **SMARD** (Bundesnetzagentur) real German grid data, and
compares the resulting event statistics against four published definitions
from the literature.

DART doesn't carry everything needed directly: it writes 100 m wind only as
10 m (`u10`/`v10`), and solar irradiance (`ssrd`) only as a monthly mean, not
3-hourly. The reconstruction pipeline below fills both gaps, trained and
validated on ERA5 — which does have the real fields — before being applied to
DART, where there's no ground truth to check against directly.

## Layout

- **[`solar/`](solar/)** — reconstructs 3-hourly `ssrd` from the fields DART
  *does* carry 3-hourly (cloud fractions + `tsr`), via a
  `HistGradientBoostingRegressor` predicting clearness index, then an exact
  monthly rescale to match DART's real monthly total. `solar/models/` (see
  its own README) holds five training iterations, v1→v5; **use v5**.
  `solar/reference/` documents the OpenIFS solar-geometry physics this
  reimplements.
- **[`wind/`](wind/)** — reconstructs 100 m wind (`u100`/`v100`) from 10 m
  wind via the log law, using ERA5 land roughness (year-cycled) over land and
  the Charnock relation over ocean. `wind/reports/FINDINGS.md` has the
  log-law-vs-regression validation writeup.
- **[`dunkelflaute/`](dunkelflaute/)** — combines the solar + wind capacity
  factors into Germany's combined CF and detects Dunkelflaute events. This is
  where the actual science question lives:
  - `scripts/core/capacity_factor.py`, `domain.py` — shared CF formulas and
    the real Germany land+EEZ boundary mask (from [`boundaries/`](boundaries/),
    not a bounding box).
  - `scripts/era5/` — the ERA5-side pipeline. Start with
    `compute_germany_dunkelflaute_2015_2026_mockert.py`, the current,
    validated version (see `data/germany_era5/README.md` for why).
  - `smard_validation/` — SMARD real-generation data and CF computation, used
    as ground truth throughout the ERA5 validation.
  - `reports/` — published HTML comparisons against the literature (open
    directly in a browser).
- **[`boundaries/`](boundaries/)** — Germany's real land (Natural Earth) and
  EEZ (Marine Regions) polygons, used by every Germany-domain script instead
  of a lat/lon bounding box.
- **[`scaling/`](scaling/)** — one-off check of TCo1279-DART's seasonal
  wind-speed/cloud-cover scaling factors.

## Literature comparison

Four published Dunkelflaute definitions are checked against this pipeline's
output — algorithm faithfully reproduced from each paper's own text, not a
secondhand summary:

- **Mockert et al. (2023)** — 48h rolling-mean combined CF < 6%, their exact
  window-expansion event-construction rule, their capacity weights.
- **Li et al. (2021)** — instantaneous wind CF < 20% *and* solar CF < 20%,
  sustained > 24h, no smoothing.
- **Kaspar et al. (2019)** — the paper Mockert calibrated their threshold
  against; instantaneous CF < 10%, ≥ 48h, no smoothing.
- **Lohmann et al. (2025)** — not a new definition but a report card on the
  others: evaluated against real grid-stress data (Energy Not Served), found
  CF-threshold methods are weak predictors (F-score 0.14) next to
  residual-load-based ones (F-score 0.41).

Results and methodology write-ups: `dunkelflaute/reports/`.

## Running on the HPC

`python` is not on `PATH` by default. Load the environment module first:

```bash
module load analysis-toolbox/python-04.2026
```

Many scripts have a matching `.sh` wrapper that does this for you — prefer the
wrapper where one exists.

Downloads use the [CDS API](https://cds.climate.copernicus.eu/how-to-api) and
need a personal access token in `~/.cdsapirc`; no credentials are stored here.

## Data

Input/output NetCDF and most intermediate `.npz`/`.csv` files are **not
tracked** in git (see `.gitignore`) — they run to hundreds of GB. Recreate
them by running the relevant `download_*` / `compute_*` / `dart_reconstruct_*`
scripts. Small, genuinely load-bearing summary files (report data, training
configs) are tracked.

## ERA5 grid note

ERA5 here is 0.25° regular lat/lon. Raw percentiles over that grid over-count
polar points — 48% of land points sit poleward of 60°N/S but only 21% of land
*area* does. All spatial averages in this repo are area-weighted by cos(lat)
unless a script says otherwise.

## License

MIT — see [LICENSE](LICENSE).
