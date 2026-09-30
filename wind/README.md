# Wind capacity factor reconstruction

Reconstructs 100 m wind (`u100`/`v100`) for TCo1279-DART, which only carries
10 m wind. ERA5 (which has real `u100`/`v100`) is the testbed used to decide
which extrapolation method to trust; SMARD real German wind generation is the
independent ground truth for the resulting capacity factor.

## Data

- **ERA5** (`data/era5/`): `u10`/`v10`/`u100`/`v100` (real, for validation),
  `fsr` (forecast surface roughness, land z0), `lsm` (land-sea mask).
- **TCo1279-DART** (1950C, 2080C, `data/dart_extract/` for single-timestamp
  pulls, `data/dart_1950c_100m/`/`data/dart_2080c_100m/` for the full
  reconstruction): `u10`/`v10` only — no `u100`/`v100`, no usable roughness
  field (see Findings).
- **SMARD** (`dunkelflaute/smard_validation/`): real German onshore/offshore
  wind generation and installed capacity.

## Method

**Log-law extrapolation** (`scripts/methods/dart_loglaw/`), the production
method: `v100 = v10 · ln(100/z0)/ln(10/z0)`. Land z0 borrows ERA5's own `fsr`
climatology (2016–2025, year-cycled onto DART's much longer run); ocean z0 is
solved iteratively from the Charnock relation using that month's own DART
`u10`/`v10`. No fitted parameters.

Two alternatives were validated against it and rejected for production, kept
for comparison (`scripts/methods/{powerlaw,comparison}/`):
- **Power law** (IEC/Moemken et al. 2018 exponents) — no roughness field
  needed at all, but no clear accuracy advantage to justify dropping the
  log law's physical grounding.
- **Regression** (ERA5-fitted `u10→u100`) — better *overall* RMSE onshore,
  but loses in the low-wind dark-doldrum regime that actually matters here
  (see Findings), and its fitted coefficients are untested on a 9 km grid
  quite unlike the 25 km one they were fit on.

## Procedure

1. Validate all three methods against real ERA5 `u100`/`v100` (multiple
   timestamps/winters, area-weighted by cos(lat)).
2. Decide the z0 source for DART, since DART's own roughness field turned out
   to be unusable (see Findings) — settled on ERA5's `fsr`, after a
   sensitivity check showed the coarsening cost is small.
3. Run the chosen method (log law) for DART 1950C and 2080C, Sep–Mar, via
   `scripts/methods/dart_loglaw/reconstruct_month.py`.
4. Validate the resulting wind capacity factor against SMARD real generation
   (`dunkelflaute/smard_validation/`), independent of the ERA5-only checks.

## Findings

- **Log law vs. real ERA5 u100/v100**: RMSE ≈ 1.19 m/s onshore / 0.60 m/s
  offshore, r = 0.97 / 0.997. Stable across seasons, no drift.
- **Log law vs. regression, low-wind (<3 m/s) regime** — the dark-doldrum
  regime this whole project cares about: the two methods disagree by metric.
  Regression wins on RMSE (its fitted intercept helps at low wind); log law
  wins on MAPE (28–31% vs 35–39% onshore) because that same intercept is a
  large *fraction* of a 1–2 m/s true value. **Log law chosen for production**
  on that basis, plus it needs no fitted coefficients to transfer to DART's
  very different grid.
- **TCo1279's own roughness field is unusable** — `sr` (param 173) is a flat
  placeholder (0.0001 m everywhere) in every DART init file checked, not a
  real field. Orography (`sdor`/`sdfor`) isn't a substitute either: it's
  terrain relief, not aerodynamic roughness (~29× apart in magnitude), and is
  exactly zero over ocean where the log law works best.
- **Borrowing ERA5's z0 is defensible**: the log-law ratio is only
  logarithmically sensitive to z0, and coarsening ERA5 z0 by the actual
  25 km→9 km mismatch costs an estimated ~0.2–0.25 m/s RMSE on top of the
  existing 1.19 m/s onshore error (~3% degradation) — small next to the
  method's own baseline error.
- **Two caveats bigger than the z0 transfer itself**, not yet addressed:
  fetch (a 100 m wind needs ~10 km upwind fetch to equilibrate; DART's 9 km
  cells are at or below that, so the local log law is on weaker footing there
  regardless of z0 source), and displacement height (ignored entirely; likely
  a first-order cause of onshore error, concentrated in tall-canopy regions).
- **Against SMARD real generation** (pooled, full year 2015–2026, Germany):

  | | RMSE | bias | r | mean CF |
  |---|---|---|---|---|
  | onshore, reconstruction | 0.098 | +0.005 | 0.911 | 0.205 |
  | onshore, real ERA5 | 0.072 | −0.044 | 0.948 | 0.205 |
  | offshore, reconstruction | 0.274 | −0.033 | 0.704 | 0.433 |
  | offshore, real ERA5 | 0.270 | −0.008 | 0.714 | 0.433 |

  Onshore reconstruction tracks real ERA5 closely (r 0.911 vs 0.948).
  Offshore is markedly noisier for *both* reconstruction and real ERA5 (r
  ~0.70–0.71) — the gap there is dominated by something other than the
  log-law reconstruction itself (turbine curve simplification, offshore
  capacity data quality — see `dunkelflaute/smard_validation/`'s notes on the
  2015 offshore capacity growth artifact), since real ERA5 has the same
  ceiling.
- **DART reconstruction status**: complete for both 1950C and 2080C, Sep–Mar,
  all years (`data/dart_1950c_100m/`, `data/dart_2080c_100m/`).

## File structure

```
scripts/
  download/       ERA5 winds/z0/fsr/lsm downloads, single timestamps or full winters
  preprocessing/  QC: orography vs. roughness, z0 distribution, time coverage
  methods/
    loglaw/       log-law reconstruction vs. real ERA5 (validation)
    powerlaw/     power-law alternative (validation)
    dart_loglaw/  production DART reconstruction (reconstruct_month.py + sbatch launchers)
    comparison/   method-vs-method scoring, including the only regression fit in the repo
  analysis/       exploratory extensions beyond the three methods (e.g. testing
                  solar radiation/cloud cover as a log-law stability-error proxy)
data/
  era5/           raw ERA5 downloads (single timestamps + training_data/ winters)
  dart_extract/   TCo1279-DART single-field extracts
  dart_1950c_100m/, dart_2080c_100m/   production reconstruction output
  results/        derived json/npz from the method-comparison scripts
figures/          era5/ (raw-field diagnostics), methods/{loglaw,powerlaw,comparison}/
jobs/             SLURM sbatch scripts (downloads)
logs/             run logs
reports/          FINDINGS.md and other write-ups
```

Every script chdirs to this `wind/` root on import, so they run from anywhere.
