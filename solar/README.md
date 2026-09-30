# Solar capacity factor reconstruction

Reconstructs 3-hourly surface downward shortwave radiation (`ssrd`) — and from
it, solar capacity factor — for TCo1279-DART, which only carries `ssrd` as a
monthly mean. ERA5 (which has real 3-hourly `ssrd`) is the training target and
testbed; SMARD real German solar generation is the independent ground truth.

## Data

- **ERA5** (`data/era5/`, `data/era5_1h/`, `data/era5_3h_mean/`): cloud
  fractions (`tcc`/`hcc`/`mcc`/`lcc`), net solar radiation (`tsr`), and real
  `ssrd`, 2015–2026. `era5_1h` is native hourly; `era5_3h_mean` is the genuine
  3-hour sum built from it (see Methods — this distinction is the whole reason
  v4/v5 exist).
- **TCo1279-DART** (1950C, 2080C, via `scripts/evaluation/dart_reconstruct_year.py`):
  the same four cloud fractions plus `tsr` at 3-hourly resolution, but `ssrd`
  only as a monthly mean — the gap this pipeline fills.
- **SMARD** (`dunkelflaute/smard_validation/`): real German solar generation
  and installed capacity, used as ground truth independent of both ERA5 and
  the reconstruction.
- **Static**: `data/analytical_tisr/` — precomputed top-of-atmosphere
  irradiance templates (pure astronomy, no reanalysis needed — see
  `scripts/core/solar_geometry.py` and `solar/reference/README.md` for the
  physics this reimplements).

## Method

A `HistGradientBoostingRegressor` predicts the clearness index
`kt = ssrd/tisr` from six features: `T` (= `tsr/tisr`, cloud-radiative ratio),
the four cloud fractions, and `mu` (solar geometry / cos-zenith proxy). The
predicted `ssrd = kt_pred × tisr` is then **exactly rescaled** so each
calendar month's total matches the real monthly `ssrd` sum bit-for-bit
(`scripts/core/rescale.py`, `constrained_rescale`) — this is a hard
constraint, not a post-processing nicety (see the memory note: rescaling is
part of the model, always evaluate after it). Capacity factor is then
`CF = clip((ssrd/accum_seconds)/1000, 0, 1)` (1000 W/m² = STC irradiance).

## Procedure

1. Train on ERA5 (`scripts/training/train_v*.py`) — five iterations, see
   `models/README.md` for the full v1→v5 lineage.
2. Validate against ERA5's own real `ssrd` (held out) and against SMARD real
   generation — both checked, since agreeing with ERA5 alone doesn't prove
   the reconstruction is right if ERA5 itself has been fed a subtly wrong
   training target (see Findings).
3. Apply the validated model (v5) to DART's cloud fields via
   `dart_reconstruct_year.py`, with the same exact monthly rescale against
   DART's own real monthly `ssrd`.

## Findings

- **The accumulation-window bug (the important one).** v1–v3 were trained
  against ERA5 `ssrd` sampled as a 1-hour accumulation taken every 3 hours —
  not the same thing as a genuine 3-hour sum, which is what DART's own
  3-hourly fields actually are. v4 fixed this (`era5_3h_mean`, genuine
  3-hour sums); v5 is the same fix restricted to the Sep–Mar training window.
  This was found and fixed *before* discovering a second, separate bug below
  — worth stating plainly since it means v1–v3's validation numbers, however
  good they looked, were scoring against a subtly wrong target.
- **A second, independent bug in the SMARD comparison** (not a solar-model
  bug, but it looked like one until diagnosed): SMARD's hourly generation was
  originally resampled to 3-hourly with a forward-looking bin
  (`label="left"`), while ERA5's own accumulated fields are backward-looking
  (labeled at the *end* of the window they cover). Fixing the resample
  convention alone raised the SMARD correlation from ~0.75 to ~0.97.
- **Current accuracy (v5, Sep–Mar 2015–2026, Germany)**: r = 0.998 vs real
  ERA5 `ssrd`-derived CF, r = 0.931 vs real SMARD generation-derived CF (real
  ERA5 itself only reaches r = 0.931 against SMARD too — so the reconstruction
  is now essentially as good as real ERA5 is at predicting the real grid, not
  a meaningfully weaker proxy for it).
- **Open, not fully resolved**: a true-3-hour ERA5 variant (`era5_true3h`,
  built from native hourly data by direct summation rather than the
  accumulation-field trick) scores *slightly worse* against SMARD than the
  original 1-hour-sampled data, even after the SMARD alignment fix. Flagged
  as an honest unresolved finding, not swept under the rug.
- **DART reconstruction status**: log-law wind is complete for both 1950C and
  2080C (Sep–Mar, all years). Solar reconstruction (v5) is still running as
  of this writing — see `jobs/dart_reconstruct_{1950c,2080c}_array.sbatch`.

## File structure

```
scripts/
  download/       ERA5 field downloads (download_era5.py, backfill scripts)
  preprocessing/  TISR templates, unit/geometry/completeness checks
  core/           shared library: reconstruct_ssrd*.py, solar_geometry.py, rescale.py
  training/       train_v1..v5_model.py, training diagnostics
  evaluation/     score against ERA5/SMARD/DART; dart_reconstruct_year.py (DART production script)
  plotting/       evaluation figures
models/           model_v1/ .. model_v5/, one folder per iteration -- see models/README.md
reference/        OpenIFS TSIR source + call-chain notes (the physics solar_geometry.py reimplements)
data/             era5/, era5_1h/, era5_3h_mean/, analytical_tisr/, dart_reconstructed/ (DART output, in progress),
                  training_cache_*/ (cached features, regenerable)
jobs/             SLURM sbatch scripts
reports/          write-ups (md/pdf)
figures/          diagnostic and evaluation plots
```

Every script chdirs to this `solar/` root on import (the `_root` snippet at
the top of each file), so they run from anywhere; SLURM jobs in `jobs/`
invoke them by their `scripts/<category>/` path.
