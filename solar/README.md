# Solar capacity factor reconstruction

Reconstructs 3-hourly surface downward shortwave radiation (ssrd) from fields
that TCo1279-DART actually carries (tsr, cloud fractions), trained and
validated on ERA5, for later use as a solar capacity-factor proxy.

## Pipeline order

1. **`scripts/download/`** — pull ERA5 fields (`download_era5.py`/`.sh`) and
   static orography (`download_era5_orography.py`).
2. **`scripts/preprocessing/`** — build the analytic TISR templates
   (`build_tisr_templates.py`), verify solar geometry and ERA5 completeness
   (`check_solar_geometry.py`, `check_era5*.py`), audit units
   (`audit_era5_units.py`), extract the ONDJF subset (`extract_ondjf.py`).
3. **`scripts/core/`** — the shared library every other script imports:
   `reconstruct_ssrd.py` (core kt-model, features, training loop),
   `reconstruct_ssrd_dart.py` (DART-specific variant), `solar_geometry.py`
   (analytic TOA irradiance), `rescale.py` (monthly-sum-constrained rescale).
4. **`scripts/training/`** — fit the models: `train_final_model.py` (v1),
   `train_v2_model.py` (v2, multi-year CV), `train_v3_model.py` (v3,
   ONDJF-only), `diag_training.py`/`diag_bias_v2.py` (training diagnostics).
5. **`scripts/evaluation/`** — score the models against ERA5/DART:
   `evaluate_era5_regional.py`, `evaluate_era5_winter.py`, `evaluate_v2_cv.py`,
   `evaluate_v3_rescaled.py`, `dart_year_eval.py`, `rmse_maps*.py`,
   `summarize_v2_eval.py`, `summarize_v3.py`, `reconstruct_and_diagnose.py`.
6. **`scripts/plotting/`** — figures from evaluation output:
   `make_v2_figures.py`, `plot_rmse_maps.py`.

Every script chdirs to this `solar/` root on import (see the `_root` snippet
at the top of each file), so they can be run from anywhere; SLURM jobs in
`jobs/` invoke them by their `scripts/<category>/` path.

## Data / model / figure layout

- `data/era5/` — raw ERA5 downloads. `data/static/`, `data/analytical_tisr/`
  — static/precomputed inputs. `data/reconstructed_ssrd/`,
  `data/dart_eval_1950*/` — reconstruction/evaluation output.
  `data/training_cache_*/` — cached training features.
- `models/model_v1/`, `model_v2/`, `model_v3/` — one folder per model
  version (v1 = single-fit baseline, v2 = multi-year CV, v3 = ONDJF-only).
- `figures/era5/` — diagnostics against raw ERA5 (error maps, TISR/scale
  checks). `figures/models/` — model evaluation figures (RMSE maps, v2/v3
  CV plots).
- `jobs/` — SLURM sbatch scripts. `reports/` — write-ups (md/pdf).
  `notebooks/` — exploratory notebooks. `logs/` — SLURM job logs.
