# ☀️ Solar Capacity Factor Reconstruction

> Reconstructs 3-hourly solar irradiance and capacity factor for TCo1279-DART — which only carries monthly-mean irradiance — trained and validated on ERA5 reanalysis and real German grid data (SMARD), for use in Dunkelflaute (dark-doldrum) risk analysis.

---

## 📌 Project Purpose

TCo1279-DART writes surface downward shortwave radiation (`ssrd`) only as a monthly mean, but the fields needed to *predict* 3-hourly `ssrd` — cloud fractions and net solar radiation — it does carry at 3-hourly resolution. This project closes that gap, and answers the questions that matter before the result can be trusted:

- Can 3-hourly `ssrd` be predicted from DART's own 3-hourly cloud/radiation fields, accurately enough to drive a solar capacity-factor calculation?
- Does the reconstruction hold up against **real** ERA5 `ssrd` (held out, never trained on)?
- Does it *also* hold up against **real German grid generation** (SMARD) — an independent check that doesn't depend on ERA5 being right?
- What exactly changed across five model iterations, and which one should be trusted?
- What's still unresolved, and what should be checked before applying this to a future-climate scenario (2080C) with no ground truth at all?

---

## 📂 Data Sources

| Source | Period | Role |
|--------|--------|------|
| **ERA5** reanalysis | 2015-01-01 – 2026-03-31, Sep–Mar months only | Training target + validation testbed |
| **TCo1279-DART 1950C** | 1950–1969 (20 years), Sep–Mar only | Present-day-analogue scenario the model is applied to |
| **TCo1279-DART 2080C** | 2080–2092 (13 years), Sep–Mar only | High-emission future scenario the model is applied to |
| **SMARD** (Bundesnetzagentur) | 2015-01-01 – 2026-09-27 (raw download); Sep–Mar subset used to match the ERA5 window | Independent ground truth, unrelated to ERA5 |
| **Boundaries** ([`boundaries/`](../boundaries/)) | Static — Natural Earth land polygon + Marine Regions EEZ polygon | Real Germany land+EEZ mask, not a lat/lon box. Not read by anything in `solar/` directly — it's applied one step downstream, in `dunkelflaute/scripts/era5/`, to turn the global `ssrd` reconstruction into the Germany-only aggregate that SMARD is actually compared against |

### Grid systems: ERA5 vs. TCo1279-DART

The model is trained on one grid and applied to a structurally different one.
Neither the grid spacing nor the grid *topology* match, and it matters for
how the reconstruction should be read.

| | ERA5 | TCo1279-DART |
|---|------|--------------|
| Grid type | Regular latitude–longitude | Cubic-octahedral reduced Gaussian (spectral truncation T1279) |
| Nominal resolution | 0.25° (~28 km at the equator, narrowing towards the poles in the longitude direction since spacing is defined in degrees, not distance) | ~9 km, uniform across latitude (Gaussian grids are constructed to keep physical cell area roughly constant, unlike a regular lat/lon grid) |
| Grid points, globally | 721 × 1440 = 1,038,240 | 6,599,680 (irregular per-latitude point count, verified directly from the reconstructed output files) |
| Point arrangement | Every latitude row has the same 1,440 longitude points | Point count per latitude row *decreases* towards the poles — points are only placed where needed to preserve resolution, not on a fixed rectangular grid |

No regridding happens anywhere in this pipeline — the model is applied
natively to each grid in turn, for the methodological reason explained under
Model Specification below. Whether that's actually *safe* across two such
different resolutions is a separate, still-open question — see Known Data
Quality Issues.

### ERA5 fields

All `ssrd`/`tsr` fields carry the **same** `units` attribute — `J m**-2` —
regardless of source or accumulation window. That's a trap, not a
convenience: see "Units and accumulation conventions" below.

| Field | Description | Unit | Timestep |
|-------|-------------|------|----------|
| `ssrd` (3-hourly) | Surface solar radiation downwards, **real** | J/m², genuine 3-hour accumulation | 3-hourly |
| `ssrd` (monthly) | Same field, ERA5's own monthly product — downloaded and used as an independent cross-check on the 3-hourly reconstruction | J/m², mean of a **1-day** accumulation | Monthly |
| `tsr` | Top net solar radiation | J/m², 3-hour accumulation | 3-hourly |
| `tisr` | TOA incident solar radiation | W/m² (computed, not accumulated) | Computed analytically, not a stored field — pure astronomy (see `scripts/core/solar_geometry.py`) |
| `tcc`/`hcc`/`mcc`/`lcc` | Total/high/medium/low cloud cover | fraction, 0–1 | 3-hourly |

### TCo1279-DART fields

| Field | Description | Unit | Timestep |
|-------|-------------|------|----------|
| `ssrd` | Surface solar radiation downwards | J/m², mean of a **3-hour** accumulation | Monthly mean only — the gap this project fills |
| `tsr` | Top net solar radiation | J/m², 3-hour accumulation | 3-hourly |
| `tcc`/`hcc`/`mcc`/`lcc` | Total/high/medium/low cloud cover | fraction, 0–1 | 3-hourly |
| `tisr` | TOA incident solar radiation | W/m² (computed) | Computed analytically, same as for ERA5 — DART carries no such field itself, which is exactly why this predictor transfers cleanly |

### SMARD fields

| Field | Description | Unit | Timestep |
|-------|-------------|------|----------|
| Solar generation | Realisierte Erzeugung | MW | Hourly (native); resampled to 3-hourly to match ERA5/DART |
| Installed capacity | Solar PV | MW | Yearly |

### Derived quantity

| Field | Description |
|-------|-------------|
| `kt` | Clearness index, `ssrd/tisr` — the model's actual prediction target |

### Units and accumulation conventions

Converting J/m² to W/m² means dividing by the number of seconds the value was
*actually* accumulated over — and that window is **not** written anywhere in
the file's own metadata; every ERA5/DART ssrd/tsr field claims the same units
string no matter which window applies. Get the divisor wrong and everything
downstream is silently off by a clean factor (e.g. 86400/10800 = 8×) while
the rescale step's own residual check still reports "perfect" agreement,
since it only confirms internal consistency with whatever divisor it was
given — not that the divisor is the right one.

| Field | Accumulation window | Divisor (seconds) |
|-------|---------------------|--------------------|
| ERA5 monthly ssrd | Mean of a 1-day accumulation | 86,400 |
| ERA5 hourly ssrd | 1 hour | 3,600 |
| ERA5 3-hourly ssrd/tsr | 3 hours (genuine sum) | 10,800 |
| DART monthly ssrd | Mean of a 3-hour accumulation | 10,800 |
| DART 3-hourly ssrd/tsr | 3 hours | 10,800 |

ERA5's monthly convention was verified directly, not assumed: `GRIB_stepType`
confirms `avgad` (average of daily accumulations), and dividing by 86,400
reproduces the 3-hourly file's own `/3600` mean to within 0.01 W/m²
(`era5_monthly_ssrd_202508` = `era5_ssrd_3h_202508` = 179.66 W/m², both
ways). DART's monthly convention is likewise read from its own
`cell_methods = "time: mean (interval: 3 h)"` attribute, identical for both
1950C and 2080C — **not** assumed to match ERA5's. This single value is
isolated in one named constant, `MONTHLY_ANCHOR_DIVISOR`
(`scripts/core/reconstruct_ssrd*.py`), specifically so the ERA5 and DART
variants of a shared script can never silently drift onto the wrong divisor.

**Full unit-conversion chain, J/m² → capacity factor:**

```
ssrd (J/m²)  ÷ accum_seconds  →  irradiance (W/m²)  ÷ 1000 (STC)  →  clip[0,1]  →  CF
```

---

## 🛠️ Technical Stack

| Layer | Tool |
|-------|------|
| **Model** | `sklearn.ensemble.HistGradientBoostingRegressor` (sklearn 1.1.1) |
| **Data handling** | xarray, netCDF4, numpy |
| **Compute** | AWI HPC (SLURM), `analysis-toolbox/python-04.2026` module |
| **Persistence** | joblib (models), NetCDF4 (reconstructed fields) |

---

## 🏗️ Pipeline

```
ERA5 (cloud fractions, tsr, real ssrd)
        │
        ▼
Phase 1: Download + Preprocess
  - Pull ERA5 fields (download_era5.py)
  - Build analytic TISR templates (build_tisr_templates.py)
  - Audit units, verify solar geometry
        │
        ▼
Phase 2: Train (scripts/training/train_v*.py)
  - Predict kt = ssrd/tisr from [T, tcc, hcc, mcc, lcc, mu]
  - HistGradientBoostingRegressor, cos(lat)-weighted
        │
        ▼
Phase 3: Constrained Rescale (scripts/core/rescale.py)
  - ssrd_pred = kt_pred × tisr, THEN exactly rescaled so each
    calendar month's sum matches the real monthly total
        │
        ▼
Phase 4: Diagnose (per-cell RMSE, bias, error maps)
  - Pre-rescale: rmse_maps.py, diag_training.py, diag_bias_v2.py
  - Post-rescale: rmse_maps_rescaled.py, reconstruct_and_diagnose.py
        │
        ▼
Phase 5: Validate
  ├──► vs. real ERA5 ssrd (held out)     -- global grid, scoreable anywhere
  └──► vs. real SMARD generation          -- Germany-only, by construction
        │
        ▼
Phase 6: Literature validation (NOT in solar/ -- the gate before DART)
  - Solar CF alone isn't what gets checked against the literature -- it's
    weighted-combined with wind onshore/offshore CF first, in dunkelflaute/,
    on the same 2015-2026 ERA5 period, THEN checked against four published
    Dunkelflaute definitions (Mockert, Li, Kaspar, Lohmann). This is the
    trustworthiness gate: only once the combined pipeline held up here did
    it get applied to DART. See the root README's dunkelflaute/ section,
    dunkelflaute/data/germany_era5/README.md, and dunkelflaute/reports/.
        │
        ▼
Phase 7: Apply to DART (dart_reconstruct_year.py)
  - Same model, same rescale, DART's own cloud fields + monthly ssrd
  - reconstruct_and_diagnose.py writes an error_map_<stamp>.npz alongside
    every reconstructed month (Phase 4's diagnostic step, run again here,
    since there's no ERA5/SMARD ground truth for DART itself)
```

### Phase 4 in detail: what each diagnostic script actually measures

| Script | Measures | Against | Rescale applied? | Scope |
|--------|----------|---------|-------------------|-------|
| `rmse_maps.py` | Per-cell RMSE + bias (W/m²), and RMSE of the monthly means | Real ERA5 3-hourly `ssrd`, held-out years | No — raw model output | Global (1°), Europe + Korea (full 0.25°) |
| `diag_training.py` | Learning curve (1M→32M rows), train-vs-held-out error, area/energy-weighted variants | Common held-out set (Jan/Apr/Jul/Oct 2023–2024) | No | Area-weighted (cos lat) globally |
| `diag_bias_v2.py` | v1 vs v2 bias, binned by `tisr`-equivalent irradiance | Same held-out ERA5 rows | No | Wherever the held-out rows fall (global sample) |
| `rmse_maps_rescaled.py` | Per-cell RMSE of the **full pipeline** (model + rescale) | Real ERA5 3-hourly `ssrd`, held-out months | **Yes** | Selectable month range (e.g. `--months 12,1,2`) |
| `reconstruct_and_diagnose.py` | Per-cell actual/predicted/absolute-error/scale-factor arrays | DART's own monthly total (there's no independent truth for DART) | Pre-rescale diagnostic, written alongside the rescaled output | Whatever grid the DART run covers — global |

The pre-rescale scripts (`rmse_maps.py`, `diag_training.py`, `diag_bias_v2.py`) answer "is the ML model itself any good?"; the post-rescale ones (`rmse_maps_rescaled.py`, `reconstruct_and_diagnose.py`) answer "after the hard monthly-sum constraint is enforced, how much residual 3-hourly error is left?" — a materially different (usually smaller) number, since the rescale can't fix within-month timing errors but does fix the mean.

### Scope: what's global vs. what's Germany-only

| Phase | Script(s) | Scope |
|-------|-----------|-------|
| 1. Download | `download_era5.py` | Global — `AREA = None` |
| 2. Train | `train_v1..v5_model.py` | Global (v3+ minus the lat/elevation mask above — not a country restriction) |
| 3. Constrained rescale | `rescale.py` | Global — no spatial logic; the monthly-sum constraint is enforced per cell, wherever that cell is |
| 4. Diagnose | `rmse_maps*.py`, `diag_*.py` | Mostly global/1°, with Europe/Korea boxes at full resolution in `rmse_maps.py` |
| 5. Validate — vs. ERA5 | `evaluate_era5_regional.py`, `evaluate_era5_winter.py`, `dart_year_eval.py` | Selectable: Global / Europe / Germany / Korea boxes |
| 5. Validate — vs. SMARD | `dunkelflaute/smard_validation/` + `boundaries/` | **Germany-only** — SMARD has no equivalent dataset for any other country in this repo |
| 7. Apply to DART | `dart_reconstruct_year.py` | Global — full native grid (`sl = slice(0, N)`), no region slice |

**Takeaway:** the model is trained, rescaled and applied to DART everywhere on Earth; the one check that doesn't depend on ERA5 being right — real reconstruction vs. real grid generation — has only ever been run for Germany.

### Corrections Applied

| Correction | Reason |
|------|--------|
| Genuine 3-hour-sum ERA5 target (`era5_3h_mean`), not 1h-accum-sampled-every-3h | v1–v3 were silently trained against a target that wasn't the same quantity DART's own 3-hourly fields represent |
| SMARD resampled `label="right", closed="right"` | Matches ERA5's backward-looking accumulation convention — using the pandas default (`label="left"`) misaligned every comparison by one bin |
| Exact constrained monthly rescale (hard constraint, not post-processing) | The reconstruction is only as trustworthy as its agreement with DART's own real monthly total — enforced exactly, not approximately |

---

## 🗺️ Model Roadmap

| Version | Training window | Domain | Key change | Status |
|---------|-----------------|--------|------------|--------|
| v1 | 3 single ERA5 months (Aug 2025, Mar 2026, Aug 2026), stride 16 | **Global**, no exclusions — includes Antarctica, the Himalaya/Andes/Tibetan Plateau, everything | Baseline `HistGradientBoostingRegressor`, single combined fit | Superseded |
| v2 | 2015–2025, all 12 months (132 months), stride 8 | **Global**, no exclusions — same as v1 | More training data than v1, same method otherwise | Superseded |
| v3 | Oct–Feb (ONDJF) only, 2015–2025 | **Global minus** cells south of 60°S *and* above 3000 m elevation (first version to mask anything) | `area`/`plain` cos(lat)-weighting variants, 3 year-block CV folds | Superseded |
| v4 | All 12 months, 2015–2026 (136 months) | **Global minus** the same mask as v3 (lat ≥ −60°, elevation ≤ 3000 m) | **Fixes the accumulation-window bug** — genuine 3-hour-sum target | Reference (full-year evaluation) |
| **v5** | **Sep–Mar, 2015–2025 (80 months)** | **Global minus** the same mask as v3/v4 | Same fix as v4, restricted to the season actually used downstream | **Current — use this** |

v1 and v2 were never masked at all — their predictions over Antarctica, high mountains, etc. are just as "in-sample" as anywhere else, for better or worse. Starting at v3, the training mask excludes cells south of 60°S and above 3000 m; predictions there (all versions v3+) are extrapolated, not validated against held-out data specifically from those cells (see Key Finding 5 below). No version has ever excluded anything based on ocean/land, hemisphere, or country — the exclusion is purely the one lat/elevation mask.

---

## 📐 Model Specification

**Features** (6): `T` (= `tsr/tisr`, cloud-radiative ratio), `tcc`, `hcc`, `mcc`, `lcc`, `mu` (solar-geometry / cos-zenith proxy).

**Target**: `kt = ssrd / tisr`, clipped to `[0, 1.5]`.

**Training config (v5)**:

| Parameter | Value |
|-----------|-------|
| Variant | `area` (cos(lat)-weighted) |
| Months | Sep–Mar (extended winter), 2015–2025 |
| Rows used | 32,000,000 |
| Mask | `lat ≥ -60°` and `elevation ≤ 3000 m` |
| `max_iter` | 250 |
| `dt_seconds` | 10,800 (3-hourly) |

**Capacity factor**: `CF = clip((ssrd / accum_seconds) / 1000, 0, 1)` — 1000 W/m² is STC irradiance.

### How `tisr` is calculated

`tisr` isn't read from any file — it's computed analytically
(`scripts/core/solar_geometry.py`) from latitude, longitude and timestamp
alone, using standard solar-geometry astronomy (Spencer 1971 Fourier-series
approximations, the same reference formulas used in solar-engineering
irradiance models, e.g. Iqbal 1983):

1. **Orbital terms** from fractional day-of-year: eccentricity correction
   `E0`, solar declination, and the equation of time.
2. **Cosine of solar zenith angle**:
   `cosz = sin(lat)·sin(decl) + cos(lat)·cos(decl)·cos(H)`, clipped to zero
   below the horizon (night), where `H` is the hour angle from solar time
   (UTC, corrected for longitude and the equation of time).
3. **Instantaneous TOA irradiance** = `SOLAR_CONSTANT (1361 W/m²) × E0 × cosz`.
4. **Accumulation**: ERA5's `tisr` is a J/m² accumulation over the window
   *ending* at its timestamp, not an instantaneous value — reproduced by
   averaging the instantaneous irradiance over 8 sub-steps within that
   window (midpoint rule) and multiplying by the window length in seconds,
   rather than a closed-form integral (simpler to get right).

Two grid-shaped variants exist: `cos_zenith_grid`/`toa_irradiance_accumulated`
for ERA5's regular `(lat, lon)` outer-product grid, and
`cos_zenith_cells`/`toa_irradiance_accumulated_cells` for DART's flat,
irregular cell list, where `lon` at cell *i* isn't independent of `lat` at
cell *i*. Both are the same physics — this is exactly why the feature
transfers to DART natively (see Grid Systems, above): it needs no stored
field on either side, only each point's own coordinates and time, so it's
evaluated directly on whichever grid is supplied, with nothing to regrid.
Accuracy is checked directly against ERA5's own real `tisr` field
(`check_solar_geometry.py`) before being trusted anywhere a truth field
doesn't exist — i.e. for DART.

---

## 🔑 Key Findings

### 1. The accumulation-window bug is why v4/v5 exist

v1–v3 trained against ERA5 `ssrd` sampled as a **1-hour accumulation taken every 3 hours** — not the same quantity as a genuine 3-hour sum, which is what DART's own 3-hourly fields actually are. This was silent: v1–v3's validation numbers looked fine because they were being scored against the same (subtly wrong) kind of target they were trained on. v4 fixed it with `era5_3h_mean`, genuine 3-hour sums; v5 is the same fix, restricted to Sep–Mar.

### 2. A second, independent bug looked like a model problem until diagnosed

SMARD's hourly generation was originally resampled to 3-hourly with a forward-looking bin (`label="left"`), while ERA5's own accumulated fields are backward-looking. Fixing the resample convention alone raised the SMARD correlation from **~0.75 to ~0.97** — before touching the model at all.

### 3. Current accuracy is essentially at the ERA5 ceiling

| Comparison | r |
|------------|---|
| v5 reconstruction vs. real ERA5 | 0.998 |
| v5 reconstruction vs. real SMARD | 0.931 |
| **Real ERA5 vs. real SMARD** | **0.931** |

The reconstruction doesn't just correlate well with ERA5 — it matches real ERA5's own ability to predict the real grid. The remaining gap to SMARD belongs to ERA5 itself, not to the reconstruction.

### 4. One finding remains open, stated rather than hidden

A genuine-3-hour ERA5 variant (`era5_true3h`, built by direct summation of native hourly data rather than the accumulation-field trick) scores *slightly worse* against SMARD than the original 1-hour-sampled data — even after the SMARD alignment fix. Not yet explained.

### 5. v3 and later carry an unvalidated-extrapolation caveat

The training mask excludes cells south of 60°S and above 3000 m elevation. Predictions in those cells are extrapolated by the model, not validated against held-out data there specifically.

---

## 💡 Recommendations Before Applying to DART-1950C / 2080C

| Scenario | Status | Recommendation |
|----------|--------|-----------------|
| **1950C** | Reconstruction in progress | Closer to the training climate (2015–2025) — lower distribution-shift risk, but still an out-of-sample test with no direct ground truth. |
| **2080C** | Reconstruction in progress | No ground truth exists for a future climate. Frame any result as *"under the assumption that today's cloud-to-irradiance physics holds"*, not as validated. |
| **Both** | — | Cheap, concrete check before trusting either: train on a subset of ERA5 years, test on the years furthest away in time, as a proxy for temporal distribution-shift risk. Large skill degradation with distance = a real warning sign; little movement = actual supporting evidence, not just an assumption. |
| **Both** | Not yet run | A spatial-resolution counterpart to the check above: coarsen ERA5's own 0.25° cloud fractions further (e.g. to ~1°) and see how much the fitted kt-relationship's skill degrades, as a proxy for whether training-resolution (~28 km) vs. application-resolution (~9 km) is a real risk here — the wind side already ran this exact style of test for z0 and got a small, quantified answer; solar hasn't yet. |
| **Both** | — | Recompute the Germany Dunkelflaute combined-CF once reconstruction finishes, using the real boundary + Mockert's exact weights + the corrected event-construction rule — the existing `dunkelflaute/scripts/compute_germany_combined_cf.py` still needs those three fixes ported over from the ERA5-side pipeline. |

---

## 📁 Project Structure

```
solar/
├── scripts/
│   ├── download/       ERA5 field downloads (download_era5.py, backfill scripts)
│   ├── preprocessing/  TISR templates, unit/geometry/completeness checks
│   ├── core/           shared library: reconstruct_ssrd*.py, solar_geometry.py, rescale.py
│   ├── training/       train_v1..v5_model.py, training diagnostics
│   ├── evaluation/     score against ERA5/SMARD/DART; dart_reconstruct_year.py (production)
│   └── plotting/       evaluation figures
├── models/              model_v1/ .. model_v5/ (see models/README.md)
├── reference/           OpenIFS TSIR source + call-chain notes
├── data/
│   ├── era5/, era5_1h/, era5_3h_mean/    ERA5 downloads
│   ├── analytical_tisr/                   precomputed TOA irradiance templates
│   ├── dart_reconstructed/                DART output (reconstruction in progress)
│   └── training_cache_*/                  cached features (regenerable)
├── jobs/                 SLURM sbatch scripts
├── reports/              write-ups (md/pdf)
└── figures/               diagnostic and evaluation plots
```

Every script `chdir`s to this `solar/` root on import (the `_root` snippet at the top of each file), so they run from anywhere; SLURM jobs in `jobs/` invoke them by their `scripts/<category>/` path.

---

## ⚙️ Technical Notes

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Predict `kt = ssrd/tisr`, not `ssrd` directly | `tisr` is pure astronomy (no reanalysis needed), so the model only has to learn the cloud-modulation factor — and it transfers cleanly to DART, which has no `tisr` field of its own but can compute it analytically |
| Exact constrained monthly rescale, always | Matching DART's real monthly total exactly is a hard constraint, not a nicety — see the standing project rule: rescaling is part of the model, evaluate only after it |
| `HistGradientBoostingRegressor` over a simpler model | Handles the nonlinear cloud→clearness relationship without manual feature engineering beyond the six base features |
| cos(lat) area weighting (`area` variant) | Un-weighted training over-represents polar cells, which are a large share of grid points but a small share of land area |
| v5 trained Sep–Mar only, not full year | Matches the season actually used downstream (the Dunkelflaute analysis), and both the ERA5 validation and DART reconstruction only need Sep–Mar |

### Known Data Quality Issues

- **True-3h anomaly** (Finding 4 above) — unresolved, flagged rather than hidden.
- **v3+ extrapolation caveat** — cells south of 60°S / above 3000 m are extrapolated, not validated.
- **Untested cross-resolution transfer**: the model's being *pointwise* means it needs no regridding to run on DART's grid (see Grid Systems, above) — but that sidesteps, rather than answers, whether the relationship it learned actually holds at a different resolution. ERA5's cloud fractions at 0.25° are area averages over a much larger footprint (~28 km) than DART's ~9 km native cells; if the true cloud–clearness relationship is nonlinear in sub-grid cloud heterogeneity — plausible, since a partly-cloudy coarse cell and a genuinely overcast fine cell can share the same mean `tcc` but very different `kt` — a model fit on coarser, more-averaged ERA5 inputs could behave differently on DART's less-averaged fields. The wind side already ran this exact style of check for its own resolution mismatch (`wind/README.md`'s z0 coarsening sensitivity test, costed at ~0.2–0.25 m/s RMSE); no analogous test has been run for solar. See Recommendations, below.
- **DART reconstruction is in progress**, not complete, as of this writing — `data/dart_reconstructed/` currently mixes 1950C and 2080C years in one flat directory; splitting it into `dart_reconstructed_1950c/`/`_2080c/` is deferred until the jobs finish, so nothing gets moved out from under an active write.

---

## 📊 Key Formulas

```
Clearness index:        kt = ssrd / tisr                    (model's prediction target)
Reconstructed ssrd:      ssrd_pred = kt_pred × tisr           (before rescale)
Exact monthly rescale:   Σ ssrd_pred(month) ≡ Σ ssrd_real(month)   (hard constraint)
Capacity factor:         CF = clip((ssrd / accum_seconds) / 1000, 0, 1)
```

---

## 📜 License

MIT — see the repository [LICENSE](../LICENSE).
