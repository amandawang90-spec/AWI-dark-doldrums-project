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
Phase 4: Validate
  ├──► vs. real ERA5 ssrd (held out)
  └──► vs. real SMARD generation (independent of ERA5 entirely)
        │
        ▼
Phase 5: Literature validation (NOT in solar/ -- the gate before DART)
  - Solar CF alone isn't what gets checked against the literature -- it's
    weighted-combined with wind onshore/offshore CF first, in dunkelflaute/,
    on the same 2015-2026 ERA5 period, THEN checked against four published
    Dunkelflaute definitions (Mockert, Li, Kaspar, Lohmann). This is the
    trustworthiness gate: only once the combined pipeline held up here did
    it get applied to DART. See the root README's dunkelflaute/ section,
    dunkelflaute/data/germany_era5/README.md, and dunkelflaute/reports/.
        │
        ▼
Phase 6: Apply to DART (dart_reconstruct_year.py)
  - Same model, same rescale, DART's own cloud fields + monthly ssrd
```

### Corrections Applied

| Correction | Reason |
|------|--------|
| Genuine 3-hour-sum ERA5 target (`era5_3h_mean`), not 1h-accum-sampled-every-3h | v1–v3 were silently trained against a target that wasn't the same quantity DART's own 3-hourly fields represent |
| SMARD resampled `label="right", closed="right"` | Matches ERA5's backward-looking accumulation convention — using the pandas default (`label="left"`) misaligned every comparison by one bin |
| Exact constrained monthly rescale (hard constraint, not post-processing) | The reconstruction is only as trustworthy as its agreement with DART's own real monthly total — enforced exactly, not approximately |

---

## 🗺️ Model Roadmap

| Version | Training window | Key change | Status |
|---------|-----------------|------------|--------|
| v1 | — | Baseline `HistGradientBoostingRegressor` | Superseded |
| v2 | 2015–2025, all months | More training data than v1 | Superseded |
| v3 | Oct–Feb (ONDJF) only | `area`/`plain` cos(lat)-weighting variants | Superseded |
| v4 | All 12 months, 2015–2026 | **Fixes the accumulation-window bug** — genuine 3-hour-sum target | Reference (full-year evaluation) |
| **v5** | **Sep–Mar, 2015–2025** | Same fix as v4, restricted to the season actually used downstream | **Current — use this** |

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
