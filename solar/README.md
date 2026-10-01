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
  - Pre-rescale ("is the model itself any good?"):
    rmse_maps.py, diag_training.py, diag_bias_v2.py
  - Post-rescale ("how much 3-hourly error is left once the monthly
    sum is forced?"): rmse_maps_rescaled.py, reconstruct_and_diagnose.py
        │
        ▼
Phase 5: Validate
  ├──► vs. real ERA5 ssrd (held out)     -- the SCRIPTS support global/regional
  │                                         scoring, but the headline number
  │                                         quoted in Key Finding 3 (r=0.998) is
  │                                         Germany-only -- no equivalent GLOBAL
  │                                         correlation figure has been computed
  │                                         and saved for v5. See Scope table below.
  └──► vs. real SMARD generation          -- Germany-only, unavoidably (SMARD has
                                              no equivalent dataset anywhere else)
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
  - Finished: 1950C 138/140 months (1950-1969), 2080C 91/91 months (2080-2092);
    Feb/Mar 1952 skipped -- DART's own tcc/tsr input files don't exist
    (1952 is spin-up, so nothing downstream is affected)
  - dart_reconstruct_year.py does NOT write error maps; the error_map_<stamp>.npz
    files in data/reconstructed_ssrd/ come from earlier reconstruct_and_diagnose.py
    runs. A check of the finished DART output itself has not been run yet.
```

**Visual report (Phase 5/6, decade view)**: [v5 vs ERA5 vs SMARD, 2015–2026](https://claude.ai/artifact/ThqZ2FGwKCWeYHTUAYHPzV) — solar, onshore, offshore and combined CF against real ERA5 and real SMARD, year by year across the full period (Key Finding 3's Germany-only table above is the single-snapshot version of this same check; this is its decade-long counterpart, and the one Phase 6's combined-CF literature validation builds on).

### Scope: what's global vs. what's Germany-only

| Phase | Script(s) | Scope |
|-------|-----------|-------|
| 1. Download | `download_era5.py` | Global — `AREA = None` |
| 2. Train | `train_v1..v5_model.py` | Global (v3+ minus the lat/elevation mask — not a country restriction) |
| 3. Constrained rescale | `rescale.py` | Global — no spatial logic |
| 4. Diagnose | `rmse_maps*.py`, `diag_*.py` | Mostly global/1°, with Europe/Korea boxes at full resolution in `rmse_maps.py` |
| 5. Validate — vs. ERA5 (**capability**) | `evaluate_era5_regional.py`, `evaluate_era5_winter.py`, `dart_year_eval.py` | Selectable: Global / Europe / Germany / Korea boxes |
| 5. Validate — vs. ERA5 (**the actual quoted number**) | `germany_solar_cf_correlation_summary.json` | **Germany-only** — no global equivalent has been computed and saved for v5 |
| 5. Validate — vs. SMARD | `dunkelflaute/smard_validation/` + `boundaries/` | **Germany-only**, unavoidably — SMARD has no equivalent dataset anywhere else in this repo |
| 7. Apply to DART | `dart_reconstruct_year.py` | Global — full native grid, no region slice |

**Takeaway**: the model is trained, rescaled, and applied to DART everywhere on Earth. Broad error diagnostics ran globally during development (Phase 4). But the one clean, quantified "does this actually work" check — the one checked against real-world grid generation, not just another reanalysis — has only ever been computed for Germany, because that's the only place SMARD exists. Key Finding 3's r=0.998/0.931 figures are both Germany numbers, paired against each other on purpose; neither is a global statistic.

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

---

## 🔑 Key Findings

### 1. The accumulation-window bug is why v4/v5 exist

v1–v3 trained against ERA5 `ssrd` sampled as a **1-hour accumulation taken every 3 hours** — not the same quantity as a genuine 3-hour sum, which is what DART's own 3-hourly fields actually are. This was silent: v1–v3's validation numbers looked fine because they were being scored against the same (subtly wrong) kind of target they were trained on. v4 fixed it with `era5_3h_mean`, genuine 3-hour sums; v5 is the same fix, restricted to Sep–Mar.

### 2. A second, independent bug looked like a model problem until diagnosed

SMARD's hourly generation was originally resampled to 3-hourly with a forward-looking bin (`label="left"`), while ERA5's own accumulated fields are backward-looking. Fixing the resample convention alone raised the SMARD correlation from **~0.75 to ~0.97** — before touching the model at all.

### 3. Current accuracy is essentially at the ERA5 ceiling (Germany — no global equivalent computed)

All three numbers below are for Germany specifically — this is the SMARD comparison, and SMARD only exists for Germany. No global version of this table exists; see the Scope table above.

| Comparison (Germany only) | r |
|------------|---|
| v5 reconstruction vs. real ERA5 | 0.998 |
| v5 reconstruction vs. real SMARD | 0.931 |
| **Real ERA5 vs. real SMARD** | **0.931** |

The reconstruction doesn't just correlate well with ERA5 over Germany — it matches real ERA5's own ability to predict the real German grid. The remaining gap to SMARD belongs to ERA5 itself, not to the reconstruction. Whether this holds up equally well elsewhere in the world is untested — there's no real-generation ground truth outside Germany in this repo to check it against.

**Visual report**: [Recon vs SMARD](https://claude.ai/artifact/WZKEpqfxir2j9nsJsZ1yF2) — the published comparison this table's numbers come from.

### 4. One finding remains open, stated rather than hidden

A genuine-3-hour ERA5 variant (`era5_true3h`, built by direct summation of native hourly data rather than the accumulation-field trick) scores *slightly worse* against SMARD than the original 1-hour-sampled data — even after the SMARD alignment fix. Not yet explained.

### 5. v3 and later carry an unvalidated-extrapolation caveat

The training mask excludes cells south of 60°S and above 3000 m elevation. Predictions in those cells are extrapolated by the model, not validated against held-out data there specifically.

---

## 💡 Recommendations Before Applying to DART-1950C / 2080C

| Scenario | Status | Recommendation |
|----------|--------|-----------------|
| **1950C** | Reconstruction finished (138/140 months; Feb/Mar 1952 inputs missing) | Closer to the training climate (2015–2025) — lower distribution-shift risk, but still an out-of-sample test with no direct ground truth. |
| **2080C** | Reconstruction finished (91/91 months) | No ground truth exists for a future climate. Frame any result as *"under the assumption that today's cloud-to-irradiance physics holds"*, not as validated. |
| **Both** | — | Cheap, concrete check before trusting either: train on a subset of ERA5 years, test on the years furthest away in time, as a proxy for temporal distribution-shift risk. Large skill degradation with distance = a real warning sign; little movement = actual supporting evidence, not just an assumption. |
| **Both** | Not yet run | A spatial-resolution counterpart to the check above: coarsen ERA5's own 0.25° cloud fractions further (e.g. to ~1°) and see how much the fitted kt-relationship's skill degrades, as a proxy for whether training-resolution (~28 km) vs. application-resolution (~9 km) is a real risk here — the wind side already ran this exact style of test for z0 and got a small, quantified answer; solar hasn't yet. |
| **Both** | — | Done: Germany combined CF and Mockert events for 1950C and 2080C (real boundary, Mockert weights, corrected event rule) are in `dunkelflaute/results/` — see `report_dunkelflaute.html`, built by `dunkelflaute/scripts/dart/compute_germany_dunkelflaute_dart_mockert.py`. The older `dunkelflaute/scripts/compute_germany_combined_cf.py` is superseded. |

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
│   ├── dart_reconstructed/                FINAL DART output (1950C + 2080C years)
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
- **DART reconstruction is finished** (jobs complete) — `data/dart_reconstructed/` still mixes 1950C and 2080C years in one flat directory; splitting it into `dart_reconstructed_1950c/`/`_2080c/` is now possible but not yet done.
- **DART output not yet checked** — monthly-sum residuals and CF ranges of the finished files have not been verified.

---

## 📊 Key Formulas

**Model**

```
Clearness index:        kt = ssrd / tisr                          (prediction target)
Cloud-radiative ratio:  T  = tsr / tisr                           (feature)
Cos-zenith proxy:       mu = tisr / (Δt · S0)                     (feature)
Reconstructed ssrd:     ssrd_pred = kt_pred × tisr                (before rescale)
Exact monthly rescale:  Σ ssrd_pred(month) ≡ Σ ssrd_real(month)   (hard constraint)
Capacity factor:        CF = clip((ssrd / accum_seconds) / 1000, 0, 1)
```

**`tisr` — analytic top-of-atmosphere irradiance** (`scripts/core/solar_geometry.py`; Spencer 1971 / Iqbal 1983)

```
γ      = 2π (doy − 1) / 365.25
E0     = 1.000110 + 0.034221 cos γ + 0.001280 sin γ + 0.000719 cos 2γ + 0.000077 sin 2γ
δ      = 0.006918 − 0.399912 cos γ + 0.070257 sin γ − 0.006758 cos 2γ + 0.000907 sin 2γ
         − 0.002697 cos 3γ + 0.001480 sin 3γ
EoT    = 229.18 (0.000075 + 0.001868 cos γ − 0.032077 sin γ − 0.014615 cos 2γ − 0.040890 sin 2γ)   [min]

H      = (t_UTC[min] + 4·lon + EoT) / 4 − 180°          (hour angle)
cos z  = max(0, sin φ sin δ + cos φ cos δ cos H)         (φ = latitude)

I(t)   = S0 · E0(t) · cos z(t)                           S0 = 1361 W/m²   (instantaneous, W/m²)
tisr(t) = Δt · (1/n) Σ_{k=0..n−1} I( t − Δt + (k + ½)·Δt/n )       [J/m²]
          (midpoint rule over the Δt window ENDING at t; DART cells: Δt = 3 h, n = 8;
           ERA5-grid function defaults: Δt = 1 h, n = 12)
```

---

## 📜 License

MIT — see the repository [LICENSE](../LICENSE).
