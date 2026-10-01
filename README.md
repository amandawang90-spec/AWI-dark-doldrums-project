# 🌙 AWI Dark Doldrums Project

> Changes in Dunkelflaute (dark-doldrum) risk under climate change: a 9 km-resolution pipeline that reconstructs the missing wind and solar fields in TCo1279-DART, validates the result against ERA5 reanalysis and real German grid data (SMARD), and compares detected events against four published definitions from the literature.

---

## 📌 Project Purpose

A *Dunkelflaute* is a period when wind and solar output are both low at once — the case that matters most for grid resilience, since the two resources are normally expected to compensate for each other. Germany, where the term originates, saw two such periods lasting 30 and 54 days between December 2022 and March 2023. This project builds the pipeline needed to detect these events in a high-resolution (~9 km) climate simulation, so that how Dunkelflaute risk changes under warming can actually be assessed.

The catch: **TCo1279-DART doesn't carry everything needed directly.** It writes 100 m wind only as 10 m (`u10`/`v10`), and solar irradiance (`ssrd`) only as a monthly mean, not 3-hourly. Both gaps have to be filled by a reconstruction trained and validated on ERA5 — which does have the real fields — before being trusted on DART, where there's no ground truth to check against directly. That reconstruction-and-validation work is most of what this repo contains; detecting and characterizing Dunkelflaute events is what it's *for*.

This is an extension of the proposal in [`Project_Proposal.md`](Project_Proposal.md), a 4-week study originally scoped to cover Germany **and** South Korea, both the 1950-control and 2080s-SSP5-8.5 climates, sub-national spatial patterns, synoptic drivers, and formal uncertainty quantification. The **Status** section below says plainly which of that is done, and which is still ahead.

---

## ✅ Status at a Glance

| Proposal item | Status |
|---|---|
| Wind reconstruction (10 m → 100 m), validated vs. ERA5 + SMARD | **Done** — log law, see [`wind/README.md`](wind/README.md) |
| Solar reconstruction (monthly → 3-hourly `ssrd`), validated vs. ERA5 + SMARD | **Done** — ML model v5, see [`solar/README.md`](solar/README.md) |
| Real Germany boundary (land + EEZ), not a bounding box | **Done** — [`boundaries/`](boundaries/) |
| Pipeline validation against ERA5 (proposal §5.8) | **Done** — Mockert threshold reproduces ~5.6 events/winter against a published ~4–6/year; Li/Kaspar/Lohmann comparisons also run, see Literature Comparison below |
| Baseline (1950C) and future (2080C) Dunkelflaute climatology for Germany, Mockert threshold | **Done** — see Key Findings below |
| Same comparison under the Li et al. (2021) dual-technology threshold, applied to DART | **Not yet** — Li/Kaspar/Lohmann so far only validated on ERA5, not re-run on 1950C/2080C |
| South Korea domain | **Not started** — only a `dunkelflaute/context/` visualization of the proposal's pattern-scaling coefficients exists, not an actual Dunkelflaute detection run |
| Sub-national spatial analysis (coast vs. inland vs. mountains) | **Not started** |
| Synoptic composite analysis (mean sea-level pressure during events) | **Not started** |
| Onshore vs. offshore wind behaviour specifically *during events* | **Not started** (onshore/offshore wind CF is validated generally; event-conditional breakdown isn't built) |
| Threshold/window sensitivity tests (CF 0.04/0.08/0.10, 24h/72h) | **Not started** |
| Uncertainty quantification (year-block bootstrap, internal consistency) | **Not started** |

In short: the hard validation work — proving the reconstructed wind and solar fields can be trusted — is done and holds up well. The comparative climate-change science the proposal was ultimately for (Germany vs. Korea, synoptic mechanisms, uncertainty bounds) is the work that's still ahead.

---

## 🔑 Key Findings

**Solar reconstruction is essentially at the ERA5 ceiling.** The v5 model reproduces real ERA5 `ssrd` at r = 0.998 over Germany, and real SMARD generation at r = 0.931 — matching ERA5's *own* ability to predict SMARD (also r = 0.931). The gap to real-world generation belongs to ERA5 itself, not to the reconstruction. See [solar/README.md](solar/README.md#-key-findings).

**Wind reconstruction is solid onshore, noisier offshore — for reasons unrelated to the method.** Log-law 100 m wind matches real ERA5 at RMSE ≈ 1.19 m/s onshore / 0.60 m/s offshore (r = 0.97 / 0.997). Against real SMARD generation, onshore capacity factor reaches r = 0.91 (vs. real ERA5's own r = 0.95); offshore is noisier for *both* the reconstruction and real ERA5 alike (r ≈ 0.70–0.71), pointing at the power-curve simplification and offshore capacity-data quality rather than the reconstruction itself. See [wind/README.md](wind/README.md#-key-findings).

**The pipeline reproduces realistic Dunkelflaute statistics, and detects a real climate-change signal for Germany.** Using the Mockert et al. (2023) threshold (48h rolling combined CF < 6%, Germany's 2024 capacity weights):

| Period | Winters | Events/winter | Mean combined CF |
|---|---|---|---|
| ERA5 (validation, 2016–2026) | 11 | 5.64 | 0.186 |
| **DART 1950C (control)** | 17 (1953–1969) | **3.47** | 0.201 |
| **DART 2080C (SSP5-8.5 future)** | 10 (2083–2092) | **5.50** | 0.199 |

The ERA5 validation figure (5.64/winter) sits inside the published ~4–6/year range, which is what the ERA5-validation step (proposal §5.8) was for. Comparing the two DART endpoints: event frequency rises from 3.47/winter under 1950-control forcing to 5.50/winter under the 2080s SSP5-8.5 future — a **~59% increase**, and the future figure lands close to where present-day ERA5 already sits. This is a genuine, if single-threshold, answer to proposal questions 1 and 2 (baseline climatology and the climate-change signal) for Germany. Source data: `dunkelflaute/results/{1950c,2080c,validation_era5_2015_2026}/summary.json`; write-up: `dunkelflaute/results/report_dunkelflaute.html`.

---

## 📂 Layout

- **[`solar/`](solar/)** — reconstructs 3-hourly `ssrd` from the fields DART *does* carry 3-hourly (cloud fractions + `tsr`), via a `HistGradientBoostingRegressor` predicting clearness index, then an exact monthly rescale to match DART's real monthly total. `solar/models/` holds five training iterations, v1→v5; **use v5**. `solar/reference/` documents the OpenIFS solar-geometry physics this reimplements.
- **[`wind/`](wind/)** — reconstructs 100 m wind (`u100`/`v100`) from 10 m wind via the log law, using ERA5 land roughness (year-cycled) over land and the Charnock relation over ocean. `wind/reports/FINDINGS.md` has the log-law-vs-regression validation writeup.
- **[`dunkelflaute/`](dunkelflaute/)** — combines the solar + wind capacity factors into Germany's combined CF and detects Dunkelflaute events. This is where the actual science question lives:
  - `scripts/core/capacity_factor.py`, `domain.py` — shared CF formulas and the real Germany land+EEZ boundary mask (from [`boundaries/`](boundaries/), not a bounding box).
  - `scripts/era5/` — the ERA5-side pipeline. Start with `compute_germany_dunkelflaute_2015_2026_mockert.py`, the current, validated version.
  - `scripts/dart/` — the DART-side pipeline (1950C/2080C), Mockert-weighted; `compute_germany_dunkelflaute_dart_mockert.py` + `make_report.py`.
  - `smard_validation/` — SMARD real-generation data and CF computation, used as ground truth throughout the ERA5 validation.
  - `results/` — finished 1950C/2080C/ERA5-validation summaries and figures (the Key Findings table above comes from here). `reports/` — the earlier Mockert/Li/Kaspar/Lohmann literature-comparison write-ups.
  - `context/` — not part of the reconstruction/validation pipeline itself, but the motivation for the 1950C-vs-2080C comparison: TCo1279-DART's climate pattern-scaling coefficients (local change in wind speed and cloud cover — the two Dunkelflaute drivers — per 1K of global warming), visualized from an externally-supplied dataset. Also shows the South Korea domain from the proposal, alongside Germany, though no Dunkelflaute detection has been run there yet.
- **[`boundaries/`](boundaries/)** — Germany's real land (Natural Earth) and EEZ (Marine Regions) polygons, used by every Germany-domain script instead of a lat/lon bounding box.

Everything else at the repo root is either a fixed project file (`LICENSE`, `Project_Proposal.md`, this `README.md`) or data/tooling excluded from version control — see Repo Hygiene below.

---

## 🧹 Repo Hygiene

- **`scratch/`** (gitignored) — a working directory for one-off, exploratory analysis that isn't part of the documented pipeline: ad hoc robustness checks (e.g. "does scoring with the properly held-out CV-fold model change the v4/v5 result?"), draft versions of HTML reports before they're published, and intermediate JSON summaries. Nothing here is load-bearing for the pipeline in `solar/`, `wind/`, or `dunkelflaute/` — if a script or finding in those folders turned out to matter, it was promoted out of `scratch/` into the real structure. Treat it as a notebook margin, not a fourth project folder.
- **Superseded outputs are archived with a README, not scattered.** `dunkelflaute/data/germany_era5/archive/` is the model for this: old combined-CF runs that predate the current methodology are kept (for traceability) in their own `archive/` subfolder with a README explaining exactly why each one is superseded and that nothing should be built on them. Follow this pattern rather than leaving stale output next to current output with no explanation.
- **2026-10-01 cleanup**: removed three directories that didn't follow either rule above — `solar/data/dart_eval_1950_check/`, `dart_eval_1950/`, and `dart_eval_1950_winter1/` were pilot/diagnostic runs from the earliest DART evaluation attempts (Sep 21–22, scoring the old v1/v2 models, before the v5 model and the production `dart_reconstruct_*` pipeline existed), undocumented and unreferenced by anything except the job scripts that produced them. Removed those job scripts (`solar/jobs/dart_eval_year.sbatch`, `dart_recon_year.sbatch`, `dart_winter_1950.sbatch`) along with their logs. Also removed `dunkelflaute/scripts/compute_germany_combined_cf.py` and its output `dunkelflaute/data/germany/` — the script was already documented as superseded (see the solar README's Recommendations table) and its single output directory, confusingly similarly named to `dunkelflaute/data/germany_era5/`, had no archive note of its own.

---

## 📖 Literature Comparison

Four published Dunkelflaute definitions are checked against this pipeline's ERA5 output — algorithm faithfully reproduced from each paper's own text, not a secondhand summary:

- **Mockert et al. (2023)** — 48h rolling-mean combined CF < 6%, their exact window-expansion event-construction rule, their capacity weights. The only definition also run on DART 1950C/2080C so far (see Key Findings).
- **Li et al. (2021)** — instantaneous wind CF < 20% *and* solar CF < 20%, sustained > 24h, no smoothing.
- **Kaspar et al. (2019)** — the paper Mockert calibrated their threshold against; instantaneous CF < 10%, ≥ 48h, no smoothing.
- **Lohmann et al. (2025)** — not a new definition but a report card on the others: evaluated against real grid-stress data (Energy Not Served), found CF-threshold methods are weak predictors (F-score 0.14) next to residual-load-based ones (F-score 0.41).

Visual reports: [Mockert Comparison](https://claude.ai/artifact/54xCd9Uaf2QZiD9jRb4yQm), [Li Comparison](https://claude.ai/artifact/3W6FQWUhVoSF8rKKxXR9pk), [Kaspar and Lohmann Comparison](https://claude.ai/artifact/5j1r6dcDGJukME3RfFsLPL). Write-ups: `dunkelflaute/reports/`.

---

## 🗺️ What's Next

Directly from the Status table above, roughly in the order the original proposal sequenced them:

1. **Re-run the Li et al. threshold on DART 1950C/2080C**, not just ERA5 — gives the dual-threshold robustness check the proposal calls for (§5.4), using machinery that already exists.
2. **South Korea domain** — extract the Korean TCo1279-DART fields, build the equivalent boundary mask, and run the same validated pipeline. The hard part (reconstruction + validation) transfers directly; what's missing is just running it on a second domain.
3. **Sub-national spatial patterns** — North Sea vs. Baltic vs. inland Germany; the data already carries this resolution, it isn't aggregated out yet.
4. **Synoptic composite analysis** — mean sea-level pressure during Dunkelflaute events vs. all-time mean, control vs. future, for the physical "why" behind the frequency change above.
5. **Uncertainty quantification** — year-block bootstrap and the internal-consistency check described in proposal §5.9; right now the 59% frequency increase is a point estimate with no reported confidence interval.
6. **Threshold/window sensitivity** — CF thresholds of 0.04/0.08/0.10 and 24h/72h windows, to see how much the headline finding depends on the specific Mockert calibration.

---

## 🖥️ Running on the HPC

`python` is not on `PATH` by default. Load the environment module first:

```bash
module load analysis-toolbox/python-04.2026
```

Many scripts have a matching `.sh` wrapper that does this for you — prefer the wrapper where one exists.

Downloads use the [CDS API](https://cds.climate.copernicus.eu/how-to-api) and need a personal access token in `~/.cdsapirc`; no credentials are stored here.

## 💾 Data

Input/output NetCDF and most intermediate `.npz`/`.csv` files are **not tracked** in git (see `.gitignore`) — they run to hundreds of GB. Recreate them by running the relevant `download_*` / `compute_*` / `dart_reconstruct_*` scripts. Small, genuinely load-bearing summary files (report data, training configs) are tracked.

## 🌐 ERA5 Grid Note

ERA5 here is 0.25° regular lat/lon. Raw percentiles over that grid over-count polar points — 48% of land points sit poleward of 60°N/S but only 21% of land *area* does. All spatial averages in this repo are area-weighted by cos(lat) unless a script says otherwise.

## 📜 License

MIT — see [LICENSE](LICENSE).
