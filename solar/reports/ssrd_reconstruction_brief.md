---
title: "3-Hourly Solar Radiation for TCo1279-DART: Method & Status"
subtitle: "Short briefing"
author: "AWI Dark-Doldrums Project"
date: "2026-09-15"
geometry: margin=2.5cm
fontsize: 11pt
---

## The problem

The TCo1279-DART model runs (1950C and 2080C) save top-of-atmosphere
shortwave radiation and cloud fractions every 3 hours, but save surface
downward shortwave (`ssrd`) — the quantity the dark-doldrums study actually
needs — only as a **monthly mean**. A monthly mean cannot show the
short-lived low-wind/low-sun events the study is looking for. Running a
real radiative transfer model to fill the gap isn't possible: it needs
aerosol and humidity profiles that DART's saved output doesn't contain.

**Approach**: statistically reconstruct the missing 3-hourly `ssrd` from
what DART *does* save, using ERA5 (which has both) to learn the
relationship, then apply it to DART.

## The method, briefly

Instead of predicting `ssrd` directly, we predict the normalized
**clearness index** $k_t = ssrd / tisr$, where `tisr` is top-of-atmosphere
incoming radiation — pure astronomy, computable exactly from lat/lon/time.
Normalizing this way keeps the target in a stable [0,1]-ish range regardless
of season, which is what lets a model trained on a few ERA5 months
generalize at all.

Predictors: TOA transmission ($tsr/tisr$), solar zenith angle, and the four
cloud fraction layers. We tested a plain cloud-fraction formula first and it
only explained 57% of the variance; adding transmission and switching to
gradient boosting (which lets clouds attenuate differently depending on
cloud type, rather than one fixed sensitivity) raised that to **98.6%**,
tested honestly on ERA5 months the model never saw during training
(leave-one-month-out validation, RMSE 32.7 W/m², bias approx. 0).

After the 3-hourly prediction, each grid cell is **rescaled so its monthly
mean exactly matches DART's own real monthly `ssrd`** — this removes any
systematic level bias and guarantees consistency with DART's own numbers.

## Applying it to DART: first test (January 1950)

Two things had to be solved specifically for DART, not needed for ERA5:

- **`tisr` lookup, not recomputation.** Since solar geometry only depends on
  day-of-year/time-of-day (not the specific year), we precomputed and
  independently verified two reusable reference files — one normal year,
  one leap year — covering every year in both DART periods exactly.
- **Correct accumulation window.** DART's fields are 3-hour accumulations,
  confirmed three independent ways (metadata, and matching known ~190 W/m²
  January climatology). Getting this wrong wouldn't crash anything — it
  would silently distort the transmission predictor.

**Result of the one-month test**: before correction, the reconstruction was
6.6% too high (204 vs. 191 W/m² real DART monthly mean) — comparable to
error levels seen during ERA5 validation. After the per-cell monthly
rescale, the corrected output matches DART's real monthly value almost
exactly (residual 0.0002 W/m²).

## The open question

All the 98.6%-accuracy numbers above are *ERA5 predicting ERA5* — they
prove the method captures how ERA5's own radiation scheme behaves. They do
**not** prove DART's radiation scheme responds to the same cloud fraction
in exactly the same way, since DART runs a different model at different
resolution. The first hint this gap is real: per-cell correction factors on
the test month ranged from 0.56× to 5.3×, wider than seen in ERA5 (though
the median is exactly 1.0 and 90% of cells fall in a much tighter 0.94–1.36
band — the extremes are a minority of outlier locations, not a global
problem).

This can't be closed directly — DART has no independent 3-hourly `ssrd`
anywhere to check against; that's the whole reason this method exists. The
practical mitigation: the monthly rescale absorbs any *systematic* bias, so
overall magnitude is trustworthy even where within-month shape might be
imperfect.

## Verdict and next steps

**Not clear yet whether the model needs changing** — one test month passing
within the expected error range isn't a failure signal, but the wide
scale-factor spread needs to be checked, not ignored. Before committing to
the full 240-month (1950C) + 156-month (2080C) run:

1. Map scale factors across a handful of contrasting months (not just one)
   and check whether large corrections cluster geographically (e.g. terrain,
   high-latitude winter) — a pattern would point to a fixable feature gap
   (e.g. adding surface albedo, which DART does carry monthly) rather than
   a fundamental problem.
2. Refactor for efficiency: the current pipeline re-reads the same
   `tisr` reference file every month even within the same calendar year.
   Restructuring to read it once per year cuts total reference-file reads
   from 396 to 33 before the full run is launched.
3. Once both checks pass, run the full two periods, one SLURM job per year.

---
*Full technical report with all validation tables and figures available on
request.*
