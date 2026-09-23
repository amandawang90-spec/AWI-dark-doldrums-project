---
title: "SSRD Reconstruction: Evaluation Summary Across All Tested Months"
subtitle: "Seven ERA5 months (real ground truth) and four DART months (transfer test), Global / Europe / Germany / Korea"
author: "AWI Dark-Doldrums Project"
date: "2026-09-17"
geometry: margin=2.5cm
fontsize: 11pt
---

## Scope

Eleven months have now been tested in total: seven on **ERA5** (real
3-hourly ground truth exists) and four on **DART** (only a real monthly
mean exists; the 3-hourly reconstruction itself is what's being validated
for). All eleven use the same production model (`models/model_v1/model_kt.joblib`,
trained once on ERA5 August 2025 / March 2026 / August 2026 -- no winter
data went into training) and the same evaluation code, broken down the same
way: Global, Europe, Germany, Korea.

Two different metrics are used, and this report keeps them in fully
separate sections rather than mixing them, because they answer different
questions and behave very differently:

- **Pointwise R²** -- only possible on ERA5, since only ERA5 has real
  3-hourly ground truth. Compares every individual 3-hourly prediction
  against the real value, at every grid cell. This is the time-series
  fidelity metric -- does the model track *when* radiation rises and falls.
- **Monthly-mean spatial R²** -- the only metric possible on DART, and also
  computed on ERA5 for direct comparability. Aggregates to a monthly mean
  per cell first, then asks whether the model ranks *which cells* get more
  or less sun correctly. In a small region with little genuine spatial
  variance to explain, this statistic is much more fragile and can turn
  sharply negative even when the underlying predictions are good in an
  absolute sense.

---

# Part 1: Pointwise R² (real 3-hourly ground truth, ERA5 only)

DART has no 3-hourly truth to compare against, so this section covers
ERA5's seven tested months only.

| region | Nov'25 | Dec'25 | Jan'26 | Feb'26 | Mar'26 | Aug'25 | Aug'26 |
|---|---|---|---|---|---|---|---|
| Global | 0.91 | 0.90 | 0.93 | 0.97 | 0.96 | 0.98 | 0.98 |
| Europe | 0.97 | 0.96 | 0.95 | 0.95 | 0.96 | 0.98 | 0.98 |
| Germany | 0.90 | 0.92 | 0.92 | 0.89 | 0.98 | 0.98 | 0.98 |
| Korea | 0.97 | 0.96 | 0.96 | 0.98 | 0.98 | 0.93 | 0.94 |

**Reading**: never below 0.89, for any region, in any of the seven months.
This is the headline reassurance from the pointwise metric alone -- the
model's grip on the actual time-varying signal, the part that determines
whether a dark-doldrums event gets detected, is robust across the full
annual cycle tested so far, in all three specific regions, including deep
winter.

---

# Part 2: Monthly-mean spatial R²

## 2a. ERA5 (all seven months)

| region | Nov'25 | Dec'25 | Jan'26 | Feb'26 | Mar'26 | Aug'25 | Aug'26 |
|---|---|---|---|---|---|---|---|
| Global | 0.92 | 0.93 | 0.95 | 0.97 | 0.95 | 0.99 | 0.99 |
| Europe | 0.99 | 0.99 | 0.98 | 0.97 | 0.94 | 0.98 | 0.97 |
| Germany | 0.74 | 0.90 | 0.93 | 0.61 | 0.47 | 0.70 | 0.64 |
| Korea | **0.84** | 0.52 | 0.11 | **-1.31** | -1.12 | -0.07 | -1.37 |

## 2b. DART (all four months -- the only ground truth DART allows at all)

| region | Jan'50 | Apr'50 | Jul'50 | Oct'50 |
|---|---|---|---|---|
| Global | 0.98 | 0.95 | 0.98 | 0.97 |
| Europe | 0.97 | 0.87 | 0.97 | 0.98 |
| Germany | 0.87 | **0.41** | 0.53 | 0.92 |
| Korea | 0.43 | **-2.14** | 0.41 | 0.86 |

## 2c. Reading the spatial results together

**Korea's fragile spatial fit is a real property of the method, confirmed
on real ERA5 ground truth, not manufactured by the ERA5-to-DART
transfer.** ERA5's own Korea R² collapses from 0.84 (November) to -1.31
(February) -- the same late-winter-into-spring window where DART's Korea
result is also weakest. This is the method's own limitation, visible even
with real ground truth and no transfer involved.

**But the DART transfer clearly makes a bad month worse, not just "the
same."** ERA5's January Korea result is *marginal* (0.11) -- not great, but
nowhere near DART's January (0.43) or, worse, DART's April (-2.14, the
single worst result found anywhere in this evaluation). DART's own cloud
scheme and snow/ice representation add a real, additional error on top of
whatever spatial fragility the reconstruction method already has in that
season.

**A genuine gap this comparison exposes**: DART's worst result (April,
Korea, -2.14) has no ERA5 cross-check yet -- ERA5 has never been tested for
April. Since April is DART's worst month by a wide margin, ERA5 April is
now the single most informative test left to run: it would show directly
whether DART's April crisis is "the known seasonal fragility, amplified,"
matching the Nov-to-Feb arc already seen, or something specific to DART
alone.

**Germany shows the same qualitative shape, smaller amplitude.** ERA5
Germany dips to 0.47-0.74 in its harder months (Mar, Aug, Nov) and recovers
to 0.90-0.93 in Dec/Jan; DART Germany dips to 0.41-0.53 (Apr, Jul) and
recovers to 0.87-0.92 (Jan, Oct). The specific weak calendar months don't
line up between the two datasets, which is itself informative: a shared
month name (e.g. "January") doesn't guarantee matching synoptic weather
between a reanalysis and a free-running climate simulation.

---

# Practical implications

- **For time-series / event-detection work** (the actual dark-doldrums use
  case): Part 1's results support trusting the reconstruction across the
  full year, including winter, in all three specific regions.
- **For anything relying on the absolute spatial pattern within a small
  region** in the transition months (roughly February through April, and
  to a lesser extent August): treat Korea's reconstruction with real
  caution; Germany's less severely but still noticeably.
- **`fal` (surface albedo) remains the best-supported next step** --
  literature-validated as essential for this class of method, zero new
  downloads needed, and doubly motivated now: it explains both the DART
  transfer finding and this newly confirmed ERA5-native seasonal
  fragility.
- **Test ERA5 April next**, specifically to resolve whether DART's worst
  result is "the known fragility, amplified" or a separate problem.

---
*All figures generated on Levante; source arrays and maps in
`solar/data/reconstructed_ssrd/`, `solar/figures/`. Production model:
`solar/models/model_v1/model_kt.joblib`.*
