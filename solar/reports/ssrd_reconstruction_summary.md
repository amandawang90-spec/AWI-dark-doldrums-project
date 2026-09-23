---
title: "Reconstructing 3-Hourly Surface Solar Radiation for TCo1279-DART"
subtitle: "Method, validation on ERA5, first application to DART, and open questions before the full 1950C / 2080C runs"
author: "AWI Dark-Doldrums Project"
date: "2026-09-15"
geometry: margin=2.5cm
fontsize: 11pt
colorlinks: true
---

# 1. The problem this solves

TCo1279-DART saves top-of-atmosphere net shortwave radiation (`tsr`) and four
cloud-fraction fields (`tcc`, `hcc`, `mcc`, `lcc`) every three hours, on its
native ~9 km reduced grid. It never saves surface downward shortwave
(`ssrd`) at that frequency — only as a monthly mean. For a dark-doldrums
study that needs to identify short, simultaneous low-wind/low-sun events,
a monthly mean is useless: the whole point is to see three-hourly
variability. Something has to bridge the gap between what DART *has* and
what the study *needs*, using only fields DART actually saved.

Running a real radiative transfer calculation was not an option: that needs
aerosol loading, full vertical humidity and temperature profiles, and other
atmospheric state that DART's saved output does not carry either. So the
approach taken is **statistical reconstruction**: learn the relationship
between what DART has (`tsr`, clouds, sun angle) and what it needs (`ssrd`)
from a dataset that has both — ERA5 — then apply that learned relationship
to DART.

# 2. Method

## 2.1 The target and predictors

Rather than predicting `ssrd` directly, the model predicts a normalised
**surface clearness index**:

$$k_t = \frac{ssrd}{tisr}$$

where `tisr` is top-of-atmosphere *incident* shortwave — pure astronomy,
computable from latitude, longitude and time alone, with no dependence on
atmospheric state. Normalising by `tisr` bounds the target to roughly
[0, 1] regardless of season or latitude, which is what lets a model fitted
on a handful of months generalise at all.

Five predictors, all physically bounded near [0, 1] regardless of climate
period:

| predictor | definition | role |
|---|---|---|
| $T$ | $tsr / tisr$ | TOA transmission — the dominant predictor |
| $\mu$ | $tisr / (\Delta t \cdot S_0)$ | mean cosine of solar zenith over the window |
| `tcc`, `hcc`, `mcc`, `lcc` | cloud fractions | add information transmission alone misses |

$S_0 = 1361$ W m$^{-2}$ (solar constant), $\Delta t$ is the accumulation
window (1 h for ERA5 as downloaded, 3 h for DART — see §5.1).

## 2.2 Why a fixed formula was not enough

A classic cloud-fraction-only solar parameterisation was tried first,
tested against real ERA5 data rather than assumed to work:

| method | variance explained (R²) | typical error |
|---|---|---|
| cloud fraction alone | 0.566 | 35.3 W m$^{-2}$ |
| all four cloud layers | 0.695 | 29.6 W m$^{-2}$ |
| + TOA transmission (linear) | 0.850 | 22.6 W m$^{-2}$ |
| + TOA transmission (gradient boosting) | **0.881–0.986** | **19.7–33 W m$^{-2}$** |

Cloud fraction alone leaves 43% of the variance unexplained. Adding
transmission, which already integrates how much of the incoming radiation
the whole atmospheric column let through, roughly halves the error on its
own. The final jump from linear regression to gradient boosting reflects a
real physical fact: a given cloud fraction attenuates sunlight very
differently depending on whether it is thin high cloud or thick low cloud.
A linear model is forced to use one fixed sensitivity everywhere; a boosted
tree ensemble can bend the fitted relationship differently across that
range.

## 2.3 Training and validation on ERA5

**Data**: three ERA5 months chosen to span the seasonal cycle — August
2025, March 2026 (equinox), August 2026 — read directly from DKRZ's local
ERA5 pool at hourly resolution on the native N320 grid, thinned to a
stride-16 sample for tractability (a per-pixel instantaneous relationship
does not need every 0.25° point to fit well).

**Validation methodology: leave-one-month-out**, not a random split. A
random split lets the model see both August months in training and get
tested on nearby days of the same month, which overstates skill.
Leave-one-month-out trains on two months and predicts the third, which
mimics the real situation on DART: no local truth exists there at all, so
the model must generalise to conditions it has not been fitted on.

**The monthly constraint.** After the leave-one-out fit, the reconstructed
3-hourly field is aggregated to a monthly mean and *rescaled, per grid
cell*, so that mean exactly equals the independently known monthly `ssrd`:

$$ssrd_{final}(t) = ssrd_{pred}(t) \times \frac{monthly_{actual}}{monthly_{pred}}$$

Multiplicative, not additive — night is exactly zero and must stay zero; an
additive correction would inject false radiation into the night hours to
hit the target mean.

**Result, pooled across all three held-out months, after the monthly
constraint:**

| metric | value |
|---|---|
| R² | 0.986 |
| RMSE | 32.7 W m$^{-2}$ |
| bias | +0.19 W m$^{-2}$ (essentially zero, by construction) |

This is the number that answers "how good is the method": the reconstructed
3-hourly field explains 98.6% of the variance in ERA5's real `ssrd`, on
months the model never trained on.

## 2.4 The production model

The leave-one-out folds exist to test the method honestly, holding out each
month in turn. A production model should use all the training signal
available, not sacrifice a third of it. `train_final_model.py` fits one
`HistGradientBoostingRegressor` on all three ERA5 months combined and saves
it (`model_kt.joblib`) for reuse — no refitting per DART month.

# 3. Making it work on DART: two things ERA5 never needed

## 3.1 `tisr` has to be a lookup, not a calculation

DART will never have a `tisr` field to read — this is genuinely unavoidable,
`tisr` depends only on where a point is and what time it is, and DART's
saved output does not include per-timestep astronomical calculations for
every cell. Two options existed: compute `tisr` analytically every time it
is needed, or precompute it once and look it up.

**Why lookup, not live computation**: `tisr` depends only on latitude,
longitude, time-of-day and day-of-year — not on which specific calendar
year it is (orbital drift operates on 10,000+ year timescales). DART's own
calendar was confirmed to be genuine Gregorian (Feb 1953 = 28 days; Feb
1956 and 1964 = 29 days, read directly from real DART output), so **two**
reusable templates — one 365-day year, one 366-day leap year — cover every
real year in both DART periods exactly, matched by day-of-year and
time-of-day. No approximation: the astronomy really is that periodic,
modulo the leap-day calendar shift.

Building these two templates at DART's full 6.6-million-cell native
resolution took substantial engineering to get right — a detailed,
independently-verified account exists in the companion **TISR Template
Audit**. Both files passed every check run against them: correct shape,
exact grid alignment with DART's own cells, zero missing values across a
full scan of both 24 GB files, physically correct polar night and midnight
sun behaviour, and an end-to-end trace confirming no cell ever got
reordered during the build.

## 3.2 The three-hour window, and why it matters more than it looks

ERA5's downloaded fields accumulate over the **1 hour** ending at each
stamp, even though sampled every 3 hours. DART's `tsr` and `ssrd` genuinely
accumulate over the full **3 hours** — confirmed empirically three
independent ways, not assumed from the file's metadata label (which
misleadingly reads `online_operation="instant"`):

1. DART's raw metadata states `cell_methods = "time: mean (interval: 3 h)"`
   for its monthly fields.
2. Cross-checking DART's own reduced-grid `tsr` against its own trusted
   monthly `tsr` product: the /10800 divisor reproduces the monthly value;
   /3600 or /86400 do not.
3. The identical check repeated for `ssrd` on both DART runs, against known
   climatology (~190 W m$^{-2}$ global mean in January): only the
   3-hour-basis divisor lands in a physically plausible range —

| assumed basis | 1950C | 2080C |
|---|---|---|
| 1 hour | 574 W m$^{-2}$ | 570 W m$^{-2}$ |
| **3 hours** | **191 W m$^{-2}$** | **190 W m$^{-2}$** |
| 24 hours | 24 W m$^{-2}$ | 24 W m$^{-2}$ |

Because the predictors are *ratios* of two quantities accumulated over the
same window ($T = tsr/tisr$, $\mu = tisr/(\Delta t \cdot S_0)$), a model
fitted on ERA5's 1-hour basis transfers to DART's 3-hour basis without
retraining — **provided** the analytic `tisr` used for DART is generated
with the matching 3-hour window, which is exactly how the two lookup
templates were built (`dt_seconds=10800`). Getting this wrong would not
crash anything; it would silently scale the transmission predictor by 3×,
which looks like unusually high transmission, not an error.

# 4. First application to DART: one month, end to end

**Test**: January 1950, TCo1279-DART-1950C, the full production pipeline —
real DART files in, real reconstructed file out, no shortcuts.

| step | result |
|---|---|
| coarse check, before any correction | reconstructed 203.8 W m$^{-2}$ vs. DART's real 191.2 W m$^{-2}$ (+6.6%) |
| exact per-cell monthly rescale | matches DART's real value to 0.000177 W m$^{-2}$ |
| output | `ssrd_reduced_3h_195001-195001.nc`, full 3-hourly field on DART's native grid |

The +6.6% pre-correction gap is comparable to what the same method showed
during ERA5 validation, so nothing broke in the transfer to a different
grid and a different model.

**What the rescaling had to correct.** Per-cell scale factors ranged from
0.56 to 5.29 — wider than on ERA5. Mapped globally, the *median* factor is
exactly 1.00 and 90% of cells fall between 0.94 and 1.36; the wide extremes
come from a small number of outlier locations, not a systematic global
problem (see the delivered scale-factor map,
`scale_factor_map_195001.png`).

# 5. Problems, and how to check them

## 5.1 The real open question: does ERA5's shape transfer to DART everywhere?

Every number reported for ERA5 in §2.3 is *ERA5 predicting ERA5*: it proves
the functional form captures how ERA5's own radiation scheme behaves. It
does **not** prove DART's radiation scheme behaves identically for the same
reported cloud fraction — DART runs a different model cycle at a different
resolution, and a coarser or finer cloud parameterisation can report the
same fraction for a physically different atmosphere. The wide scale-factor
range on the one tested month is the first concrete evidence this gap is
real, not just theoretical.

**This cannot be closed directly.** DART has no 3-hourly `ssrd` anywhere to
check the shape against — that absence is the entire reason this method
exists. Two things partially cover for the gap without eliminating it:

- The exact monthly rescale absorbs any *systematic* bias between the two
  models' radiation schemes, so the corrected magnitude is trustworthy even
  where the pre-correction shape is off.
- It does **not** fix a *shape* problem: if DART's day-to-day cloud
  variability behaves differently from ERA5's, the within-month pattern
  can still be wrong even after the level is corrected.

## 5.2 How to actually check whether the model is good enough

In order of how much they tell you, cheapest first:

1. **Map the scale factors, every month, not just one.** Already built for
   January 1950 (§4). If large corrections cluster over specific
   geography — mountains, coastlines, high-latitude winter — that mirrors
   the *exact* pattern already found on the wind side of this project (onshore
   errors roughly double offshore, concentrated over terrain), and would
   mean the cloud-radiation relationship itself needs attention there, not
   just a level correction.
2. **Run several more months, across seasons.** One month cannot separate "this
   particular month was unusual" from "this is a persistent regional bias."
   A cheap next step: repeat the coarse-gate + rescale test on a summer
   month and one from the 2080 scenario run, and compare scale-factor maps
   across all of them for consistency.
3. **Check day/night and diurnal shape directly** in the output file — night
   must be exactly zero everywhere, and the shape within a day should vary
   smoothly, no artefacts at the model's dawn/dusk boundary.
4. **Consider adding forecast albedo (`fal`)** as a feature. DART carries it
   monthly, and it was flagged early as the natural fix for a known failure
   mode of `tsr`-based predictors: a clear sky over bright snow and an
   overcast sky over dark ground can look radiatively similar from space.
   If the large scale-factor cells concentrate at high latitude in winter,
   this is the first thing to try.

## 5.3 Does the model need to change?

**Not yet, on the evidence so far.** One month passing the coarse gate at
+6.6% and landing on an exact match after correction is not a failure
signal — it is within the same range the method showed on ERA5 itself. The
wide scale-factor spread is a flag to investigate (§5.2), not proof the
model is wrong. The right test before deciding anything: run the checks in
§5.2 across a handful of contrasting months first. If the scale-factor
pattern is stable and geographically sensible (e.g. concentrated exactly
where the wind analysis already found terrain-driven error), the model is
doing its job and the correction step is doing its. If it is essentially
random or implausibly large everywhere, that would be the signal to revisit
the feature set or retrain including a DART-appropriate predictor.

# 6. What is still needed before the full 1950C / 2080C runs

1. **Efficiency.** The one-month test took 22 minutes, 10 of them just
   reading the `tisr` template. That cost repeats identically for every
   month sharing a calendar year, since a full year uses the entire
   template exactly once. Run month-by-month as built now, 396 months
   would mean re-scanning the template 396 times — roughly 65 hours on
   lookups alone. Restructuring to process a full calendar year per job,
   reading the template once and reusing it for all twelve months, cuts
   that to 33 scans total (one per year across both periods).
2. **Broader validation before committing.** At least a small spread of
   additional months (§5.2, item 2) run and checked before launching the
   full 240 + 156 month set, so a systematic problem is caught on a handful
   of months rather than discovered after processing decades.
3. **Parallelisation strategy.** Each month is fully independent of every
   other month once the per-year template read is amortised — the natural
   unit of parallelism is one SLURM job per year (33 total), not
   multiprocessing within a single job. This deliberately avoids the class
   of rare, hard-to-diagnose forking/memory issues that the `tisr` template
   build ran into and eventually resolved by removing concurrency
   entirely — a lesson worth carrying forward rather than relearning.
4. **A decision on the scale-factor finding.** Once §5.2's checks are in,
   decide whether the current feature set is sufficient or whether adding
   `fal` (or another DART-native predictor) is worth a retrain before the
   two decades-long runs, rather than after.

---

*Prepared for internal project tracking. Companion document: TISR Template
Audit (verification of the two lookup templates referenced in §3.1).*
