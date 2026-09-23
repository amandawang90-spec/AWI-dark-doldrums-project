---
title: "Status: 3-Hourly Solar Reconstruction for TCo1279-DART"
subtitle: "Where it stands, the rescaling design, and whether to improve before scaling up"
author: "AWI Dark-Doldrums Project"
date: "2026-09-15"
geometry: margin=2.5cm
fontsize: 11pt
---

## In one paragraph

DART saves 3-hourly TOA radiation and clouds but only monthly surface
radiation (`ssrd`). We trained a gradient-boosting model on ERA5 (which has
both) to predict `ssrd` from TOA transmission, cloud fractions and solar
geometry, validated it honestly on ERA5 held-out months (98.6% of variance
explained), then ran it once on real DART data (Jan 1950, TCo1279-DART-1950C)
end to end. It ran cleanly and produced a physically sane result. The open
question is not "does it run" — it does — but "does the ERA5-trained
relationship between clouds/transmission and surface radiation actually
match how DART's radiation scheme behaves," and the one test we have so far
gives a mixed but interpretable answer, detailed below.

## How the monthly value is actually used — and why not as a training input

Worth addressing directly: DART's known monthly `ssrd` **is** used, but as
an exact post-hoc correction, not as an input to the ML model. After the
model predicts 3-hourly values, each grid cell's predicted values are
multiplied by a single factor so that their monthly mean exactly equals
DART's real monthly `ssrd` for that cell:

$$ssrd_{final}(t) = ssrd_{pred}(t) \times \frac{ssrd_{monthly,\,real}}{ssrd_{monthly,\,pred}}$$

Feeding the monthly value into the model itself as a feature instead was
considered and is deliberately not what we did, for a concrete reason:
doing so would only make the model's *aggregate* output roughly consistent
with the real monthly mean, not exactly consistent, because a gradient
boosting model has no built-in mechanism to guarantee an aggregation
identity — you'd typically still need a correction step afterward to close
the residual gap. Since we need day-to-day totals to add up to a known,
trusted number (this feeds capacity-factor calculations later), an exact,
provable correction is preferable to an approximate one. Concretely: the
post-hoc rescale currently used closes the gap to 0.0002 W/m$^2$ — far
tighter than a model could be trusted to hit on its own, and unlike a
learned feature, it's independently checkable in one line for any month
without touching the model.

There's a genuine limitation this exposes, though, and it is the crux of
the current open question:

**The rescale is a single number per grid cell per month — it corrects the
overall level, not the shape.** If DART's real day-to-day cloud variability
drives surface radiation differently than ERA5's does (different radiation
scheme, different resolution), a monthly correction cannot fix that; it can
only make the monthly total right while leaving the distribution of that
total across the month's individual 3-hour steps still shaped by whatever
the ERA5-trained model thinks is typical. This is exactly why the
per-cell correction factors are the diagnostic to watch, not just an
implementation detail.

## What the one test month actually showed

Before any correction, the reconstructed January 1950 field averaged 204
W/m$^2$ against DART's real monthly value of 191 W/m$^2$ — about 6.6% high,
which is in line with the residual error already seen when the same model
was tested on ERA5's own held-out months, so nothing about the DART
transfer looks broken on its face.

The per-cell correction factors needed to close that gap ranged from
0.56x to 5.3x — wider than anything seen within ERA5. But the distribution
is reassuring in its middle: the median factor is exactly 1.00, and 90% of
all cells need a correction between 0.94x and 1.36x, comparable to normal
model error. The extreme values are a minority of specific locations, not
a global symptom. That pattern — a well-behaved bulk with a smaller number
of large local outliers — is the same shape of problem already documented
on the wind side of this project, where onshore/terrain cells showed
roughly double the error of open water. It's a reasonable working
hypothesis that the same physical drivers (terrain, coastline, possibly
high-latitude winter snow/ice albedo) are responsible here too, but that
hasn't been confirmed yet for solar — the map exists but hasn't been
cross-checked against geography yet.

## Can the model be improved before running more months?

Yes, and cheaply, before committing to the full 240 + 156 month run:

1. **Check whether the outliers are geographic first**, by overlaying the
   scale-factor map against terrain and coastline. If the pattern matches
   the known wind-side terrain effect, that's a specific, fixable
   hypothesis rather than a diffuse "the model might be wrong" worry.
2. **Add surface albedo (`fal`) as a feature.** DART carries this monthly.
   It directly addresses a known blind spot of the current feature set: a
   clear sky over bright snow/ice and an overcast sky over dark ground can
   produce a similar TOA signature, which transmission and cloud fraction
   alone can't always distinguish. If winter/high-latitude cells are
   over-represented among the outliers, this is the most likely single
   improvement available without retraining architecture.
3. **Run 3-4 more contrasting months** (different season, and one from the
   2080C run) before deciding anything permanent. One month cannot tell us
   whether an outlier pattern is a persistent regional bias or a one-off.
   This is a half-day of compute, not a redesign.
4. **Deliberately not** doing: switching the rescale itself to use a
   different divisor, an additive correction, or folding the monthly value
   into training — none of these address a shape problem, only a level
   problem the current rescale already solves exactly.

## Bottom line

The pipeline works end-to-end and the one real test is within expected
error bounds after correction. The right next step is diagnostic, not a
rebuild: confirm whether the wide correction-factor spread is geographic
and systematic (fixable, likely via adding albedo) or scattered and
unpredictable (would need a harder look at the feature set) — using a
handful of additional test months — before launching the full two-period
run.
