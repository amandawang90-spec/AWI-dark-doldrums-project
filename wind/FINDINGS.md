# u100/v100 reconstruction from u10/v10 — findings so far

Goal: reconstruct 100 m winds from 10 m winds for TCo1279-DART, for dark-doldrum
(low-wind) research. ERA5 (which has real u100/v100) is used as the testbed to
decide which method to trust.

## 1. Methods compared

Validated on 3 ERA5 timestamps: 20250830_0000, 20260301_0000, 20260830_0000.
All scores area-weighted by cos(lat). Regression fit on 80% of 10°×10° spatial
blocks, both methods scored on the held-out 20%.

**Log-law alone vs. real ERA5 u100/v100** (`loglaw_reconstruct.py`):

| region   | u100 RMSE | v100 RMSE | r     |
|----------|-----------|-----------|-------|
| onshore  | ~1.19 m/s | ~1.10 m/s | 0.97  |
| offshore | ~0.60 m/s | ~0.54 m/s | 0.997 |

Stable across all three months — no seasonal drift. Offshore ~2× better than
onshore; onshore error concentrates over mountains and forest.

**Log-law vs. regression** (`compare_loglaw_vs_regression.py`):

- Overall RMSE: **regression wins onshore** (~15–25% lower); offshore ~tie.
- Low wind (<3 m/s), the dark-doldrum regime — the two metrics disagree:
  - **RMSE**: regression wins onshore (0.86–0.91 vs 0.94–0.95);
    log-law wins offshore (0.37–0.39 vs 0.42–0.48).
  - **MAPE**: log-law wins everywhere (28–31% vs 35–39% onshore,
    15% vs 18–24% offshore).
- Why they disagree: regression's fitted intercept adds a near-constant offset.
  Negligible at high wind, but a large *fraction* of a 1–2 m/s true value.

Neither method is precise in the low-wind regime (MAPE 15–39%). State as a limitation.

## 2. Roughness (z0) — the blocker

- **TCo1279 has no usable z0.** `sr` (param 173) is a flat placeholder,
  0.0001 m at all 6,599,680 points, in every init file checked
  (oifs-43r3 DART + non-DART, oifs-48r1). Verified by map, with `sdor` as a
  control that renders correctly on the same grid → the flatness is the data.
- **Orography ≠ roughness.** `sdor`/`sdfor` (params 160/74) are populated, but are
  terrain relief (land median 8.7 m) not aerodynamic roughness (ERA5 land median
  0.30 m) — ~29× apart, and `sdor` is *exactly zero over ocean*, so it cannot serve
  as z0 where the log-law works best. Only `sr`(173)/`fsr`(244) carry the CF name
  `surface_roughness_length`.
- **What IS populated** in `ICMGGhf05INIT`: `cvl`, `cvh`, `tvl`, `tvh` (vegetation
  cover/type), `lai_lv`/`lai_hv` (in `ICMCLhf05INIT`), `lsm`, `chnk` (Charnock),
  orography terms. These are the raw material IFS uses to *compute* z0.
- **Land/sea mask for the DART grid**: `L128.msk`/`A128.msk` in
  `/albedo/work/projects/p_pool_oasis/cy43r3/TCO1279-DART/masks.nc`, native grid,
  28.2% land. Also `lsm` in the init file. No regridding needed.

## 3. Can ERA5's z0 be applied to TCo1279?

**Yes, defensibly** (`z0_scale_sensitivity.py`):

- Log-law ratio R = ln(100/z0)/ln(10/z0) is only *logarithmically* sensitive:
  a 10× z0 error costs ~19% in wind at the smooth end, but ~76% at the rough end.
- Coarsening ERA5 z0 by 4× (25→111 km) costs only ~0.29–0.32 m/s RMSE at 100 m.
  Our actual mismatch is 2.8× (25→9 km) → expect ~0.2–0.25 m/s, i.e. ~3%
  degradation on top of the existing 1.19 m/s onshore error. Negligible offshore.

**Two bigger caveats than the z0 transfer itself:**

1. **Fetch.** An internal boundary layer grows ~1:100, so a 100 m wind needs ~10 km
   of upwind fetch to equilibrate. ERA5 at 25 km > fetch (local log law defensible);
   TCo1279 at 9 km is *at or below* it — neighbouring boxes aren't independent. The
   local log law is on weaker footing at 9 km regardless of which z0 is supplied.
2. **Displacement height is being ignored.** z0 ≈ 0.1h, d ≈ 0.7h. For a 20 m forest
   d ≈ 14 m, so the 10 m level sits *below* d and `ln((10−d)/z0)` is undefined. We
   currently ignore d entirely — likely a first-order cause of the onshore error,
   concentrated in the same regions where performance is worst.

## 4. Open / next steps

- Locate the `atm_reduced_3h_{10u,10v}_3h_{YYYYMM}-{YYYYMM}.nc` files (path still
  unknown; filesystem too large to blind-search).
- Decide z0 source: borrow ERA5 `fsr` (cheap, ~3% penalty) **or** derive from
  `cvl/cvh/tvl/tvh` + IFS lookup table + Charnock over ocean (DART-native).
- Regression transfer: ERA5-fitted coefficients applied to DART u10/v10, stratified
  onshore/offshore by `lsm`. Note this assumes ERA5's fitted relationship carries to
  a 9 km model — untested.
- Consider adding displacement height over tall canopy.
- `sdor` would likely be a good *predictor of log-law error* (errors track terrain),
  useful if a residual-correction layer is added.

## 5. Environment notes

- `python` is not on PATH. Every script must `module load analysis-toolbox/python-04.2026`
  (python 3.12, xarray 2026.4.0, cfgrib/ecCodes). No `ncdump`.
- `/albedo/pool/oifs-43r3/...` and `/albedo/work/projects/p_pool_openifs/oifs-43r3/...`
  are the same files (same inode), just two mount paths.
- ERA5 test data is 0.25° regular lat/lon (721×1440). Beware: raw percentiles over
  this grid over-count polar points (48% of land points are poleward of 60°, but only
  21% of land area) — always area-weight.
