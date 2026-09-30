# Solar model versions — v1 through v5

All versions predict the clearness index `kt = ssrd/tisr` from cloud cover
(`tcc`, `hcc`, `mcc`, `lcc`) plus solar geometry (`mu`) and a cloud-radiative
ratio (`T`), using a `HistGradientBoostingRegressor`. Predicted `ssrd` is then
exactly monthly-rescaled to match the real monthly total — see
`solar/scripts/core/rescale.py`; this is a hard constraint, not a
post-processing nicety. Physics background: `solar/reference/README.md`.

**Use v5** for anything new — it's what the current Germany ERA5 validation
and the TCo1279-DART reconstruction both run on.

| Version | Months trained on | Key change |
|---|---|---|
| v1 | — | Baseline. |
| v2 | 2015–2025, all months | More training data than v1; same method. |
| v3 | Oct–Feb (ONDJF) only | `area` (cos(lat)-weighted) and `plain` variants. |
| v4 | All 12 months, 2015–2026 | **Fixes the accumulation-window bug**: trained on the genuine 3-hour-sum ERA5 ssrd/tsr (`era5_3h_mean`), not the 1-hour-accumulation-sampled-every-3-hours that v1–v3 used without realizing it wasn't the same thing. This is the fix that matters — DART's own 3-hourly fields are genuine 3h accumulations, so v1–v3 were silently training on a mismatched target. |
| **v5** | Sep–Mar (extended winter) only, 2015–2025 | Same corrected target as v4, restricted to the Sep–Mar window (reuses v4's data cache). Current default — training window matches the season actually used in the Dunkelflaute analysis, and both the Germany ERA5 pipeline and the DART reconstruction only need Sep–Mar anyway. |

`v4` is still referenced by a few full-year evaluation scripts; `v1`–`v3` are
kept only so earlier evaluation reports/figures stay reproducible — don't
build new analysis on them.
