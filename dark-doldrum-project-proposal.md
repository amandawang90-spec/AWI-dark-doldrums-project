# Research Proposal

# Changes in Dark Doldrum (Dunkelflaute) Events Under High-Emission Climate Change: A 9 km Resolution Analysis for Europe and South Korea

**An extension study of the AWI research project: How warming could change wind and solar resources**

Author: Jing Wang
Affiliation: Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven
Date: September 2026

---

## 1. Background 

### 1.1 The Dunkelflaute challenge

The global energy transition relies increasingly on wind and solar power. Both resources are inherently variable, and grid stability depends on the assumption that when one source is weak, the other often compensates. Dark doldrum events — known in German as Dunkelflaute — violate that assumption: they are sustained periods when both wind and solar resources fall below usable levels simultaneously. During such events, electricity systems must draw on storage, backup generation or imports, making Dunkelflaute events critical for infrastructure planning and energy security.

Northern and Central Europe are particularly vulnerable. Winter days are short and solar irradiance is low, so wind carries a disproportionate share of renewable supply. When a persistent high-pressure system settles over the region — bringing calm winds and often overcast skies — both resources fail together. Germany's Bundesnetzagentur and ENTSO-E have identified Dunkelflaute as a key challenge for system adequacy. Between December 2022 and March 2023, two extended dark periods lasting 30 and 54 days were observed in Germany.

South Korea faces an analogous but climatologically distinct challenge. The country's energy transition targets substantial increases in wind and solar capacity under the 10th Basic Plan for Electricity Supply and Demand. South Korea's climate is shaped by the East Asian monsoon, producing seasonal patterns fundamentally different from those in Europe: hot, humid summers with monsoon cloud cover that reduces solar output, and cold winters when high-pressure systems from Siberia can suppress both wind and solar across the Korean peninsula simultaneously. The complex coastline and mountainous interior make South Korea an ideal test case for high-resolution climate modelling of renewable resource deficits.

### 1.2 Dunkelflaute definitions in the literature

There is no single standard definition. The two most widely adopted approaches are:

**Combined CF threshold (Mockert et al., 2023):** A Dunkelflaute event occurs when the 48-hour running mean of the combined (capacity-weighted) wind and solar capacity factor falls below 0.06. This threshold was calibrated to reproduce the event frequency reported by Kaspar et al. (2019), and the 48-hour window focuses on multi-day events that pose the greatest challenge for energy storage (Schmidt et al., 2019). This definition identifies approximately 4 severe events per year in Germany. The methodology has been widely adopted in subsequent studies (e.g. Strnad et al., 2025).

**Individual technology CF threshold (Li et al., 2021):** A Dunkelflaute occurs when both wind CF and solar PV CF individually fall below 20% during a given period. This threshold identifies approximately 5–10 events per year lasting longer than one day in Germany, and is the most cited definition for climatological studies. Events under this definition are less severe individually but more frequent, providing better statistical sampling.

**Evaluation against grid adequacy (Lohmann et al., 2025):** A systematic evaluation of detection methods against actual grid stress data (expected energy not served) found that no simple meteorological threshold perfectly predicts grid stress — the F-score peaks at only 0.14 — because grid adequacy also depends on demand, interconnection and plant availability. Detection performs best near a 23% individual-technology CF threshold with a 48-hour duration criterion.

### 1.3 Gap in current research

Most climate projections of renewable resources examine wind and solar changes separately. The AWI research project on wind and solar resources uses AWI-CM3 to assess future wind and solar resources, but its wind–solar complementarity assessment is limited to the monthly scale. It explicitly notes that its available radiation data do not support weather-scale analysis of joint low-wind-low-solar events.

Furthermore, existing Dunkelflaute climatologies focus almost exclusively on Europe, particularly the North Sea and Baltic regions. No study has applied the same event-based methodology to compare Dunkelflaute risk across fundamentally different climate regimes using high-resolution climate model output.

This extension study fills both gaps: it analyses Dunkelflaute events at sub-daily to daily timescales using 9 km resolution AWI-CM3 output, and applies the same methodology to both Europe and South Korea.

### 1.4 Relationship to existing work

This study builds on:

**The AWI project on wind and solar resource**, which provides the scientific framework and methodological conventions.

**Moon et al. (2025)**, "Earth's future climate and its variability simulated at 9 km global resolution" (Earth Syst. Dynam., 16, 1103–1134), which describes the AWI-CM3 high-resolution simulation protocol. The HR data originate from the AWI–IBS (Institute for Basic Science, Busan, South Korea) collaboration. Moon et al. demonstrate that 9 km resolution substantially increases regional climate information over topographically complex terrain and coastal zones — directly relevant to both the European and Korean study domains.

**The existing AWI Dunkelflaute detection codebase**, which implements the Mockert et al. (2023) methodology — a 48-hour moving average combined CF threshold of 0.06, with Germany's 2024 installed capacity weights and pre-built configuration for TCo1279 model data.

---

## 2. Research questions

### 2.1 Full research framework

The complete set of questions this programme could address:

1. Baseline Dunkelflaute climatology under early-industrial (1950) forcing
2. Climate change signal under SSP5-8.5
3. Temporal evolution through the 21st century
4. Natural variability context from the long MR control
5. Resolution sensitivity (9 km vs 31 km)
6. Scenario dependence (SSP1-2.6, SSP3-7.0, SSP5-8.5)
7. Onshore vs offshore wind behaviour during events
8. Cross-regional comparison: Europe vs East Asia

### 2.2 Scope of this 4-week study

This study focuses on questions 1, 2, 7 and 8 using high-resolution (TCo1279-DART, ~9 km) data only:

- Compare Dunkelflaute events between the 1950 control and 2080s SSP5-8.5 at 9 km across Europe and South Korea
- Characterise differences in event frequency, duration and seasonality between the two regions
- Assess onshore vs offshore wind behaviour during events

Medium-resolution analysis, additional scenarios, temporal evolution and the full natural variability assessment from the 184-year MR control are documented as direct future extensions.

---

## 3. Study domains

### 3.1 Europe

The European domain covers approximately 35°N to 72°N, 15°W to 35°E, capturing the full gradient from Mediterranean to Arctic conditions, including all major European renewable energy markets. Country-level statistics are extracted by aggregating gridded results within national boundaries.

Europe provides the highest installed wind + solar capacity density globally, strong latitude-driven contrasts in solar availability, Atlantic-influenced vs continental climate regimes, and complex coastal features (North Sea, Norwegian fjords, Mediterranean islands) that test the value of 9 km resolution.

### 3.2 South Korea

The Korean domain covers approximately 33°N to 39°N, 124°E to 132°E, encompassing the Korean peninsula and surrounding seas (Yellow Sea, East Sea / Sea of Japan, Korea Strait). Country-level statistics are extracted for South Korea specifically.

South Korea provides a fundamentally different climate regime: East Asian monsoon with distinct wet/dry seasons, Siberian high-pressure influence in winter and typhoon season in late summer. The peninsula has a complex coastline and mountainous interior (~70% of land area) — an ideal test case for 9 km resolution benefits. Growing offshore wind development in the Yellow Sea and southern coast provides policy relevance.

### 3.3 Why this cross-regional comparison matters

Applying the same methodology, model, thresholds and pipeline to two climatologically distinct regions provides a generalisability test. If the Dunkelflaute response to warming is similar in both regions, that suggests a robust signal. If it differs, the contrasting climate drivers (NAO / Atlantic blocking vs monsoon / Siberian High) help explain why. Both regions are actively planning large-scale renewable deployment and need Dunkelflaute risk information under climate change.

---

## 4. Data and simulations

### 4.1 What "1950 control" means

The IPCC defines "pre-industrial" as the 1850–1900 global average. Pre-industrial CO₂ was approximately 280 ppm. By 1950, atmospheric CO₂ had risen to approximately 310 ppm — about 11% above pre-industrial — and global temperature was roughly 0.2–0.3°C above the 1850–1900 baseline.

The 1950 control therefore represents **early industrial conditions**, not a pristine pre-industrial state. However, the vast majority of anthropogenic warming occurs after 1950: CO₂ has since risen from 310 to over 420 ppm, with an additional ~0.9°C of warming by the 2020s, and approximately 4–5°C more by the 2080s under SSP5-8.5 (Moon et al., 2025).

The 1950-to-2080s comparison captures the bulk of the anthropogenic climate change signal. Throughout this study, "control" refers to the 1950-forcing simulation and "future" to the 2080s SSP5-8.5 simulation.

### 4.2 Available data: TCo1279-DART (~9 km)

The data originate from the AWI–IBS collaboration described in Moon et al. (2025). Both simulations use the TCo1279 atmospheric resolution (~9 km) coupled with the DART variable-resolution ocean mesh (4–25 km). Data are global; European and Korean domains are extracted during processing.

**TCo1279-DART-1950C (control)**

| Grid | Frequency | Variables | Years |
|---|---|---|---|
| Native (6,599,680 cells) | 3-hourly | 10u, 10v, 2t, cp, lsp, tcc, hcc, mcc, lcc, tsr, w850 | 1950–1969 |
| 5136 × 2560 (remapped) | 3-hourly | msl | 1950–1969 |
| 5136 × 2560 (remapped) | Monthly | fal, sst, tcwv, ssrd, ssrc, strc, strd, tsrc, ttr, ttrc | 1950–1969 |
| 5136 × 2560 (remapped) | Monthly | tsr | 1966–1969 only |

**TCo1279-DART-2080C (SSP5-8.5 future)**

| Grid | Frequency | Variables | Years |
|---|---|---|---|
| Native (6,599,680 cells) | 3-hourly | 10u, 10v, 2t, cp, lsp, tcc, hcc, mcc, lcc, tsr, w850 | 2080–2092 |
| 5136 × 2560 (remapped) | 3-hourly | msl | 2080–2092 |
| 5136 × 2560 (remapped) | Monthly | fal, sst, tcwv, ssrd, ssrc, str, strc, strd, tsr, tsrc, ttr, ttrc | 2080–2092 |
| 5136 × 2560 (remapped) | Monthly | ssr | 2082–2092 only |

### 4.3 Usable analysis periods and spin-up exclusion

The HR simulations are time-slice experiments branched from the MR transient. When the model resolution jumps from 31 km to 9 km, the atmosphere and land surface need time to adjust: small-scale features such as coastal wind patterns, orographic effects and soil moisture distributions that the 9 km grid can resolve were not present in the 31 km initial conditions. During this "spin-up" period, the model develops physically consistent fine-scale patterns, so the output is not yet representative of the model's equilibrium climate at 9 km.

Following Moon et al. (2025), the first 2 years of each simulation are excluded from analysis.

| Simulation | Total period | Spin-up exclusion | Usable period | Usable years |
|---|---|---|---|---|
| 1950 Control | 1950–1969 | 1950–1951 | 1952–1969 | 18 years |
| 2080s SSP5-8.5 | 2080–2092 | 2080–2081 | 2082–2092 | 11 years |

The AWI research project on wind and solar resources document notes that sensitivity to the length of the exclusion period (1 year vs 2 years vs 3 years) will be tested. For consistency with the published Moon et al. protocol, this study uses the 2-year exclusion as the default.

### 4.4 Key variables for Dunkelflaute analysis

| Variable | Code | Available at | Used for |
|---|---|---|---|
| 10 m eastward wind | 10u | 3h native | Wind speed → wind CF |
| 10 m northward wind | 10v | 3h native | Wind speed → wind CF |
| 2 m temperature | 2t | 3h native | Solar PV temperature correction |
| Total cloud cover | tcc | 3h native | Surface solar estimation |
| High cloud cover | hcc | 3h native | Surface solar estimation (low attenuation) |
| Medium cloud cover | mcc | 3h native | Surface solar estimation (moderate attenuation) |
| Low cloud cover | lcc | 3h native | Surface solar estimation (strong attenuation) |
| TOA net solar radiation | tsr | 3h native | Surface solar estimation |
| Surface solar rad. downward | ssrd | Monthly remapped | Validation of 3h solar estimate |
| Mean sea-level pressure | msl | 3h remapped | Synoptic context (if time permits) |

### 4.5 Why HR only in this study

The short internship requires sharp focus. The HR data provide the highest available resolution (9 km) and the highest temporal frequency (3-hourly) — the best data for Dunkelflaute detection. The MR data at 31 km with 6-hourly wind are valuable for natural variability assessment and resolution comparison, but these analyses are deferred. The pipeline developed here will make that extension straightforward.

### 4.6 Why SSP5-8.5 only

HR simulations exist exclusively under SSP5-8.5. Running additional scenarios requires approximately 676,000 core hours per 10-year chunk (Moon et al., 2025), far beyond the scope of this short-term internship. SSP5-8.5 provides the strongest climate change signal, maximising detectability.

---

## 5. Methods

### 5.1 Wind capacity factor

Wind capacity factors are computed separately for **onshore** and **offshore** grid cells, reflecting different turbine technologies and wind environments.

**Step 1 — Wind speed:**

v₁₀ = √(u₁₀² + v₁₀²)

**Step 2 — Hub-height extrapolation (power law):**

v_hub = v₁₀ × (H_hub / 10)^α

| Parameter | Onshore | Offshore |
|---|---|---|
| Hub height H_hub | 100 m | 150 m |
| Wind shear exponent α | 1/7 ≈ 0.143 | 0.11 |

**Step 3 — Simplified turbine power curve:**

| Wind speed range | CF_wind(v) |
|---|---|
| v < v_in (cut-in) | 0 |
| v_in ≤ v < v_rated | (v³ − v_in³) / (v_rated³ − v_in³) |
| v_rated ≤ v ≤ v_out | 1.0 |
| v > v_out (cut-out) | 0 |

| Parameter | Onshore | Offshore |
|---|---|---|
| v_in | 3 m/s | 3 m/s |
| v_rated | 12 m/s | 13 m/s |
| v_out | 25 m/s | 25 m/s |

**Step 4 — Apply per sample, then average:**
The power curve is applied to each 3-hourly sample before temporal averaging, preserving the nonlinear relationship. CF(v̄) ≠ mean(CF(v)).

The model's land-sea mask classifies grid cells. For South Korea, offshore areas include the Yellow Sea, southern coast and East Sea development zones.

### 5.2 Solar PV capacity factor

**Primary (simplified) formula:**

CF_PV(t) ≈ G_est(t) / 1000

where G_est(t) is the estimated surface solar radiation and 1000 W/m² is the standard test condition irradiance.

**With temperature correction (sensitivity test):**

CF_PV(t) = [G_est(t) / 1000] × [1 + γ × (T_2m(t) + c_T × G_est(t) − 25°C)]

where γ ≈ −0.005 per °C and c_T ≈ 0.035 °C·m²/W. Temperature correction is relevant for South Korea's hot summers and for assessing whether warming reduces PV efficiency during Dunkelflaute shoulder seasons.

### 5.3 Estimating sub-daily surface solar radiation

Surface solar radiation (ssrd) is available only monthly. Sub-daily estimates use the 3-hourly cloud cover and TOA radiation fields.

**Cloud-attenuation model:**

G_est(t) = S_clear(t) × K_clear × (1 − a_low × lcc − a_mid × mcc − a_high × hcc)

where S_clear(t) is the theoretical clear-sky TOA incoming solar (from solar geometry: latitude, day of year, hour of day), K_clear ≈ 0.75 is clear-sky atmospheric transmittance, and a_low ≈ 0.7, a_mid ≈ 0.5, a_high ≈ 0.3 are cloud-type attenuation coefficients.

**Validation:** 3-hourly estimates are aggregated to monthly means and compared with the actual monthly ssrd for both simulations and both study domains. Systematic biases are corrected with monthly scaling factors. Dual-domain validation is important because the cloud-attenuation relationship may behave differently under European (Atlantic-dominated) and Korean (monsoon-dominated) cloud regimes.

### 5.4 Dunkelflaute event definition

This study uses two complementary threshold approaches:

#### Primary threshold: Mockert et al. (2023)

**Combined capacity factor:**

CF_combined(t) = w_solar × CF_PV(t) + w_onshore × CF_wind,onshore(t) + w_offshore × CF_wind,offshore(t)

A Dunkelflaute event occurs when the **48-hour moving average** of CF_combined falls below **0.06**. This identifies periods when the combined renewable fleet operates at less than 6% of rated capacity for at least two consecutive days — the severe events most relevant for storage sizing and backup planning.

The 48-hour window was chosen by Mockert et al. because energy storage becomes most problematic for renewable shortfalls lasting longer than 2 days (Schmidt et al., 2019). The threshold of 0.06 was calibrated to reproduce the Dunkelflaute frequency observed by Kaspar et al. (2019).

**Capacity weights (Germany 2024 reference):** w_solar = 0.577, w_onshore = 0.369, w_offshore = 0.054 (source: Bundesnetzagentur). South Korea-specific weights are applied for the Korean domain where available; otherwise equal weighting (1/3 each) serves as a neutral alternative.

#### Secondary threshold: Li et al. (2021)

A Dunkelflaute time step occurs when **both** wind CF < 20% **and** solar PV CF < 20% simultaneously. Events are consecutive time steps meeting both conditions.

This threshold identifies more frequent, less severe events (~5–10 per year in Germany lasting >1 day vs ~4 for Mockert). The larger event count provides better statistical sampling — critical given the short analysis windows (18 and 11 usable years).

#### Rationale for using both

Both thresholds are applied to all analyses. If spatial patterns and climate change signals agree across thresholds, the findings are robust. If they diverge, the divergence itself is a finding: it reveals where results depend on whether one is measuring severe events (Mockert) vs moderate-but-more-frequent events (Li).

#### Additional sensitivity tests

- Combined CF thresholds: 0.04, 0.08, 0.10
- Moving-average windows: 24 h, 72 h
- Country-specific capacity weights

#### Polar night handling

During polar night (solar elevation ≤ 0° for the entire day), solar CF is zero by definition. Such periods are classified separately as extended-darkness low-wind events. This affects the northernmost European domain (above ~67°N) in mid-winter. South Korea (33–39°N) is unaffected.

### 5.5 Event metrics

For each grid cell, season (DJF, MAM, JJA, SON), country and simulation:

- **Event frequency:** Distinct events per year or season
- **Mean and median duration:** Days
- **Maximum duration:** Longest event in the analysis period
- **90th-percentile duration:** Duration exceeded by 10% of events
- **Dunkelflaute fraction:** Share of time classified as Dunkelflaute (%)
- **Spatial coherence:** Fraction of country area simultaneously below CF threshold (%)

### 5.6 Quantifying the climate change signal

Δ_metric = metric_2080s − metric_1950control

Reported as absolute differences, percentage changes, and per degree of global warming (dividing by the global mean temperature difference between simulations).

### 5.7 Uncertainty estimation

**Year-block bootstrap:** Whole years are resampled with replacement from each analysis period. The seasonal cycle and within-year weather sequences are preserved; events are not joined across resampled year boundaries. The median response and 5th–95th percentile range are reported.

**Internal consistency check:** Without the long MR control, a full natural variability envelope cannot be computed in this phase. Instead, the 18-year control is split into two non-overlapping segments and event metrics are compared. If the difference between control sub-periods is comparable to the control-to-future difference, the climate signal cannot be confidently distinguished from internal variability.

**Minimum sample size:** Grid cells or seasons with fewer than ~20 events are flagged as having insufficient data for robust statistics. This is particularly relevant for the Mockert threshold (~4 events/year × 11 future years ≈ 44 total events before splitting by season).

---

## 6. Limitations

**Only two time periods with usable HR data.** Ideally, this study would analyse multiple time slices across the full 21st century — including the HR 2000s historical slice and intermediate periods (2030s, 2060s) — to track how Dunkelflaute risk evolves with progressive warming. However, the available data prevent this. The HR 2000s slice (TCo1279-DART-2000) contains only daily-mean 2 m temperature — no wind, no radiation, no cloud cover at sub-daily resolution — making it unusable for Dunkelflaute detection. The 2030s and 2060s HR slices are not present in the accessible data archive. As a result, the study is limited to comparing two endpoints: the 1950 control and the 2080s SSP5-8.5. This means the analysis captures the full magnitude of the climate change signal but cannot determine how the Dunkelflaute response develops through the century — whether changes emerge gradually, accelerate after a threshold, or appear only under strong warming. Filling this gap would require either locating the missing HR slices or running the same analysis on the MR transient (2015–2099), which has 6-hourly wind data spanning the full century.

**Short analysis windows.** Even the two available time slices are relatively short: 18 usable years for the 1950 control and 11 usable years for the 2080s future (after excluding the 2-year spin-up). Longer windows would improve sampling of rare, long-duration events and provide more robust seasonal statistics. Under the stricter Mockert threshold (~4 events/year), the future period yields only approximately 44 total events before splitting by season or country — marginal for robust frequency estimates. The Li et al. threshold partially mitigates this by capturing more events. Bootstrap uncertainty estimates and minimum sample-size masking are applied throughout, but cannot fully compensate for the short record length.

**Early-industrial baseline.** The 1950 control includes CO₂ at ~310 ppm (~11% above the pre-industrial ~280 ppm). The climate change signal slightly underestimates total change since pre-industrial times.

**Estimated sub-daily solar.** Surface solar is reconstructed from TOA radiation and cloud cover. The cloud-attenuation relationship may behave differently under European and East Asian monsoon cloud regimes; dual-domain validation addresses this partially.

**No full natural variability assessment.** Without the long MR control, only a limited internal consistency check is possible. The full 184-year natural variability envelope is a priority future extension.

**Single emission scenario.** Only SSP5-8.5. Response under weaker pathways is deferred.

**Single model family.** AWI-CM3 only. Multi-model comparison would strengthen confidence.

**Fixed capacity weights.** Germany's 2024 mix is the primary reference. Country-specific weights for all European countries are not applied in this phase.

**Meteorological potential only.** Actual electricity shortfalls depend on demand, storage, interconnection and installed capacity.

---

## 7. Future extensions

These extensions require no new climate simulations:

- **MR analysis and natural variability:** Apply the pipeline to TCo319 (31 km) control and SSP5-8.5; use the 184-year MR control for robust natural variability estimates.
- **Resolution comparison:** Systematically compare 9 km and 31 km Dunkelflaute results.
- **Additional scenarios:** Apply the pipeline to MR SSP1-2.6 and SSP3-7.0.
- **Temporal evolution:** Use MR SSP5-8.5 transient (2015–2099) to track decadal changes.
- **Global extension:** Apply to other regions (broader East Asia, North America, Australia).
- **Energy-system coupling:** Combine with demand profiles and storage models.
- **Country-specific capacity weights:** Apply national weights for all countries.
- **Stability-corrected wind extrapolation:** Replace the power-law assumption with Monin-Obukhov similarity theory.

---

## 8. Expected contribution

This study provides the first comparative analysis of Dunkelflaute events under climate change at 9 km atmospheric resolution across two climatologically distinct regions. It delivers:

- **Baseline Dunkelflaute climatologies** for Europe and South Korea at unprecedented resolution, identifying spatial hotspots, seasonal patterns and event characteristics.
- **Climate change projections** showing how event frequency, duration and seasonality change under approximately 5°C of global warming, with country-level results.
- **Cross-regional comparison** revealing whether the Dunkelflaute response to warming is consistent between Atlantic-influenced European and monsoon-influenced East Asian climates.
- **Dual-threshold robustness:** Results reported under both the Mockert et al. (2023) severe-event threshold and the Li et al. (2021) moderate-event threshold, showing where conclusions are threshold-independent.
- **Onshore vs offshore characterisation** of wind behaviour during events in both regions.

---

## References

Kaspar, F., Borsche, M., Pfeifroth, U., Trentmann, J., Drücke, J., and Becker, P.: A climatological assessment of balancing effects and shortfall risks of photovoltaics and wind energy in Germany and Europe, Adv. Sci. Res., 16, 119–128, https://doi.org/10.5194/asr-16-119-2019, 2019.

Li, B., Basu, S., Watson, S. J., and Russchenberg, H. W. J.: A Brief Climatology of Dunkelflaute Events over and Surrounding the North and Baltic Sea Areas, Energies, 14, 6508, https://doi.org/10.3390/en14206508, 2021.

Lohmann, J., et al.: Evaluation of 'Dunkelflaute' event detection methods considering grid operators' needs, Environ. Res.: Energy, https://doi.org/10.1088/2753-3751/adcf29, 2025.

Mockert, F., Grams, C. M., Brown, T., and Kaspar, F.: Meteorological conditions during periods of low wind speed and insolation in Germany, Meteorol. Z., 32, 181–197, https://doi.org/10.1127/metz/2023/2141, 2023.

Moon, J.-Y., Streffing, J., Lee, S.-S., et al.: Earth's future climate and its variability simulated at 9 km global resolution, Earth Syst. Dynam., 16, 1103–1134, https://doi.org/10.5194/esd-16-1103-2025, 2025.

Schmidt, O., Melchior, S., Hawkes, A., and Staffell, I.: Projecting the future levelized cost of electricity storage technologies, Joule, 3, 81–100, https://doi.org/10.1016/j.joule.2018.12.008, 2019.

---

