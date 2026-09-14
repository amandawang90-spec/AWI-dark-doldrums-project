# TSIR (TOA incident solar radiation) in OpenIFS 48r1

Copied from `oifs48r1/ifs-source` (branch `cray_fiat_wall_fix`).

## Call chain

| Step | File (original path) | What it does |
|------|----------------------|--------------|
| 1 | `updtim.F90` (`arpifs/utility/`) | Date/time -> orbital phase `RTETA`, declination `RDECLI = RDS(ZTETA)`, Sun–Earth distance `RDEASO = RRS(ZTETA)`, hour angle `RWSOVR = (REQTIM+RHGMT)*2π/RDAY`. Scaled solar constant `RII0 = RI0 * (REA/RDEASO)**2` (~l. 320–355). Radiation-step values `RCODECM/RSIDECM/RWSOVRM` at ~l. 703–718. Optional `SHR_ORB_UPD` if `LCORBMD`. |
| 1b | `updtier.F90` (`arpifs/phys_ec/`) | Updates the radiation-time-step declination/hour-angle terms (`RCODECM`, `RSIDECM`, `RCOVSRM`, `RSIVSRM`). |
| 1c | `fctast.func.h` (`arpifs/function/`) | Statement functions `RTETA`, `RDS`, `RRS`, `RET` (orbit, declination, distance, equation of time). |
| 2 | `cos_sza.F90` (`arpifs/phys_ec/`) | Cosine of solar zenith angle: `mu0 = sinδ sinφ − cosδ cosφ cos(ω+λ)`, clipped at 0 (l. 151–153); optional time-step-averaged mu0 with sunrise/sunset handling (l. 168–226). |
| 3 | `radheatn.F90` (`arpifs/phys_radi/`) | `ZI0 = RII0 * PMU0` (l. 288–298), then `PTINCF = ZI0 * PTINCFI` (l. 577) = TSIR. |
| 4 | `postphy_layer.F90` (`arpifs/phys_ec/`) | Passes `PRAD%PTINCF` on to the diagnostics/output. |

## In short

TSIR = S0 · (a/r)² · max(0, cos θz), where

- S0 and (a/r)² come from `updtim.F90`
- cos θz comes from `cos_sza.F90` (latitude, longitude, declination, hour angle)
- the product is formed in `radheatn.F90`
