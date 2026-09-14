"""Download the ERA5 fields needed for the solar capacity-factor reconstruction.

Replaces download_era5_clouds_3h.py, download_era5_monthly.py and
download_era5_missing.py, which together fetched an overlapping set in three
passes, and an earlier version of this script that also carried 2t and a
monthly clear-sky trio no longer used.

WHAT IS FETCHED, AND WHY EACH FIELD IS HERE
-------------------------------------------
3-hourly, the reconstruction itself:
  tsr   top net solar radiation      predictor -- dominant one; DART has it 3-hourly
  tcc   total cloud cover            predictor
  hcc   high cloud cover             predictor
  mcc   medium cloud cover           predictor
  lcc   low cloud cover              predictor
  ssrd  surface solar rad. downwards TARGET -- what the fit predicts
  tisr  TOA incident solar radiation normaliser for both ratios -- tsr/tisr and
                                     ssrd/tisr are what the fit actually compares,
                                     not the raw values; dropping this breaks the
                                     method. Rides in the same request as ssrd, so
                                     keeping it costs no extra download.

monthly:
  ssrd  surface solar rad. downwards for a direct look at the monthly field, and as
                                     a cheap independent check on any 3-hourly-to-
                                     monthly aggregation code. NOTE: averaging the
                                     3-hourly ssrd above over a month reproduces this
                                     to two decimal places (179.66 W/m2 either way,
                                     Aug 2025) -- so this file is a convenience, not
                                     new information.
  fal   forecast albedo              separates cloud darkening from snow brightening
                                     in tsr; DART carries fal monthly so it may
                                     enter the fit as a slowly varying predictor

Not fetched: 2t (cell temperature -- only needed for the later capacity-factor
step, not for reconstructing ssrd) and the clear-sky diagnostics ssrc/tsrc (never
enter the fit; DART doesn't carry them 3-hourly either). Add back if a specific
next step needs them.

The 3-hourly predictors mirror exactly what TCo1279-DART-1950C/2080C carry on the
native reduced grid, which is the constraint the whole method is built around: a
predictor ERA5 has but DART lacks can never enter the transfer function.

UNITS -- THREE different divisors; do not mix them
-------------------------------------------------------------------------------
  ERA5 3-hourly here  : accumulated over the hour ENDING at the stamp
                        W m-2 = value / 3600
  ERA5 monthly here   : mean DAILY accumulation for the month
                        W m-2 = value / 86400      (global ssrd -> ~180 W m-2)
  TCo1279-DART monthly: mean of its 3-hourly accumulations
                        W m-2 = value / 10800      (global ssrd -> ~191 W m-2)
Applies to ssrd and tisr. Both ERA5 numbers are checked by check_era5.py, which
fails loudly if a future CDS change alters the convention.

Sampling ERA5 3-hourly gives 1-hour accumulations 3 h apart, NOT 3-hour totals.
That is fine because both accumulated fields are sampled the same way, so the
window cancels in the ratios ssrd/tisr and tsr/tisr that the fit uses. The residual
caveat is that the relation is fitted on 1-hour statistics and applied to DART's
3-hour means; state it as a limitation.

tcc/hcc/mcc/lcc/fal are fractions (0-1), all instantaneous at the stamp, which
matches DART's convention exactly.

Accumulated and instantaneous variables go in SEPARATE requests on purpose: mixing
them makes CDS return a zip holding one .nc per stepType instead of a plain file.

REQUIREMENTS
------------
cdsapi >= 0.7.2 with the datapi backend. The legacy 0.5.x client posts to the
retired /resources/ endpoint and fails with "'tuple' object is not callable".
Credentials go in ~/.cdsapirc (url + key); see
https://cds.climate.copernicus.eu/how-to-api

Usage:
    python3 download_era5.py 202508 202603 202608      # fetch, skipping what exists
    python3 download_era5.py --force 202508            # re-fetch even if present
    python3 download_era5.py --list                    # show the plan, fetch nothing

Output (one file per group per month, in DATA_DIR):
    era5_clouds_3h_<YYYYMM>.nc     tcc hcc mcc lcc
    era5_tsr_3h_<YYYYMM>.nc        tsr
    era5_ssrd_3h_<YYYYMM>.nc       ssrd tisr
    era5_monthly_ssrd_<YYYYMM>.nc  ssrd
    era5_monthly_fal_<YYYYMM>.nc   fal
"""

import os
import sys

DATA_DIR = "data/era5"

SL = "reanalysis-era5-single-levels"
SL_MONTHLY = "reanalysis-era5-single-levels-monthly-means"

TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]   # 00,03,...,21 UTC
DAYS = [f"{d:02d}" for d in range(1, 32)]          # CDS ignores days a month lacks

AREA = None            # [north, west, south, east]; None = global

# (tag, dataset, variables) -- one CDS request each, one file each.
# Split by stepType so CDS returns a plain .nc rather than a zip.
GROUPS = [
    ("clouds_3h", SL, ["total_cloud_cover", "high_cloud_cover",
                       "medium_cloud_cover", "low_cloud_cover"]),
    ("tsr_3h", SL, ["top_net_solar_radiation"]),
    ("ssrd_3h", SL, ["surface_solar_radiation_downwards",
                     "toa_incident_solar_radiation"]),
    ("monthly_ssrd", SL_MONTHLY, ["surface_solar_radiation_downwards"]),
    ("monthly_fal", SL_MONTHLY, ["forecast_albedo"]),
]


def build_request(dataset, variables, year, month):
    req = {"variable": variables, "year": year, "month": month,
           "data_format": "netcdf", "download_format": "unarchived"}
    if dataset == SL_MONTHLY:
        req["product_type"] = "monthly_averaged_reanalysis"
        req["time"] = "00:00"          # required by the form, ignored for monthly
    else:
        req["product_type"] = "reanalysis"
        req["day"] = DAYS
        req["time"] = TIMES
    if AREA is not None:
        req["area"] = AREA
    return req


def main(argv):
    force = "--force" in argv
    dry = "--list" in argv
    months = [a for a in argv if not a.startswith("-")]
    if not months:
        sys.exit(__doc__.split("Usage:")[1].strip())

    plan = [(f"{DATA_DIR}/era5_{tag}_{m}.nc", ds, v, m)
            for m in months for tag, ds, v in GROUPS]

    if dry:
        for target, _, variables, _ in plan:
            state = "present" if os.path.exists(target) else "MISSING"
            print(f"  {state:8s} {target:38s} {', '.join(variables)}")
        return 0

    os.makedirs(DATA_DIR, exist_ok=True)
    import cdsapi
    client = cdsapi.Client()

    skipped, done, failed = [], [], []
    for target, dataset, variables, stamp in plan:
        if os.path.exists(target) and not force:
            print(f"--- skip (exists) {target}", flush=True)
            skipped.append(target)
            continue
        print(f"--> {target}", flush=True)
        try:
            client.retrieve(dataset,
                            build_request(dataset, variables, stamp[:4], stamp[4:6]),
                            target)
            done.append(target)
        except Exception as exc:
            # Keep going: one unpublished month must not abort the rest.
            print(f"    FAILED: {exc}", flush=True)
            failed.append(target)

    print(f"\n{len(done)} downloaded, {len(skipped)} skipped, {len(failed)} failed.")
    for t in failed:
        print(f"  missing: {t}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
