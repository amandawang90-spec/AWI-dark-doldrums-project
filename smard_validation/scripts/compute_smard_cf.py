"""
Compute capacity factors from SMARD's actual generation, on the same 3-hourly
UTC grid as the ERA5 pipeline, for direct comparison with cf_*_recon/cf_*_real.

Steps:
  1. SMARD hourly MW is already an average-MW-for-that-hour figure.
  2. Resample to 3-hourly mean MW, bin-left-labeled so bin "2024-10-01 00:00"
     covers hours [00:00, 03:00) -- the same timestamps ERA5's 3-hourly files use.
  3. CF = generation (MW) / installed capacity (MW), held fixed at the 2024
     value for the whole winter (not interpolated, not stepped at Jan 1) --
     one constant denominator throughout, per instruction.
  4. Sanity check every value falls in [0, 1] and report winter means.
  5. Align to the ERA5 pipeline's own 3-hourly timestamps for direct comparison.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
GEN_DIR = f"{ROOT}/data/smard/realisierte_erzeugung"
ERA5_PATH = f"{ROOT}/data/combined_cf_era5_2025_2025.npz"
OUT_PATH = f"{ROOT}/data/cf_smard_2024_2025_3hourly.csv"

TECHS = {"wind_onshore": "cf_onshore_smard", "wind_offshore": "cf_offshore_smard", "solar": "cf_solar_smard"}
WEIGHTS = dict(solar=0.577, onshore=0.369, offshore=0.054)

# 2024 installed capacity (MW), held constant for the whole winter -- see
# data/smard/installierte_erzeugungsleistung_history/smard_installed_*_year_history.csv, year==2024
CAPACITY_MW_2024 = {"wind_onshore": 59841.0, "wind_offshore": 8456.0, "solar": 76601.0}


def resample_3hourly_mean(df):
    # ERA5's own accumulated fields are labeled by the END of their window (timestamp T
    # covers (T-3h, T]), confirmed via GRIB_stepType=accum metadata -- so SMARD's bins
    # must be labeled the same way (right-closed, right-labeled) to line up with ERA5's
    # timestamps. label="left" (the previous version of this function) put SMARD's bin
    # "03:00" over [03:00,06:00) instead -- a full 3-hour misalignment against ERA5.
    return df.set_index("timestamp_utc")["value_mw"].resample("3h", label="right", closed="right").mean()


def main():
    cf = {}
    for tech, col in TECHS.items():
        df = pd.read_csv(f"{GEN_DIR}/smard_{tech}_hour_2024_2025.csv", parse_dates=["timestamp_utc"])
        gen_3h = resample_3hourly_mean(df)
        cf[col] = gen_3h / CAPACITY_MW_2024[tech]

    out = pd.DataFrame(cf)
    out["cf_combined_smard"] = (
        WEIGHTS["solar"] * out["cf_solar_smard"]
        + WEIGHTS["onshore"] * out["cf_onshore_smard"]
        + WEIGHTS["offshore"] * out["cf_offshore_smard"]
    )

    # sanity check: every CF must fall in [0, 1]
    for col in out.columns:
        bad = out[(out[col] < 0) | (out[col] > 1)]
        if len(bad):
            print(f"WARNING: {col} has {len(bad)} values outside [0,1]:")
            print(bad[[col]])
        else:
            print(f"OK: {col} stays within [0, 1] for all {len(out)} rows")

    # align to the ERA5 pipeline's own 3-hourly timestamps
    era5 = np.load(ERA5_PATH, allow_pickle=True)
    era5_time = pd.to_datetime([str(t) for t in era5["time"]]).tz_localize("UTC")
    aligned = out.reindex(era5_time)
    aligned.index.name = "timestamp_utc"
    n_missing = aligned.isna().any(axis=1).sum()
    print(f"\nAligned to {len(era5_time)} ERA5 timestamps -- {n_missing} rows missing SMARD data")

    aligned.to_csv(OUT_PATH)
    print(f"\n{'':20s} {'mean':>8s} {'min':>8s} {'max':>8s}")
    labels = {"cf_solar_smard": "Solar", "cf_onshore_smard": "Onshore",
              "cf_offshore_smard": "Offshore", "cf_combined_smard": "Combined"}
    for col, label in labels.items():
        v = aligned[col]
        print(f"{label:20s} {v.mean():8.3f} {v.min():8.3f} {v.max():8.3f}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
