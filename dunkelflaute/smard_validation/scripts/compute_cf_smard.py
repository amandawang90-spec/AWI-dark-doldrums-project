"""
Compute capacity factors from SMARD's actual generation, using a single fixed
installed-capacity figure (the 2024 value) as the denominator throughout the
winter 2024-2025 window -- avoiding the artificial Jan-1 step in the raw
installed-capacity series (see the "Capacity Steps" report) since the user
wants one constant denominator for this comparison rather than a mid-winter
jump.

cf_combined uses the SAME fixed weights as the ERA5 pipeline
(dunkelflaute/scripts/era5/compute_germany_combined_cf_era5.py: WEIGHTS =
solar 0.577 / onshore 0.369 / offshore 0.054) so cf_combined_smard is
comparable to cf_combined_recon/cf_combined_real -- same formula, different
CF inputs.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
GEN_DIR = f"{ROOT}/data/smard/realisierte_erzeugung"
OUT_PATH = f"{ROOT}/data/cf_smard_2024_2025.csv"

# 2024 installed capacity (MW), held constant for the whole winter -- see
# data/smard/installierte_erzeugungsleistung_history/smard_installed_*_year_history.csv, year==2024
CAPACITY_MW_2024 = {"wind_onshore": 59841.0, "wind_offshore": 8456.0, "solar": 76601.0}

WEIGHTS = dict(solar=0.577, onshore=0.369, offshore=0.054)


def main():
    series = {}
    for tech in ["wind_onshore", "wind_offshore", "solar"]:
        df = pd.read_csv(f"{GEN_DIR}/smard_{tech}_hour_2024_2025.csv", parse_dates=["timestamp_utc"])
        df = df.set_index("timestamp_utc")
        series[tech] = df["value_mw"] / CAPACITY_MW_2024[tech]

    out = pd.DataFrame({
        "cf_solar_smard": series["solar"],
        "cf_onshore_smard": series["wind_onshore"],
        "cf_offshore_smard": series["wind_offshore"],
    })
    out["cf_combined_smard"] = (
        WEIGHTS["solar"] * out["cf_solar_smard"]
        + WEIGHTS["onshore"] * out["cf_onshore_smard"]
        + WEIGHTS["offshore"] * out["cf_offshore_smard"]
    )
    out = out.reset_index().rename(columns={"timestamp_utc": "timestamp_utc"})
    out.to_csv(OUT_PATH, index=False)

    print(f"n={len(out)}  {out.timestamp_utc.min()} .. {out.timestamp_utc.max()}")
    print(f"{'':20s} {'mean':>8s} {'min':>8s} {'max':>8s}")
    for col, label in [("cf_solar_smard", "Solar"), ("cf_onshore_smard", "Onshore"),
                        ("cf_offshore_smard", "Offshore"), ("cf_combined_smard", "Combined")]:
        v = out[col]
        print(f"{label:20s} {v.mean():8.3f} {v.min():8.3f} {v.max():8.3f}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
