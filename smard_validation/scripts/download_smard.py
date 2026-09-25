"""
Download actual German electricity generation (realisierte Erzeugung) from
SMARD for the same winter 2024-2025 period as the ERA5 combined-CF validation
run, to compare against smard_validation/data/combined_cf_era5_2025_2025.npz.
See smard_common.py for how the chunked chart_data API is queried.
"""
import os

import pandas as pd

import smard_common as sc

FILTERS = {4067: "wind_onshore", 1225: "wind_offshore", 4068: "solar"}

PERIOD_START = pd.Timestamp("2024-10-01T00:00:00Z")   # winter 2024-2025, same as the ERA5 run
PERIOD_END = pd.Timestamp("2025-03-01T00:00:00Z")      # exclusive; covers all of Feb 28 2025

OUT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/smard/realisierte_erzeugung"


if __name__ == "__main__":
    sc.run(FILTERS, PERIOD_START, PERIOD_END, OUT_DIR, file_prefix="smard")
