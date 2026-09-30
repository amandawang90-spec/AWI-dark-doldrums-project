"""
Download actual German solar generation (realisierte Erzeugung) from SMARD for
2015-2026 -- same filter/parameters as download_smard.py's winter 2024-2025
pull, just solar over a much longer range (companion to
download_smard_wind_2015_2026.py).
See smard_common.py for how the chunked chart_data API is queried.
"""
import os

import pandas as pd

import smard_common as sc

FILTERS = {4068: "solar"}

PERIOD_START = pd.Timestamp("2015-01-01T00:00:00Z")
PERIOD_END = pd.Timestamp("2027-01-01T00:00:00Z")  # exclusive; SMARD simply has nothing past "now"

OUT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/smard/realisierte_erzeugung"


if __name__ == "__main__":
    sc.run(FILTERS, PERIOD_START, PERIOD_END, OUT_DIR, file_prefix="smard")
