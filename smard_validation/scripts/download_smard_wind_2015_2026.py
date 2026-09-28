"""
Download actual German wind generation (realisierte Erzeugung) from SMARD for
2015-2026 -- same filters/parameters as download_smard.py's winter 2024-2025
pull, just wind onshore/offshore (no solar) over a much longer range.
See smard_common.py for how the chunked chart_data API is queried.
"""
import os

import pandas as pd

import smard_common as sc

FILTERS = {4067: "wind_onshore", 1225: "wind_offshore"}

PERIOD_START = pd.Timestamp("2015-01-01T00:00:00Z")
PERIOD_END = pd.Timestamp("2027-01-01T00:00:00Z")  # exclusive; SMARD simply has nothing past "now"

OUT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/smard/realisierte_erzeugung"


if __name__ == "__main__":
    sc.run(FILTERS, PERIOD_START, PERIOD_END, OUT_DIR, file_prefix="smard")
