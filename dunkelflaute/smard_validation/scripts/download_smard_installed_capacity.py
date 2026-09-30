"""
Download installed generation capacity (installierte Erzeugungsleistung) from
SMARD for the same winter 2024-2025 period and technologies as
download_smard.py's realized-generation pull, so MW generation can be divided
by MW capacity to get a capacity factor comparable to cf_*_real/cf_*_recon.

Filter IDs come from https://www.smard.de/app/chart_configuration/market_data_configuration.json
(Stromerzeugung -> Installierte Erzeugungsleistung); they're a different set
than the realized-generation filters in download_smard.py, despite both
living under "Stromerzeugung" on the site.

SMARD's own source data for this dataset is yearly (one figure per year) --
the hourly endpoint just repeats that year's value across every hour, so the
resulting series is a step function that only changes at a year boundary.
That's still exactly what's needed here: one capacity value per hour, aligned
with the generation timestamps.
"""
import os

import pandas as pd

import smard_common as sc

FILTERS = {186: "wind_onshore", 4076: "wind_offshore", 188: "solar"}

PERIOD_START = pd.Timestamp("2024-10-01T00:00:00Z")   # same winter 2024-2025 period
PERIOD_END = pd.Timestamp("2025-03-01T00:00:00Z")      # exclusive; covers all of Feb 28 2025

OUT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/smard/installierte_erzeugungsleistung"


if __name__ == "__main__":
    sc.run(FILTERS, PERIOD_START, PERIOD_END, OUT_DIR, file_prefix="smard_installed")
