"""
Download the full-history installed generation capacity (installierte
Erzeugungsleistung) for Germany, year resolution, one figure per calendar
year -- SMARD's native resolution for this dataset (see
download_smard_installed_capacity.py for why the hourly endpoint for the
same filters is just this series upsampled with repeated values).

Each fetched chunk is a single point [timestamp, value] where timestamp is
Dec 31 23:00 UTC of the year *before* the value's effective year (Dec 31
2023 23:00 UTC = Jan 1 2024 00:00 CET, so that point's value is what's in
force through calendar year 2024).
"""
import os

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"
REGION = "DE"
RESOLUTION = "year"
FILTERS = {186: "wind_onshore", 4076: "wind_offshore", 188: "solar"}

OUT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/smard/installierte_erzeugungsleistung_history"


def effective_year(timestamp_ms):
    return pd.Timestamp(timestamp_ms, unit="ms", tz="UTC").year + 1


def download_history(session, filter_id, name):
    idx = session.get(f"{BASE}/{filter_id}/{REGION}/index_{RESOLUTION}.json", timeout=30)
    idx.raise_for_status()
    timestamps = idx.json()["timestamps"]

    rows = []
    for ts in timestamps:
        r = session.get(f"{BASE}/{filter_id}/{REGION}/{filter_id}_{REGION}_{RESOLUTION}_{ts}.json", timeout=30)
        r.raise_for_status()
        for t, v in r.json()["series"]:
            if v is not None:
                rows.append((effective_year(t), v))

    df = pd.DataFrame(rows, columns=["year", "value_mw"]).sort_values("year").reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/smard_installed_{name}_year_history.csv"
    df.to_csv(out_path, index=False)
    print(f"{name:15s} {len(df):3d} years  {df.year.min()}..{df.year.max()}  -> {out_path}")
    return df


def main():
    session = requests.Session()
    for filter_id, name in FILTERS.items():
        download_history(session, filter_id, name)


if __name__ == "__main__":
    main()
