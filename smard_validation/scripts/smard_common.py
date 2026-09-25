"""
Shared SMARD (Bundesnetzagentur) chart_data download helpers.

SMARD's chart_data API has no direct date-range query: index_{resolution}.json
lists every chunk's start timestamp, and you fetch whichever chunks overlap the
period -- including the chunk starting before it, since its data run past its
own start into the period. This works the same way for every SMARD dataset
(realized generation, installed capacity, ...); only the filter IDs differ
per dataset (see https://www.smard.de/app/chart_configuration/market_data_configuration.json).
"""
import os
import time

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"
REGION = "DE"
RESOLUTION = "hour"


def period_tag(period_start, period_end):
    return f"{period_start.year}_{(period_end - pd.Timedelta(days=1)).year}"


def chunks_for_period(index_timestamps, start_ms, end_ms):
    before = [t for t in index_timestamps if t <= start_ms]
    after = [t for t in index_timestamps if start_ms < t < end_ms]
    return ([before[-1]] if before else []) + after


def download_filter(session, filter_id, name, start_ms, end_ms, out_dir, file_prefix, tag):
    index_url = f"{BASE}/{filter_id}/{REGION}/index_{RESOLUTION}.json"
    idx = session.get(index_url, timeout=30)
    idx.raise_for_status()
    timestamps = idx.json()["timestamps"]

    rows = []
    for chunk_ts in chunks_for_period(timestamps, start_ms, end_ms):
        url = f"{BASE}/{filter_id}/{REGION}/{filter_id}_{REGION}_{RESOLUTION}_{chunk_ts}.json"
        r = session.get(url, timeout=30)
        r.raise_for_status()
        for t, v in r.json()["series"]:
            if v is not None and start_ms <= t < end_ms:
                rows.append((t, v))
        time.sleep(0.3)

    df = pd.DataFrame(rows, columns=["timestamp_ms", "value_mw"]).drop_duplicates("timestamp_ms")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[["timestamp_utc", "value_mw"]].sort_values("timestamp_utc").reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{file_prefix}_{name}_hour_{tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"{name:15s} {len(df):5d} rows  {df['timestamp_utc'].min()} .. {df['timestamp_utc'].max()}"
          f"  -> {out_path}")
    return df


def run(filters, period_start, period_end, out_dir, file_prefix):
    start_ms = int(period_start.timestamp() * 1000)
    end_ms = int(period_end.timestamp() * 1000)
    tag = period_tag(period_start, period_end)
    session = requests.Session()
    for filter_id, name in filters.items():
        download_filter(session, filter_id, name, start_ms, end_ms, out_dir, file_prefix, tag)
