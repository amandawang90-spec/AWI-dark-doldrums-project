"""Download wind (onshore/offshore) and solar generation and installed capacity from the Energy-Charts API (Fraunhofer ISE,
https://api.energy-charts.info, free, no key) for the country this folder is named after, 2015 -> now.

Energy-Charts republishes the TSO/ENTSO-E generation data; the resolution is whatever the country reports (15 min for NL,
30 min for IE, hourly for PL, SE, NO). Not every country has every type (e.g. no offshore wind or solar reported for IE/NO).
  data/energycharts_<cc>_production_<year>.csv   UTC time + one column per wind/solar type [MW]
  data/energycharts_<cc>_installed_power.csv     installed capacity per wind/solar type [GW], yearly (the API has no monthly series)
The API rate-limits aggressively (HTTP 429): the script spaces requests and backs off. Existing past years are skipped.
"""
import os
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.basename(os.path.dirname(HERE))          # e.g. poland_validation
CC = {"poland": "pl", "netherlands": "nl", "ireland": "ie", "sweden": "se", "norway": "no"}[FOLDER.replace("_validation", "")]
OUT = f"{HERE}/../data"
API = "https://api.energy-charts.info"
KEEP = ("wind", "solar")
GAP = 10


def get(path, **params):
    wait = 30
    for attempt in range(12):
        r = requests.get(f"{API}/{path}", params=params, timeout=180)
        if r.status_code == 429:
            print(f"  429, waiting {wait}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 300)
            continue
        r.raise_for_status()
        time.sleep(GAP)
        return r.json()
    raise RuntimeError(f"giving up on {path} {params}")


def production(year):
    j = get("public_power", country=CC, start=f"{year}-01-01T00:00+00:00", end=f"{year}-12-31T23:59+00:00")
    df = pd.DataFrame({"time_utc": pd.to_datetime(j["unix_seconds"], unit="s")})
    for p in j["production_types"]:
        if any(k in p["name"].lower() for k in KEEP):
            df[p["name"]] = p["data"]
    return df


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    now = pd.Timestamp.now()
    for year in range(2015, now.year + 1):
        path = f"{OUT}/energycharts_{CC}_production_{year}.csv"
        if os.path.exists(path) and year < now.year:
            print(f"{CC} {year}: exists, skipped", flush=True)
            continue
        df = production(year)
        df.to_csv(path, index=False)
        step = df.time_utc.diff().mode().iloc[0] if len(df) > 1 else None
        print(f"{CC} {year}: {len(df)} rows, step {step}, cols {list(df.columns[1:])}, "
              f"{df.time_utc.min()} .. {df.time_utc.max()}", flush=True)
    try:
        j = get("installed_power", country=CC, time_step="yearly", installation_decommission="false")
        df = pd.DataFrame({"time": j["time"]})
        for p in j["production_types"]:
            if any(k in p["name"].lower() for k in KEEP):
                df[p["name"]] = p["data"]
        df.to_csv(f"{OUT}/energycharts_{CC}_installed_power.csv", index=False)
        print(f"{CC} installed power: {len(df)} rows, {df.time.iloc[0]} .. {df.time.iloc[-1]}, cols {list(df.columns[1:])}", flush=True)
    except Exception as e:
        print(f"{CC} installed power FAILED: {e}", flush=True)
