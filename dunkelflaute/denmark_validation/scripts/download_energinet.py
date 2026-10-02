"""Download Danish (Energinet, Energi Data Service) hourly wind/solar production and monthly installed capacity, 2015-.

  ProductionConsumptionSettlement  hourly, per price area (DK1, DK2): offshore/onshore wind and solar production [MWh]
  CapacityPerMunicipality          monthly, per municipality: offshore/onshore wind and solar capacity [MW]
Free, no API key. The API rate-limits (HTTP 429); the script waits as long as the response asks and retries.
Existing yearly files are skipped, so it can be re-run to resume.
"""
import json
import os
import re
import sys
import time

import pandas as pd
import requests

API = "https://api.energidataservice.dk/dataset"
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/../data"
PROD_COLS = ["HourUTC", "HourDK", "PriceArea", "OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh",
             "OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh", "SolarPowerLt10kW_MWh", "SolarPowerGe10Lt40kW_MWh",
             "SolarPowerGe40kW_MWh", "SolarPowerSelfConMWh"]
START_YEAR = 2015


def get(session, dataset, **params):
    url = f"{API}/{dataset}"
    for attempt in range(30):
        r = session.get(url, params=params, timeout=120)
        if r.status_code == 429:
            m = re.search(r"(\d+) seconds", r.text)
            wait = int(m.group(1)) + 5 if m else 60
            print(f"  429, waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["records"]
    raise RuntimeError(f"giving up on {dataset} {params}")


def download_production(session):
    this_year = pd.Timestamp.now().year
    for year in range(START_YEAR, this_year + 1):
        path = f"{OUT}/energinet_production_hourly_{year}.csv"
        if os.path.exists(path) and year < this_year:
            print(f"{year}: exists, skipped", flush=True)
            continue
        recs = get(session, "ProductionConsumptionSettlement", start=f"{year}-01-01T00:00", end=f"{year + 1}-01-01T00:00",
                   columns=",".join(PROD_COLS), sort="HourUTC asc", limit=0)
        df = pd.DataFrame(recs)
        df.to_csv(path, index=False)
        print(f"{year}: {len(df)} rows ({df.HourUTC.min()} .. {df.HourUTC.max()}) -> {path}", flush=True)
        time.sleep(5)


def download_capacity(session):
    recs = get(session, "CapacityPerMunicipality", start=f"{START_YEAR}-01-01T00:00", sort="Month asc", limit=0)
    df = pd.DataFrame(recs)
    df.to_csv(f"{OUT}/energinet_capacity_per_municipality_monthly.csv", index=False)
    tot = df.groupby("Month")[["OffshoreWindCapacity", "OnshoreWindCapacity", "SolarPowerCapacity"]].sum()
    tot.to_csv(f"{OUT}/energinet_capacity_denmark_monthly.csv")
    print(f"capacity: {len(df)} rows, {tot.index.min()} .. {tot.index.max()}", flush=True)
    print(tot.iloc[[0, -1]], flush=True)


if __name__ == "__main__":
    s = requests.Session()
    download_capacity(s)
    download_production(s)
