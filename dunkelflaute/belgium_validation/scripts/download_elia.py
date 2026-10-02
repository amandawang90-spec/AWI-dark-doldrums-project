"""Download Belgian (Elia Open Data) wind and solar measured production at 15-minute resolution.

  ods031  wind  (historical): offshore/onshore x region x grid connection type, 2015-01-01 -> now
  ods032  solar (historical): per region plus a 'Belgium' total, only from 2020-09-01 (Elia publishes nothing earlier)
Columns: datetime (UTC), measured [MW, mean over the 15 min], monitoredcapacity [MW, capacity Elia monitors at that time,
which can deviate from the installed capacity], loadfactor [measured / monitoredcapacity].
Free, no API key (CC BY 4.0). One CSV per dataset and year; existing past years are skipped so the script can resume.
"""
import os
import time

import pandas as pd
import requests

BASE = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/../data"
DATASETS = {
    "wind": ("ods031", "datetime,offshoreonshore,region,gridconnectiontype,measured,monitoredcapacity,loadfactor", 2015),
    "solar": ("ods032", "datetime,region,measured,monitoredcapacity,loadfactor", 2020),
}


def export(session, dataset, select, year):
    params = {"select": select, "order_by": "datetime asc", "delimiter": ",",
              "where": f'datetime>="{year}-01-01T00:00:00Z" and datetime<"{year + 1}-01-01T00:00:00Z"'}
    for attempt in range(5):
        r = session.get(f"{BASE}/{dataset}/exports/csv", params=params, timeout=600)
        if r.status_code == 429 or r.status_code >= 500:
            print(f"  HTTP {r.status_code}, retry in 60s", flush=True)
            time.sleep(60)
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"giving up on {dataset} {year}")


if __name__ == "__main__":
    s = requests.Session()
    this_year = pd.Timestamp.now().year
    for name, (dataset, select, first_year) in DATASETS.items():
        for year in range(first_year, this_year + 1):
            path = f"{OUT}/elia_{name}_15min_{year}.csv"
            if os.path.exists(path) and year < this_year:
                print(f"{name} {year}: exists, skipped", flush=True)
                continue
            content = export(s, dataset, select, year)
            with open(path, "wb") as f:
                f.write(content)
            n = content.count(b"\n") - 1
            print(f"{name} {year}: {n} rows -> {path}", flush=True)
            time.sleep(2)
