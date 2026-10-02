"""Download Finnish (Fingrid Open Data) wind production, wind capacity and the solar estimate, 2015-.
NOT YET RUN: the API needs a free subscription key (HTTP 401 without it). Get one at https://data.fingrid.fi (or
avoindata@fingrid.fi) and run:   FINGRID_API_KEY=<key> python3 download_fingrid.py
Endpoint and parameters follow the Fingrid API v1 docs as I remember them; check them at https://developer-data.fingrid.fi/apis
if the first request fails. Alternatively download the same datasets manually (CSV) from https://data.fingrid.fi/en/data?datasets=<id>.

Datasets (ids from data.fingrid.fi):
  75   Wind power generation, 15 min (hourly before 2023-06-13), MW, from 2014-11-28; measured parks + estimate for the rest
  268  Total production capacity used in the wind power forecast (installed wind capacity proxy)
  248  Solar power generation forecast, 15 min (hourly before 2023-05-31), from 2017-02-24. This is a MODEL estimate, not a
       measurement: Fingrid publishes no measured national solar production.
"""
import os
import sys
import time

import pandas as pd
import requests

URL = "https://data.fingrid.fi/api/datasets/{id}/data"
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/../data"
DATASETS = {"wind_production": 75, "wind_capacity": 268, "solar_estimate": 248}

key = os.environ.get("FINGRID_API_KEY")
if not key:
    sys.exit("Set FINGRID_API_KEY (free key from data.fingrid.fi).")
s = requests.Session()
s.headers["x-api-key"] = key
for name, ds in DATASETS.items():
    for year in range(2015, pd.Timestamp.now().year + 1):
        path = f"{OUT}/fingrid_{name}_{year}.csv"
        if os.path.exists(path) and year < pd.Timestamp.now().year:
            continue
        r = s.get(URL.format(id=ds), params={"startTime": f"{year}-01-01T00:00:00Z", "endTime": f"{year + 1}-01-01T00:00:00Z",
                                             "format": "csv", "pageSize": 20000, "sortBy": "startTime", "sortOrder": "asc"}, timeout=300)
        if r.status_code == 429:
            time.sleep(70); r = s.get(r.url, timeout=300)
        r.raise_for_status()
        open(path, "wb").write(r.content)
        print(name, year, len(r.content), "bytes", flush=True)
        time.sleep(3)
