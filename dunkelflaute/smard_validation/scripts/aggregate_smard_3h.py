"""
Aggregate SMARD hourly generation (wind_onshore, wind_offshore, solar) to a
true 3-hourly MEAN -- same convention as the ERA5 ssrd/tsr hourly->3-hourly
aggregation: each output stamp at 00,03,...,21 is the mean of the 3 hourly
values ending at that stamp (e.g. 03:00 = mean(01:00, 02:00, 03:00)), correctly
crossing day boundaries (00:00 = mean of 22:00,23:00 the previous day + 00:00).

Written to a SEPARATE directory (realisierte_erzeugung_3h/) so hourly and
3-hourly files stay distinguishable -- source files in realisierte_erzeugung/
are untouched.

Usage:
    python3 aggregate_smard_3h.py               # every smard_*_hour_*.csv found
    python3 aggregate_smard_3h.py smard_wind_onshore_hour_2015_2026.csv
"""
import glob
import os
import re
import sys

import pandas as pd

SRC_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/smard/realisierte_erzeugung"
OUT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/smard/realisierte_erzeugung_3h"

NAME_RE = re.compile(r"^(smard_.+)_hour_(.+)\.csv$")


def aggregate_one(src_path):
    fname = os.path.basename(src_path)
    m = NAME_RE.match(fname)
    if not m:
        print(f"    SKIP {fname}: doesn't match smard_<name>_hour_<tag>.csv")
        return
    name, tag = m.groups()
    out_path = f"{OUT_DIR}/{name}_3h_{tag}.csv"

    df = pd.read_csv(src_path, parse_dates=["timestamp_utc"])
    df = df.set_index("timestamp_utc").sort_index()

    n_hourly = len(df)
    # closed='right', label='right': bin (t-3h, t] -> labeled t, i.e. the mean
    # of the 3 hourly values ENDING at t (hours t-2,t-1,t) -- matches the ERA5
    # 3-hourly-mean convention exactly, and correctly spans midnight.
    agg = df["value_mw"].resample("3h", label="right", closed="right").mean()
    agg = agg.dropna()

    os.makedirs(OUT_DIR, exist_ok=True)
    agg.to_frame("value_mw").reset_index().rename(columns={"index": "timestamp_utc"}) \
        .to_csv(out_path, index=False)
    print(f"{name:15s} {n_hourly:6d} hourly -> {len(agg):6d} 3-hourly  "
          f"{agg.index.min()} .. {agg.index.max()}  -> {out_path}")


def main(argv):
    if argv:
        paths = [f"{SRC_DIR}/{a}" for a in argv]
    else:
        paths = sorted(glob.glob(f"{SRC_DIR}/smard_*_hour_*.csv"))

    for p in paths:
        aggregate_one(p)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
