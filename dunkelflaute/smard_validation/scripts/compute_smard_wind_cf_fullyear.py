"""
SMARD wind CF (onshore + offshore), every year 2015-2026, using EACH YEAR'S OWN
installed capacity (not held fixed at one year, unlike the original single-winter
compute_smard_cf.py) -- capacity roughly doubled onshore and grew ~9x offshore
over this span, so a fixed denominator would badly distort a multi-year series.

CF(t) = generation_MW(t) / installed_capacity_MW(year of t)

Resampled to the same 3-hourly grid as ERA5 (label="right", closed="right" --
ERA5's accumulated fields are labeled by the END of their window, see
compute_smard_cf.py's note on this).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
GEN_DIR = f"{ROOT}/data/smard/realisierte_erzeugung"
CAP_DIR = f"{ROOT}/data/smard/installierte_erzeugungsleistung_history"
OUT_PATH = f"{ROOT}/data/cf_smard_fullyear_2015_2026.csv"

TECHS = {"wind_onshore": "cf_onshore_smard", "wind_offshore": "cf_offshore_smard"}


def resample_3hourly_mean(df):
    return df.set_index("timestamp_utc")["value_mw"].resample("3h", label="right", closed="right").mean()


def main():
    cap = {}
    for tech in TECHS:
        c = pd.read_csv(f"{CAP_DIR}/smard_installed_{tech.replace('wind_','wind_')}_year_history.csv")
        cap[tech] = dict(zip(c["year"], c["value_mw"]))

    cf = {}
    for tech, col in TECHS.items():
        df = pd.read_csv(f"{GEN_DIR}/smard_{tech}_hour_2015_2026.csv", parse_dates=["timestamp_utc"])
        gen_3h = resample_3hourly_mean(df)
        capacity_series = gen_3h.index.year.map(cap[tech])
        cf[col] = gen_3h.values / np.array(capacity_series, dtype=float)
        cf[col] = pd.Series(cf[col], index=gen_3h.index)

    out = pd.DataFrame(cf)
    for col in out.columns:
        bad = out[(out[col] < 0) | (out[col] > 1.02)]
        if len(bad):
            print(f"WARNING: {col} has {len(bad)} values outside [0,1.02]")
        else:
            print(f"OK: {col} stays within [0, 1.02] for all {len(out)} rows")

    out.index.name = "timestamp_utc"
    out.to_csv(OUT_PATH)
    print(f"\nn={len(out)}  {out.index.min()} .. {out.index.max()}")
    print(f"{'':20s} {'mean':>8s} {'min':>8s} {'max':>8s}")
    for col, label in [("cf_onshore_smard", "Onshore"), ("cf_offshore_smard", "Offshore")]:
        v = out[col].dropna()
        print(f"{label:20s} {v.mean():8.3f} {v.min():8.3f} {v.max():8.3f}")
    print(f"\nCapacity used (MW): onshore={cap['wind_onshore']}")
    print(f"Capacity used (MW): offshore={cap['wind_offshore']}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
