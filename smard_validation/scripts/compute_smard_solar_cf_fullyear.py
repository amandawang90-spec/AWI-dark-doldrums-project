"""SMARD solar CF, every year 2015-2026, using EACH YEAR'S OWN installed
capacity (not held fixed at one year, unlike the original single-winter
compute_smard_cf.py) -- German solar capacity grew from 37.3 GW (2015) to
86.4 GW (2025), so a fixed denominator would badly distort a multi-year series.
Mirrors compute_smard_wind_cf_fullyear.py exactly, for solar.

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
OUT_PATH = f"{ROOT}/data/cf_smard_solar_fullyear_2015_2026.csv"


def resample_3hourly_mean(df):
    return df.set_index("timestamp_utc")["value_mw"].resample("3h", label="right", closed="right").mean()


def main():
    cap = pd.read_csv(f"{CAP_DIR}/smard_installed_solar_year_history.csv")
    cap = dict(zip(cap["year"], cap["value_mw"]))

    df = pd.read_csv(f"{GEN_DIR}/smard_solar_hour_2015_2026.csv", parse_dates=["timestamp_utc"])
    gen_3h = resample_3hourly_mean(df)
    capacity_series = gen_3h.index.year.map(cap)
    cf = gen_3h.values / np.array(capacity_series, dtype=float)
    out = pd.DataFrame({"cf_solar_smard": cf}, index=gen_3h.index)
    out.index.name = "timestamp_utc"

    bad = out[(out["cf_solar_smard"] < 0) | (out["cf_solar_smard"] > 1.02)]
    if len(bad):
        print(f"WARNING: cf_solar_smard has {len(bad)} values outside [0,1.02]")
    else:
        print(f"OK: cf_solar_smard stays within [0, 1.02] for all {len(out)} rows")

    out.to_csv(OUT_PATH)
    print(f"\nn={len(out)}  {out.index.min()} .. {out.index.max()}")
    v = out["cf_solar_smard"].dropna()
    print(f"{'Solar':20s} mean={v.mean():.3f} min={v.min():.3f} max={v.max():.3f}")
    print(f"Capacity used (MW): {cap}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
