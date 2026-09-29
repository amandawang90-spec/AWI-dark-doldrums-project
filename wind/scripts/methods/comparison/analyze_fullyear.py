"""
Turn the additive statistics from three_methods_fullyear.py into metrics: full
calendar year pooled, a Sep-Mar "wide winter" subset, and the original Oct-Feb
subset (for a same-months, different-year-range comparison against the old
10/20-winter pipelines). Writes data/results/three_methods_fullyear_2015_2025_metrics.json.
Does not touch any of the old winter-only stats/metrics files.

Usage: python analyze_fullyear.py [stats.json] [out.json]
"""

import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)

import json
import sys
from collections import defaultdict
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/results/three_methods_fullyear_2015_2025_stats.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/results/three_methods_fullyear_2015_2025_metrics.json"

d = json.load(open(SRC))
stats = d["stats"]
METHODS = ["log_law", "regression", "power_law"]
REGIONS = ["onshore", "offshore"]
SUBSETS = ["overall", "low_wind"]

SEPMAR = {"09", "10", "11", "12", "01", "02", "03"}
OCTFEB = {"10", "11", "12", "01", "02"}


def metrics(v):
    n, Sw, Swe, Swe2, Swt, Swp, Swtt, Swpp, Swtp, Swape = v
    mt, mp = Swt / Sw, Swp / Sw
    cov = Swtp / Sw - mt * mp
    r = cov / np.sqrt((Swtt / Sw - mt ** 2) * (Swpp / Sw - mp ** 2))
    sst = Swtt - Swt ** 2 / Sw
    return dict(rmse=float(np.sqrt(Swe2 / Sw)), bias=float(Swe / Sw), mape=float(Swape / Sw * 100),
                r=float(r), r2=float(1 - Swe2 / sst), nrmse=float(np.sqrt(Swe2 / Sw) / mt),
                mean_true=float(mt), n=int(n))


def group(keyfn):
    acc = defaultdict(lambda: np.zeros(10))
    for k, v in stats.items():
        ym, season, region, method, subset = k.split("|")
        keep = keyfn(ym, season)
        if keep is None:
            continue
        acc[keep + (region, method, subset)] += np.array(v)
    return {k: metrics(v) for k, v in acc.items()}


def month_of(ym):
    return ym[5:7]


pooled_fullyear = group(lambda ym, s: ())
pooled_sepmar = group(lambda ym, s: () if month_of(ym) in SEPMAR else None)
pooled_octfeb = group(lambda ym, s: () if month_of(ym) in OCTFEB else None)
by_season = group(lambda ym, s: (s,))
by_month = group(lambda ym, s: (month_of(ym),))
by_year = group(lambda ym, s: (ym[:4],))

out = {
    "fit": d["fit"],
    "n_months_total": len(set(k.split("|")[0] for k in stats)),
    "pooled_fullyear": {f"{r}|{m}|{s}": pooled_fullyear[(r, m, s)] for r in REGIONS for m in METHODS for s in SUBSETS},
    "pooled_sepmar": {f"{r}|{m}|{s}": pooled_sepmar[(r, m, s)] for r in REGIONS for m in METHODS for s in SUBSETS},
    "pooled_octfeb": {f"{r}|{m}|{s}": pooled_octfeb[(r, m, s)] for r in REGIONS for m in METHODS for s in SUBSETS},
    "by_season": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in by_season.items()},
    "by_month": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in by_month.items()},
    "by_year": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in by_year.items()},
}
json.dump(out, open(OUT, "w"), indent=2)
print(f"saved {OUT}")

print("\n=== FULL YEAR (all 12 months, 2015-2025) pooled ===")
print(f"{'region':<9}{'subset':<9}" + "".join(f"{m:>28}" for m in METHODS))
for r in REGIONS:
    for s in SUBSETS:
        cells = [f"RMSE {pooled_fullyear[(r,m,s)]['rmse']:.3f} MAPE {pooled_fullyear[(r,m,s)]['mape']:5.1f}% b{pooled_fullyear[(r,m,s)]['bias']:+.2f}" for m in METHODS]
        print(f"{r:<9}{s:<9}" + "".join(f"{c:>28}" for c in cells))

print("\n=== SEP-MAR wide winter (2015-2025) pooled ===")
for r in REGIONS:
    for s in SUBSETS:
        cells = [f"RMSE {pooled_sepmar[(r,m,s)]['rmse']:.3f} MAPE {pooled_sepmar[(r,m,s)]['mape']:5.1f}% b{pooled_sepmar[(r,m,s)]['bias']:+.2f}" for m in METHODS]
        print(f"{r:<9}{s:<9}" + "".join(f"{c:>28}" for c in cells))

print("\n=== OCT-FEB, but 2015-2025 only (same months as old pipeline, different year range) ===")
for r in REGIONS:
    for s in SUBSETS:
        cells = [f"RMSE {pooled_octfeb[(r,m,s)]['rmse']:.3f} MAPE {pooled_octfeb[(r,m,s)]['mape']:5.1f}% b{pooled_octfeb[(r,m,s)]['bias']:+.2f}" for m in METHODS]
        print(f"{r:<9}{s:<9}" + "".join(f"{c:>28}" for c in cells))
