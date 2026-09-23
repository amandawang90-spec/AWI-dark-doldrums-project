"""
Turn the additive statistics from three_methods_10winters.py into metrics, pooled /
per winter / per calendar month, and test whether the earlier pattern holds in
every winter. Writes data/results/three_methods_winters_metrics.json for the artifact.

Usage: python analyze_winters.py [stats.json] [out.json]
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

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/results/three_methods_10winters_stats.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/results/three_methods_winters_metrics.json"

d = json.load(open(SRC))
stats = d["stats"]
METHODS = ["log_law", "regression", "power_law"]
REGIONS = ["onshore", "offshore"]
SUBSETS = ["overall", "low_wind"]


def metrics(v):
    n, Sw, Swe, Swe2, Swt, Swp, Swtt, Swpp, Swtp, Swape = v
    mt, mp = Swt / Sw, Swp / Sw
    cov = Swtp / Sw - mt * mp
    r = cov / np.sqrt((Swtt / Sw - mt ** 2) * (Swpp / Sw - mp ** 2))
    sst = Swtt - Swt ** 2 / Sw  # weighted total sum of squares of the true 100 m speed
    return dict(rmse=float(np.sqrt(Swe2 / Sw)), bias=float(Swe / Sw), mape=float(Swape / Sw * 100),
                r=float(r), r2=float(1 - Swe2 / sst), nrmse=float(np.sqrt(Swe2 / Sw) / mt),
                mean_true=float(mt), n=int(n))


def group(keyfn):
    acc = defaultdict(lambda: np.zeros(10))
    for k, v in stats.items():
        ym, winter, region, method, subset = k.split("|")
        acc[keyfn(ym, winter) + (region, method, subset)] += np.array(v)
    return {k: metrics(v) for k, v in acc.items()}


pooled = group(lambda ym, w: ())
by_winter = group(lambda ym, w: (w,))
by_month = group(lambda ym, w: (ym[5:],))  # "10","11","12","01","02"
winters = sorted({k[0] for k in by_winter})
months = ["10", "11", "12", "01", "02"]

print("=== HOW GOOD ARE THEY: correlation r, R^2 (1 - SSE/SST, penalises bias and scale), RMSE / mean wind ===")
print(f"{'region':<9}{'subset':<9}{'method':<12}{'r':>7}{'r^2':>7}{'R^2':>8}{'nRMSE':>8}{'mean 100m':>11}")
for r_ in REGIONS:
    for s_ in SUBSETS:
        for m_ in METHODS:
            x = pooled[(r_, m_, s_)]
            print(f"{r_:<9}{s_:<9}{m_:<12}{x['r']:>7.3f}{x['r'] ** 2:>7.3f}{x['r2']:>8.3f}{x['nrmse'] * 100:>7.1f}%{x['mean_true']:>10.2f}")
print()
print("=== POOLED over all winters ===")
print(f"{'region':<9}{'subset':<9}" + "".join(f"{m:>26}" for m in METHODS))
for r in REGIONS:
    for s in SUBSETS:
        cells = []
        for m in METHODS:
            x = pooled[(r, m, s)]
            cells.append(f"RMSE {x['rmse']:.3f} MAPE {x['mape']:5.1f}% b{x['bias']:+.2f}")
        print(f"{r:<9}{s:<9}" + "".join(f"{c:>26}" for c in cells))
print()


def best(metric, region, subset, source, key, lowest=True):
    vals = {m: source[key + (region, m, subset)][metric] for m in METHODS}
    vals_cmp = {m: abs(v) for m, v in vals.items()} if metric == "bias" else vals
    return (min if lowest else max)(vals_cmp, key=vals_cmp.get)


# pattern checks, per winter
checks = [
    ("regression best overall RMSE onshore", "rmse", "onshore", "overall", "regression"),
    ("regression best overall RMSE offshore", "rmse", "offshore", "overall", "regression"),
    ("log law best low-wind RMSE offshore", "rmse", "offshore", "low_wind", "log_law"),
    ("log law best low-wind MAPE offshore", "mape", "offshore", "low_wind", "log_law"),
    ("power law best low-wind MAPE onshore", "mape", "onshore", "low_wind", "power_law"),
    ("power law WORST overall RMSE offshore", "rmse", "offshore", "overall", "power_law"),
]
print("=== PATTERN CHECK: how many of the winters agree ===")
summary = {}
for name, metric, region, subset, expect in checks:
    hits = []
    for w in winters:
        if "WORST" in name:
            vals = {m: by_winter[(w, region, m, subset)][metric] for m in METHODS}
            got = max(vals, key=vals.get)
        else:
            got = best(metric, region, subset, by_winter, (w,))
        hits.append(got == expect)
    summary[name] = dict(holds=int(sum(hits)), of=len(hits), winters_failing=[w for w, h in zip(winters, hits) if not h])
    print(f"  {name:<42} {sum(hits)}/{len(hits)}   failing: {summary[name]['winters_failing'] or '-'}")

pb = [by_winter[(w, "offshore", "power_law", "overall")]["bias"] for w in winters]
print(f"  power law offshore overall bias per winter: min {min(pb):+.2f}  max {max(pb):+.2f}  (all positive: {all(b > 0 for b in pb)})")
summary["power_law_offshore_bias_range"] = [min(pb), max(pb)]

print("\n=== PER WINTER: onshore low-wind MAPE / offshore low-wind MAPE ===")
print(f"{'winter':<9}" + "".join(f"{m+' on':>14}" for m in METHODS) + "".join(f"{m+' off':>14}" for m in METHODS))
for w in winters:
    row = [by_winter[(w, "onshore", m, "low_wind")]["mape"] for m in METHODS] + \
          [by_winter[(w, "offshore", m, "low_wind")]["mape"] for m in METHODS]
    print(f"{w:<9}" + "".join(f"{x:>14.1f}" for x in row))

print("\n=== BY CALENDAR MONTH (pooled over winters): onshore low-wind MAPE ===")
print(f"{'month':<7}" + "".join(f"{m:>12}" for m in METHODS) + "   | offshore low-wind MAPE")
for mo in months:
    on = [by_month[(mo, "onshore", m, "low_wind")]["mape"] for m in METHODS]
    off = [by_month[(mo, "offshore", m, "low_wind")]["mape"] for m in METHODS]
    print(f"{mo:<7}" + "".join(f"{x:>12.1f}" for x in on) + "   | " + "  ".join(f"{x:5.1f}" for x in off))

fit = d["fit"]
out = dict(fit=fit, months=months, winters=winters, methods=METHODS,
           pooled={"|".join(k): v for k, v in pooled.items()},
           by_winter={"|".join(k): v for k, v in by_winter.items()},
           by_month={"|".join(k): v for k, v in by_month.items()},
           pattern=summary)
json.dump(out, open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}")
