"""Is the v5 model's raw kt-prediction good on its own, BEFORE the exact monthly
rescale forces each month's total to match truth? Answers a real gap: the only
pre-vs-post-rescale comparison in the repo (rmse_maps.py vs rmse_maps_rescaled.py)
covers v1/v2 only, predating v4/v5 and the accumulation-window fix. Reuses the
Germany v5 infra directly so the post-rescale numbers here are comparable to
solar/README's Key Finding 3 table (same model, same region, same period).

p = kt_pred * tisr               (pre-rescale -- raw model output)
q = constrained_rescale(p, ...)  (post-rescale -- what Key Finding 1 reports)

Promoted out of scratch/ since its numbers are quoted in solar/README.md Key
Finding 2; was previously an ad hoc check only.

Usage: python3 prerescale_vs_postrescale_germany.py
Output: models/model_v5/evaluation/prerescale_vs_postrescale_germany_v5.npz
"""
import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)

import os, sys
import numpy as np
import joblib

sys.path.insert(0, _os.path.dirname(_root) + "/dunkelflaute/scripts/era5")
from compute_germany_solar_cf_v5_sepmar import (load_month, germany_land_mask, month_available,
                                                  MONTHS, YEARS, FINAL_FEATURES, DAY_MU,
                                                  TRUE3H_SECONDS, KMAX)
from rescale import constrained_rescale
from solar_geometry import SOLAR_CONSTANT

MODEL = joblib.load("models/model_v5/model_kt_v5.joblib")["model"]


def reconstruct_both(m, model):
    ntime, nlat, nlon = m["tsr"].shape
    ncell = nlat * nlon
    tisr = m["tisr"].reshape(ntime, ncell).astype("float64")
    tisr_flat = m["tisr"].ravel()
    mu = tisr_flat / (TRUE3H_SECONDS * SOLAR_CONSTANT)
    day = mu > DAY_MU
    feat = {
        "tcc": m["tcc"].ravel()[day], "hcc": m["hcc"].ravel()[day],
        "mcc": m["mcc"].ravel()[day], "lcc": m["lcc"].ravel()[day],
        "mu": mu[day], "T": np.clip(m["tsr"].ravel()[day] / tisr_flat[day], 0, 1.5),
    }
    X = np.column_stack([feat[c] for c in FINAL_FEATURES])
    kt_pred = np.clip(model.predict(X), 0, 1.5)

    p = np.zeros(ntime * ncell, dtype="float64")
    p[day] = kt_pred * tisr_flat[day]
    p = p.reshape(ntime, ncell)

    real = m["ssrd_real"].reshape(ntime, ncell)
    target = real.sum(axis=0)

    hours = (m["times"] - m["times"].astype("datetime64[D]")) / np.timedelta64(1, "h") - 0.5
    local = (hours[:, None] + np.tile(m["lon"], nlat)[None, :] / 15.0) % 24.0
    tw = np.maximum(np.cos(2 * np.pi * (local - 12.0) / 24.0), 0.0) + 1e-9

    q, diag = constrained_rescale(p, tisr, target, kmax=KMAX, twilight_w=tw)
    return p.reshape(ntime, nlat, nlon), q.reshape(ntime, nlat, nlon), diag


def main():
    mask = germany_land_mask()
    w = mask.astype(float)  # cos(lat) weight folded in below via broadcasting against lat
    all_times, pre, post, real_l = [], [], [], []
    for y in YEARS:
        for m in MONTHS:
            if not month_available(y, m):
                continue
            mm = load_month(y, m)
            p, q, diag = reconstruct_both(mm, MODEL)
            cosw = np.cos(np.deg2rad(mm["lat"]))[:, None] * np.ones((1, len(mm["lon"]))) * mask
            wsum = cosw.sum()
            pre.append((p * cosw[None]).sum(axis=(1, 2)) / wsum)
            post.append((q * cosw[None]).sum(axis=(1, 2)) / wsum)
            real_l.append((mm["ssrd_real"] * cosw[None]).sum(axis=(1, 2)) / wsum)
            all_times.append(mm["times"])
            print(f"  {y}-{m:02d} rescale_err={diag['max_rel_sum_err']:.1e}", flush=True)

    t = np.concatenate(all_times)
    pre, post, real = np.concatenate(pre), np.concatenate(post), np.concatenate(real_l)
    # convert J/m2 (3h sum) -> W/m2
    pre_w, post_w, real_w = pre / TRUE3H_SECONDS, post / TRUE3H_SECONDS, real / TRUE3H_SECONDS

    for label, pred in (("PRE-rescale (raw model)", pre_w), ("POST-rescale (what Key Finding 1 reports)", post_w)):
        err = pred - real_w
        rmse = np.sqrt(np.mean(err ** 2))
        bias = err.mean()
        r = np.corrcoef(pred, real_w)[0, 1]
        r2 = 1 - np.sum(err ** 2) / np.sum((real_w - real_w.mean()) ** 2)
        print(f"{label}: n={len(pred)} mean_real={real_w.mean():.2f} W/m2 "
              f"RMSE={rmse:.2f} W/m2 bias={bias:+.2f} W/m2 r={r:.4f} R2={r2:.4f}")

    out = "models/model_v5/evaluation/prerescale_vs_postrescale_germany_v5.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, time=t, pre=pre_w, post=post_w, real=real_w)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
