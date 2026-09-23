"""Constrained exact monthly rescale (part of the model, not post-processing).

For every cell: find q_t (t = the 3-hourly steps of the month) with
    sum_t q_t == target_sum            exactly (float64)
    0 <= q_t <= kmax * tisr_t          physical ceiling; night (tisr = 0) stays 0
    q_t proportional to the ML prediction p_t wherever the cap is not active
Implemented as iterative water-filling: scale, clip the steps that exceed their cap at the cap,
re-scale the rest to absorb the remainder, repeat.  Cells whose prediction is (nearly) all zero
still get the amount spread over the daylight steps in proportion to tisr (tiny cap-proportional
floor added to p), so no cell is left unscaled.

Arrays are (time, cell).  Returns q and a diagnostics dict.
"""

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, "data")):
    _root = _os.path.dirname(_root)
_os.chdir(_root)
_sys.path.insert(0, _os.path.join(_root, "scripts", "core"))

import numpy as np


def constrained_rescale(p, tisr, target_sum, kmax=1.1, twilight_w=None, max_iter=60):
    """twilight_w (time, cell): weights (largest near local noon) used ONLY for cells whose analytic tisr is zero all
    month but whose real monthly sum is > 0 (polar-night edge: twilight light the analytic geometry does not see)."""
    p = np.asarray(p, dtype="float64")
    tisr = np.asarray(tisr, dtype="float64")
    target = np.asarray(target_sum, dtype="float64")
    tisr_sum = tisr.sum(axis=0)
    twi = (tisr_sum <= 0) & (target > 0)                                     # nothing analytic to scale: twilight cells
    kcell = np.full(target.shape, float(kmax))
    raised = (~twi) & (target > kmax * tisr_sum * (1 + 1e-12))               # ceiling too low for this cell's real total
    kcell[raised] = target[raised] / tisr_sum[raised] * (1 + 1e-9)
    cap = kcell[None, :] * tisr
    p = np.maximum(p, 0.0) + 1e-9 * cap                                      # floor proportional to tisr; only matters if p ~ 0
    q = np.zeros_like(p)
    active = cap > 0
    remaining = np.where(twi, 0.0, target)
    for _ in range(max_iter):
        psum = np.where(active, p, 0.0).sum(axis=0)
        s = np.divide(remaining, psum, out=np.zeros_like(remaining), where=psum > 0)
        cand = s[None, :] * p
        over = active & (cand > cap * (1 + 1e-12))
        if not over.any():
            q = np.where(active, cand, q)
            break
        q = np.where(over, cap, q)
        remaining = np.maximum(remaining - np.where(over, cap, 0.0).sum(axis=0), 0.0)
        active = active & ~over
    else:
        raise RuntimeError("water-filling did not converge")
    if twi.any():
        w = np.ones_like(p) if twilight_w is None else np.asarray(twilight_w, dtype="float64")
        wt = np.where(twi[None, :], w, 0.0)
        q = np.where(twi[None, :], target[None, :] * wt / np.maximum(wt.sum(axis=0, keepdims=True), 1e-300), q)
    got = q.sum(axis=0)
    rel = np.abs(got - target)[target > 0] / target[target > 0]
    normal = ~(twi | raised)
    kt = np.where(tisr > 0, q / np.maximum(tisr, 1e-30), 0.0)
    diag = dict(max_abs_sum_err=float(np.abs(got - target).max()),
                max_rel_sum_err=float(rel.max()) if rel.size else 0.0,
                n_cells=int(target.size), n_ceiling_raised=int(raised.sum()), n_twilight=int(twi.sum()),
                max_kt_normal=float(kt[:, normal].max()) if normal.any() else 0.0,
                max_kt_raised=float(kt[:, raised].max()) if raised.any() else 0.0,
                night_max_normal=float(q[:, normal][tisr[:, normal] <= 0].max()) if (tisr[:, normal] <= 0).any() else 0.0)
    return q, diag


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    T, N = 248, 5000
    tisr = np.maximum(0, np.sin(np.linspace(0, 2 * np.pi * 31, T))[:, None] * rng.uniform(0.2, 1.0, N)[None, :]) * 1e7
    tisr[:, :100] = 0                                                          # polar night cells
    kt = np.clip(rng.uniform(0.05, 0.9, (T, N)), 0, 1.5)
    real = kt * tisr
    p = np.clip(kt * rng.uniform(0.3, 1.8, (T, N)), 0, 1.5) * tisr           # deliberately wrong prediction
    p[:, 200:210] = 0                                                          # cells with an all-zero prediction
    tgt = real.sum(axis=0)
    tgt[300] = tisr[:, 300].sum() * 2.0                                        # impossible under the ceiling -> ceiling raised
    tgt[5] = 1e-3                                                              # polar-night cell with real twilight light
    q, d = constrained_rescale(p, tisr, tgt, kmax=1.1, twilight_w=np.tile(np.maximum(np.cos(np.linspace(0, 2 * np.pi * 31, T)), 0)[:, None], (1, N)))
    print("self-test:", d)
    assert d["max_rel_sum_err"] < 1e-12 and d["night_max_normal"] == 0.0 and d["max_kt_normal"] <= 1.1 + 1e-9
    assert d["n_ceiling_raised"] >= 1 and d["n_twilight"] >= 1
    assert abs(q[:, 5].sum() - 1e-3) < 1e-15 and abs(q[:, 300].sum() - tgt[300]) / tgt[300] < 1e-12
    real_target = real.sum(axis=0)
    print("cell with an all-zero prediction: sum error", abs(q[:, 205].sum() - real_target[205]) / real_target[205])
    print("PASSED")
