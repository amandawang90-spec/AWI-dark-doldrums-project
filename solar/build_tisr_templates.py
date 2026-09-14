"""Build two reusable analytic-tisr templates for TCo1279-DART: one ordinary
year (365 days), one leap year (366 days), at DART's real native grid and its
real 3-hour accumulation window. Every actual month in both TCo1279-DART-1950C
(1950-1969) and TCo1279-DART-2080C (2080-2092) reuses whichever of these two
matches its calendar type -- see match_dart_timestamp() below -- instead of
being recomputed or stored 396 times over.

WHY TWO TEMPLATES ARE ENOUGH, NOT 33 YEARS
-------------------------------------------------------------------------------
tisr depends only on latitude, longitude, time-of-day and day-of-year. The
calendar YEAR itself (1950 vs 2080) plays no role in the astronomy -- orbital
drift operates on 10,000+ year timescales. The only thing that differs between
individual years is whether they're a leap year, which shifts which calendar
date maps to which day-of-year from March onward. DART's own files confirmed
this is a real, standard Gregorian calendar (Feb 1953 = 28 days, Feb 1956 and
1964 = 29 days), so two templates, matched by leap-year status, are exact, not
an approximation.

WHY NOT JUST RECOMPUTE ON THE FLY EVERY TIME
-------------------------------------------------------------------------------
Tried first: one month at full 6.6M-cell resolution, unchunked, was still
running after 3.5 minutes with memory past 40 GB and climbing -- killed before
it finished. Two causes, both fixed here:
  1. The grid-outer-product implementation doesn't apply to DART's grid at all
     -- it's a flat list of paired (lat_i, lon_i) cells, not a lat x lon
     rectangle (see cos_zenith_cells in solar_geometry.py).
  2. Holding the whole (nt, 6.6M) array in memory at once is what caused the
     blow-up. This script chunks over cells (CHUNK_CELLS) and parallelises
     across chunks (N_WORKERS) instead.
float32 throughout the heavy arrays (~2x speedup, cos() benchmarked at 317M
vs 75M elements/s float32 vs float64 on this node) plus 24-way parallelism
(load average ~27/256 at build time -- considerate headroom) brings both
templates from a ~4.3 hour single-threaded, float64 estimate down to minutes.

OUTPUT
------
data/analytical_tisr/tisr_template_365day.nc   (ordinary year, reference 2001)
data/analytical_tisr/tisr_template_366day.nc   (leap year,     reference 2000)
Both on DART's real native grid (6,599,680 cells, read from an actual DART
file so it's guaranteed identical to what the reconstruction will index into),
dt_seconds=10800 matching DART's real tsr/ssrd 3-hour accumulation window
(confirmed empirically -- see reconstruct_ssrd.py's DT comment).

Usage: python3 build_tisr_templates.py
"""

import itertools
import os
import shutil
import time
from multiprocessing import Pool

import netCDF4 as nc
import numpy as np
import xarray as xr

from solar_geometry import toa_irradiance_accumulated_cells

OUT_DIR = "data/analytical_tisr"
# NOT /tmp: on compute nodes that's a tmpfs capped at 63 GB (the login node's
# /tmp is a real 218 GB disk -- easy to miss, since it looked fine there).
# Dispatching all jobs to the pool at once meant up to N_WORKERS*chunk-size
# temp files could exist simultaneously, and near the end of a template a
# burst of near-simultaneous finishes outran the 63 GB cap -- 10-11 chunks per
# template failed with "not enough free space" and were silently left as fill
# value in the output (silently in the sense that the run didn't crash, but
# the .nc files ARE missing ~15-17% of their cells -- check any old output
# against `failed` before trusting it). Fixed two ways: this uses the
# project's scratch filesystem instead (effectively uncapped), AND the
# dispatch loop below now caps in-flight jobs at N_WORKERS so at most that
# many temp files can exist at once regardless of which filesystem holds them.
TMP_DIR = f"/scratch/a/{os.environ.get('USER', 'a270321')}/tisr_build_chunks"
GRID_SOURCE = ("/work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C/"
               "outdata/oifs/atm_reduced_3h_tcc_3h_195001-195001.nc")
DT_SECONDS = 10800.0     # DART's real tsr/ssrd accumulation window (3 h)

# 8->4 substeps roughly halves compute; checked against a 16-substep reference
# first: RMSE 3.6 W/m2-equivalent, max diff 22 W/m2 at the single worst point
# of the whole grid -- small next to this method's existing multi-W/m2 to
# tens-of-W/m2 tolerance, worth it given the time pressure to get this built.
N_SUBSTEPS = 4
CHUNK_CELLS = 100_000

# History: on the shared LOGIN node (~90 interactive users, load ~27-29),
# parallelism didn't help at all -- 5 chunks under 24 workers took as long as
# 1 chunk alone. Moved to SLURM's `shared` partition next, which reserves
# CPUs, but that node itself turned out to be nearly full (248/256 cores
# already allocated to other users' jobs) AND its 940 MB/CPU memory ratio led
# to an actual OUT_OF_MEMORY kill. Moved to `compute` with a whole node
# exclusively next (257 GB, no other job's cores) and tried 64 workers -- that
# ALSO hit OUT_OF_MEMORY (sacct showed the batch step's real state), because
# each worker holds several full (nt, CHUNK_CELLS) float32 arrays at once --
# acc plus the transient temporaries inside cos_zenith_cells's expression
# evaluation. Measured PRECISELY in isolation (resource.getrusage, no
# contention): 8.02 GB peak for one chunk at real size (100,000 cells x 2920
# steps). 24 concurrent workers -> ~192 GB steady-state, which fits in the
# 257 GB node, but repeated real runs at 24 (and even 16) hit hard OOM kills
# anyway (confirmed via bash's own "Killed" message -- SIGKILL from the
# kernel, not a Python exception) -- transient spikes during array-expression
# evaluation (several temporaries alive briefly before GC) push the real PEAK
# well above the 8 GB steady-state figure when many workers hit that spike
# together. Worse: a worker OOM-killed mid-task leaves multiprocessing.Pool
# waiting forever for a result that will never arrive, with NO error printed
# -- a silent hang, not a crash, which is what made this so slow to diagnose.
# --mem=0 in the sbatch script fixed a SEPARATE real bug (an implicit ~117 GB
# cap even under --exclusive) but wasn't sufficient alone. 12 workers gives a
# generous ~2.5x safety margin (241 GB granted / 12 ~= 20 GB per worker) over
# the measured 8 GB steady-state peak.
# DROPPED to 4 after 12, 16, 24, and even 64 all failed the SAME way
# despite three different fixes (--mem=0, scratch-dir temp files,
# maxtasksperchild=1): a chunk of chunks around the ~2nd-wave mark always
# stalls, memory stays low when it happens (ruling out OOM), and it
# persists even with a guaranteed-fresh process per task (ruling out
# worker-reuse state corruption) -- something about this node/account's
# environment throttles sustained concurrency above roughly this level
# in a way not yet root-caused. 4 is the ONE configuration DIRECTLY
# tested with zero failures, across 4 full waves (16 real-size chunks,
# maxtasksperchild=1) -- not extrapolated from a smaller test like the
# earlier 12/16/24 choices were. Slower (~4x fewer parallel chunks), but
# proven, and getting a complete, correct file matters more here than
# shaving minutes off a two-run job.
# 1 = fully sequential, no multiprocessing at all (see the N_WORKERS<=1
# branch below) -- the only configuration with a perfect track record.
# Slower (~50-60 min for both templates vs the ~15 min hoped for with real
# parallelism) but every single chunk is guaranteed to actually complete.
N_WORKERS = 1
CHUNK_TIMEOUT_S = 300     # generous vs the ~20-60s/chunk seen so far; a hang past
                          # this is reported by name instead of hanging forever

TEMPLATES = [(2001, 365, "365day"), (2000, 366, "366day")]   # (ref year, ndays, tag)


def _worker(args):
    """Compute one chunk and save it to a LOCAL file -- returning only a small
    index/path tuple, not the ~1 GB array. First attempt returned the array
    through multiprocessing.Pool's own pickle-based pipe and stalled: most of
    24 workers sat at ~0% CPU after 8+ minutes with not one chunk written,
    because Pool.imap collects every result through a single pickled channel,
    and that channel is the bottleneck for GB-sized numpy arrays, not the CPU.
    """
    idx, lat_chunk, lon_chunk, times = args
    result = toa_irradiance_accumulated_cells(lat_chunk, lon_chunk, times,
                                              dt_seconds=DT_SECONDS, n_substeps=N_SUBSTEPS)
    path = f"{TMP_DIR}/chunk_{idx:04d}.npy"
    np.save(path, result)
    return idx, path


def build_one(ref_year, ndays, tag, lat, lon, n_cells):
    """Two fully SEPARATE phases, on purpose:
      1. COMPUTE all chunks via the Pool, writing each to its own temp file.
         NO netCDF4/HDF5 file is open anywhere near the Pool during this phase.
      2. WRITE everything into the target .nc file, sequentially, in this
         process alone, AFTER the Pool has fully shut down.
    Why: every fork-based attempt with the target file open DURING computation
    (writing chunks into it as they completed, interleaved with maxtasksperchild=1
    continuously forking new workers) stalled partway through, consistently
    memory-healthy but permanently stuck -- the documented "HDF5/fork" hazard,
    which manifests specifically when a fork happens while HDF5 has active
    internal state (mid-write, B-tree/chunk-cache updates), not merely while a
    file is open-and-idle. Decoupling means no fork ever happens while HDF5 is
    doing anything, at the cost of needing ~77 GB of scratch space to hold all
    66 chunks before the write pass (fine on the project's scratch filesystem).
    """
    target = f"{OUT_DIR}/tisr_template_{tag}.nc"
    log(f"--- {tag} (reference year {ref_year}, {ndays} days) -> {target}")

    times = (np.datetime64(f"{ref_year}-01-01T03:00:00", "s")
             + np.arange(ndays * 8).astype("timedelta64[h]") * 3)
    nt = times.size
    seconds_since_start = (times - times[0]).astype("timedelta64[s]").astype("float64")

    bounds = list(range(0, n_cells, CHUNK_CELLS)) + [n_cells]
    jobs = [(i, lat[bounds[i]:bounds[i + 1]], lon[bounds[i]:bounds[i + 1]], times)
            for i in range(len(bounds) - 1)]
    os.makedirs(TMP_DIR, exist_ok=True)
    total = len(jobs)

    # ---------- PHASE 1: compute only, no netCDF4 file open anywhere ----------
    t0 = time.time()
    done = 0
    failed = []
    chunk_paths = {}

    if N_WORKERS <= 1:
        # NO multiprocessing at all -- not Pool(1), a plain function call in a
        # for loop. Every multiprocessing configuration tried (Pool sizes from
        # 4 to 64, maxtasksperchild, spawn vs fork, with/without the 2-phase
        # netCDF split above) eventually stalled a handful of chunks in, at a
        # DIFFERENT chunk index each run (24, then 20, then 39) -- the
        # non-determinism itself points to a rare race in repeated forking,
        # not a fixable resource threshold. Direct single-process calls were
        # the only thing with a perfect track record across every test run
        # (3 isolated trials, one clean 40-chunk sequential-equivalent run) --
        # slower, but every chunk is guaranteed to actually run to completion.
        for job in jobs:
            idx = job[0]
            try:
                _, path = _worker(job)
                chunk_paths[idx] = path
                done += 1
                log(progress_bar(done, total, t0, label="compute"))
            except Exception as exc:
                log(f"    !!! chunk {idx} FAILED: {exc!r} -- continuing with the rest")
                failed.append(idx)
    else:
        with Pool(N_WORKERS, maxtasksperchild=1) as pool:
            job_iter = iter(jobs)
            pending = {}

            def _submit_next():
                job = next(job_iter, None)
                if job is not None:
                    pending[job[0]] = pool.apply_async(_worker, (job,))

            for job in itertools.islice(job_iter, N_WORKERS):
                pending[job[0]] = pool.apply_async(_worker, (job,))
            while pending:
                for idx in list(pending):
                    res = pending[idx]
                    if not res.ready():
                        continue
                    del pending[idx]
                    try:
                        _, path = res.get(timeout=1)
                        chunk_paths[idx] = path
                        done += 1
                        log(progress_bar(done, total, t0, label="compute"))
                    except Exception as exc:
                        log(f"    !!! chunk {idx} FAILED: {exc!r} -- continuing with the rest")
                        failed.append(idx)
                    _submit_next()
                if pending:
                    time.sleep(2)
                    stuck = [i for i, r in pending.items()
                             if time.time() - t0 > CHUNK_TIMEOUT_S]
                    if stuck:
                        log(f"    !!! chunk(s) {stuck} exceeded {CHUNK_TIMEOUT_S}s -- "
                            f"likely dead (OOM?) worker, abandoning and continuing")
                        for i in stuck:
                            del pending[i]
                            _submit_next()
                            failed.append(i)
    compute_elapsed = time.time() - t0
    log(f"    compute phase done: {done}/{total} chunks, {compute_elapsed:.0f}s")

    # ---------- PHASE 2: write everything, sequentially, Pool fully closed ----------
    t1 = time.time()
    ds = nc.Dataset(target, "w", format="NETCDF4")
    ds.createDimension("cell", n_cells)
    ds.createDimension("time_counter", nt)
    v_lat = ds.createVariable("lat", "f4", ("cell",), zlib=True, complevel=4)
    v_lon = ds.createVariable("lon", "f4", ("cell",), zlib=True, complevel=4)
    v_time = ds.createVariable("time_counter", "f8", ("time_counter",))
    v_time.units = f"seconds since {ref_year}-01-01 03:00:00"
    v_time.calendar = "gregorian"
    v_tisr = ds.createVariable("tisr", "f4", ("time_counter", "cell"),
                               zlib=True, complevel=4,
                               chunksizes=(nt, min(CHUNK_CELLS, n_cells)))
    v_tisr.units = "J m-2"
    v_tisr.long_name = "TOA incident shortwave, analytic (Spencer 1971 solar geometry)"
    v_tisr.comment = (f"Reusable template: matches any real DART {ndays}-day year "
                       f"by calendar day-of-year and time-of-day; the reference "
                       f"year number {ref_year} itself carries no physical meaning.")
    v_lat[:] = lat
    v_lon[:] = lon
    v_time[:] = seconds_since_start

    for n, idx in enumerate(sorted(chunk_paths)):
        arr = np.load(chunk_paths[idx])
        v_tisr[:, bounds[idx]:bounds[idx + 1]] = arr
        os.remove(chunk_paths[idx])
        if (n + 1) % 10 == 0 or n + 1 == len(chunk_paths):
            log(progress_bar(n + 1, len(chunk_paths), t1, label="write  "))
    ds.close()
    write_elapsed = time.time() - t1

    if failed:
        log(f"    {len(failed)} chunk(s) never completed: {sorted(failed)} -- "
            f"re-run with a smaller N_WORKERS or CHUNK_CELLS to fill these in")
    size_gb = os.path.getsize(target) / 1e9
    elapsed = compute_elapsed + write_elapsed
    status = "INCOMPLETE" if failed else "OK"
    log(f"    done ({status}): {size_gb:.2f} GB, {elapsed:.0f}s total "
        f"({compute_elapsed:.0f}s compute + {write_elapsed:.0f}s write)")
    return elapsed, failed


def progress_bar(done, total, t0, width=30, label=""):
    """One text line: [#####.....] 34/66 (51.5%)  elapsed 5m20s  ETA 5m01s
    Printed fresh every chunk rather than in-place (\\r doesn't render usefully
    in a log file read with cat/tail), so this is 66 lines per template rather
    than one updating bar -- the log itself IS the progress bar here.
    """
    frac = done / total
    filled = int(width * frac)
    bar = "#" * filled + "." * (width - filled)
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done else float("nan")

    def fmt(s):
        m, s = divmod(int(s), 60)
        return f"{m}m{s:02d}s"

    prefix = f"{label} " if label else ""
    return f"    {prefix}[{bar}] {done}/{total} ({100*frac:5.1f}%)  elapsed {fmt(elapsed)}  ETA {fmt(eta)}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.rmtree(TMP_DIR, ignore_errors=True)     # clear any stale chunks from a killed run
    log(f"reading DART's real native grid from {GRID_SOURCE}")
    grid = xr.open_dataset(GRID_SOURCE)
    lat, lon = grid.lat.values.astype("float32"), grid.lon.values.astype("float32")
    n_cells = lat.size
    log(f"grid: {n_cells:,} cells")
    grid.close()

    total_elapsed = 0.0
    any_failed = False
    for i, (ref_year, ndays, tag) in enumerate(TEMPLATES):
        elapsed, failed = build_one(ref_year, ndays, tag, lat, lon, n_cells)
        total_elapsed += elapsed
        any_failed = any_failed or bool(failed)
        if i == 0 and len(TEMPLATES) > 1:
            # first template's rate is the best estimate for the second (same
            # grid, nearly the same day count -- 365 vs 366)
            log(f">>> first template took {elapsed/60:.1f} min -> "
                f"both templates combined, expect roughly {2*elapsed/60:.1f} min total\n")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    if any_failed:
        log(f"\nFINISHED WITH GAPS, {total_elapsed/60:.1f} min total -- "
            f"at least one template is missing real cells; do not use until re-run cleanly")
        raise SystemExit(1)
    log(f"\nboth templates done, {total_elapsed/60:.1f} min total")


if __name__ == "__main__":
    main()
