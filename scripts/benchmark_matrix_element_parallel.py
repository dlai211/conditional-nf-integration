"""Measure matrix-element throughput: single core versus a multiprocessing pool.

Why this exists
---------------
The matrix element is evaluated one event per Python call
(``MadGraphMatrixElement.__call__``), which makes it the wall-clock bottleneck
for both training and validation on this pipeline. Moving the flow to a GPU
changed a smoke run from 303 s to 304 s, because the flow is not where the time
is — see ``plan.md`` §2.

The MANGO throughput gate is ">=10x versus a 32-core CPU importance-sampling
reference". Before any Alpaka number can be judged against it, we need to know
how fast the CPU side actually goes as cores are added. That is what this
script measures. It is the baseline, not a competitor to Alpaka.

Run from inside the standalone directory, like every MadGraph driver here
(``lha_read.f`` resolves ``./Cards/`` against the CWD):

    cd generated/gg_ttbar_standalone
    PYTHONPATH=<repo> python <repo>/scripts/benchmark_matrix_element_parallel.py \
        --config <repo>/configs/processes/gg_ttbar.json \
        --points 200000 --workers 1,2,4,8,16

The points are uniform in the unit hypercube at a fixed sqrt(s_hat); this
measures raw evaluation throughput, not an integral, so the value is irrelevant
beyond keeping the matrix element in a well-defined region.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Set once per worker by _init_worker.
_EVALUATOR = None


def _init_worker(config_path: str, root: str, standalone: str) -> None:
    """Build the evaluator once per worker, then reuse it for every chunk.

    Each process needs its own handle on the f2py extension — the module object
    is not shareable across processes — and its own CWD inside the standalone,
    because MadGraph resolves Cards/ relative to the working directory.
    """
    global _EVALUATOR
    os.chdir(standalone)

    from madgraph_process_integrand import ConfiguredMadGraphIntegrand

    spec = json.loads(Path(config_path).read_text())
    _EVALUATOR = ConfiguredMadGraphIntegrand(spec, Path(root))


def _evaluate_chunk(payload):
    unit, sqrt_s = payload
    return float(np.sum(_EVALUATOR.log_integrand(unit, sqrt_s)))


def _time_pool(config_path, root, standalone, unit, sqrt_s, workers, chunk, repeats):
    """Return the best wall-clock time over `repeats` for a pool of `workers`."""
    starts = list(range(0, len(unit), chunk))
    payloads = [
        (unit[start:start + chunk], sqrt_s[start:start + chunk]) for start in starts
    ]

    best = float("inf")
    checksum = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        with Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(config_path, root, standalone),
        ) as pool:
            checksum = sum(pool.map(_evaluate_chunk, payloads))
        best = min(best, time.perf_counter() - t0)
    return best, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="process config JSON")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--energy", type=float, default=None,
                        help="sqrt(s_hat) in GeV; defaults to the first held-out point")
    parser.add_argument("--workers", default="1,2,4,8,16",
                        help="comma-separated core counts to measure")
    parser.add_argument("--chunk", type=int, default=4_000,
                        help="points per task; smaller balances better, larger has less overhead")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat each configuration and keep the best time")
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()

    if not Path("Cards").is_dir():
        raise SystemExit(
            "Cards/ not found in the current directory. Run this from inside the "
            "standalone, e.g. cd generated/gg_ttbar_standalone"
        )

    spec = json.loads(Path(args.config).read_text())
    root = str(Path(args.root).resolve())
    standalone = str((Path(args.root) / spec["standalone"]).resolve())

    from madgraph_process_integrand import ConfiguredMadGraphIntegrand

    evaluator = ConfiguredMadGraphIntegrand(spec, Path(args.root))
    dim = evaluator.dimension

    energy = args.energy
    if energy is None:
        energy = float(spec["condition"]["held_out"][0])

    rng = np.random.default_rng(args.seed)
    unit = rng.random((args.points, dim))
    sqrt_s = np.full(args.points, energy, dtype=np.float64)

    cores = os.cpu_count() or 1
    worker_counts = [int(w) for w in args.workers.split(",") if w.strip()]

    print(f"process          = {spec['name']}")
    print(f"dimension        = {dim}")
    print(f"points           = {args.points:,}")
    print(f"sqrt(s_hat)      = {energy:g} GeV")
    print(f"cores available  = {cores}")
    print(f"chunk            = {args.chunk:,} points/task")
    print()

    # In-process baseline. Run first, while the machine is otherwise idle.
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        evaluator.log_integrand(unit, sqrt_s)
        times.append(time.perf_counter() - t0)
    baseline = min(times)
    baseline_rate = args.points / baseline

    print(f"{'workers':>8} {'seconds':>10} {'evals/s':>14} {'speedup':>9} {'efficiency':>11}")
    print("-" * 56)
    print(f"{1:>8} {baseline:>10.2f} {baseline_rate:>14,.0f} {1.00:>9.2f} {1.0:>10.1%}")

    for workers in worker_counts:
        if workers == 1:
            continue
        if workers > cores:
            print(f"{workers:>8}  skipped: only {cores} cores available")
            continue
        elapsed, _ = _time_pool(
            args.config, root, standalone, unit, sqrt_s, workers, args.chunk, args.repeats
        )
        rate = args.points / elapsed
        speedup = baseline / elapsed
        efficiency = speedup / workers
        print(
            f"{workers:>8} {elapsed:>10.2f} {rate:>14,.0f} "
            f"{speedup:>9.2f} {efficiency:>10.1%}"
        )

    print()
    print("Baseline row is in-process (no pool overhead, no pickling). Pool rows")
    print("include inter-process transfer of the sample array, which is small")
    print("relative to the matrix-element cost but not zero.")


if __name__ == "__main__":
    main()
