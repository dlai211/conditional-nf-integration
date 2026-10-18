"""Measure how a training/evaluation step splits between the flow and the matrix element.

Why this exists
---------------
Porting the flow to Alpaka is only worth doing in proportion to the flow's share
of the wall clock. Indirect evidence says that share is small (moving the flow
from CPU to an RTX 3060 changed a smoke run from 303 s to 304 s), but the
indirect evidence does not say *how* small, and does not say how the share
changes with dimension.

This script measures it directly, per process and per batch size:

    flow  = sample from the proposal + evaluate log q + backward through log q
    ME    = evaluate the MadGraph matrix element on the same batch

Both are timed on the same points, so the ratio is the number that matters:
if ME/(flow+ME) is 95 %, an infinitely fast flow buys 5 %.

Run from inside the standalone directory, like every MadGraph driver:

    cd generated/uu_ww_4l_standalone
    PYTHONPATH=<repo> python <repo>/scripts/profile_flow_vs_matrix_element.py \
        --config <repo>/configs/processes/uu_ww_4l.json \
        --checkpoint <repo>/results/uu_ww_4l_copula_thorough_gpu/madgraph_uu_ww_4l_8_final_model.pt \
        --device cuda --points 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
# `run_adaptive_integrand_structure_search` lives in experiments/, which is on
# sys.path when the experiment scripts run but not when this one does.
for _path in (PROJECT, PROJECT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_adaptive_integrand_structure_search as adaptive
from madgraph_process_integrand import ConfiguredMadGraphIntegrand

COND_DIM = 3


def condition_features(z: float) -> np.ndarray:
    return np.array([z, z * z, math.log1p(z) / math.log(2.0)], dtype=np.float32)


def load_model(checkpoint_path: Path, dimension: int, device: torch.device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    adaptive.DIM = dimension
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeats):
    best = float("inf")
    out = None
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--energy", type=float, default=None)
    parser.add_argument("--uniform-fraction", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not Path("Cards").is_dir():
        raise SystemExit("run from inside the standalone directory (Cards/ not found)")

    spec = json.loads(args.config.read_text())
    evaluator = ConfiguredMadGraphIntegrand(spec, args.root)
    dimension = evaluator.dimension

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, dimension, device)

    energy = args.energy
    if energy is None:
        energy = float(spec["condition"]["held_out"][0])
    minimum = float(spec["condition"]["minimum"])
    maximum = float(spec["condition"]["maximum"])

    z = (energy - minimum) / (maximum - minimum)
    feature = torch.as_tensor(condition_features(z), device=device).reshape(1, COND_DIM)
    cond = feature.expand(args.points, COND_DIM)

    print(f"process      = {spec['name']}")
    print(f"dimension    = {dimension}")
    print(f"device       = {device}")
    print(f"points       = {args.points:,}")
    print(f"sqrt(s_hat)  = {energy:g} GeV")
    print()

    # --- flow: sample + log q ------------------------------------------------
    def do_sample():
        with torch.no_grad():
            return adaptive.sample_from_proposal(
                proposal_model=model, n=args.points, cond=cond, device=device,
                uniform_fraction=args.uniform_fraction,
            )

    t_sample, (unit, logq) = timed(do_sample, device, args.repeats)

    # --- flow: log_prob forward ---------------------------------------------
    # Use the sampled points; they are the batch a KL step actually scores.
    x = unit.detach().requires_grad_(True)

    def do_logprob():
        with torch.enable_grad():
            out = model.log_prob(x, cond)
            return out

    t_logprob, logprob = timed(do_logprob, device, args.repeats)

    # --- flow: backward through log_prob ------------------------------------
    def do_backward():
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        with torch.enable_grad():
            loss = -model.log_prob(x, cond).reshape(-1).mean()
            loss.backward()
        return float(loss.detach())

    t_backward, _ = timed(do_backward, device, args.repeats)

    # --- matrix element on the same points ----------------------------------
    unit_numpy = unit.detach().cpu().double().numpy()
    energies = np.full(args.points, energy, dtype=np.float64)
    t_me, _ = timed(lambda: evaluator.log_integrand(unit_numpy, energies), torch.device("cpu"), args.repeats)

    flow = t_sample + t_logprob + t_backward
    total = flow + t_me

    def line(name, seconds):
        share = 100.0 * seconds / total
        rate = args.points / seconds if seconds > 0 else float("inf")
        print(f"{name:<28} {seconds * 1e3:>10.1f} ms {share:>8.1f}% {rate:>14,.0f}/s")

    print(f"{'stage':<28} {'time':>13} {'share':>9} {'throughput':>15}")
    print("-" * 70)
    line("flow: sample from proposal", t_sample)
    line("flow: log_prob forward", t_logprob)
    line("flow: log_prob backward", t_backward)
    line("flow TOTAL", flow)
    line("matrix element (CPU)", t_me)
    print("-" * 70)
    print(f"{'per batch (flow + ME)':<28} {total * 1e3:>10.1f} ms {100.0:>8.1f}%")
    print()
    print(f"Matrix element is {100.0 * t_me / total:.1f}% of the step.")
    print(f"A free (zero-cost) flow would speed this step up by {total / t_me:.2f}x.")
    print(f"A 10x faster flow would speed it up by {total / (t_me + flow / 10.0):.2f}x.")


if __name__ == "__main__":
    main()
