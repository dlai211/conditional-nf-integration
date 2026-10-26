"""Dump the telemetry consumed by web_visualizer/.

The web app is static: it must not import torch or MadGraph. This script runs
the model and the matrix element once, downsamples what the browser needs, and
writes ``web_visualizer/telemetry_data.json`` (plus a ``.js`` wrapper so the
page also works when opened directly from disk).

The MadGraph Fortran library resolves ``Cards/ident_card.dat`` relative to the
CWD, so this script must be started from inside the standalone directory:

    cd generated/gg_ttbar_standalone
    PYTHONPATH=C:/Users/laiji/test/conditional-nf-integration \
      "C:/Users/laiji/.conda/envs/workspace/python.exe" \
      C:/Users/laiji/test/conditional-nf-integration/scripts/dump_web_telemetry.py

Output schema (abridged):

    meta          process, checkpoint, device, schedule, grid, sample counts
    conditions    [400, 650, 1000, 1500, 1950]
    stages        per condition: u, x_marg, u_copula, x_copula, kinematic
    annealing     per condition: x0, x1, log10_density[beta][y][x]
    weights       per condition: log10_w, ess_sample, nf/sobol reference values
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
for path in (PROJECT, PROJECT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

import run_adaptive_integrand_structure_search as adaptive  # noqa: E402
from madgraph_process_integrand import (  # noqa: E402
    ConfiguredMadGraphIntegrand,
    load_process_spec,
)

DEFAULT_CONFIG = PROJECT / "configs" / "processes" / "gg_ttbar.json"
DEFAULT_CHECKPOINT = (
    PROJECT
    / "results"
    / "gg_ttbar_copula_smoke_gpu"
    / "madgraph_gg_ttbar_2_final_model.pt"
)
DEFAULT_VALIDATION = PROJECT / "results" / "gg_ttbar_validation_smoke_gpu" / "summary.csv"
DEFAULT_OUTPUT = PROJECT / "web_visualizer" / "telemetry_data.json"
CONDITIONS = (400.0, 650.0, 1000.0, 1500.0, 1950.0)
BETAS = (0.0, 0.08, 0.25, 0.55, 1.0)
N_STAGE = 5000
N_WEIGHTS = 2000
N_GRID = 80
UNIFORM_FRACTION = 0.10
SEED = 20261022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage-points", type=int, default=N_STAGE)
    parser.add_argument("--weight-points", type=int, default=N_WEIGHTS)
    parser.add_argument("--grid-points", type=int, default=N_GRID)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def condition_tensor(energy: float, count: int, minimum: float, maximum: float,
                    device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    z = (energy - minimum) / (maximum - minimum)
    feature = torch.tensor(
        [z, z * z, math.log1p(z) / math.log(2.0)], dtype=dtype, device=device
    )
    return feature.reshape(1, 3).expand(count, -1)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    adaptive.DIM = 2
    model = adaptive.make_real_model(checkpoint["graph"], None, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def marginal_forward(model, values: torch.Tensor, cond: torch.Tensor, index: int):
    out, _ = model.marginals[index].forward(values, cond)
    return out.reshape(-1).clamp(0.0, 1.0)


def stages_for(model, cond: torch.Tensor, base: torch.Tensor):
    """Apply marginals only, then the full copula + marginals, to one base."""
    x_marg = torch.empty_like(base)
    for j in range(2):
        x_marg[:, j] = marginal_forward(model, base[:, j:j + 1], cond, j)

    u_copula = base.clone()
    # StructuredCopula.context(parents=[0]) = [u0, cond].
    context = torch.cat([u_copula[:, [0]], cond], dim=1)
    u_copula[:, 1] = (
        model.copula.conditionals["1"]
        .forward(base[:, 1:2], context)[0]
        .reshape(-1)
        .clamp(0.0, 1.0)
    )
    x_copula = torch.empty_like(base)
    for j in range(2):
        x_copula[:, j] = marginal_forward(model, u_copula[:, j:j + 1], cond, j)
    return x_marg, u_copula, x_copula


def kinematic_observables(evaluator, points: np.ndarray, energy: float):
    final, _ = evaluator.phase_space.map(points, np.full(len(points), energy))
    first = final[:, 0, :]
    pt = np.hypot(first[:, 1], first[:, 2])
    energy_component = np.maximum(first[:, 0], 1e-12)
    pz = np.clip(first[:, 3], -energy_component + 1e-9, energy_component - 1e-9)
    eta = 0.5 * np.log((energy_component + pz) / (energy_component - pz))

    total = final.sum(axis=1)
    mass_squared = total[:, 0] ** 2 - np.sum(total[:, 1:] ** 2, axis=1)
    mtt = np.sqrt(np.maximum(mass_squared, 0.0))
    return {"pt": pt, "eta": eta, "mtt": mtt}


def annealing_grid(evaluator, energy: float, grid_points: int):
    axis = np.linspace(0.0, 1.0, grid_points)
    x0, x1 = np.meshgrid(axis, axis)
    unit = np.stack([x0.ravel(), x1.ravel()], axis=1)
    log_f = evaluator.log_integrand(unit, np.full(len(unit), energy))
    log_f = log_f.reshape(grid_points, grid_points)  # [x1, x0]

    finite = np.isfinite(log_f)
    floor = float(np.min(log_f[finite]) - 4.0) if finite.any() else -50.0
    log_f = np.where(finite, log_f, floor)

    profiles = []
    for beta in BETAS:
        logp = beta * log_f
        peak = float(np.max(logp))
        log_norm = peak + math.log(float(np.sum(np.exp(logp - peak))))
        density = np.exp(logp - log_norm)
        profiles.append(np.log10(np.maximum(density, 1e-12)))
    return axis, profiles


def weight_diagnostics(model, evaluator, cond: torch.Tensor, energy: float,
                       count: int, device: torch.device):
    x, log_r = adaptive.sample_from_proposal(
        proposal_model=model,
        n=count,
        cond=cond,
        device=device,
        uniform_fraction=UNIFORM_FRACTION,
    )
    points = x.detach().cpu().double().numpy()
    log_f = evaluator.log_integrand(points, np.full(count, energy))
    log_w = log_f - log_r.detach().cpu().double().numpy()

    finite = np.isfinite(log_w)
    if not finite.any():
        return {"log10_w": [-12.0] * count, "ess_sample": 0.0, "finite": 0}
    floor = float(np.min(log_w[finite]) - 3.0 * math.log(10.0))
    clipped = np.where(finite, log_w, floor)
    normalized = np.exp(clipped - np.max(clipped))
    ess = float((normalized.sum() ** 2) / (count * np.square(normalized).sum()))
    return {
        "log10_w": [round(float(value), 4) for value in clipped / math.log(10.0)],
        "ess_sample": round(ess, 6),
        "finite": int(finite.sum()),
    }


def reference_rows(summary_path: Path):
    rows: dict[float, dict[str, dict[str, float]]] = {}
    if not summary_path.exists():
        return rows
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            energy = float(row["energy_gev"])
            method = row["method"]
            rows.setdefault(energy, {})[method] = {
                "estimate_pb": float(row["estimate_pb"]),
                "stderr_pb": float(row["stderr_pb"]),
                "ess": float(row["normalized_ess"]) if row["normalized_ess"] else None,
            }
    return rows


def marginal_peaks(model, cond_values, minimum: float, maximum: float, dtype):
    grid = torch.linspace(1e-4, 1.0 - 1e-4, 2000, dtype=dtype, device=cond_values.device)
    peaks = []
    with torch.no_grad():
        for j in range(2):
            _, logdet = model.marginals[j].inverse(grid, cond_values)
            peaks.append(float(torch.exp(-logdet).max()))
    return peaks


def rounded(points: np.ndarray, digits: int = 6):
    return [[round(float(a), digits), round(float(b), digits)] for a, b in points]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = load_process_spec(args.config)
    standalone = Path(spec["standalone"])
    if not standalone.is_absolute():
        standalone = PROJECT / standalone
    cwd = Path.cwd().resolve()
    if cwd != standalone.resolve() and standalone.resolve() not in cwd.parents:
        raise SystemExit(
            f"Run this from inside {standalone} (CWD is {cwd}). "
            "The Fortran library resolves Cards/ident_card.dat relative to the CWD."
        )

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    dtype = next(model.parameters()).dtype
    evaluator = ConfiguredMadGraphIntegrand(spec, PROJECT)
    minimum = float(spec["condition"]["minimum"])
    maximum = float(spec["condition"]["maximum"])
    references = reference_rows(args.validation_summary)

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    stages: dict[str, dict] = {}
    annealing: dict[str, dict] = {}
    weights: dict[str, dict] = {}
    condition_meta: list[dict] = []

    for energy in CONDITIONS:
        key = f"{energy:g}"
        cond_stage = condition_tensor(
            energy, args.stage_points, minimum, maximum, device, dtype
        )
        base = torch.rand(
            (args.stage_points, 2), generator=generator, device=device, dtype=dtype
        )
        with torch.no_grad():
            x_marg, u_copula, x_copula = stages_for(model, cond_stage, base)
        x_marg_np = x_marg.detach().cpu().double().numpy()
        u_copula_np = u_copula.detach().cpu().double().numpy()
        x_copula_np = x_copula.detach().cpu().double().numpy()
        base_np = base.detach().cpu().double().numpy()
        kinematics = kinematic_observables(evaluator, x_copula_np, energy)

        stages[key] = {
            "u": rounded(base_np),
            "x_marg": rounded(x_marg_np),
            "u_copula": rounded(u_copula_np),
            "x_copula": rounded(x_copula_np),
            "kinematic": {
                "pt": [round(float(v), 4) for v in kinematics["pt"]],
                "eta": [round(float(v), 4) for v in kinematics["eta"]],
                "mtt": [round(float(v), 4) for v in kinematics["mtt"]],
            },
        }

        axis, profiles = annealing_grid(evaluator, energy, args.grid_points)
        annealing[key] = {
            "x0": [round(float(v), 6) for v in axis],
            "x1": [round(float(v), 6) for v in axis],
            "log10_density": [
                [[round(float(v), 4) for v in row] for row in profile]
                for profile in profiles
            ],
        }

        cond_weights = condition_tensor(
            energy, args.weight_points, minimum, maximum, device, dtype
        )
        diagnostics = weight_diagnostics(
            model, evaluator, cond_weights, energy, args.weight_points, device
        )
        ref = references.get(energy, {})
        nf = ref.get("conditional_nf")
        sobol = ref.get("sobol_qmc")
        pull = None
        if nf and sobol:
            denominator = math.hypot(nf["stderr_pb"], sobol["stderr_pb"])
            if denominator > 0:
                pull = (nf["estimate_pb"] - sobol["estimate_pb"]) / denominator
        weights[key] = {
            **diagnostics,
            "nf": nf,
            "sobol": sobol,
            "pull": None if pull is None else round(float(pull), 4),
        }

        cond_scalar = condition_tensor(energy, 1, minimum, maximum, device, dtype)
        peaks = marginal_peaks(model, cond_scalar, minimum, maximum, dtype)
        condition_meta.append(
            {
                "energy": energy,
                "marginal_peaks": [round(value, 4) for value in peaks],
            }
        )
        print(
            f"{energy:6.0f} GeV  ESS={diagnostics['ess_sample']:.4f}  "
            f"finite weights={diagnostics['finite']}/{args.weight_points}  "
            f"marginal peaks={peaks[0]:.2f}, {peaks[1]:.2f}",
            flush=True,
        )

    payload = {
        "meta": {
            "process": spec["name"],
            "description": spec.get("description", ""),
            "checkpoint": str(args.checkpoint),
            "validation_summary": str(args.validation_summary),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": str(device),
            "model_dtype": str(dtype).replace("torch.", ""),
            "n_bins": 32,
            "dimension": 2,
            "conditions": list(CONDITIONS),
            "betas": list(BETAS),
            "stage_points": args.stage_points,
            "weight_points": args.weight_points,
            "grid_points": args.grid_points,
            "uniform_fraction": UNIFORM_FRACTION,
            "seed": args.seed,
            "schedule_note": (
                "The beta grid starts at 0.00 for the uniform-base visual; the "
                "smoke training schedule itself starts at 0.02."
            ),
        },
        "conditions": list(CONDITIONS),
        "condition_meta": condition_meta,
        "stages": stages,
        "annealing": annealing,
        "weights": weights,
    }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    output.write_text(text, encoding="utf-8")
    wrapper = output.with_suffix(".js")
    wrapper.write_text("window.TELEMETRY_DATA = " + text + ";\n", encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {wrapper} ({wrapper.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
