#!/usr/bin/env python3
"""Cross-check the hadronic NF integral against the exact partonic scan.

The NF integration and the native MadEvent total disagree by ~1.5x.  This
script removes the NF from that comparison: it convolves the *exact* partonic
MadEvent cross sections (``results/zj_hadronic_madevent_interpolation``) with
the same LHAPDF luminosity used by the NF integration, and compares the result
with the NF grid integral (``results/zj_hadronic_integral/grid_*.csv``).

If the exact convolution agrees with the NF grid, the discrepancy is between
this LHAPDF convolution and the native ``p p > z j`` run, not between the NF
and the matrix element.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
for _path in (PROJECT, PROJECT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import lhapdf

from experiments.integrate_hadronic_process import luminosity
from validate_madgraph_process_integral import GEV2_TO_PB


CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]


def trapz(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    weights = np.zeros_like(x_arr)
    weights[1:-1] = 0.5 * (x_arr[2:] - x_arr[:-2])
    weights[0] = 0.5 * (x_arr[1] - x_arr[0])
    weights[-1] = 0.5 * (x_arr[-1] - x_arr[-2])
    return float(np.sum(weights * y_arr))


def read_exact(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (float(row["energy_gev"]), float(row["madevent_pb"]))
            for row in csv.DictReader(handle)
        ]


def read_grid(path: Path, range_name: str) -> tuple[list[float], list[float], list[float]]:
    energies, values, ess = [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["range"] != range_name:
                continue
            energies.append(float(row["sqrts_hat"]))
            values.append(
                float(row["luminosity"]) * float(row["sigma_gev2"]) * GEV2_TO_PB
            )
            ess.append(float(row["ess"]))
    return energies, values, ess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT / "configs/processes/zj_hadronic_low.json",
    )
    parser.add_argument(
        "--exact-dir", type=Path,
        default=PROJECT / "results/zj_hadronic_madevent_interpolation",
    )
    parser.add_argument(
        "--integration-dir", type=Path,
        default=PROJECT / "results/zj_hadronic_integral",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/zj_hadronic_integral/normalization_audit.csv",
    )
    parser.add_argument("--y-points", type=int, default=200)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    beam = config["beam"]
    pdf = lhapdf.mkPDF(beam["pdf"], int(beam.get("member", 0)))
    q2 = float(beam["scale"]) ** 2
    sqrts = float(beam["sqrt_s"])
    low_max = float(config["condition"]["maximum"])
    y_nodes, y_weights = np.polynomial.legendre.leggauss(args.y_points)
    pairs = {item["name"]: item["pdf_pairs"] for item in config["channels"]}

    rows = []
    for channel in CHANNELS:
        exact = read_exact(args.exact_dir / channel / "comparison.csv")
        low_exact = [(e, v) for e, v in exact if e <= low_max]
        high_exact = [(e, v) for e, v in exact if e >= low_max]
        low_lum = np.array([
            luminosity(pdf, e, sqrts, q2, pairs[channel], y_nodes, y_weights)
            for e, _ in low_exact
        ])
        high_lum = np.array([
            luminosity(pdf, e, sqrts, q2, pairs[channel], y_nodes, y_weights)
            for e, _ in high_exact
        ])
        exact_low = trapz(
            [e for e, _ in low_exact],
            low_lum * np.array([v for _, v in low_exact]),
        )
        exact_high = trapz(
            [e for e, _ in high_exact],
            high_lum * np.array([v for _, v in high_exact]),
        )

        grid_path = args.integration_dir / f"grid_{channel}.csv"
        low_e, low_v, low_ess = read_grid(grid_path, "low")
        high_e, high_v, high_ess = read_grid(grid_path, "high")
        nf_low = trapz(low_e, low_v)
        nf_high = trapz(high_e, high_v)
        exact_total = exact_low + exact_high
        nf_total = nf_low + nf_high
        rows.append({
            "channel": channel,
            "nf_low_pb": nf_low,
            "nf_high_pb": nf_high,
            "nf_total_pb": nf_total,
            "exact_low_pb": exact_low,
            "exact_high_pb": exact_high,
            "exact_total_pb": exact_total,
            "nf_over_exact": nf_total / exact_total,
            "low_integration_ess": float(np.mean(low_ess)),
            "high_integration_ess": float(np.mean(high_ess)),
        })
        print(
            f"{channel:8s} NF={nf_total:9.2f} pb  "
            f"exact={exact_total:9.2f} pb  "
            f"NF/exact={nf_total / exact_total:.5f}"
        )

    total_nf = sum(row["nf_total_pb"] for row in rows)
    total_exact = sum(row["exact_total_pb"] for row in rows)
    low_nf = sum(row["nf_low_pb"] for row in rows)
    high_nf = sum(row["nf_high_pb"] for row in rows)
    low_exact = sum(row["exact_low_pb"] for row in rows)
    high_exact = sum(row["exact_high_pb"] for row in rows)
    print()
    print(f"low  NF={low_nf:.2f} pb  exact={low_exact:.2f} pb  ratio={low_nf/low_exact:.5f}")
    print(f"high NF={high_nf:.2f} pb  exact={high_exact:.2f} pb  ratio={high_nf/high_exact:.5f}")
    print(f"total NF={total_nf:.2f} pb  exact={total_exact:.2f} pb  ratio={total_nf/total_exact:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
