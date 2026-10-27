#!/usr/bin/env python3
"""Plots for the hadronic Z + jet NF campaign.

Two diagnostics:

1. Training loss and ESS against cumulative wall-clock time, one figure per
   sqrt(s-hat) range, for the six subprocess groups.
2. The condition-interpolation check: the NF partonic cross section at 50
   evenly spaced sqrt(s-hat) points per range, overlaid with the explicit
   MadEvent evaluations at the same points.

The training logs are produced by ``run_hadronic_balanced_one.sbatch`` through
``scripts/timestamp_log.py``, so each line starts with ``[elapsed_seconds]``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]

# Colour-blind-safe categorical slots, assigned in fixed order.  Line style
# and markers are varied too, so identity is never colour-alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]

NF_COLOR = "#2a78d6"
EXACT_COLOR = "#eb6834"


def parse_training_log(path: Path):
    """Return KL loss/ESS, chi2 ESS, and final ESS records with elapsed time."""
    kl, chi2, final = [], [], []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        # The timestamp filter pads the elapsed time, e.g. "[  163.373]".
        match = re.match(r"\[\s*(?P<t>[0-9.]+)\]\s*(?P<body>.*)", raw)
        if not match:
            continue
        elapsed = float(match.group("t"))
        body = match.group("body")
        kl_match = re.match(
            r"step\s+(\d+)\s+NLL=([-\d.eE+]+)\s+source ESS=([\d.]+)",
            body,
        )
        if kl_match:
            kl.append((elapsed, float(kl_match.group(2)), float(kl_match.group(3))))
            continue
        chi2_match = re.match(
            r"chi2 step\s+(\d+)\s+loss=([-\d.eE+]+)\s+local ESS=([\d.]+)\s+global ESS=([\d.]+)",
            body,
        )
        if chi2_match:
            chi2.append((elapsed, float(chi2_match.group(2)), float(chi2_match.group(4))))
            continue
        final_match = re.match(r"ESS\s+=\s+([\d.]+)", body)
        if final_match:
            final.append((elapsed, float(final_match.group(1))))
    return kl, chi2, final


def latest_log(logs_dir: Path, range_name: str, channel: str) -> Path | None:
    pattern = str(logs_dir / f"hadronic-balanced-hb-{range_name}-{channel}-*.log")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    return Path(paths[-1]) if paths else None


def training_figure(logs_dir: Path, range_name: str, output: Path) -> None:
    fig, (loss_ax, ess_ax) = plt.subplots(
        2, 1, figsize=(7.6, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    found = False
    ess_values: list[float] = []
    for index, channel in enumerate(CHANNELS):
        path = latest_log(logs_dir, range_name, channel)
        if path is None:
            print(f"missing log: {range_name} {channel}")
            continue
        kl, chi2, final = parse_training_log(path)
        if not kl:
            print(f"no KL records: {path}")
            continue
        found = True
        color = COLORS[index]
        style = LINESTYLES[index]
        marker = MARKERS[index]
        times = np.array([row[0] for row in kl]) / 60.0
        losses = np.array([row[1] for row in kl])
        ess_kl = np.array([row[2] for row in kl])
        ess_values.extend(ess_kl.tolist())
        loss_ax.plot(times, losses, color=color, linestyle=style, lw=1.6,
                     label=channel)
        ess_ax.plot(times, ess_kl, color=color, linestyle=style, lw=1.4,
                    marker=marker, markevery=max(1, len(times) // 5), ms=3.2,
                    label=channel)
        if chi2:
            times_c = np.array([row[0] for row in chi2]) / 60.0
            ess_c = np.array([row[2] for row in chi2])
            ess_values.extend(ess_c.tolist())
            ess_ax.plot(times_c, ess_c, color=color, linestyle="--", lw=1.0,
                        alpha=0.85)
        if final:
            t_final, ess_final = final[-1]
            ess_values.append(ess_final)
            ess_ax.scatter([t_final / 60.0], [ess_final], color=color,
                           marker=marker, s=30, zorder=5, edgecolor="white",
                           linewidth=0.6)
    if not found:
        plt.close(fig)
        return
    loss_ax.set_ylabel("KL loss (NLL)")
    loss_ax.set_title(
        f"Training trajectory — {range_name} sqrt(s-hat) range\n"
        "solid: KL NLL; dashed: chi2 fine-tuning ESS; dots: final ESS",
        fontsize=11,
    )
    ess_ax.set_ylabel("ESS")
    ess_ax.set_xlabel("cumulative compute time (min)")
    if ess_values:
        ess_ax.set_ylim(max(0.0, min(ess_values) - 0.05), 1.0)
    else:
        ess_ax.set_ylim(0.0, 1.02)
    for ax in (loss_ax, ess_ax):
        ax.grid(True, color="#000000", alpha=0.10, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    loss_ax.legend(ncol=2, fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"loss_ess_{range_name}.{suffix}", dpi=170,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output / f'loss_ess_{range_name}.png'}")


def read_nf_interpolation(path: Path):
    if not path.is_file():
        return None
    groups: dict[float, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            energy = float(row["sqrts_hat_gev"])
            groups.setdefault(energy, []).append(
                (float(row["nf_pb"]), float(row["nf_err_pb"]))
            )
    energies = np.array(sorted(groups))
    values = np.array([sum(item[0] for item in groups[e]) for e in energies])
    errors = np.array(
        [float(np.sqrt(sum(item[1] ** 2 for item in groups[e]))) for e in energies]
    )
    return energies, values, errors


def read_exact_interpolation(directory: Path):
    groups: dict[float, list[tuple[float, float]]] = {}
    for channel in CHANNELS:
        path = directory / channel / "comparison.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                energy = float(row["energy_gev"])
                groups.setdefault(energy, []).append(
                    (float(row["madevent_pb"]), float(row["madevent_error_pb"]))
                )
    if not groups:
        return None
    energies = np.array(sorted(groups))
    values = np.array([sum(item[0] for item in groups[e]) for e in energies])
    errors = np.array(
        [float(np.sqrt(sum(item[1] ** 2 for item in groups[e]))) for e in energies]
    )
    return energies, values, errors


def interpolation_figure(nf_path: Path, exact_dir: Path, range_name: str,
                         output: Path, xlim: tuple[float, float]) -> None:
    nf = read_nf_interpolation(nf_path)
    exact = read_exact_interpolation(exact_dir)
    if nf is None:
        print(f"missing NF interpolation: {nf_path}")
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    energies, values, errors = nf
    ax.plot(energies, values, color=NF_COLOR, lw=1.8, label="conditional NF")
    ax.fill_between(energies, values - errors, values + errors,
                    color=NF_COLOR, alpha=0.16, linewidth=0)
    if exact is not None:
        e_energy, e_value, e_error = exact
        mask = (e_energy >= xlim[0]) & (e_energy <= xlim[1])
        ax.errorbar(e_energy[mask], e_value[mask], yerr=e_error[mask],
                    fmt="o", color=EXACT_COLOR, ms=4.2, lw=0.9,
                    capsize=2.0, label="MadEvent (partonic)")
    ax.set_xlim(*xlim)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    ax.set_ylabel(r"partonic $\sigma$ (pb)")
    ax.set_title(
        f"Condition interpolation — {range_name} range "
        f"[{xlim[0]:g}, {xlim[1]:g}] GeV",
        fontsize=11,
    )
    ax.grid(True, which="both", color="#000000", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"interpolation_{range_name}.{suffix}", dpi=170,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output / f'interpolation_{range_name}.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir", type=Path,
        default=Path("/storage/users/dlai/mango-linux/logs"),
    )
    parser.add_argument(
        "--interpolation-dir", type=Path,
        default=PROJECT / "results/zj_hadronic_integral",
    )
    parser.add_argument(
        "--madevent-dir", type=Path,
        default=PROJECT / "results/zj_hadronic_madevent_interpolation",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/zj_hadronic_integral/plots",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    training_figure(args.logs_dir, "low", args.output)
    training_figure(args.logs_dir, "high", args.output)
    interpolation_figure(
        args.interpolation_dir / "interpolation_low.csv",
        args.madevent_dir, "low", args.output, (114.0, 250.0),
    )
    interpolation_figure(
        args.interpolation_dir / "interpolation_high.csv",
        args.madevent_dir, "high", args.output, (200.0, 2000.0),
    )


if __name__ == "__main__":
    main()
