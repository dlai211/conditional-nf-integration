#!/usr/bin/env python3
"""Plots for the hadronic Z + jet NF campaign.

Three diagnostics:

1. Training loss and ESS against cumulative wall-clock time, one figure per
   sqrt(s-hat) range, for the six subprocess groups.  These require the server
   logs; the final-summary bar chart below works without them.
2. The condition-interpolation check: the six-channel summed partonic cross
   section at 50 evenly spaced sqrt(s-hat) points per range, overlaid with the
   explicit MadEvent evaluations, plus a lower NF/MadEvent ratio panel.
3. Final training/held-out ESS and integration-grid ESS, split by range and
   channel, from the saved summaries and grid CSVs.

The training logs are produced by ``run_hadronic_balanced_one.sbatch`` through
``scripts/timestamp_log.py``, so each line starts with ``[elapsed_seconds]``.
When the logs are present, the script also writes
``training_history_{low,high}.csv`` with the parsed time series.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT = Path(__file__).resolve().parents[1]
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]
CHANNEL_LABELS = {
    "gu_zu": "g(u+c)",
    "gd_zd": "g(d+s)",
    "gux_zux": "g(u~+c~)",
    "gdx_zdx": "g(d~+s~)",
    "uux_zg": "(u+c)(u~+c~)",
    "ddx_zg": "(d+s)(d~+s~)",
}

# Colour-blind-safe categorical slots, assigned in fixed order.  Markers are
# varied too, so identity is never colour-alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
RANGE_COLORS = {"low": "#2a78d6", "high": "#eb6834"}

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
    history_rows: list[dict] = []
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
        for elapsed, loss, ess in kl:
            history_rows.append({
                "range": range_name, "channel": channel, "stage": "kl",
                "elapsed_seconds": elapsed, "value": loss, "ess": ess,
            })
        color = COLORS[index]
        marker = MARKERS[index]
        times = np.array([row[0] for row in kl]) / 60.0
        losses = np.array([row[1] for row in kl])
        ess_kl = np.array([row[2] for row in kl])
        ess_values.extend(ess_kl.tolist())
        # Colour identifies the channel; line style identifies the stage.
        loss_ax.plot(times, losses, color=color, linestyle="-", lw=1.6,
                     label=CHANNEL_LABELS.get(channel, channel))
        ess_ax.plot(times, ess_kl, color=color, linestyle="-", lw=1.4,
                    alpha=0.9)
        if chi2:
            times_c = np.array([row[0] for row in chi2]) / 60.0
            ess_c = np.array([row[2] for row in chi2])
            ess_values.extend(ess_c.tolist())
            for elapsed, loss, ess in chi2:
                history_rows.append({
                    "range": range_name, "channel": channel, "stage": "chi2",
                    "elapsed_seconds": elapsed, "value": loss, "ess": ess,
                })
            ess_ax.plot(times_c, ess_c, color=color, linestyle="--", lw=1.0,
                        alpha=0.85)
        if final:
            t_final, ess_final = final[-1]
            ess_values.append(ess_final)
            history_rows.append({
                "range": range_name, "channel": channel, "stage": "final",
                "elapsed_seconds": t_final, "value": "", "ess": ess_final,
            })
            ess_ax.scatter([t_final / 60.0], [ess_final], color=color,
                           marker=marker, s=30, zorder=5, edgecolor="white",
                           linewidth=0.6)
    if not found:
        plt.close(fig)
        return
    loss_ax.set_ylabel("KL loss (NLL)")
    loss_ax.set_title(
        f"Training trajectory — {range_name} sqrt(s-hat) range\n"
        "Colour = channel. Top: KL NLL. Bottom: ESS "
        "(solid = KL stage, dashed = chi2 stage, dots = final held-out).",
        fontsize=10.5,
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
    stage_handles = [
        Line2D([0], [0], color="0.25", lw=1.6, ls="-", label="KL stage"),
        Line2D([0], [0], color="0.25", lw=1.0, ls="--", label="chi2 stage"),
        Line2D([0], [0], color="0.25", marker="o", ls="none", ms=6,
               label="final held-out"),
    ]
    ess_ax.legend(handles=stage_handles, fontsize=8, frameon=False,
                  loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"loss_ess_{range_name}.{suffix}", dpi=170,
                    bbox_inches="tight")
    plt.close(fig)
    history_path = output / f"training_history_{range_name}.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["range", "channel", "stage", "elapsed_seconds",
                        "value", "ess"],
        )
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"wrote {output / f'loss_ess_{range_name}.png'}")
    print(f"wrote {history_path}")


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


def read_exact_interpolation(directory: Path, scan_energies: np.ndarray | None = None):
    groups: dict[float, list[tuple[float, float]]] = {}
    for channel in CHANNELS:
        path = directory / channel / "comparison.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                energy = float(row["energy_gev"])
                # comparison.csv contains both the low scan [114,250] and the
                # high scan [200,2000].  Keep only the scan that matches the
                # NF interpolation grid, otherwise the high plot gets a dense
                # cluster of low-scan points between 200 and 250 GeV.
                if scan_energies is not None and not np.isclose(
                    energy, scan_energies, rtol=0.0, atol=1e-5
                ).any():
                    continue
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
    exact = read_exact_interpolation(exact_dir, nf[0] if nf is not None else None)
    if nf is None:
        print(f"missing NF interpolation: {nf_path}")
        return
    fig, (ax, ratio_ax) = plt.subplots(
        2, 1, figsize=(7.4, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
    )
    energies, values, errors = nf
    ax.plot(energies, values, color=NF_COLOR, lw=1.8,
            label="conditional NF (six-channel sum)")
    ax.fill_between(energies, values - errors, values + errors,
                    color=NF_COLOR, alpha=0.16, linewidth=0)
    if exact is not None:
        e_energy, e_value, e_error = exact
        mask = (e_energy >= xlim[0]) & (e_energy <= xlim[1])
        ax.errorbar(e_energy[mask], e_value[mask], yerr=e_error[mask],
                    fmt="o", color=EXACT_COLOR, ms=4.2, lw=0.9,
                    capsize=2.0, label="MadEvent (partonic, six-channel sum)")
        # Ratio panel: NF / MadEvent at the exact scan points.
        nf_at_e = np.interp(e_energy[mask], energies, values)
        nf_err_at_e = np.interp(e_energy[mask], energies, errors)
        ratio = nf_at_e / e_value[mask]
        ratio_err = ratio * np.sqrt(
            (nf_err_at_e / nf_at_e) ** 2 + (e_error[mask] / e_value[mask]) ** 2
        )
        ratio_ax.errorbar(e_energy[mask], ratio, yerr=ratio_err,
                          fmt="o", color=EXACT_COLOR, ms=3.6, lw=0.8,
                          capsize=1.8)
        ratio_ax.axhline(1.0, color="0.35", lw=0.8, linestyle=":")
        ratio_ax.set_ylabel("NF / MadEvent")
        ratio_ax.set_ylim(
            min(0.98, float(np.nanmin(ratio)) - 0.004),
            max(1.02, float(np.nanmax(ratio)) + 0.004),
        )
        ratio_ax.grid(True, color="#000000", alpha=0.10, linewidth=0.6)
        ratio_ax.spines["top"].set_visible(False)
        ratio_ax.spines["right"].set_visible(False)
    else:
        ratio_ax.set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_yscale("log")
    ax.set_ylabel(r"six-channel summed partonic $\sigma$ (pb)")
    ax.set_title(
        f"Six-channel summed partonic cross section — {range_name} range "
        f"[{xlim[0]:g}, {xlim[1]:g}] GeV",
        fontsize=11,
    )
    ax.grid(True, which="both", color="#000000", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ratio_ax.set_xlabel(r"$\sqrt{\hat s}$ (GeV)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"interpolation_{range_name}.{suffix}", dpi=170,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output / f'interpolation_{range_name}.png'}")


def read_final_summary(range_name: str, channel: str):
    path = (
        PROJECT / "results"
        / f"zj_hadronic_{range_name}_{channel}_copula_balanced_gpu"
        / f"madgraph_zj_hadronic_{range_name}_{channel}_2_final_summary.csv"
    )
    if not path.is_file():
        return None
    row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
    return float(row["ess"]), float(row["std_logw"])


def read_grid_ess(integration_dir: Path, range_name: str, channel: str):
    path = integration_dir / f"grid_{channel}.csv"
    if not path.is_file():
        return None
    values = [
        float(row["ess"])
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if row["range"] == range_name
    ]
    return float(np.mean(values)) if values else None


def ess_summary_figure(integration_dir: Path, output: Path) -> None:
    """Training/held-out ESS and integration-grid ESS by range and channel."""
    rows = []
    for channel in CHANNELS:
        low = read_final_summary("low", channel)
        high = read_final_summary("high", channel)
        low_grid = read_grid_ess(integration_dir, "low", channel)
        high_grid = read_grid_ess(integration_dir, "high", channel)
        if low is None or high is None:
            print(f"missing final summary for {channel}")
            continue
        rows.append({
            "channel": channel,
            "low_training_ess": low[0],
            "high_training_ess": high[0],
            "low_std_logw": low[1],
            "high_std_logw": high[1],
            "low_integration_ess": low_grid,
            "high_integration_ess": high_grid,
        })
    if not rows:
        return

    labels = [CHANNEL_LABELS.get(row["channel"], row["channel"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, (train_ax, int_ax) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), sharey=True,
    )
    for ax, low_key, high_key, title in (
        (train_ax, "low_training_ess", "high_training_ess",
         "Training / held-out ESS"),
        (int_ax, "low_integration_ess", "high_integration_ess",
         "Integration ESS (60-point grid)"),
    ):
        low_values = [row[low_key] for row in rows]
        high_values = [row[high_key] for row in rows]
        low_bars = ax.bar(x - width / 2, low_values, width,
                          color=RANGE_COLORS["low"], label="low [114, 250] GeV")
        high_bars = ax.bar(x + width / 2, high_values, width,
                           color=RANGE_COLORS["high"], label="high [200, 2000] GeV")
        for bars in (low_bars, high_bars):
            for bar in bars:
                value = bar.get_height()
                if value is not None and np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                            f"{value:.2f}", ha="center", va="bottom",
                            fontsize=6.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.08)
        ax.grid(True, axis="y", color="#000000", alpha=0.10, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    train_ax.set_ylabel("ESS")
    fig.suptitle("ESS by channel and range", fontsize=12)
    fig.legend(
        handles=[
            Patch(color=RANGE_COLORS["low"], label="low [114, 250] GeV"),
            Patch(color=RANGE_COLORS["high"], label="high [200, 2000] GeV"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2,
        frameon=False, fontsize=9,
    )
    fig.text(
        0.5, 0.01,
        "Training/held-out ESS is lower in the high range. Integration ESS is "
        "high in both ranges; the low 114 GeV threshold point is the only "
        "low-ESS grid point.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.90))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"ess_summary.{suffix}", dpi=170,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output / 'ess_summary.png'}")

    csv_path = output.parent / "ess_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


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
    ess_summary_figure(args.interpolation_dir, args.output)


if __name__ == "__main__":
    main()
