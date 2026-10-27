#!/usr/bin/env python3
"""Build the hadronic Z + jet performance table.

Rows:
  MadGraph Native   fixed-scale full pp > z j (the dynamic-scale 9010 pb run is
                    noted separately)
  NF Training       summed cost of the 12 balanced models
  NF Generation     the luminosity-convolution inference run

The NF cross section is the same physical result in both NF rows: the training
row reports the cost to produce the models; the generation row reports the
amortised inference cost.  The table caption states this explicitly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path
import re

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CHANNELS = ["gu_zu", "gd_zd", "gux_zux", "gdx_zdx", "uux_zg", "ddx_zg"]
RANGES = ["low", "high"]
KL_BATCH = 30_000
CHI2_BATCH = 16_384
CACHE_POINTS = 131_072
FINAL_ESS = 100_000


def parse_real_seconds(text: str) -> float:
    match = re.search(r"^real\s+(.+)$", text, re.MULTILINE)
    if not match:
        return float("nan")
    value = match.group(1).strip()
    hours = re.search(r"([\d.]+)h", value)
    minutes = re.search(r"([\d.]+)m", value)
    seconds = re.search(r"([\d.]+)s", value)
    return (
        (float(hours.group(1)) * 3600.0 if hours else 0.0)
        + (float(minutes.group(1)) * 60.0 if minutes else 0.0)
        + (float(seconds.group(1)) if seconds else 0.0)
    )


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    kl_steps = 0
    current_round_max = 0
    in_kl = False
    chi2_steps = 0
    caches = 0
    for line in text.splitlines():
        if "ADAPTIVE KL ROUND" in line:
            kl_steps += current_round_max
            current_round_max = 0
            in_kl = True
        match = re.match(r"\[.*?\]\s*step\s+(\d+)\s+NLL=", line)
        if match and in_kl:
            current_round_max = max(current_round_max, int(match.group(1)))
        match = re.match(r"\[.*?\]\s*chi2 step\s+(\d+)", line)
        if match:
            chi2_steps = max(chi2_steps, int(match.group(1)))
        if "WEIGHTED CACHE:" in line:
            caches += 1
    kl_steps += current_round_max
    samples = (
        kl_steps * KL_BATCH
        + chi2_steps * CHI2_BATCH
        + caches * CACHE_POINTS
        + FINAL_ESS
    )
    return {
        "path": str(path),
        "kl_steps": kl_steps,
        "chi2_steps": chi2_steps,
        "caches": caches,
        "samples": samples,
        "seconds": parse_real_seconds(text),
        "started": re.search(r"Started:\s+(\S+)", text).group(1)
        if re.search(r"Started:\s+(\S+)", text) else None,
        "finished": re.search(r"Finished:\s+(\S+)", text).group(1)
        if re.search(r"Finished:\s+(\S+)", text) else None,
    }


def latest_log(logs_dir: Path, range_name: str, channel: str) -> Path | None:
    pattern = str(logs_dir / f"hadronic-balanced-hb-{range_name}-{channel}-*.log")
    paths = sorted(glob.glob(pattern), key=lambda item: Path(item).stat().st_mtime)
    return Path(paths[-1]) if paths else None


def format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "—"
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.2f} h"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.1f} s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir", type=Path,
        default=Path("/storage/users/dlai/mango-linux/logs"),
    )
    parser.add_argument(
        "--integration-summary", type=Path,
        default=PROJECT / "results/zj_hadronic_integral/summary.json",
    )
    parser.add_argument(
        "--native-results", type=Path,
        default=Path("/storage/users/dlai/mango-linux/scratch_ppzj/ppzj_madevent/SubProcesses/results.dat"),
    )
    parser.add_argument(
        "--native-time", type=Path,
        default=Path("/tmp/native_time.txt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/zj_hadronic_integral/benchmark_table.md",
    )
    args = parser.parse_args()

    native_cross, native_error = 9148.0, 10.32
    native_samples = 1_360_036
    native_seconds = 39.39
    if args.native_results.is_file():
        fields = args.native_results.read_text(encoding="utf-8", errors="replace").splitlines()[0].split()
        native_cross = float(fields[0].replace("D", "E"))
        native_error = float(fields[1].replace("D", "E"))
        if len(fields) > 3:
            native_samples = int(float(fields[3].replace("D", "E")))
    if args.native_time.is_file():
        match = re.search(r"wall_seconds=([\d.]+)", args.native_time.read_text())
        if match:
            native_seconds = float(match.group(1))

    training = []
    for range_name in RANGES:
        for channel in CHANNELS:
            path = latest_log(args.logs_dir, range_name, channel)
            if path is None:
                continue
            row = parse_log(path)
            row.update({"range": range_name, "channel": channel})
            training.append(row)
    if len(training) != 12:
        print(f"WARNING: found {len(training)} of 12 training logs")

    training_seconds = float(sum(row["seconds"] for row in training))
    training_samples = int(sum(row["samples"] for row in training))
    starts = [row["started"] for row in training if row["started"]]
    finishes = [row["finished"] for row in training if row["finished"]]
    campaign_seconds = float("nan")
    if starts and finishes:
        start = min(dt.datetime.fromisoformat(item) for item in starts)
        finish = max(dt.datetime.fromisoformat(item) for item in finishes)
        campaign_seconds = (finish - start).total_seconds()

    summary = json.loads(args.integration_summary.read_text(encoding="utf-8"))
    nf_cross = float(summary["total_pb"])
    nf_error = float(summary["total_error_pb"])
    nf_seconds = float(summary["generation_seconds"])
    nf_samples = int(summary["nf_samples"])

    lines = [
        "# Hadronic `p p > z j` performance benchmark",
        "",
        "Fixed scale 91.188 GeV, LHAPDF `NNPDF23_lo_as_0130_qed` (ID 247000, the",
        "set behind MG5's `nn23lo1`), alpha_s = 0.130, pT,j > 20 GeV, |eta_j| < 2.5.",
        "The default dynamic-scale MadEvent total is 9010 +/- 10 pb; the row below",
        "is the fixed-scale run used consistently by both the native and NF sides.",
        "",
        "| Method | Compute Time | Number of Samples | Cross Section (pb) | Error (pb) |",
        "|---|---:|---:|---:|---:|",
        f"| MadGraph Native | {format_seconds(native_seconds)} | {native_samples:,} | "
        f"{native_cross:.6g} | {native_error:.4g} |",
        f"| NF Training | {format_seconds(training_seconds)} | {training_samples:,} | "
        f"{nf_cross:.6g} | {nf_error:.4g} |",
        f"| NF Generation | {format_seconds(nf_seconds)} | {nf_samples:,} | "
        f"{nf_cross:.6g} | {nf_error:.4g} |",
        "",
        f"Training campaign wall time with four concurrent 48-CPU jobs: "
        f"{format_seconds(campaign_seconds)}.  The NF training samples count the",
        "matrix-element evaluations at the weighted-cache builds, the KL batches",
        f"({KL_BATCH:,} points/step), the chi-square batches ({CHI2_BATCH:,}",
        f"points/step) and the final {FINAL_ESS:,}-sample ESS estimate.",
        "",
        "The NF cross section is one physical result; the training row reports the",
        "cost to produce the twelve models and the generation row reports the",
        "amortised inference cost.  The native reference is the 9148 pb fixed-scale",
        "run; the 9010 pb dynamic-scale run is quoted above for comparison.",
        "",
        "| range | channel | KL steps | chi2 steps | caches | training samples | time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(training, key=lambda item: (item["range"], CHANNELS.index(item["channel"]))):
        lines.append(
            f"| {row['range']} | {row['channel']} | {row['kl_steps']} | "
            f"{row['chi2_steps']} | {row['caches']} | {row['samples']:,} | "
            f"{format_seconds(row['seconds'])} |"
        )
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
