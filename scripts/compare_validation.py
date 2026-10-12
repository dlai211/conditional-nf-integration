#!/usr/bin/env python3
"""Compare a validation run against the curated reference in results_public/.

Usage
-----
    python scripts/compare_validation.py \
        --mine  results/gg_ttbar_validation_smoke \
        --theirs results_public/gg_ttbar_validation

Both directories must contain a `summary.csv` in the format written by
`experiments/validate_madgraph_process_integral.py`:

    energy_gev,method,estimate_gev2,stderr_gev2,estimate_pb,stderr_pb,
    relative_stderr,normalized_ess,replicate,evaluations,seconds

The reference directory for each process is:

    gg_ttbar   -> results_public/gg_ttbar_validation
    gg_tth     -> results_public/gg_tth_5d_validation
    uu_ww_4l   -> results_public/uu_ww_4l_validation

Where a Sobol column is present on both sides it is compared directly: the
integrals are a property of the generated matrix element and the phase-space
map, not of the trained flow, so agreement there is the end-to-end check on
the build.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    summary = path / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"No summary.csv in {path}")
    frame = pd.read_csv(summary)
    missing = {"energy_gev", "method", "estimate_pb", "normalized_ess"} - set(frame.columns)
    if missing:
        raise ValueError(f"{summary} lacks columns: {sorted(missing)}")
    return frame


def pick(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = frame[frame["method"] == method].copy()
    return selected.sort_values("energy_gev").reset_index(drop=True)


def relative_difference(mine: float, theirs: float) -> float:
    if theirs == 0:
        return float("nan")
    return abs(mine - theirs) / abs(theirs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mine", type=Path, required=True,
                        help="directory holding this run's summary.csv")
    parser.add_argument("--theirs", type=Path, required=True,
                        help="reference directory, e.g. results_public/gg_ttbar_validation")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    mine, theirs = load(args.mine), load(args.theirs)
    title = args.title or f"{args.mine.name}  vs  {args.theirs.name}"
    print(f"\n{title}\n{'=' * len(title)}")

    # ---- Sobol: the build-level check -------------------------------------
    my_sobol, ref_sobol = pick(mine, "sobol_qmc"), pick(theirs, "sobol_qmc")
    if len(my_sobol) and len(ref_sobol):
        merged = my_sobol.merge(
            ref_sobol, on="energy_gev", suffixes=("_mine", "_ref")
        )
        print("\nSobol (validates the matrix element + phase space, not the flow)")
        print(f"{'E [GeV]':>9} {'mine [pb]':>20} {'reference [pb]':>20} {'rel. diff':>12}")
        worst = 0.0
        for row in merged.itertuples():
            diff = relative_difference(row.estimate_pb_mine, row.estimate_pb_ref)
            worst = max(worst, diff)
            print(f"{row.energy_gev:>9.1f} {row.estimate_pb_mine:>20.10g} "
                  f"{row.estimate_pb_ref:>20.10g} {diff:>12.2e}")
        verdict = "IDENTICAL" if worst == 0 else f"max rel. diff {worst:.2e}"
        print(f"  -> {verdict}")

    # ---- NF: the flow-level comparison ------------------------------------
    my_nf, ref_nf = pick(mine, "conditional_nf"), pick(theirs, "conditional_nf")
    if len(my_nf) and len(ref_nf):
        merged = my_nf.merge(ref_nf, on="energy_gev", suffixes=("_mine", "_ref"))
        print("\nConditional NF")
        print(f"{'E [GeV]':>9} {'ESS/N mine':>12} {'ESS/N ref':>11} "
              f"{'est. mine [pb]':>18} {'est. ref [pb]':>18} {'pull':>8}")
        for row in merged.itertuples():
            combined = (row.stderr_pb_mine ** 2 + row.stderr_pb_ref ** 2) ** 0.5
            pull = (row.estimate_pb_mine - row.estimate_pb_ref) / combined if combined else float("nan")
            print(f"{row.energy_gev:>9.1f} {row.normalized_ess_mine:>12.5f} "
                  f"{row.normalized_ess_ref:>11.5f} {row.estimate_pb_mine:>18.8g} "
                  f"{row.estimate_pb_ref:>18.8g} {pull:>+8.2f}")
        print("\n  (The reference column is the authors' fully trained `thorough`")
        print("   model. This run is a reduced training budget, so a lower ESS/N")
        print("   here is a budget effect, not a method failure.)")
    print()


if __name__ == "__main__":
    main()
