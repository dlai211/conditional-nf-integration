"""Draw the anatomy of the conditional cubic-Hermite marginal.

Four panels, all from the trained `gg_ttbar` checkpoint:

  A  the density q(x | sqrt(s_hat)) the model learns, at three energies
  B  the forward map u -> x, with the 32 bins and one highlighted
  C  how the bins are built: softmax -> widths/heights -> cumulative knots
  D  one bin in detail: the cubic Hermite curve and its PCHIP slopes

Each panel is labelled with the function in nf/conditional_cubic_1d.py that
produces it, so the figure doubles as a map of the code.

    PYTHONPATH=... python scripts/visualize_conditional_marginal.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from nf.conditional_cubic_1d import ConditionalCubicFlow1D  # noqa: E402

DEFAULT_CHECKPOINT = (
    PROJECT
    / "results"
    / "gg_ttbar_copula_thorough_gpu"
    / "madgraph_gg_ttbar_2_final_model.pt"
)
ENERGIES = (400.0, 1000.0, 1950.0)
SELECTED_ENERGY = 1000.0
MINIMUM, MAXIMUM = 360.0, 2000.0

# Validated categorical slots (light mode); see the dataviz palette reference.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e5e4df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
COLORS = (BLUE, ORANGE, AQUA)


def load_marginal(checkpoint_path: Path) -> ConditionalCubicFlow1D:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["marginal_state_dict"]
    hidden = state["marginals.0.net.0.weight"].shape[0]
    model = ConditionalCubicFlow1D(cond_dim=3, hidden=hidden, n_bins=32).double()
    prefix = "marginals.0."
    model.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    )
    model.eval()
    return model


def condition(energy: float) -> torch.Tensor:
    z = (energy - MINIMUM) / (MAXIMUM - MINIMUM)
    return torch.tensor(
        [[z, z * z, math.log1p(z) / math.log(2.0)]], dtype=torch.float64
    )


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def code_tag(ax, text: str, loc: str = "lower right") -> None:
    x, y, ha, va = {
        "lower right": (0.98, 0.04, "right", "bottom"),
        "lower left": (0.02, 0.04, "left", "bottom"),
        "upper right": (0.98, 0.96, "right", "top"),
        "upper left": (0.02, 0.96, "left", "top"),
        "middle left": (0.02, 0.52, "left", "center"),
        "lower right raised": (0.98, 0.30, "right", "bottom"),
    }[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        family="monospace",
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.85),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "figures" / "conditional_marginal_anatomy",
    )
    args = parser.parse_args()

    model = load_marginal(args.checkpoint)

    # ------------------------------------------------------------------
    # Data for all panels
    # ------------------------------------------------------------------
    x_grid = torch.linspace(1e-5, 1.0 - 1e-5, 4000, dtype=torch.float64)
    densities = {}
    with torch.no_grad():
        for energy in ENERGIES:
            _, logabsdet = model.inverse(x_grid, condition(energy))
            densities[energy] = torch.exp(-logabsdet).numpy()

        cond = condition(SELECTED_ENERGY)
        widths, heights, cumwidths, cumheights = model._params(cond)
        slopes = model._derivatives(widths, heights)
        widths = widths[0].numpy()
        heights = heights[0].numpy()
        cumw = cumwidths[0].numpy()
        cumh = cumheights[0].numpy()
        d = slopes[0].numpy()

        # The bin whose Hermite curve departs most from its chord: it shows
        # the curvature that the polynomial adds.
        t_probe = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
        deviations = []
        for k in range(len(widths)):
            xk, _ = model._hermite(
                t_probe,
                torch.full_like(t_probe, cumh[k]),
                torch.full_like(t_probe, heights[k]),
                torch.full_like(t_probe, widths[k]),
                torch.full_like(t_probe, d[k]),
                torch.full_like(t_probe, d[k + 1]),
            )
            chord = cumh[k] + heights[k] * t_probe
            deviations.append(float((xk - chord).abs().max()))
        bin_index = int(np.argmax(deviations))

        u_star = 0.5 * (cumw[bin_index] + cumw[bin_index + 1])
        t_star = (u_star - cumw[bin_index]) / widths[bin_index]
        u_curve = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
        x_curve, _ = model.forward(u_curve, cond)
        x_star, _ = model.forward(torch.tensor([u_star], dtype=torch.float64), cond)
        x_star = float(x_star[0])

        # Hermite curve inside the selected bin, for panel D.
        t_grid = torch.linspace(0.0, 1.0, 400, dtype=torch.float64)
        y0 = cumh[bin_index]
        bin_width = widths[bin_index]
        bin_height = heights[bin_index]
        d0 = d[bin_index]
        d1 = d[bin_index + 1]
        x_bin, dx_bin = model._hermite(
            t_grid,
            torch.full_like(t_grid, y0),
            torch.full_like(t_grid, bin_height),
            torch.full_like(t_grid, bin_width),
            torch.full_like(t_grid, d0),
            torch.full_like(t_grid, d1),
        )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.845, bottom=0.075, wspace=0.22, hspace=0.40)

    fig.text(0.07, 0.965, "Anatomy of the conditional cubic-Hermite marginal",
             fontsize=17, color=INK, ha="left", va="top", fontweight="semibold")
    fig.text(0.07, 0.925,
             "gg → t t̄, marginal 0, thorough checkpoint — what each stage of "
             "nf/conditional_cubic_1d.py does",
             fontsize=10.5, color=INK2, ha="left", va="top")
    fig.text(0.07, 0.018,
             "Source: results/gg_ttbar_copula_thorough_gpu. Blue/orange/aqua "
             "are the three energies; panels B–D use 1000 GeV.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    # ------------------------------------------------------------------
    # A — the density
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    style_axis(ax)
    ax.set_yscale("log")
    x_np = x_grid.numpy()
    for energy, color in zip(ENERGIES, COLORS):
        ax.plot(x_np, densities[energy], color=color, linewidth=2.0,
                solid_capstyle="round", label=f"{energy:g} GeV")

    # Direct labels at the trough, where the three curves are separated.
    trough_x = 0.5
    trough_index = int(np.argmin(np.abs(x_np - trough_x)))
    for energy, color, offset in zip(ENERGIES, COLORS, ((12, 12), (38, 2), (12, -14))):
        y = densities[energy][trough_index]
        ax.plot([trough_x], [y], "o", color=color, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)
        ax.annotate(
            f"{energy:g} GeV",
            xy=(trough_x, y),
            xytext=offset,
            textcoords="offset points",
            color=INK2,
            fontsize=9,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE, edgecolor="none", alpha=0.9),
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.1, 22.0)
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("x  (the marginal variable, unit interval)", fontsize=10, color=INK2)
    ax.set_ylabel(r"$q(x\,|\,\sqrt{\hat{s}})$", fontsize=11, color=INK2)
    ax.set_title("What the marginal models", fontsize=12.5, color=INK,
                 loc="left", pad=26, fontweight="semibold")
    ax.text(0.0, 1.015, "density at three energies; one model, conditioned on √ŝ",
            transform=ax.transAxes, fontsize=9.2, color=MUTED, ha="left", va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center",
              ncol=3, handlelength=1.6, columnspacing=1.4, bbox_to_anchor=(0.5, 0.94))
    code_tag(ax, "inverse → exp(−logdet)", loc="lower left")

    # ------------------------------------------------------------------
    # B — the forward map
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    style_axis(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
            label="identity")
    ax.plot(u_curve.numpy(), x_curve.numpy(), color=ORANGE, linewidth=2.0,
            solid_capstyle="round", label="u → x")

    # Every fourth knot, kept faint so the eye still follows the curve.
    for k in range(0, 33, 4):
        ax.axvline(cumw[k], color=GRID, linewidth=0.7, zorder=0)
        ax.axhline(cumh[k], color=GRID, linewidth=0.7, zorder=0)

    ax.add_patch(
        Rectangle(
            (cumw[bin_index], cumh[bin_index]),
            widths[bin_index],
            heights[bin_index],
            facecolor=ORANGE,
            alpha=0.18,
            edgecolor=ORANGE,
            linewidth=1.4,
            zorder=4,
        )
    )
    ax.plot([u_star], [x_star], "o", color=ORANGE, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=6)
    ax.annotate(
        f"bin {bin_index}\nu = {u_star:.3f} → x = {x_star:.3f}",
        xy=(u_star, x_star),
        xytext=(18, -28),
        textcoords="offset points",
        fontsize=9,
        color=INK2,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9),
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("u  (base sample)", fontsize=10, color=INK2)
    ax.set_ylabel("x  (physical sample)", fontsize=10, color=INK2)
    ax.set_title("The map the code evaluates", fontsize=12.5, color=INK,
                 loc="left", pad=26, fontweight="semibold")
    ax.text(0.0, 1.015, "monotone by construction; inverse = read it vertically",
            transform=ax.transAxes, fontsize=9.2, color=MUTED, ha="left", va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right",
              handlelength=1.6)
    code_tag(ax, "forward → x, log|dx/du|", loc="upper left")

    # ------------------------------------------------------------------
    # C — bin construction
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    style_axis(ax)
    ax.grid(False)
    ax.set_xlim(0.0, 1.12)
    ax.set_ylim(-0.35, 2.35)
    for row, values, label in ((1.25, widths, "widths"), (0.25, heights, "heights")):
        start = 0.0
        for k, value in enumerate(values):
            ax.add_patch(
                Rectangle(
                    (start, row),
                    value,
                    0.62,
                    facecolor=ORANGE,
                    alpha=0.95 if k == bin_index else 0.42,
                    edgecolor=SURFACE,
                    linewidth=1.2,
                )
            )
            start += value
        ax.text(0.0, row + 0.72, label, fontsize=9.5, color=INK2, ha="left", va="bottom")
        ax.text(1.015, row + 0.31, "Σ = 1", fontsize=8.5, color=MUTED, ha="left", va="center")

    top_center = cumw[bin_index] + 0.5 * widths[bin_index]
    bottom_center = cumh[bin_index] + 0.5 * heights[bin_index]
    ax.plot([top_center, bottom_center], [1.25 + 0.31, 0.25 + 0.31],
            color=INK2, linewidth=1.0, linestyle=(0, (2, 3)), zorder=3)
    ax.text(0.5 * (top_center + bottom_center) + 0.012, 1.06, "same bin",
            fontsize=8, color=MUTED, ha="left", va="center",
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=0.2))

    ax.annotate(
        f"bin {bin_index}",
        xy=(cumw[bin_index] + 0.5 * widths[bin_index], 1.25 + 0.62),
        xytext=(cumw[bin_index] + 0.5 * widths[bin_index], 2.02),
        fontsize=8.5,
        color=INK2,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9),
    )
    ax.set_xlabel("cumulative fraction of the unit interval", fontsize=10, color=INK2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([])
    ax.set_title("How the bins are built", fontsize=12.5, color=INK,
                 loc="left", pad=26, fontweight="semibold")
    ax.text(0.0, 1.015, "softmax → widths/heights → cumulative knots (the last is forced to 1)",
            transform=ax.transAxes, fontsize=9.2, color=MUTED, ha="left", va="bottom")
    code_tag(ax, "_params", loc="lower right")

    # ------------------------------------------------------------------
    # D — one bin in detail
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    style_axis(ax)
    t_np = t_grid.numpy()
    x_np_bin = x_bin.numpy()
    chord = y0 + bin_height * t_np
    ax.plot(t_np, x_np_bin, color=ORANGE, linewidth=2.2, solid_capstyle="round",
            label="x(t)")
    ax.plot(t_np, chord, color=MUTED, linewidth=1.0,
            linestyle=(0, (4, 3)), label="chord")

    tangent_t0 = np.linspace(0.0, 0.28, 60)
    tangent_y0 = y0 + bin_width * d0 * tangent_t0
    tangent_t1 = np.linspace(0.72, 1.0, 60)
    tangent_y1 = (y0 + bin_height) + bin_width * d1 * (tangent_t1 - 1.0)
    ax.plot(tangent_t0, tangent_y0, color=INK2, linewidth=1.1,
            linestyle=(0, (2, 2)), label="endpoint tangents")
    ax.plot(tangent_t1, tangent_y1, color=INK2, linewidth=1.1,
            linestyle=(0, (2, 2)), label="_nolegend_")

    ax.plot([t_star], [x_star], "o", color=ORANGE, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)

    stats = (
        f"bin {bin_index}\n"
        f"width  = {bin_width:.4f}\n"
        f"height = {bin_height:.4f}\n"
        f"d0     = {d0:.4f}\n"
        f"d1     = {d1:.4f}"
    )
    ax.text(0.03, 0.97, stats, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.2, family="monospace", color=INK2,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                      edgecolor=GRID, linewidth=0.8, alpha=0.95))

    all_y = np.concatenate([x_np_bin, chord, tangent_y0, tangent_y1, [x_star]])
    low, high = float(all_y.min()), float(all_y.max())
    margin = 0.10 * (high - low)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("t  (position inside the bin)", fontsize=10, color=INK2)
    ax.set_ylabel("x  (physical value)", fontsize=10, color=INK2)
    ax.set_title("Inside one bin", fontsize=12.5, color=INK,
                 loc="left", pad=26, fontweight="semibold")
    ax.text(0.0, 1.015, "cubic Hermite with PCHIP slopes; dotted = endpoint tangents",
            transform=ax.transAxes, fontsize=9.2, color=MUTED, ha="left", va="bottom")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right",
              handlelength=1.6)
    code_tag(ax, "_derivatives + _hermite", loc="lower right raised")

    # ------------------------------------------------------------------
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor=SURFACE)
    fig.savefig(output.with_suffix(".svg"), facecolor=SURFACE)
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
