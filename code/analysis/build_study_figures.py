"""Build the published success-rate and post-softmax attention figures."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image


INK = "#172520"
FOREST = "#173c36"
COPPER = "#b05635"
MUTED = "#55645b"
PAPER = "#fdfcf8"
RULE = "#d6dcd4"
CONDITIONS = (
    ("base", "Base VLM", FOREST, "o", "-"),
    ("finetuned", "Mixed VQA + LIBERO", COPPER, "D", "--"),
)


def draw_curves(data, output):
    with (data / "libero.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.75), sharey=True)
    fig.subplots_adjust(left=.085, right=.975, top=.73, bottom=.17, wspace=.22)
    fig.text(.03, .905, "LIBERO rollout success", fontsize=18, family="serif")
    legend = [Line2D([], [], color=c, marker=m, linestyle=line, lw=2.15, ms=6)
              for _, _, c, m, line in CONDITIONS]
    fig.legend(legend, [c[1] for c in CONDITIONS], ncol=2, frameon=False,
               loc="center right", bbox_to_anchor=(.98, .924), fontsize=10,
               handlelength=2.6, columnspacing=1.5)
    fig.add_artist(Line2D([.03, .975], [.855, .855], transform=fig.transFigure,
                         color=RULE, lw=.8))
    for ax, (scope, title) in zip(axes, [("frozen", "Frozen VLM"), ("full", "Full VLA")]):
        ax.set_title(title, loc="left", y=1.035, fontsize=12, fontweight="bold")
        for arm, _, color, marker, line in CONDITIONS:
            selected = sorted((r for r in rows if r["scope"] == scope and
                               r["suite"] == "all" and r["start_vlm"] == arm),
                              key=lambda r: int(r["updates"]))
            updates = [int(r["updates"]) / 1000 for r in selected]
            if updates != [4, 8, 12] or any(int(r["episodes"]) != 2000 for r in selected):
                raise ValueError(f"Incomplete condition: {scope}/{arm}")
            success = [100 * int(r["successes"]) / int(r["episodes"]) for r in selected]
            ax.plot(updates, success, color=color, ls=line, marker=marker,
                    lw=2.2, ms=6, markeredgecolor=PAPER, markeredgewidth=.75)
            for i, (x, y) in enumerate(zip(updates, success)):
                # Place the lower series' labels below the line after it crosses the other arm.
                below = (scope == "full" and arm == "finetuned" and i > 0) or (
                    scope == "frozen" and arm == "base" and i > 0)
                ax.annotate(f"{y:.2f}", (x, y), xytext=(0, -14 if below else 7),
                            textcoords="offset points", ha="center", fontsize=9.5, color=color)
        ax.set(xlim=(3.3, 12.7), ylim=(45, 91), xticks=[4, 8, 12],
               xticklabels=["4k", "8k", "12k"], yticks=[50, 60, 70, 80, 90])
        ax.grid(axis="y", color=RULE, lw=.7)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0, labelcolor=MUTED, pad=5, labelsize=9)
        ax.set_xlabel("Optimizer updates", fontsize=9.5, color=MUTED, labelpad=7)
    axes[0].set_ylabel("Success rate (%)", fontsize=9.5, color=MUTED, labelpad=7)
    fig.savefig(output / "libero-curves.svg", metadata={"Date": None})
    plt.close(fig)


def draw_attention(data, output):
    with np.load(data / "attention-map-input.npz") as source:
        maps = {key: source[key] for key in source}
    views = [np.asarray(Image.open(data / f"attention-view-{i}.png").convert("RGB"))
             for i in range(2)]
    fig, axes = plt.subplots(2, 5, figsize=(10.6, 5.15))
    fig.subplots_adjust(left=.071, right=.98, top=.745, bottom=.12, wspace=.05, hspace=.075)
    fig.text(.03, .93, "Where the first action query attends", fontsize=18, family="serif")
    fig.text(.977, .935, "Frozen VLM   /   8k updates   /   Frame 70", fontsize=9,
             color=MUTED, ha="right")
    norm = Normalize(0, max(map_.max() for map_ in maps.values()))
    cmap = LinearSegmentedColormap.from_list("visual_attention",
        ["#251b32", "#5c344f", "#9a4b43", "#cc7348", "#eda85a", "#ffe5a2"])
    for label, start, stop, color in [("Observation", 0, 0, MUTED),
                                      ("Base VLM", 1, 2, FOREST),
                                      ("Mixed VQA + LIBERO", 3, 4, COPPER)]:
        x0 = axes[0, start].get_position().x0
        x1 = axes[0, stop].get_position().x1
        fig.text(x0, .834, label, fontsize=11, color=color, fontweight="bold")
        fig.add_artist(Line2D([x0, x1], [.815, .815], transform=fig.transFigure,
                             color=color, lw=1.2))
    names = (None, "base_14", "base_15", "finetuned_14", "finetuned_15")
    headers = ["Original RGB", "SA · layer 14", "CA · layer 15", "SA · layer 14", "CA · layer 15"]
    for row, rgb in enumerate(views):
        for col, group in enumerate(names):
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.set(xticks=[], yticks=[])
            ax.spines[:].set_visible(False)
            if row == 0:
                ax.set_title(headers[col], fontsize=9, loc="left", color=MUTED, pad=5)
            if col == 0:
                ax.set_ylabel(("Scene camera", "Wrist camera")[row], fontsize=9.5,
                              labelpad=8, color=INK, fontweight="bold")
                continue
            weights = maps[group][row]
            if weights.shape != (8, 8):
                raise ValueError(f"Unexpected attention grid: {weights.shape}")
            h, w = rgb.shape[:2]
            # All panels share the same raw probability scale and 8×8 bin boundaries.
            ax.imshow(weights, cmap=cmap, norm=norm, interpolation="nearest", alpha=.64,
                      extent=(-.5, w-.5, h-.5, -.5))
            ax.text(.045, .045, f"Σ {weights.sum():.1%}", transform=ax.transAxes,
                    fontsize=8, color="white", bbox={"facecolor": INK, "edgecolor": "none",
                    "boxstyle": "round,pad=.22", "alpha": .86})
    bar = fig.add_axes([.63, .065, .347, .017])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar,
                      orientation="horizontal", ticks=[0, .004, .008, .012],
                      format=PercentFormatter(1, decimals=1))
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, color=MUTED, labelcolor=MUTED, labelsize=8, pad=3)
    fig.text(.072, .059, "Σ  Total attention assigned to the camera", fontsize=9, color=MUTED)
    fig.text(.61, .066, "Per token", ha="right", fontsize=9, color=MUTED)
    fig.savefig(output / "attention-map.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans", "font.serif": ["DejaVu Serif"],
        "text.color": INK, "axes.titlecolor": INK, "axes.axisbelow": True,
        "axes.facecolor": PAPER, "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
        "svg.fonttype": "path", "svg.hashsalt": "libero-capstone-figures-v2",
    }):
        draw_curves(args.data, args.out)
        draw_attention(args.data, args.out)


if __name__ == "__main__":
    main()
