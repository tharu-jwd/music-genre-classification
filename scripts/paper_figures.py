"""Redraw paper figures from saved analysis tables without recomputing metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_ORDER = ("cnn", "full", "no-supervision", "without-instrument", "without-timbre",
                "without-rhythm", "without-harmony")
CONFIG_LABELS = {"cnn": "Direct CNN", "full": "Full model", "no-supervision": "No concept supervision",
                 **{f"without-{name}": f"Without {name}" for name in ("instrument", "timbre", "rhythm", "harmony")}}
GENRES = ("classical", "electronic", "folk", "hiphop", "jazz", "rock")
BLUE = "#17608A"
GREY = "#444444"
ORANGE = "#BA5A24"


def _save(fig, directory, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight",
                    dpi=400 if extension == "png" else None, facecolor="white")


def _clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7)
    ax.set_axisbelow(True)


def render_figures(destination, summary, differences, *, runs=None, bootstrap=None, concepts=None):
    """Use existing seed summaries and intervals; never fit models or bootstrap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    captions = []
    settings = {"font.family": "serif", "font.serif": ["DejaVu Serif"],
                "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                "ytick.labelsize": 10, "legend.fontsize": 8,
                "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
                "svg.fonttype": "none", "savefig.facecolor": "white"}
    with plt.rc_context(settings):
        table = summary.set_index("configuration")
        order = [name for name in CONFIG_ORDER if name in table.index]
        if order:
            fig, ax = plt.subplots(figsize=(6.0, 3.8))
            means = table.loc[order, "macro_average_precision_mean"].to_numpy(float) * 100
            deviations = table.loc[order, "macro_average_precision_sd"].to_numpy(float) * 100
            limits = []
            for y, (name, mean, deviation) in enumerate(zip(order, means, deviations)):
                color = BLUE if name == "full" else GREY
                if runs is not None and not runs.empty:
                    points = runs.loc[runs.configuration == name, "macro_average_precision"].to_numpy(float) * 100
                    offsets = np.linspace(-0.11, 0.11, len(points))
                    ax.scatter(points, y + offsets, marker="x", color="#8B8B8B", s=24,
                               linewidths=1.0, zorder=3)
                    limits.extend(points)
                ax.errorbar(mean, y, xerr=deviation if np.isfinite(deviation) else None,
                            fmt="o", markersize=5.5, color=color, elinewidth=1.4,
                            capsize=3, zorder=4)
                value = f"{mean:.2f} ± {deviation:.2f}" if np.isfinite(deviation) else f"{mean:.2f}"
                ax.text(1.035, y, value, transform=ax.get_yaxis_transform(), va="center", fontsize=9)
                limits.extend([mean - (deviation if np.isfinite(deviation) else 0),
                               mean + (deviation if np.isfinite(deviation) else 0)])
            ax.set_yticks(np.arange(len(order)), [CONFIG_LABELS[name] for name in order])
            ax.set_ylim(len(order) - 0.6, -0.65)
            lower = 2 * np.floor((min(limits) - 0.7) / 2)
            upper = 2 * np.ceil((max(limits) + 0.7) / 2)
            ax.set_xlim(max(0, lower), min(100, upper))
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.set_xlabel("Test macro average precision (%)")
            ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
            ax.text(1.035, 1.025, "Mean ± SD", transform=ax.transAxes, fontsize=9)
            legend = [Line2D([], [], color=GREY, marker="o", linewidth=1.4, markersize=5,
                             label="Mean ± seed SD")]
            if runs is not None:
                legend.append(Line2D([], [], color="#8B8B8B", marker="x", linestyle="none",
                                     label="Individual seeds"))
            ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(-0.02, 1.025),
                      ncol=2, frameon=False, borderaxespad=0)
            _clean_axis(ax)
            fig.tight_layout()
            _save(fig, destination, "genre_performance")
            plt.close(fig)
            captions.append("**genre_performance:** Test macro average precision for seven configurations. "
                            "Dots and horizontal bars show the mean and sample standard deviation across seeds "
                            "42, 123, and 456; crosses show individual runs. The horizontal axis is zoomed and "
                            "explicitly labelled. Standard-deviation bars are not confidence intervals.")

        if differences is not None and not differences.empty:
            removals = differences[differences.comparison.str.startswith("removal-")]
            matrix = removals.pivot_table(index="comparison", columns="genre", values="delta_ap") * 100
            names = [f"removal-{name}" for name in ("instrument", "timbre", "rhythm", "harmony") if f"removal-{name}" in matrix.index]
            columns = [name for name in GENRES if name in matrix.columns]
            if names and columns:
                matrix = matrix.loc[names, columns]
                limit = max(float(np.nanmax(np.abs(matrix.to_numpy()))), 0.1)
                fig, ax = plt.subplots(figsize=(6.0, 3.35))
                cmap = plt.get_cmap("BrBG")
                norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
                im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
                ax.set_xticks(np.arange(len(columns)), ["Hip-hop" if name == "hiphop" else name.capitalize() for name in columns])
                ax.set_yticks(np.arange(len(names)), ["Without " + name.removeprefix("removal-") for name in names])
                for row in range(len(names)):
                    for col in range(len(columns)):
                        value = matrix.iloc[row, col]
                        if not np.isfinite(value):
                            continue
                        red, green, blue, _ = cmap(norm(value))
                        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                        ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=10,
                                color="white" if luminance < 0.5 else "#171717")
                ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
                ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
                ax.grid(which="minor", color="white", linewidth=1.3)
                ax.tick_params(which="both", length=0, pad=7)
                colorbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.10, pad=0.24)
                colorbar.set_label("Full − branch-removal AP (percentage points)", fontsize=9)
                ax.set_title("Positive: full model better   |   Negative: removal better", fontsize=9, pad=12)
                fig.tight_layout()
                _save(fig, destination, "branch_removal_heatmap")
                plt.close(fig)
                captions.append("**branch_removal_heatmap:** Mean per-genre AP difference between the full model "
                                "and each retrained branch-removal model, averaged across three seeds. Values "
                                "are percentage points; positive values favour the full model. These cells do "
                                "not show confidence intervals or causal effects, and removals change capacity.")

        bootstrap = bootstrap or {}
        comparisons = [("full-vs-cnn", "Full − direct CNN"),
                       ("concept-supervision", "Full − no supervision"),
                       *((f"removal-{name}", f"Full − without {name}") for name in ("instrument", "timbre", "rhythm", "harmony"))]
        present = [(key, label) for key, label in comparisons if key in bootstrap]
        if present:
            fig, ax = plt.subplots(figsize=(6.0, 3.5))
            edges = [0]
            for y, (key, _) in enumerate(present):
                item = bootstrap[key]
                center, low, high = [100 * item[field] for field in ("mean_seed_difference", "lower", "upper")]
                # Percentile intervals need not contain their point estimate.
                ax.hlines(y, low, high, color=BLUE, linewidth=1.6)
                ax.vlines([low, high], y - 0.08, y + 0.08, color=BLUE, linewidth=1.1)
                ax.plot(center, y, "o", color=BLUE, markersize=5.5)
                ax.text(1.035, y, f"{center:+.2f} [{low:+.2f}, {high:+.2f}]",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=8.5)
                edges.extend([low, high, center])
            ax.axvline(0, color=GREY, linestyle=(0, (3, 3)), linewidth=1)
            ax.set_yticks(np.arange(len(present)), [label for _, label in present])
            ax.set_ylim(len(present) - 0.6, -0.7)
            span = max(max(edges) - min(edges), 0.5)
            ax.set_xlim(min(edges) - 0.12 * span, max(edges) + 0.12 * span)
            ax.set_xlabel("Difference in test macro AP (percentage points)")
            ax.text(1.035, 1.035, "Estimate [95% CI]", transform=ax.transAxes, fontsize=9)
            ax.set_title("Paired comparisons", loc="left", fontsize=10, pad=13)
            ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
            _clean_axis(ax)
            fig.tight_layout()
            _save(fig, destination, "paired_comparisons")
            plt.close(fig)
            captions.append("**paired_comparisons:** Seed-averaged macro AP differences with 95% percentile "
                            "paired track-bootstrap intervals. Positive values favour the full model. The same "
                            "resampled tracks are used for both configurations and all seeds; intervals describe "
                            "track-sampling uncertainty conditional on the fitted models, assume track independence, "
                            "and have no multiple-comparison correction. Branch removals change capacity.")

        if concepts is not None and not concepts.empty:
            selected = [("full", "model", "Supervised", BLUE),
                        ("no-supervision", "model", "No supervision", ORANGE),
                        ("full", "training-reference", "Constant reference", "#999999")]
            fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1, 2.4]})
            bar_width = 0.23
            for offset, (config, source, label, color) in enumerate(selected):
                subset = concepts[(concepts.configuration == config) & (concepts.source == source)].set_index("branch")
                if "instrument" in subset.index and pd.notna(subset.loc["instrument", "macro_average_precision_mean"]):
                    row = subset.loc["instrument"]
                    axes[0].bar((offset - 1) * bar_width, row.macro_average_precision_mean * 100,
                                width=bar_width, color=color,
                                yerr=row.macro_average_precision_sd * 100 if pd.notna(row.macro_average_precision_sd) else None,
                                capsize=2, error_kw={"elinewidth": 0.9}, label=label)
                families = [name for name in ("timbre", "rhythm", "harmony") if name in subset.index]
                for name in families:
                    row = subset.loc[name]
                    if pd.isna(row.rmse_standardized_mean):
                        continue
                    x = ("timbre", "rhythm", "harmony").index(name)
                    axes[1].bar(x + (offset - 1) * bar_width, row.rmse_standardized_mean,
                                width=bar_width, color=color,
                                yerr=row.rmse_standardized_sd if pd.notna(row.rmse_standardized_sd) else None,
                                capsize=2, error_kw={"elinewidth": 0.9})
            axes[0].set_xticks([0], ["Instrument"])
            axes[0].set_ylabel("Macro AP (%)")
            axes[0].set_title("(a) Binary concepts\nHigher is better", fontsize=9)
            axes[1].set_xticks([0, 1, 2], ["Timbre", "Rhythm", "Harmony"])
            axes[1].set_ylabel("Standardised RMSE")
            axes[1].set_title("(b) Continuous descriptors\nLower is better", fontsize=9)
            for ax in axes:
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
                _clean_axis(ax)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
            fig.tight_layout(rect=(0, 0.10, 1, 1))
            _save(fig, destination, "concept_prediction_quality")
            plt.close(fig)
            captions.append("**concept_prediction_quality:** Full supervised versus no-concept-supervision "
                            "models and constant training-derived references. Panel (a) shows instrument macro "
                            "AP; panel (b) shows standardised descriptor RMSE. Bars show seed means and error "
                            "bars show sample seed SD. AP and RMSE use different axes and cannot be compared "
                            "numerically with each other. Alignment with these targets is not proof of faithful explanations.")

    (destination / "figure_captions.md").write_text("# Figure captions\n\n" + "\n\n".join(captions) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    source = args.analysis_dir
    def table(name, required=False):
        path = source / f"{name}.csv"
        if path.exists():
            return pd.read_csv(path)
        if required:
            raise FileNotFoundError(path)
        return None
    report = json.loads((source / "analysis.json").read_text())
    destination = args.out_dir or source / "paper-figures"
    render_figures(destination, table("summary", True), table("per_genre_differences", True),
                   runs=table("runs"), bootstrap=report.get("bootstrap"), concepts=table("concept_summary"))
    print(f"Paper figures saved to {destination}; original metrics and bootstrap intervals were not recomputed.")


if __name__ == "__main__":
    main()
