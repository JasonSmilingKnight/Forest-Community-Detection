"""Compare four methods at alpha=0 and theta proportional to n^-1.

Creates a 1x4 arrival-time PDF (t^-1, 1/(t log t), t^-3/2, t^-2)
and a 1x4 initial-degree PDF (d^2, d^2(log d+1), d^3, d^4). The logarithmic weight uses
w(1)=w(2)=1/(2 log 2). Points are arithmetic means of
nonmissing trial errors, using the metrics already computed in the CSVs.
The weighted metrics exclude unclassified nodes and optimize label alignment
separately for each weighting scheme; no metrics are recomputed here.

Run from the repository root:
    python output/analyze_results_alpha0_rate1_weights.py
    python output/analyze_results_alpha0_rate1_weights.py --c 2

Input CSVs and output PDFs default to this script's directory. Use
--data-dir and --output-dir to override them. Figures are saved as PDF only.
Missing files/columns are reported and skipped, as in the existing plots.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

from analyze_results_all_methods_mean import METHODS, METHOD_STYLES


WEIGHT_GROUPS = {
    "arrival_time": [
        ("weighted_overall", r"$w_v=t_v^{-1}$"),
        ("weighted_overall_t1_log", r"$w_v=(t_v\log t_v)^{-1}$"),
        ("weighted_overall_t3_2", r"$w_v=t_v^{-3/2}$"),
        ("weighted_overall_r2", r"$w_v=t_v^{-2}$"),
    ],
    "degree": [
        ("weighted_overall_deg2", r"$w_v=d_v^2$"),
        ("weighted_overall_deg2_log", r"$w_v=d_v^2(\log d_v+1)$"),
        ("weighted_overall_deg3", r"$w_v=d_v^3$"),
        ("weighted_overall_deg4", r"$w_v=d_v^4$"),
    ],
}


def load_means(data_dir, c):
    """Load only alpha=0, rate_1 and average each metric by sample size."""
    means = {}
    cache = {}
    metrics = [metric for group in WEIGHT_GROUPS.values() for metric, _ in group]
    for method, (template, strategy) in METHODS.items():
        filename = template.format(alpha="0")
        if c == 2:
            filename = filename.replace("scaled_theta", "scaled_theta_c2")
        path = data_dir / filename
        if not path.exists():
            print(f"[skip] {method}: {filename} not found")
            continue
        if path not in cache:
            cache[path] = pd.read_csv(path)
        df = cache[path]
        df = df.loc[(df["strategy"] == strategy) & (df["theta_rate"] == "rate_1")]
        available = [metric for metric in metrics if metric in df.columns]
        for metric in set(metrics) - set(available):
            print(f"[skip] {method}: missing column {metric}")
        if df.empty or not available:
            print(f"[skip] {method}: no usable alpha=0, rate_1 data")
            continue
        means[method] = df.groupby("n", sort=True)[available].mean()
    return means


def plot_weights(means, group, output_dir, c):
    panels = WEIGHT_GROUPS[group]
    all_n = sorted({n for df in means.values() for metric, _ in panels
                    if metric in df for n in df[metric].dropna().index})
    if not all_n:
        print(f"[skip] {group}: no nonmissing data to plot")
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(13 * len(panels) / 3, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.25, top=0.875, wspace=0.1)
    positions = {n: i for i, n in enumerate(all_n)}
    handles = {}
    for ax, (metric, title) in zip(axes, panels):
        ax.set_title(title, fontsize=14, pad=8)
        for method in METHODS:
            if method not in means or metric not in means[method]:
                continue
            values = means[method][metric].dropna()
            if values.empty:
                continue
            line, = ax.plot(
                [positions[n] for n in values.index], values.to_numpy(),
                linestyle="--", linewidth=1.8, markersize=5,
                label=method, **METHOD_STYLES[method],
            )
            handles[method] = line
        if not ax.lines:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_xticks(np.arange(len(all_n)))
        ax.set_xticklabels([f"{n / 1000:g}k" if n >= 1000 else f"{n:g}" for n in all_n])
        ax.tick_params(axis="y", labelsize=13)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    # Same panel aspect ratio and two positive ticks plus zero as one boxplot row.
    ymax = max(float(line.get_ydata().max()) for ax in axes for line in ax.lines)
    ymax = max(ymax * 1.12, 0.02)
    axes[0].set_ylim(0, ymax)
    axes[0].set_yticks([0, round(ymax * 0.25, 2), round(ymax * 0.75, 2)])
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ordered = [method for method in METHODS if method in handles]
    fig.legend([handles[m] for m in ordered], ordered, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=len(ordered), frameon=False, fontsize=11)
    stem = output_dir / f"mean_1x{len(panels)}_c{c}_alpha0_rate1_{group}_weights"
    path = stem.with_suffix(".pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--c", type=int, choices=(1, 2), default=1)
    parser.add_argument("--data-dir", type=Path, default=script_dir)
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means = load_means(args.data_dir, args.c)
    produced = [plot_weights(means, group, args.output_dir, args.c)
                for group in WEIGHT_GROUPS]
    if not any(produced):
        parser.exit(1, "No figures generated: check input CSVs and weighted metric columns.\n")


if __name__ == "__main__":
    main()
