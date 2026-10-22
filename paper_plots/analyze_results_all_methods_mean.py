"""
Combined 3x3 mean-trend plot across all four methods for the c=1
scaled-theta sweep family: Distance Recovery (leaf-stripping's own
distance-threshold recovery), SCORE, Gao-DCBM, and Monte Carlo.

Rows = alpha (-0.5, 0, 2), columns = theta rate (rate_1, rate_3_4,
rate_1_2) -- same 3x3 layout as analyze_results_scaled_theta.py's
combined box plots. Each subplot shows one dashed line per method: the
MEAN of a chosen metric across that cell's trials, at each sample size
n. Unlike a box plot, only the mean is drawn -- no spread/whiskers --
so all four methods can be overlaid and compared directly in one panel.

Run this after the leaf-stripping-only sweep, the SCORE/Gao-DCBM
benchmark sweep, and the Monte Carlo sweep have all produced their raw
CSVs for a given alpha (place/copy them alongside this script, same as
analyze_results_scaled_theta.py expects). A method missing its CSV for a
given alpha is skipped with a warning rather than failing the whole plot,
so you can run this against a partial set of finished jobs.

Usage:
    python analyze_results_all_methods_mean.py

Produces one PDF per metric in METRICS_TO_PLOT below (first_50,
highdeg_50, depth_1, the arrival-weighted overall metrics weighted_overall
/ weighted_overall_t1_log / weighted_overall_r2 / weighted_overall_t3_2, and
the degree-weighted overall metrics weighted_overall_deg2_log /
weighted_overall_deg3 / weighted_overall_deg2 / weighted_overall_deg4, by
default). Pass a metric name as an argument to plot just that one
instead: `python ... overall_classified`.
Any column present in the raw CSVs works: first_10, first_50,
highdeg_10, highdeg_50, root, depth_1, depth_2, overall_classified,
weighted_overall, weighted_overall_t1_log, weighted_overall_r2,
weighted_overall_t3_2, weighted_overall_deg2_log, weighted_overall_deg3,
weighted_overall_deg2, weighted_overall_deg4, ...
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ALPHA_VALUES = [2, 0, -0.5]
ALPHA_FILE_STR = {-0.5: "-0.5", 0: "0", 2: "2"}
METRICS_TO_PLOT = [
    'first_50', 'highdeg_50', 'depth_1',
    'weighted_overall', 'weighted_overall_t1_log', 'weighted_overall_t3_2', 'weighted_overall_r2',
    'weighted_overall_deg2', 'weighted_overall_deg2_log', 'weighted_overall_deg3', 'weighted_overall_deg4',
]
RATE_NAMES = ["rate_1", "rate_3_4", "rate_1_2"]
RATE_TITLES = {
    'rate_1':   r'$\theta \asymp n^{-1}$',
    'rate_3_4': r'$\theta \asymp n^{-3/4}$',
    'rate_1_2': r'$\theta \asymp n^{-1/2}$',
}
ALPHA_ROW_LABELS = {
    -0.5: r'$\alpha = -0.5$',
    0:    r'$\alpha = 0$',
    2:    r'$\alpha = 2$',
}

# Preserve legacy names for unchanged metrics; new weights use new columns.
METRIC_FILE_LABELS = {
    'weighted_overall': 'weighted_overall_t1',
    'weighted_overall_t1_log': 'weighted_overall_t1_log',
    'weighted_overall_r2': 'weighted_overall_t2',
    'weighted_overall_t3_2': 'weighted_overall_t3_2',
}

# Display name -> (raw-CSV filename template, strategy value inside that CSV)
METHODS = {
    "Distance Recovery": ("distance_recovery_raw_data_experiment{alpha}_scaled_theta.csv",
                           "with_protected"),
    "SCORE":             ("distance_recovery_raw_data_experiment{alpha}_scaled_theta_benchmarks.csv",
                           "SCORE"),
    "GMZZ":          ("distance_recovery_raw_data_experiment{alpha}_scaled_theta_benchmarks.csv",
                           "Gao-DCBM"),
    "Monte Carlo":       ("distance_recovery_raw_data_experiment{alpha}_scaled_theta_montecarlo.csv",
                           "MonteCarlo"),
}

METHOD_STYLES = {
    "Distance Recovery": dict(color="tab:blue",   marker="o"),
    "SCORE":             dict(color="tab:orange", marker="s"),
    "GMZZ":          dict(color="#333333",  marker="^"),
    "Monte Carlo":       dict(color="tab:red",    marker="D"),
}


def load_method_means(data_dir, metric):
    """
    Returns {method_name: {(alpha, theta_rate): {n: mean_value}}}.

    Skips a (method, alpha) combination whose CSV isn't present yet
    (e.g. that sweep hasn't finished/been copied over) instead of
    failing the whole plot.
    """
    means = {method: {} for method in METHODS}

    for method, (fname_template, strategy_value) in METHODS.items():
        for alpha in ALPHA_VALUES:
            fname = fname_template.format(alpha=ALPHA_FILE_STR[alpha])
            fpath = os.path.join(data_dir, fname)
            if not Path(fpath).exists():
                print(f"  [skip] {method}, alpha={alpha}: {fname} not found")
                continue

            df = pd.read_csv(fpath)
            df = df[df['strategy'] == strategy_value]
            if metric not in df.columns:
                print(f"  [skip] {method}, alpha={alpha}: column '{metric}' not in {fname}")
                continue

            for theta_rate in RATE_NAMES:
                rate_df = df[df['theta_rate'] == theta_rate]
                for n in sorted(rate_df['n'].unique()):
                    vals = rate_df.loc[rate_df['n'] == n, metric].dropna()
                    if len(vals) == 0:
                        continue
                    means[method].setdefault((alpha, theta_rate), {})[int(n)] = vals.mean()

    return means


def plot_combined_mean(means, metric, output_dir="."):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey="row")
    fig.subplots_adjust(hspace=0.08, wspace=0.1)

    all_n = sorted({n for method_data in means.values()
                     for cell in method_data.values() for n in cell})
    if not all_n:
        print("No data found for any method/alpha -- nothing to plot.")
        plt.close(fig)
        return
    x_labels = [f"{n // 1000}k" if n >= 1000 else str(n) for n in all_n]
    x_pos = np.arange(len(all_n))
    n_to_x = dict(zip(all_n, x_pos))

    legend_handles = {}

    for row_idx, alpha in enumerate(ALPHA_VALUES):
        for col_idx, theta_rate in enumerate(RATE_NAMES):
            ax = axes[row_idx][col_idx]

            for method in METHODS:
                cell = means[method].get((alpha, theta_rate), {})
                if not cell:
                    continue
                ns = sorted(cell.keys())
                xs = [n_to_x[n] for n in ns]
                ys = [cell[n] for n in ns]
                style = METHOD_STYLES[method]
                line, = ax.plot(
                    xs, ys, linestyle="--", marker=style["marker"],
                    color=style["color"], linewidth=1.8, markersize=5,
                    label=method,
                )
                legend_handles[method] = line

            ax.set_xticks(x_pos)
            if row_idx == 2:
                ax.set_xticklabels(x_labels, fontsize=10)
            ax.tick_params(axis="y", labelsize=13, labelleft=(col_idx == 0))
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)

            if row_idx == 0:
                ax.set_title(RATE_TITLES[theta_rate], fontsize=14, pad=6)
            if col_idx == 0:
                ax.set_ylabel(ALPHA_ROW_LABELS[alpha], fontsize=14, labelpad=8)

    # Match the boxplot spacing and show zero and the two existing y values on the left.
    # Each row shares a scale; calculate limits once after all curves are added.
    for row in axes:
        ymax = max(float(line.get_ydata().max()) for ax in row for line in ax.lines) if any(ax.lines for ax in row) else 1.0
        ymax = max(ymax * 1.12, 0.02)
        row[0].set_ylim(0, ymax)
        row[0].set_yticks([0, round(ymax * 0.25, 2), round(ymax * 0.75, 2)])
        row[0].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))

    if legend_handles:
        fig.legend(
            [legend_handles[m] for m in METHODS if m in legend_handles],
            [m for m in METHODS if m in legend_handles],
            loc="lower center", ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.03),
            fontsize=12, frameon=False,
        )

    file_label = METRIC_FILE_LABELS.get(metric, metric)
    fig_file = os.path.join(output_dir, f"mean_3x3_{file_label}.pdf")
    plt.savefig(fig_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_file}")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    metrics = [sys.argv[1]] if len(sys.argv) > 1 else METRICS_TO_PLOT

    for metric in metrics:
        print(f"\n--- Metric: {metric} ---")
        means = load_method_means(script_dir, metric)
        plot_combined_mean(means, metric, output_dir=script_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
