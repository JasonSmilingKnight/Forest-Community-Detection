"""
Independent analysis script for experiment-0.5_scaled_theta results.
Loads raw CSV data and generates final tables/reports.
Can be run anytime after the experiment completes.

Usage:
    python analyze_results_scaled_theta.py

Output:
    - distance_recovery_results_experiment-0.5_scaled_theta_comprehensive.txt (formatted report)
    - distance_recovery_results_experiment-0.5_scaled_theta_median.txt (median report)
    - Box plots for misclassification rates
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Keep imports within this standalone package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utilities import (
    save_all_tables_to_txt,
    print_results_table,
    print_success_failure_breakdown,
    print_comparison_table,
    print_per_strategy_summary,
    aggregate_trial_results,
    aggregate_trial_results_median,
)


def get_theta_coef(rate, n_ref=1000, theta_ref=0.01):
    """
    Compute coefficient so that coef * n^(-rate) = theta_ref at n = n_ref.
    """
    coef = theta_ref * pow(n_ref, rate)
    return coef


def load_raw_data(raw_data_file="distance_recovery_raw_data_experiment-0.5_scaled_theta.csv"):
    """
    Load raw data from CSV file and organize by (n, theta_rate) keys.
    Uses the theta_rate column to properly group data, avoiding key collision
    at the reference point where all theta values are identical.

    Returns
    -------
    all_raw_data : dict
        {strategy_name: {(n, theta_rate): {'cluster_counts': [...], 'metrics_list': [...], 'theta': float}}}
    """
    if not Path(raw_data_file).exists():
        print(f"ERROR: Raw data file not found: {raw_data_file}")
        sys.exit(1)

    print(f"Loading raw data from: {raw_data_file}")
    df = pd.read_csv(raw_data_file)

    print(f"\nCSV Columns: {df.columns.tolist()}")

    # Check theta_rate column
    if 'theta_rate' not in df.columns:
        print("ERROR: CSV missing 'theta_rate' column. Re-run experiment to generate updated CSV.")
        sys.exit(1)

    print(f"\nTheta rates found: {df['theta_rate'].unique().tolist()}")

    # Data summary
    print("\nData summary:")
    for rate in sorted(df['theta_rate'].unique()):
        rate_df = df[df['theta_rate'] == rate]
        for n in sorted(rate_df['n'].unique()):
            n_rows = len(rate_df[rate_df['n'] == n])
            theta_val = rate_df[rate_df['n'] == n]['theta'].iloc[0]
            print(f"  {rate}, n={n}: {n_rows} rows, theta={theta_val:.6f}")

    # Build nested dict keyed by (n, theta_rate)
    all_raw_data = {}

    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        all_raw_data[strategy] = {}

        for _, group in strategy_data.groupby(['n', 'theta_rate']):
            n = int(group['n'].iloc[0])
            theta_rate = group['theta_rate'].iloc[0]
            theta_val = group['theta'].iloc[0]

            key = (n, theta_rate)

            # Extract cluster counts and metrics
            all_raw_data[strategy][key] = {
                'cluster_counts': group['cluster_count'].tolist(),
                'metrics_list': [],
                'theta': theta_val
            }

            # Get metric columns
            exclude_cols = ['strategy', 'n', 'theta', 'theta_rate', 'trial',
                            'cluster_count', 'component_ratio', 'K']
            metric_cols = [col for col in group.columns if col not in exclude_cols]

            for _, row in group.iterrows():
                metrics = {}
                has_metrics = False
                for col in metric_cols:
                    if pd.notna(row[col]):
                        metrics[col] = row[col]
                        has_metrics = True
                if has_metrics:
                    all_raw_data[strategy][key]['metrics_list'].append(metrics)

            print(f"  Loaded ({n}, {theta_rate}): "
                  f"{len(all_raw_data[strategy][key]['cluster_counts'])} trials, "
                  f"{len(all_raw_data[strategy][key]['metrics_list'])} with metrics")

    print(f"\nSuccessfully loaded data for {len(all_raw_data)} strategy(ies)")
    return all_raw_data


def compute_results_from_raw(all_raw_data, K, repeats, L_list):
    """
    Compute final results from raw data using mean aggregation.
    Uses (n, theta_rate) keys directly — no np.isclose matching needed.
    """
    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    all_results = []

    for strategy_name, aggregated_data in all_raw_data.items():
        print(f"\nProcessing strategy: {strategy_name}")

        for (n_size, theta_rate), data in sorted(aggregated_data.items(),
                                                  key=lambda x: (rate_names.index(x[0][1])
                                                                  if x[0][1] in rate_names else 99,
                                                                  x[0][0])):
            cluster_counts = data['cluster_counts']
            metrics_list = data['metrics_list']
            theta = data['theta']

            result = aggregate_trial_results(
                cluster_counts=cluster_counts,
                metrics_list=metrics_list,
                K=K,
                repeats=repeats,
                strategy_name=strategy_name,
                n=n_size,
                theta=theta,
                L_list=L_list
            )
            result['theta_rate'] = theta_rate
            all_results.append(result)

            success_rate = len(metrics_list) / repeats if repeats > 0 else 0.0
            avg_clusters = np.mean(cluster_counts) if cluster_counts else 0.0
            print(f"  n={n_size} | theta={theta:.6f} ({theta_rate}): "
                  f"Valid={len(metrics_list)}/{repeats} "
                  f"(Success={success_rate:.1%}), "
                  f"AvgK={avg_clusters:.2f}")

    if not all_results:
        print("\nWARNING: No results found.")
        return pd.DataFrame()

    df_results = pd.DataFrame(all_results)

    # Reorder columns
    fixed_cols = ["strategy", "n", "theta", "theta_rate", "valid_trials", "success_rate", "avg_clusters"]
    metric_cols = [c for c in df_results.columns if c not in fixed_cols]
    df_results = df_results[[c for c in fixed_cols if c in df_results.columns] + sorted(metric_cols)]

    return df_results


def compute_results_from_raw_median(all_raw_data, K, repeats, L_list):
    """
    Compute final results from raw data using median aggregation.
    """
    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    all_results = []

    for strategy_name, aggregated_data in all_raw_data.items():
        for (n_size, theta_rate), data in sorted(aggregated_data.items(),
                                                  key=lambda x: (rate_names.index(x[0][1])
                                                                  if x[0][1] in rate_names else 99,
                                                                  x[0][0])):
            cluster_counts = data['cluster_counts']
            metrics_list = data['metrics_list']
            theta = data['theta']

            result = aggregate_trial_results_median(
                cluster_counts=cluster_counts,
                metrics_list=metrics_list,
                K=K,
                repeats=repeats,
                strategy_name=strategy_name,
                n=n_size,
                theta=theta,
                L_list=L_list
            )
            result['theta_rate'] = theta_rate
            all_results.append(result)

    if not all_results:
        return pd.DataFrame()

    df_results = pd.DataFrame(all_results)

    fixed_cols = ["strategy", "n", "theta", "theta_rate", "valid_trials", "success_rate", "median_clusters"]
    metric_cols = [c for c in df_results.columns if c not in fixed_cols]
    df_results = df_results[[c for c in fixed_cols if c in df_results.columns] + sorted(metric_cols)]

    return df_results


def generate_boxplots(all_raw_data, n_ref=1000, theta_ref=0.01, alpha=0):
    """
    Generate box plots showing misclassification rates.
    One PDF per (metric, theta_rate) combination. No title, y-axis = Misclassification rate.
    Filenames include alpha value and theta rate.
    """
    print(f"\nGenerating box plots...")

    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    rate_file_labels = {
        'rate_1': 'n_inv_1',
        'rate_3_4': 'n_inv_3_4',
        'rate_1_2': 'n_inv_1_2',
    }

    # Metrics to plot
    metrics_to_plot = [
        'first_50',
        'highdeg_50',
        'depth_1',
    ]

    alpha_str = str(alpha).replace('-', 'neg').replace('.', '_')

    for metric_name in metrics_to_plot:
        for theta_rate in rate_names:
            fig, ax = plt.subplots(figsize=(12, 6))

            data_for_plot = []
            n_labels = []

            # Get all n values for this rate
            all_n_values = set()
            for strategy_name in all_raw_data.keys():
                for (n, rate) in all_raw_data[strategy_name].keys():
                    if rate == theta_rate:
                        all_n_values.add(n)

            for n in sorted(all_n_values):
                misclass_rates = []

                for strategy_name in all_raw_data.keys():
                    key = (n, theta_rate)
                    if key in all_raw_data[strategy_name]:
                        metrics_list = all_raw_data[strategy_name][key]['metrics_list']
                        for metrics in metrics_list:
                            if metric_name in metrics:
                                misclass_rates.append(metrics[metric_name])

                if misclass_rates:
                    data_for_plot.append(misclass_rates)
                    n_labels.append(str(int(n)))
                    print(f"  {metric_name} for n={n}, {theta_rate}: {len(misclass_rates)} trials, "
                          f"mean={np.mean(misclass_rates):.4f}")

            if data_for_plot:
                bp = ax.boxplot(data_for_plot, tick_labels=n_labels, patch_artist=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5),
                               flierprops=dict(marker='o', markerfacecolor='gray',
                                               markersize=4, alpha=0.5))

                # Y-axis limits based on whiskers
                whisker_min = float('inf')
                whisker_max = float('-inf')
                for line in bp['whiskers']:
                    ydata = line.get_ydata()
                    whisker_min = min(whisker_min, min(ydata))
                    whisker_max = max(whisker_max, max(ydata))

                padding = whisker_max * 0.12
                ax.set_ylim([0, min(1, whisker_max + padding)])

                ax.set_ylabel('Misclassification rate', fontsize=12)
                ax.set_xlabel('Sample size (n)', fontsize=12)
                ax.grid(axis='y', alpha=0.3)

                fig_file = f"boxplot_alpha{alpha_str}_{metric_name}_{rate_file_labels[theta_rate]}.pdf"
                plt.tight_layout()
                plt.savefig(fig_file, dpi=150, bbox_inches='tight')
                print(f"  Saved: {fig_file}")
                plt.close()
            else:
                print(f"  WARNING: No data for {theta_rate}, metric={metric_name}")
                plt.close()


def load_all_alpha_data(data_dir):
    """
    Load CSVs for alpha=-0.5, 0, 2 from data_dir.
    Returns dict: {alpha_label: all_raw_data}
    """
    alpha_files = {
        -0.5: "distance_recovery_raw_data_experiment-0.5_scaled_theta.csv",
        0:    "distance_recovery_raw_data_experiment0_scaled_theta.csv",
        2:    "distance_recovery_raw_data_experiment2_scaled_theta.csv",
    }

    all_alpha_data = {}
    for alpha_val, fname in alpha_files.items():
        fpath = os.path.join(data_dir, fname)
        if not Path(fpath).exists():
            print(f"WARNING: Missing file for alpha={alpha_val}: {fpath}")
            continue
        print(f"\nLoading alpha={alpha_val} from: {fpath}")
        all_alpha_data[alpha_val] = load_raw_data(fpath)

    return all_alpha_data


def generate_combined_boxplots(all_alpha_data, output_dir="."):
    """
    3x3 grid per metric: rows = alpha (-0.5, 0, 2), cols = theta rate.
    One PDF per metric. Clean axes: few large ticks, no x-axis label.
    """
    print(f"\nGenerating 3x3 combined box plots (alpha=-0.5, 0, 2)...")

    alpha_vals = [2, 0, -0.5]
    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    rate_titles = {
        'rate_1':   r'$\theta \propto n^{-1}$',
        'rate_3_4': r'$\theta \propto n^{-3/4}$',
        'rate_1_2': r'$\theta \propto n^{-1/2}$',
    }
    alpha_row_labels = {
        -0.5: r'$\alpha = -0.5$',
        0:    r'$\alpha = 0$',
        2:    r'$\alpha = 2$',
    }
    metrics_to_plot = ['first_50', 'highdeg_50', 'depth_1']

    # Collect all n values once
    all_n_values = set()
    for raw_data in all_alpha_data.values():
        for strategy_name in raw_data:
            for (n, _) in raw_data[strategy_name]:
                all_n_values.add(n)
    sorted_n = sorted(all_n_values)
    # Abbreviated x-tick labels (e.g. 1k, 2k, ...)
    x_labels = [f"{int(n)//1000}k" if int(n) >= 1000 else str(int(n)) for n in sorted_n]

    for metric_name in metrics_to_plot:
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey="row", sharex=True)
        fig.subplots_adjust(hspace=0.08, wspace=0.1)

        row_upper = [0., 0., 0.]
        for row_idx, alpha_val in enumerate(alpha_vals):
            raw_data = all_alpha_data.get(alpha_val, {})

            for col_idx, theta_rate in enumerate(rate_names):
                ax = axes[row_idx][col_idx]

                # Gather data per n
                data_for_plot = []
                for n in sorted_n:
                    vals = []
                    for strategy_name in raw_data:
                        key = (n, theta_rate)
                        if key in raw_data[strategy_name]:
                            for metrics in raw_data[strategy_name][key]['metrics_list']:
                                if metric_name in metrics:
                                    vals.append(metrics[metric_name])
                    data_for_plot.append(vals if vals else [float('nan')])

                bp = ax.boxplot(
                    data_for_plot,
                    tick_labels=x_labels,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                    flierprops=dict(marker='o', markerfacecolor='gray',
                                    markersize=3, alpha=0.5),
                )

                ax.tick_params(axis='y', labelsize=13)
                if row_idx == 2:
                    ax.tick_params(axis='x', labelsize=10)
                ax.set_xlabel('')
                ax.grid(axis='y', alpha=0.25, linewidth=0.8)

                # Column header (theta rate) on top row only
                if row_idx == 0:
                    ax.set_title(rate_titles[theta_rate], fontsize=14, pad=6)

                # Row label (alpha) on left column only
                if col_idx == 0:
                    ax.set_ylabel(alpha_row_labels[alpha_val], fontsize=14, labelpad=8)
                else:
                    ax.set_ylabel('')

                # Y limits from whiskers
                whisker_min, whisker_max = float('inf'), float('-inf')
                for line in bp['whiskers']:
                    ydata = line.get_ydata()
                    whisker_min = min(whisker_min, min(ydata))
                    whisker_max = max(whisker_max, max(ydata))
                if whisker_min < whisker_max:
                    pad = whisker_max * 0.12
                    row_upper[row_idx] = max(row_upper[row_idx], min(1, whisker_max + pad))

        # A common scale per alpha row, with two y-ticks at the left edge.
        for row_idx in range(3):
            ymax = row_upper[row_idx] or 1.
            for col_idx in range(3):
                ax = axes[row_idx][col_idx]
                ax.set_ylim(0, ymax)
                ax.set_yticks([0, round(ymax * .25, 2), round(ymax * .75, 2)])
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
                ax.tick_params(axis='y', left=col_idx == 0, labelleft=col_idx == 0)
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(True)

        fig_file = os.path.join(output_dir, f"boxplot_3x3_{metric_name}.pdf")
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig_file}")
        plt.close()


def print_results_table_median(df, title="COMPLETE RESULTS TABLE (MEDIAN)"):
    """Print formatted table of median results with updated metrics."""
    depth_cols = [col for col in df.columns if col.startswith('depth_')]
    depth_cols_sorted = sorted(depth_cols, key=lambda x: int(x.split('_')[1]))
    highdeg_cols = [col for col in df.columns if col.startswith('highdeg_')]
    highdeg_cols_sorted = sorted(highdeg_cols, key=lambda x: int(x.split('_')[1]))

    display_cols = [
        'strategy', 'n', 'theta', 'theta_rate', 'valid_trials', 'success_rate', 'median_clusters',
        'first_10', 'first_50'
    ] + highdeg_cols_sorted + [
        'root'
    ] + depth_cols_sorted + [
        'overall_classified'
    ]

    display_cols = [col for col in display_cols if col in df.columns]
    df_display = df[display_cols].copy()

    # Sort by n first (group same n together), then theta_rate
    if 'theta_rate' in df_display.columns:
        df_display = df_display.sort_values(['n', 'theta_rate']).reset_index(drop=True)

    def format_float(x, decimals=4):
        if pd.isna(x):
            return "NaN"
        return f"{x:.{decimals}f}"

    def format_percent(x):
        if pd.isna(x):
            return "NaN"
        return f"{x:.1%}"

    df_display['theta'] = df_display['theta'].apply(lambda x: f"{x:.6f}")
    df_display['success_rate'] = df_display['success_rate'].apply(format_percent)
    if 'median_clusters' in df_display.columns:
        df_display['median_clusters'] = df_display['median_clusters'].apply(lambda x: format_float(x, 2))
    for col in ['first_10', 'first_50', 'overall_classified', 'root']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_float)
    for col in highdeg_cols_sorted + depth_cols_sorted:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_float)

    column_renames = {
        'strategy': 'Strategy',
        'n': 'n',
        'theta': 'θ',
        'theta_rate': 'Rate',
        'valid_trials': 'Valid',
        'success_rate': 'Success%',
        'median_clusters': 'MedK',
        'first_10': 'First10',
        'first_50': 'First50',
        'overall_classified': 'Overall',
        'root': 'Root',
    }
    for col in highdeg_cols_sorted:
        num = col.split('_')[1]
        column_renames[col] = f'HiDeg{num}'
    for depth_col in depth_cols_sorted:
        depth_num = depth_col.split('_')[1]
        column_renames[depth_col] = f'D{depth_num}'

    df_display.rename(columns=column_renames, inplace=True)

    print("\n" + "=" * 200)
    print(title)
    print("=" * 200)
    print()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)

    print(df_display.to_string(index=False))
    print("\n" + "=" * 200)


def save_median_txt(df_results_median, n_ref=1000, theta_ref=0.01):
    """Save median results to a separate TXT file."""
    if len(df_results_median) == 0:
        print("WARNING: No median results to save.")
        return

    txt_file = "distance_recovery_results_experiment-0.5_scaled_theta_median.txt"

    with open(txt_file, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("=" * 200)
        print("COMPREHENSIVE EXPERIMENTAL RESULTS (MEDIAN) - SCALED THETA")
        print(f"Reference point: n={n_ref}, theta={theta_ref}")
        print(f"Generated: {pd.Timestamp.now()}")
        print("=" * 200)
        print("\n\n")

        print_results_table_median(df_results_median, title="COMPLETE RESULTS TABLE (MEDIAN)")
        print("\n\n")

        print("=" * 200)
        print("END OF REPORT (MEDIAN)")
        print("=" * 200)

        sys.stdout = original_stdout

    print(f"Median report saved to: {txt_file}")


def save_outputs(df_results, n_list, get_theta_list, K, repeats, all_raw_data, n_ref=1000, theta_ref=0.01):
    """Save output files (TXT report + box plots)."""
    # Save comprehensive TXT
    print(f"\n{'=' * 80}")
    txt_file = save_all_tables_to_txt(
        df_results,
        n_list,
        get_theta_list,
        K,
        repeats,
        output_prefix="distance_recovery_results_experiment-0.5_scaled_theta"
    )

    # Print to console
    print("\n\n" + "=" * 200)
    print("CONSOLE OUTPUT - FORMATTED TABLES (SCALED THETA)")
    print(f"Reference point: n={n_ref}, theta={theta_ref}")
    print("=" * 200)

    print_results_table(df_results, title="COMPLETE RESULTS TABLE (SCALED THETA)")
    print_success_failure_breakdown(df_results, K, repeats)
    print_comparison_table(df_results, n_list, get_theta_list)
    print_per_strategy_summary(df_results, repeats)

    # Generate box plots
    generate_boxplots(all_raw_data, n_ref, theta_ref, alpha=2)


def main():
    """Generate only the combined 3x3 boxplot PDFs from the per-alpha raw CSVs."""
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")

    # === COMBINED BOX PLOTS (alpha = 2, 0, -0.5) ===
    results_dir = script_dir
    print(f"\n{'=' * 80}")
    print("GENERATING COMBINED BOX PLOTS")
    print("=" * 80)
    all_alpha_data = load_all_alpha_data(results_dir)
    if all_alpha_data:
        generate_combined_boxplots(all_alpha_data, output_dir=results_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    np.random.seed(14)
    main()
