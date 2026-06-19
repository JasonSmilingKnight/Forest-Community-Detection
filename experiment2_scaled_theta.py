"""
Main parallel runner for distance recovery experiments (recursive version).
SCALED THETA VERSION: All theta values start from the same value at n=1000,
but decrease at different rates as n increases.
Alpha = 2 (super-linear preferential attachment).

Parallelizes repeats across multiple cores using multiprocessing.
No pre-computed noise matrix - generates noise on-the-fly per trial.
"""

import numpy as np


def get_theta_coef(rate, n_ref=1000, theta_ref=None):
    """
    Compute coefficient so that coef * n^(-rate) = theta_ref at n = n_ref.

    Parameters
    ----------
    rate : float
        The exponent rate (positive value, e.g., 0.5 for n^(-1/2), 0.75 for n^(-3/4))
    n_ref : int
        Reference n value where all theta should be equal (default: 1000)
    theta_ref : float or None
        Target theta value at n_ref. If None, uses n_ref^(-3/4) as default.

    Returns
    -------
    coef : float
        Coefficient such that coef * n^(-rate) = theta_ref at n = n_ref

    Examples
    --------
    >>> coef = get_theta_coef(rate=1.0, n_ref=1000, theta_ref=1000**(-3/4))
    >>> theta = coef * pow(n, -1.0)  # equals 1000^(-3/4) when n=1000

    >>> coef = get_theta_coef(rate=0.5)  # uses default theta_ref = 1000^(-3/4)
    >>> theta = coef * pow(n, -0.5)  # equals 1000^(-3/4) when n=1000
    """
    if theta_ref is None:
        theta_ref = pow(n_ref, -3/4)  # default reference

    # We want: coef * n_ref^(-rate) = theta_ref
    # So: coef = theta_ref / n_ref^(-rate) = theta_ref * n_ref^(rate)
    coef = theta_ref * pow(n_ref, rate)
    return coef
import pandas as pd
import os
from multiprocessing import Pool
from functools import partial
import pickle
from tqdm import tqdm

from leaf_stripping_recovery import generate_recursive_forest
from experiment_distance_recovery_recursive_new import (
    extract_subgraph_and_compute_depth,
    run_single_trial_for_scale,
    aggregate_trial_results,
)
from utilities import (
    save_all_tables_to_txt,
    print_results_table,
    print_success_failure_breakdown,
    print_comparison_table,
    print_per_strategy_summary,
)


def run_single_trial(trial_id, n_list, scale_configs, theta_configs_by_scale, K, alpha,
                     b, loop_bound, select_by, L_list,
                     check_protected, recovery_threshold_divisor):
    """
    Run a single trial across all scales for a given strategy (recursive version).
    No pre-computed noise matrix - generates noise on-the-fly.

    Returns aggregated results for all (n, theta) combinations.

    Parameters
    ----------
    trial_id : int
        Trial number
    n_list : list
        List of graph sizes to test
    scale_configs : dict
        Configuration for each scale
    theta_configs_by_scale : dict
        Theta values for each scale
    K, alpha, b, loop_bound, min_component_size, select_by, L_list : config
        Experiment configuration
    check_protected : bool
        Whether to use protected node removal
    recovery_threshold_divisor : float
        Divisor for recovery threshold

    Returns
    -------
    trial_results : dict
        Aggregated results from this trial across all scales
    """
    # Set unique seed for each trial to avoid repeated results
    np.random.seed(42 + trial_id)

    K = 2
    n_base = max(n_list)

    # Generate base forest once per trial
    forest_base, components_base, assignments_base = generate_recursive_forest(
        n=n_base * K, K=K, alpha=alpha
    )

    trial_results = {}

    # Process each scale
    for n_size in n_list:
        n_per_component = scale_configs[n_size]['n_per_component']
        theta_list = theta_configs_by_scale[n_size]

        # Extract subgraph once per scale
        forest, components, _ = extract_subgraph_and_compute_depth(
            forest_base, n_per_component, K
        )

        # Process each theta value
        for theta_idx, theta in enumerate(theta_list):
            cluster_count, metrics = run_single_trial_for_scale(
                forest=forest,
                components=components,
                n_per_component=n_per_component,
                theta=theta,
                K=K,
                b=b,
                loop_bound=loop_bound,
                min_component_size=n_size // 10,
                select_by=select_by,
                L_list=L_list,
                check_protected_removal=check_protected,
                recovery_threshold_divisor=recovery_threshold_divisor
            )

            # Store results with (n, theta_idx) tuple as key to avoid collision at reference point
            key = (n_size, theta_idx)
            if key not in trial_results:
                trial_results[key] = {'cluster_counts': [], 'metrics_list': [], 'theta': theta}

            trial_results[key]['cluster_counts'].append(cluster_count)
            if metrics is not None:
                trial_results[key]['metrics_list'].append(metrics)

    return trial_results


def aggregate_parallel_results(trial_results_list, n_list, theta_configs_by_scale,
                               K, repeats, strategy_name, L_list):
    """
    Aggregate results from all trials (keeps raw metrics, no averaging).

    Parameters
    ----------
    trial_results_list : list of dict
        Raw results from each trial
    n_list, theta_configs_by_scale, K, repeats, strategy_name, L_list : config
        Experiment configuration

    Returns
    -------
    aggregated_by_scale : dict
        Aggregated data by (n, theta) key, with raw metrics preserved
    """
    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    aggregated_by_scale = {}

    # Initialize structure
    for n_size in n_list:
        theta_list = theta_configs_by_scale[n_size]
        for theta_idx, theta in enumerate(theta_list):
            key = (n_size, theta_idx)
            aggregated_by_scale[key] = {
                'cluster_counts': [],
                'metrics_list': [],
                'theta': theta
            }

    # Aggregate across trials
    for trial_result in trial_results_list:
        for key, data in trial_result.items():
            if key in aggregated_by_scale:
                aggregated_by_scale[key]['cluster_counts'].extend(data['cluster_counts'])
                aggregated_by_scale[key]['metrics_list'].extend(data['metrics_list'])

    # Print progress
    for n_size in n_list:
        theta_list = theta_configs_by_scale[n_size]
        for theta_idx, theta in enumerate(theta_list):
            key = (n_size, theta_idx)
            cluster_counts = aggregated_by_scale[key]['cluster_counts']
            metrics_list = aggregated_by_scale[key]['metrics_list']

            success_rate = len(metrics_list) / repeats if repeats > 0 else 0.0
            avg_clusters = np.mean(cluster_counts) if cluster_counts else 0.0
            first_10_err = metrics_list[0].get('first_10', np.nan) if metrics_list else np.nan

            print(f"{strategy_name} | n={n_size} | θ={theta:.6f} ({rate_names[theta_idx]}): "
                  f"Valid={len(metrics_list)}/{repeats} "
                  f"(Success={success_rate:.1%}), "
                  f"AvgK={avg_clusters:.2f}, "
                  f"First10={first_10_err:.4f}")

    return aggregated_by_scale


def compute_final_results(aggregated_data, n_list, theta_configs_by_scale,
                          K, repeats, strategy_name, L_list):
    """
    Compute final results by averaging raw metrics.

    Parameters
    ----------
    aggregated_data : dict
        Raw aggregated data from aggregate_parallel_results()
    n_list, theta_configs_by_scale, K, repeats, strategy_name, L_list : config
        Experiment configuration

    Returns
    -------
    all_results : list of dict
        Final aggregated results with averaged metrics
    """
    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    all_results = []

    for n_size in n_list:
        theta_list = theta_configs_by_scale[n_size]
        for theta_idx, theta in enumerate(theta_list):
            key = (n_size, theta_idx)
            if key in aggregated_data:
                cluster_counts = aggregated_data[key]['cluster_counts']
                metrics_list = aggregated_data[key]['metrics_list']

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
                result['theta_rate'] = rate_names[theta_idx]
                all_results.append(result)

    return all_results


def main_parallel(num_cores=10):
    """
    Main experiment runner with parallel repeats (recursive version).
    SCALED THETA: All theta values equal 1000^(-3/4) at n=1000.

    Parameters
    ----------
    num_cores : int
        Number of CPU cores to use for parallel processing
    """
    print("=" * 80)
    print("DISTANCE-BASED RECOVERY EXPERIMENT: PA Model (Recursive, Parallel Repeats)")
    print("SCALED THETA VERSION: All theta equal at n=1000, different decay rates")
    print("Alpha = 2 (super-linear preferential attachment)")
    print(f"Parallel cores: {num_cores}")
    print("=" * 80)

    # === Fixed parameters ===
    K = 2
    repeats = 200
    b = 10
    loop_bound = 0
    select_by = "initial"
    L_list = [10, 50]
    alpha = 2
    recovery_threshold_divisor = 2.0

    # === Scale configuration ===
    scale_configs = {
        500: {'n_per_component': 250},
        1000: {'n_per_component': 500},
        2000: {'n_per_component': 1000},
        3000: {'n_per_component': 1500},
        4000: {'n_per_component': 2000},
        5000: {'n_per_component': 2500},
        6000: {'n_per_component': 3000},
        7000: {'n_per_component': 3500},
        8000: {'n_per_component': 4000},
        9000: {'n_per_component': 4500},
        10000: {'n_per_component': 5000},
        12000: {'n_per_component': 6000},
        14000: {'n_per_component': 7000},
        16000: {'n_per_component': 8000},
        18000: {'n_per_component': 9000},
        20000: {'n_per_component': 10000}
    }

    n_list = [1000, 2000, 3000, 4000, 5000]
    strategies = [True]

    # === Reference point for scaled theta ===
    # All theta values will equal theta_ref at n=n_ref
    n_ref = 1000
    theta_ref = 0.01  # Reference theta value at n=1000

    # === Theta configuration (SCALED) ===
    # Use get_theta_coef to compute scaling coefficients for each rate
    coef_rate_1 = get_theta_coef(rate=1.0, n_ref=n_ref, theta_ref=theta_ref)
    coef_rate_3_4 = get_theta_coef(rate=3/4, n_ref=n_ref, theta_ref=theta_ref)
    coef_rate_1_2 = get_theta_coef(rate=1/2, n_ref=n_ref, theta_ref=theta_ref)

    def get_theta_list(n):
        theta_rate_1 = coef_rate_1 * pow(n, -1)       # fastest decrease
        theta_rate_3_4 = coef_rate_3_4 * pow(n, -3/4) # medium (equals 1.0 * n^(-3/4))
        theta_rate_1_2 = coef_rate_1_2 * pow(n, -1/2) # slowest decrease
        return [theta_rate_1, theta_rate_3_4, theta_rate_1_2]

    # === Generate theta configs ===
    theta_configs_by_scale = {}
    for n in n_list:
        theta_configs_by_scale[n] = get_theta_list(n)

    print(f"\n{'=' * 80}")
    print(f"Configuration (SCALED THETA):")
    print(f"  K = {K} components")
    print(f"  Repeats = {repeats}")
    print(f"  Strategies = {['no_protected', 'with_protected']}")
    print(f"  Reference point: n={n_ref}, theta={theta_ref:.6f}")
    print(f"  Theta rates: -1 (fast), -3/4 (medium), -1/2 (slow)")
    print(f"  Scales = {n_list}")
    for n in n_list:
        thetas = theta_configs_by_scale[n]
        print(f"    n={n}: theta values = [{thetas[0]:.6f}, {thetas[1]:.6f}, {thetas[2]:.6f}]")
    print(f"{'=' * 80}\n")

    # === Run experiments ===
    all_raw_data = {}  # Store raw aggregated data for all strategies

    for check_protected in strategies:
        strategy_name = "with_protected" if check_protected else "no_protected"

        print(f"\n{'=' * 80}")
        print(f"STRATEGY: {strategy_name} (using {num_cores} cores)")
        print(f"{'=' * 80}\n")

        # Create partial function with fixed parameters
        trial_runner = partial(
            run_single_trial,
            n_list=n_list,
            scale_configs=scale_configs,
            theta_configs_by_scale=theta_configs_by_scale,
            K=K,
            alpha=alpha,
            b=b,
            loop_bound=loop_bound,
            select_by=select_by,
            L_list=L_list,
            check_protected=check_protected,
            recovery_threshold_divisor=recovery_threshold_divisor
        )

        # Run trials in parallel with progress tracking
        print(f"Running {repeats} trials in parallel...")
        print(f"Progress message every 20 trials...\n")

        trial_results_list = []
        progress_interval = 20

        with Pool(num_cores) as pool:
            # Use imap_unordered for progress tracking
            for trial_num, trial_result in enumerate(
                tqdm(
                    pool.imap_unordered(trial_runner, range(repeats)),
                    total=repeats,
                    desc=f"{strategy_name:20s}",
                    unit="trial"
                ),
                start=1
            ):
                trial_results_list.append(trial_result)

                # Print progress message every N trials
                if trial_num % progress_interval == 0 or trial_num == repeats:
                    print(f"  Completed {trial_num}/{repeats} trials")

        print(f"All {repeats} trials completed for {strategy_name}.\n")
        print(f"All trials completed. Aggregating raw metrics...")

        # Aggregate results (keeps raw metrics)
        aggregated_data = aggregate_parallel_results(
            trial_results_list,
            n_list,
            theta_configs_by_scale,
            K,
            repeats,
            strategy_name,
            L_list
        )

        # Store raw data for later analysis
        all_raw_data[strategy_name] = aggregated_data

    # === SAVE RAW DATA TO CSV (Human-readable format) ===
    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    print(f"\nExporting individual trial data to CSV...")
    raw_data_list = []
    for strategy_name in ["no_protected", "with_protected"]:
        if strategy_name in all_raw_data:
            aggregated_data = all_raw_data[strategy_name]
            for (n_size, theta_idx), data in aggregated_data.items():
                cluster_counts = data['cluster_counts']
                metrics_list = data['metrics_list']
                theta = data['theta']

                # Add entry for each trial
                for trial_idx, cc in enumerate(cluster_counts, start=1):
                    row = {
                        'strategy': strategy_name,
                        'n': n_size,
                        'theta': theta,
                        'theta_rate': rate_names[theta_idx],
                        'trial': trial_idx,
                        'cluster_count': cc
                    }

                    # Add metrics if available for this trial
                    if trial_idx - 1 < len(metrics_list):
                        metrics = metrics_list[trial_idx - 1]
                        row.update(metrics)

                    raw_data_list.append(row)

    df_raw = pd.DataFrame(raw_data_list)
    raw_data_file = "distance_recovery_raw_data_experiment2_scaled_theta.csv"
    df_raw.to_csv(raw_data_file, index=False)
    print(f"CSV exported: {raw_data_file} ({len(df_raw)} records)")

    # === COMPUTE FINAL RESULTS (averages metrics) ===
    print(f"\n{'=' * 80}")
    print("Computing final aggregated results...")
    print(f"{'=' * 80}\n")

    df_results_list = []
    for strategy_name in ["no_protected", "with_protected"]:
        if strategy_name in all_raw_data:
            aggregated_data = all_raw_data[strategy_name]
            strategy_results = compute_final_results(
                aggregated_data,
                n_list,
                theta_configs_by_scale,
                K,
                repeats,
                strategy_name,
                L_list
            )
            df_results_list.extend(strategy_results)

    # Create summary DataFrame
    df_results = pd.DataFrame(df_results_list)

    # Reorder columns for clarity
    fixed_cols = ["strategy", "n", "theta", "valid_trials", "success_rate", "avg_clusters"]
    metric_cols = [c for c in df_results.columns if c not in fixed_cols]
    df_results = df_results[fixed_cols + sorted(metric_cols)]

    # === SAVE TEXT OUTPUT ===
    txt_file = f"distance_recovery_results_experiment2_scaled_theta_comprehensive.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("=" * 200)
        print(f"COMPREHENSIVE EXPERIMENTAL RESULTS (Recursive Version - SCALED THETA)")
        print(f"Alpha = {alpha}")
        print(f"All theta values equal {theta_ref:.6f} at n={n_ref}")
        print(f"Rates: -1 (fast decay), -3/4 (medium), -1/2 (slow decay)")
        print(f"Generated: {pd.Timestamp.now()}")
        print("=" * 200)
        print("\n\n")

        print_results_table(df_results, title="COMPLETE RESULTS TABLE (SCALED THETA)")
        print("\n\n")

        print_success_failure_breakdown(df_results, K, repeats)
        print("\n\n")

        print_comparison_table(df_results, n_list, get_theta_list)
        print("\n\n")

        print_per_strategy_summary(df_results, repeats)
        print("\n\n")

        print("=" * 200)
        print("END OF REPORT")
        print("=" * 200)

        sys.stdout = original_stdout

    print(f"Text report saved to: {txt_file}\n")

    # === PRINT TO CONSOLE ===
    print("\n" + "=" * 200)
    print("FORMATTED RESULTS - CONSOLE OUTPUT (SCALED THETA)")
    print("=" * 200)

    print_results_table(df_results, title="COMPLETE RESULTS TABLE (SCALED THETA)")
    print_success_failure_breakdown(df_results, K, repeats)
    print_comparison_table(df_results, n_list, get_theta_list)
    print_per_strategy_summary(df_results, repeats)

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE (SCALED THETA)")
    print(f"Raw data: {raw_data_file}")
    print(f"Text report: {txt_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import sys
    np.random.seed(15)
    main_parallel(num_cores=25)
