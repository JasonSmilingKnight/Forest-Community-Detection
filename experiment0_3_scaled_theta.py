"""
Main parallel runner for distance recovery experiments (recursive version).
K=3 UNBALANCED + SCALED THETA VERSION:
- Three trees with 3:2:1 size ratio (unbalanced)
- All theta values start from 0.01 at n=1200, but decrease at different rates

Parallelizes repeats across multiple cores using multiprocessing.
No pre-computed noise matrix - generates noise on-the-fly per trial.
"""

import numpy as np


def get_theta_coef(rate, n_ref=1200, theta_ref=None):
    """
    Compute coefficient so that coef * n^(-rate) = theta_ref at n = n_ref.
    """
    if theta_ref is None:
        theta_ref = pow(n_ref, -3/4)

    coef = theta_ref * pow(n_ref, rate)
    return coef

import pandas as pd
import os
from multiprocessing import Pool
from functools import partial
import pickle
from tqdm import tqdm

from leaf_stripping_recovery import generate_recursive_forest, Leaf_stripping, add_noise_edges
from experiment_distance_recovery_recursive_new import (
    aggregate_trial_results,
)
from utilities import (
    save_all_tables_to_txt,
    print_results_table,
    print_success_failure_breakdown,
    print_comparison_table,
    print_per_strategy_summary,
)
import networkx as nx


def extract_subgraph_and_compute_depth_unbalanced(forest, n_component_sizes, K):
    """
    Extract subgraph with different number of nodes from each component (unbalanced).

    For K=3 with 3:2:1 ratio:
    - Component 0: n_component_sizes[0] nodes (largest)
    - Component 1: n_component_sizes[1] nodes (medium)
    - Component 2: n_component_sizes[2] nodes (smallest)
    """
    selected_nodes = []
    original_components = []

    n_full_per_component = forest.vcount() // K

    for k in range(K):
        start_idx = k * n_full_per_component
        n_to_extract = n_component_sizes[k]
        component_nodes = list(range(start_idx, start_idx + n_to_extract))
        selected_nodes.extend(component_nodes)
        original_components.append(component_nodes)

    subforest = forest.induced_subgraph(selected_nodes)

    sub_components = []
    cumulative = 0
    for k in range(K):
        sub_components.append(list(range(cumulative, cumulative + n_component_sizes[k])))
        cumulative += n_component_sizes[k]

    if "tree" in subforest.es.attributes():
        tree_edges = [(e.source, e.target) for e in subforest.es if e["tree"]]
    else:
        tree_edges = [(e.source, e.target) for e in subforest.es]

    G_tree = nx.Graph()
    G_tree.add_edges_from(tree_edges)

    node_depth = {}
    for component in sub_components:
        root = min(component)
        if len(component) > 1:
            depths = nx.single_source_shortest_path_length(G_tree, root)
            node_depth.update(depths)
        else:
            node_depth[root] = 0

    return subforest, sub_components, node_depth


def compute_misclassification_metrics_unbalanced(
    new_labels, n_component_sizes, total_nodes, node_depth, L_list, initial_degrees=None
):
    """
    Compute misclassification metrics for different node subsets (unbalanced version).
    Each subset finds its own best label permutation independently.
    """
    from itertools import permutations

    metrics = {}
    K = len(n_component_sizes)
    unique_labels = [lab for lab in np.unique(new_labels) if lab != 0]

    # True cluster assignment based on component boundaries
    true_assignment = np.zeros(total_nodes, dtype=int)
    cumulative = 0
    for k in range(K):
        for i in range(n_component_sizes[k]):
            true_assignment[cumulative + i] = k
        cumulative += n_component_sizes[k]

    if len(unique_labels) < K:
        for L in L_list:
            metrics[f"first_{L}"] = np.nan
            metrics[f"highdeg_{L}"] = np.nan
        metrics["overall_classified"] = np.nan
        for d in range(3):
            if d == 0:
                metrics["root"] = np.nan
            else:
                metrics[f"depth_{d}"] = np.nan
        return metrics

    def best_error_for_subset(subset_indices):
        if len(subset_indices) == 0:
            return np.nan
        subset_indices = np.array(subset_indices)
        subset_true = true_assignment[subset_indices]
        subset_pred = new_labels[subset_indices]
        best_err = float('inf')
        for perm in permutations(range(K)):
            expected = np.array([unique_labels[perm[t]] for t in subset_true])
            err = np.mean(subset_pred != expected)
            if err < best_err:
                best_err = err
        return best_err

    # Component boundaries
    component_boundaries = []
    cumulative = 0
    for k in range(K):
        component_boundaries.append((cumulative, cumulative + n_component_sizes[k]))
        cumulative += n_component_sizes[k]

    # First-arriving metrics (node index = arrival order by construction)
    for L in L_list:
        first_L_indices = []
        for k in range(K):
            component_start, component_end = component_boundaries[k]
            component_nodes = list(range(component_start, component_end))
            first_L_indices.extend(component_nodes[:min(L, len(component_nodes))])
        metrics[f"first_{L}"] = best_error_for_subset(first_L_indices)

    # Highest-degree metrics (top-L by degree across the whole network)
    if initial_degrees is not None:
        for L in L_list:
            all_nodes = list(range(total_nodes))
            highdeg_L_indices = sorted(all_nodes, key=lambda i: initial_degrees.get(i, 0), reverse=True)[:L]
            metrics[f"highdeg_{L}"] = best_error_for_subset(highdeg_L_indices)

    # Overall classified
    classified = np.where(new_labels != 0)[0]
    classified = classified[classified < total_nodes]
    metrics["overall_classified"] = best_error_for_subset(classified)

    # Depth metrics (capped at 0-2)
    max_depth = max(node_depth.values()) if node_depth else 0
    for d in range(min(max_depth + 1, 3)):
        depth_nodes = [i for i in range(total_nodes) if node_depth.get(i, -1) == d]
        err = best_error_for_subset(depth_nodes)
        if d == 0:
            metrics["root"] = err
        else:
            metrics[f"depth_{d}"] = err

    return metrics


def run_single_trial_for_scale_unbalanced(
    forest, components, n_component_sizes, theta, K, b, loop_bound,
    min_component_size, select_by, L_list, check_protected_removal, recovery_threshold_divisor=5.0
):
    """
    Run a single trial for a specific scale (unbalanced version for K=3).
    """
    total_nodes = sum(n_component_sizes)

    # Add noise to a copy of the forest
    graf_with_noise = forest.copy()
    add_noise_edges(graf_with_noise, theta)

    # Compute node depths
    if "tree" in graf_with_noise.es.attributes():
        tree_edges = [(e.source, e.target) for e in graf_with_noise.es if e["tree"]]
    else:
        tree_edges = [(e.source, e.target) for e in graf_with_noise.es]

    G_tree = nx.Graph()
    G_tree.add_edges_from(tree_edges)

    node_depth = {}
    for component in components:
        root = min(component)
        if len(component) > 1:
            depths = nx.single_source_shortest_path_length(G_tree, root)
            node_depth.update(depths)
        else:
            node_depth[root] = 0

    # Run leaf stripping
    A_noisy = np.array(graf_with_noise.get_adjacency().data)
    A_noisy[A_noisy > 1] = 1

    # Compute initial degrees for highdeg metrics
    initial_degrees = {i: int(A_noisy[i].sum()) for i in range(total_nodes)}

    result = Leaf_stripping(
        A_input=A_noisy,
        K=K,
        b=b,
        d=loop_bound,
        select_by=select_by,
        min_component_size=min_component_size,
        recover=True,
        recover_mode="distance_threshold",
        threshold_by="order",
        mode="True",
        check_protected_removal=check_protected_removal,
        recovery_threshold_divisor=recovery_threshold_divisor
    )

    if len(result) == 5:
        cluster_count, new_labels, tmp, component_sizes, reference_nodes_per_cluster = result
    else:
        cluster_count, new_labels, tmp, component_sizes = result

    # Compute metrics if successful
    if cluster_count >= K:
        metrics = compute_misclassification_metrics_unbalanced(
            new_labels, n_component_sizes, total_nodes, node_depth, L_list,
            initial_degrees=initial_degrees
        )
        return cluster_count, metrics
    else:
        return cluster_count, None


def run_single_trial(trial_id, n_list, scale_configs, theta_configs_by_scale, K, alpha,
                     b, loop_bound, select_by, L_list,
                     check_protected, recovery_threshold_divisor):
    """
    Run a single trial across all scales for a given strategy (K=3 unbalanced + scaled theta version).
    No pre-computed noise matrix - generates noise on-the-fly.
    """
    np.random.seed(42 + trial_id)

    n_base = max(n_list)
    # Base forest needs enough nodes for the largest component
    n_base_per_component = max(scale_configs[n_base]['n_component_sizes'])

    # Generate base forest once per trial (with enough nodes for K=3)
    forest_base, components_base, assignments_base = generate_recursive_forest(
        n=n_base_per_component * K, K=K, alpha=alpha
    )

    trial_results = {}

    # Process each scale
    for n_size in n_list:
        n_component_sizes = scale_configs[n_size]['n_component_sizes']
        theta_list = theta_configs_by_scale[n_size]

        # Extract subgraph with unbalanced sizes
        forest, components, _ = extract_subgraph_and_compute_depth_unbalanced(
            forest_base, n_component_sizes, K
        )

        # Process each theta value (use index to avoid key collision at reference point)
        for theta_idx, theta in enumerate(theta_list):
            cluster_count, metrics = run_single_trial_for_scale_unbalanced(
                forest=forest,
                components=components,
                n_component_sizes=n_component_sizes,
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

            # Use (n_size, theta_idx) as key to avoid collision when theta values are equal
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
    """
    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    aggregated_by_scale = {}

    for n_size in n_list:
        theta_list = theta_configs_by_scale[n_size]
        for theta_idx, theta in enumerate(theta_list):
            key = (n_size, theta_idx)
            aggregated_by_scale[key] = {
                'cluster_counts': [],
                'metrics_list': [],
                'theta': theta
            }

    for trial_result in trial_results_list:
        for key, data in trial_result.items():
            if key in aggregated_by_scale:
                aggregated_by_scale[key]['cluster_counts'].extend(data['cluster_counts'])
                aggregated_by_scale[key]['metrics_list'].extend(data['metrics_list'])

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
    Main experiment runner with parallel repeats (K=3 unbalanced 3:2:1 + scaled theta version).
    All theta values equal theta_ref at n=n_ref, with different decay rates.
    """
    print("=" * 80)
    print("DISTANCE-BASED RECOVERY EXPERIMENT: PA Model (Recursive, Parallel)")
    print("K=3 UNBALANCED + SCALED THETA VERSION:")
    print("  - 3:2:1 component size ratio")
    print("  - All theta equal at n=1200, different decay rates")
    print(f"Parallel cores: {num_cores}")
    print("=" * 80)

    # === Fixed parameters ===
    K = 3
    repeats = 200
    b = 15
    loop_bound = 0
    select_by = "initial"
    L_list = [10, 50]
    alpha = 0
    recovery_threshold_divisor = 2.0

    # === Scale configuration (K=3 UNBALANCED 3:2:1 ratio) ===
    # For total n nodes with K=3: large = n/2, medium = n/3, small = n/6
    scale_configs = {
        600:   {'n_component_sizes': [300, 200, 100]},
        1000:  {'n_component_sizes': [500, 333, 167]},
        1200:  {'n_component_sizes': [600, 400, 200]},
        1800:  {'n_component_sizes': [900, 600, 300]},
        2000:  {'n_component_sizes': [1000, 667, 333]},
        2400:  {'n_component_sizes': [1200, 800, 400]},
        3000:  {'n_component_sizes': [1500, 1000, 500]},
        3600:  {'n_component_sizes': [1800, 1200, 600]},
        4000:  {'n_component_sizes': [2000, 1333, 667]},
        4200:  {'n_component_sizes': [2100, 1400, 700]},
        4800:  {'n_component_sizes': [2400, 1600, 800]},
        5000:  {'n_component_sizes': [2500, 1667, 833]},
        5400:  {'n_component_sizes': [2700, 1800, 900]},
        6000:  {'n_component_sizes': [3000, 2000, 1000]},
        7200:  {'n_component_sizes': [3600, 2400, 1200]},
        9000:  {'n_component_sizes': [4500, 3000, 1500]},
        10000: {'n_component_sizes': [5000, 3333, 1667]},
        12000: {'n_component_sizes': [6000, 4000, 2000]},
    }

    n_list = [1200, 2400, 3600, 4800, 6000]
    strategies = [True]

    # === Reference point for scaled theta ===
    n_ref = 1200
    theta_ref = 0.01

    # === Theta configuration (SCALED) ===
    coef_rate_1 = get_theta_coef(rate=1.0, n_ref=n_ref, theta_ref=theta_ref)
    coef_rate_3_4 = get_theta_coef(rate=3/4, n_ref=n_ref, theta_ref=theta_ref)
    coef_rate_1_2 = get_theta_coef(rate=1/2, n_ref=n_ref, theta_ref=theta_ref)

    def get_theta_list(n):
        theta_rate_1 = coef_rate_1 * pow(n, -1)
        theta_rate_3_4 = coef_rate_3_4 * pow(n, -3/4)
        theta_rate_1_2 = coef_rate_1_2 * pow(n, -1/2)
        return [theta_rate_1, theta_rate_3_4, theta_rate_1_2]

    # === Generate theta configs ===
    theta_configs_by_scale = {}
    for n in n_list:
        theta_configs_by_scale[n] = get_theta_list(n)

    print(f"\n{'=' * 80}")
    print(f"Configuration (K=3 UNBALANCED 3:2:1 + SCALED THETA):")
    print(f"  K = {K} components")
    print(f"  Component ratio = 3:2:1 (large:medium:small)")
    print(f"  Repeats = {repeats}")
    print(f"  Reference point: n={n_ref}, theta={theta_ref:.6f}")
    print(f"  Theta rates: -1 (fast), -3/4 (medium), -1/2 (slow)")
    print(f"  Scales = {n_list}")
    for n in n_list:
        sizes = scale_configs[n]['n_component_sizes']
        thetas = theta_configs_by_scale[n]
        ratios = [s / sizes[-1] for s in sizes]
        ratio_str = ":".join([f"{r:.1f}" for r in ratios])
        print(f"    n={n}: sizes={sizes} (ratio {ratio_str}), "
              f"theta=[{thetas[0]:.6f}, {thetas[1]:.6f}, {thetas[2]:.6f}]")
    print(f"{'=' * 80}\n")

    # === Run experiments ===
    all_raw_data = {}

    for check_protected in strategies:
        strategy_name = "with_protected" if check_protected else "no_protected"

        print(f"\n{'=' * 80}")
        print(f"STRATEGY: {strategy_name} (using {num_cores} cores)")
        print(f"{'=' * 80}\n")

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

        print(f"Running {repeats} trials in parallel...")
        print(f"Progress message every 20 trials...\n")

        trial_results_list = []
        progress_interval = 20

        with Pool(num_cores) as pool:
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

                if trial_num % progress_interval == 0 or trial_num == repeats:
                    print(f"  Completed {trial_num}/{repeats} trials")

        print(f"All {repeats} trials completed for {strategy_name}.\n")
        print(f"All trials completed. Aggregating raw metrics...")

        aggregated_data = aggregate_parallel_results(
            trial_results_list,
            n_list,
            theta_configs_by_scale,
            K,
            repeats,
            strategy_name,
            L_list
        )

        all_raw_data[strategy_name] = aggregated_data

    # === SAVE RAW DATA TO CSV ===
    print(f"\nExporting individual trial data to CSV...")
    raw_data_list = []
    rate_names = ["rate_1", "rate_3_4", "rate_1_2"]
    for strategy_name in ["no_protected", "with_protected"]:
        if strategy_name in all_raw_data:
            aggregated_data = all_raw_data[strategy_name]
            for (n_size, theta_idx), data in aggregated_data.items():
                cluster_counts = data['cluster_counts']
                metrics_list = data['metrics_list']
                theta = data['theta']
                theta_rate = rate_names[theta_idx]

                for trial_idx, cc in enumerate(cluster_counts, start=1):
                    row = {
                        'strategy': strategy_name,
                        'n': n_size,
                        'theta': theta,
                        'theta_rate': theta_rate,
                        'trial': trial_idx,
                        'cluster_count': cc,
                        'K': K,
                        'component_ratio': '3:2:1'
                    }

                    if trial_idx - 1 < len(metrics_list):
                        metrics = metrics_list[trial_idx - 1]
                        row.update(metrics)

                    raw_data_list.append(row)

    df_raw = pd.DataFrame(raw_data_list)
    raw_data_file = "distance_recovery_raw_data_experiment0_3_scaled_theta.csv"
    df_raw.to_csv(raw_data_file, index=False)
    print(f"CSV exported: {raw_data_file} ({len(df_raw)} records)")

    # === COMPUTE FINAL RESULTS ===
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

    df_results = pd.DataFrame(df_results_list)

    fixed_cols = ["strategy", "n", "theta", "valid_trials", "success_rate", "avg_clusters"]
    metric_cols = [c for c in df_results.columns if c not in fixed_cols]
    df_results = df_results[fixed_cols + sorted(metric_cols)]

    # === SAVE TEXT OUTPUT ===
    txt_file = f"distance_recovery_results_experiment0_3_scaled_theta_comprehensive.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("=" * 200)
        print(f"COMPREHENSIVE EXPERIMENTAL RESULTS (K=3 Recursive Version - SCALED THETA + UNBALANCED 3:2:1)")
        print(f"All theta values equal {theta_ref:.6f} at n={n_ref}")
        print(f"Rates: -1 (fast decay), -3/4 (medium), -1/2 (slow decay)")
        print(f"Component ratio: 3:2:1 (large:medium:small)")
        print(f"Generated: {pd.Timestamp.now()}")
        print("=" * 200)
        print("\n\n")

        print_results_table(df_results, title="COMPLETE RESULTS TABLE (K=3 SCALED THETA + UNBALANCED 3:2:1)")
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
    print("FORMATTED RESULTS - CONSOLE OUTPUT (K=3 SCALED THETA + UNBALANCED 3:2:1)")
    print("=" * 200)

    print_results_table(df_results, title="COMPLETE RESULTS TABLE (K=3 SCALED THETA + UNBALANCED 3:2:1)")
    print_success_failure_breakdown(df_results, K, repeats)
    print_comparison_table(df_results, n_list, get_theta_list)
    print_per_strategy_summary(df_results, repeats)

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE (K=3 SCALED THETA + UNBALANCED 3:2:1)")
    print(f"Raw data: {raw_data_file}")
    print(f"Text report: {txt_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import sys
    np.random.seed(200)
    main_parallel(num_cores=25)
