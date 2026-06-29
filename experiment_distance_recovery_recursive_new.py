"""
Experiment evaluating distance-based recovery for leaf-stripping
on PA mixture graphs with varying n and theta, comparing:
check_protected_removal=True (early protected-node removal)
check_protected_removal=False (standard leaf-stripping)

UPDATED: Uses generate_recursive_forest + add_noise_edges
Generates n=4000 once per trial, then extracts n=2000 and n=1000 subgraphs
with separate noise addition for each size.
"""

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import trange
from collections import defaultdict
import sys
import igraph as ig
from io import StringIO
from contextlib import redirect_stdout

from leaf_stripping_recovery import (
    Leaf_stripping,
    generate_recursive_forest,
    add_noise_edges,
)


# ============================================================================
# TABLE PRINTING UTILITIES
# ============================================================================

def save_all_tables_to_txt(df_results, n_list, get_theta_list, K, repeats, output_prefix="results"):
    """
    Save all formatted tables to a single comprehensive TXT file.
    """
    txt_file = f"{output_prefix}_comprehensive.txt"
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        
        print("=" * 200)
        print(f"COMPREHENSIVE EXPERIMENTAL RESULTS")
        print(f"Generated: {pd.Timestamp.now()}")
        print("=" * 200)
        print("\n\n")
        
        print_results_table(df_results, title="COMPLETE RESULTS TABLE")
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
    
    print(f"\nComprehensive report saved to: {txt_file}")
    return txt_file


def print_results_table(df, title="COMPLETE RESULTS TABLE"):
    """Print a comprehensive formatted table of experimental results."""
    depth_cols = [col for col in df.columns if col.startswith('depth_')]
    depth_cols_sorted = sorted(depth_cols, key=lambda x: int(x.split('_')[1]))
    highdeg_cols = [col for col in df.columns if col.startswith('highdeg_')]
    highdeg_cols_sorted = sorted(highdeg_cols, key=lambda x: int(x.split('_')[1]))

    display_cols = [
        'strategy', 'n', 'theta', 'valid_trials', 'success_rate', 'avg_clusters',
        'first_10', 'first_50'
    ] + highdeg_cols_sorted + [
        'root'
    ] + depth_cols_sorted + [
        'overall_classified'
    ]

    # Include theta_rate in display if available
    if 'theta_rate' in df.columns:
        display_cols.insert(display_cols.index('theta') + 1, 'theta_rate')
    display_cols = [col for col in display_cols if col in df.columns]
    df_display = df[display_cols].copy()

    # Sort by theta_rate (group same rate together), then by n
    if 'theta_rate' in df_display.columns:
        df_display = df_display.sort_values(['theta_rate', 'n']).reset_index(drop=True)
    else:
        df_display = df_display.sort_values(['theta', 'n']).reset_index(drop=True)

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
    df_display['avg_clusters'] = df_display['avg_clusters'].apply(lambda x: format_float(x, 2))
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
        'valid_trials': 'Valid',
        'success_rate': 'Success%',
        'avg_clusters': 'AvgK',
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


def print_success_failure_breakdown(df, K, repeats):
    """Print a detailed breakdown of success/failure statistics."""
    count_cols = [col for col in df.columns if col.startswith('count_')]
    
    print("\n" + "=" * 120)
    print("SUCCESS/FAILURE BREAKDOWN BY CONFIGURATION")
    print("=" * 120)
    print()
    
    for _, row in df.iterrows():
        print(f"Strategy: {row['strategy']:15s} | n={int(row['n']):5d} | θ={row['theta']:.6f}")
        
        actual_trials = int(row['valid_trials'] / row['success_rate']) if row['success_rate'] > 0 else repeats
        print(f"  Success: {row['success_rate']:6.1%} ({int(row['valid_trials'])}/{actual_trials} trials) | Avg K: {row['avg_clusters']:.2f}")
        
        if count_cols:
            dist_parts = []
            for k in range(K + 1):
                col = f'count_{k}'
                if col in row and not pd.isna(row[col]):
                    dist_parts.append(f"K={k}:{int(row[col])}")
            col_plus = f'count_{K}plus'
            if col_plus in row and not pd.isna(row[col_plus]):
                dist_parts.append(f"K>{K}:{int(row[col_plus])}")
            if dist_parts:
                print(f"  Distribution: {' | '.join(dist_parts)}")
        
        if row['success_rate'] > 0 and not pd.isna(row.get('first_10', np.nan)):
            print(f"  Performance (successful trials): ", end="")
            metrics = []
            for metric, label in [('first_10', 'F10'), ('first_50', 'F50'),
                                 ('highdeg_10', 'HD10'), ('highdeg_50', 'HD50'),
                                 ('overall_classified', 'Overall'), ('root', 'Root')]:
                if metric in row and not pd.isna(row[metric]):
                    metrics.append(f"{label}={row[metric]:.4f}")
            print(" | ".join(metrics))
        else:
            print(f"  Performance: N/A (all trials failed)")
        print()


def print_comparison_table(df, n_list, get_theta_list):
    """Print side-by-side comparison of strategies."""
    print("\n" + "=" * 180)
    print("STRATEGY COMPARISON (with_protected vs no_protected)")
    print("=" * 180)
    print()
    
    comparison_rows = []
    
    for n in n_list:
        theta_list = get_theta_list(n)
        for theta in theta_list:
            subset = df[(df['n'] == n) & (np.isclose(df['theta'], theta))]
            
            if len(subset) == 2:
                no_prot = subset[subset['strategy'] == 'no_protected'].iloc[0]
                with_prot = subset[subset['strategy'] == 'with_protected'].iloc[0]
                
                def safe_format(val, fmt=".4f"):
                    if pd.isna(val):
                        return "NaN"
                    return f"{val:{fmt}}"
                
                comparison_rows.append({
                    'n': int(n),
                    'theta': f"{theta:.6f}",
                    'no_success': f"{no_prot['success_rate']:.1%}",
                    'with_success': f"{with_prot['success_rate']:.1%}",
                    'diff_success': f"{(with_prot['success_rate']-no_prot['success_rate'])*100:+.1f}%",
                    'no_avgK': f"{no_prot['avg_clusters']:.2f}",
                    'with_avgK': f"{with_prot['avg_clusters']:.2f}",
                    'no_F10': safe_format(no_prot.get('first_10', np.nan)),
                    'with_F10': safe_format(with_prot.get('first_10', np.nan)),
                    'diff_F10': safe_format(with_prot['first_10'] - no_prot['first_10'], "+.4f") if not pd.isna(with_prot.get('first_10', np.nan)) and not pd.isna(no_prot.get('first_10', np.nan)) else "N/A",
                    'no_Overall': safe_format(no_prot['overall_classified']),
                    'with_Overall': safe_format(with_prot['overall_classified']),
                    'diff_Overall': safe_format(with_prot['overall_classified'] - no_prot['overall_classified'], "+.4f") if not pd.isna(with_prot['overall_classified']) and not pd.isna(no_prot['overall_classified']) else "N/A",
                })
            elif len(subset) == 1:
                row = subset.iloc[0]
                strategy = row['strategy']
                
                def safe_val(is_strategy, val, fmt=".4f"):
                    if not is_strategy:
                        return "N/A"
                    if pd.isna(val):
                        return "NaN"
                    if fmt.endswith('%'):
                        return f"{val:.1%}"
                    return f"{val:{fmt}}"
                
                is_no_prot = (strategy == 'no_protected')
                is_with_prot = (strategy == 'with_protected')
                
                comparison_rows.append({
                    'n': int(n),
                    'theta': f"{theta:.6f}",
                    'no_success': safe_val(is_no_prot, row['success_rate'], '.1%'),
                    'with_success': safe_val(is_with_prot, row['success_rate'], '.1%'),
                    'diff_success': "N/A",
                    'no_avgK': safe_val(is_no_prot, row['avg_clusters'], '.2f'),
                    'with_avgK': safe_val(is_with_prot, row['avg_clusters'], '.2f'),
                    'no_F10': safe_val(is_no_prot, row.get('first_10', np.nan)),
                    'with_F10': safe_val(is_with_prot, row.get('first_10', np.nan)),
                    'diff_F10': "N/A",
                    'no_Overall': safe_val(is_no_prot, row['overall_classified']),
                    'with_Overall': safe_val(is_with_prot, row['overall_classified']),
                    'diff_Overall': "N/A",
                })
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    comparison_df.columns = [
        'n', 'θ',
        'No_Succ%', 'With_Succ%', 'Δ_Succ',
        'No_AvgK', 'With_AvgK',
        'No_F10', 'With_F10', 'Δ_F10',
        'No_Overall', 'With_Overall', 'Δ_Overall'
    ]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 180)


def print_per_strategy_summary(df, repeats):
    """Print aggregated statistics per strategy."""
    print("\n" + "=" * 100)
    print("PER-STRATEGY AGGREGATE PERFORMANCE")
    print("=" * 100)
    print()
    
    for strategy in df['strategy'].unique():
        strategy_df = df[df["strategy"] == strategy]
        print(f"Strategy: {strategy}")
        print("-" * 100)
        
        print(f"  Configurations tested: {len(strategy_df)}")
        print(f"  Avg success rate:      {strategy_df['success_rate'].mean():.1%}")
        print(f"  Avg valid trials:      {strategy_df['valid_trials'].mean():.1f}/{repeats}")
        print(f"  Avg clusters found:    {strategy_df['avg_clusters'].mean():.2f}")
        print()
        
        print(f"  Average Misclassification Rates (across successful trials):")
        for metric in ["first_10", "first_50", "highdeg_10", "highdeg_50", "overall_classified", "root"]:
            if metric in strategy_df.columns:
                avg = strategy_df[metric].mean()
                std = strategy_df[metric].std()
                print(f"    {metric:20s}: {avg:.4f} ± {std:.4f}")
        
        print()


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def compute_misclassification_metrics(
    new_labels, n_per_component, total_nodes, node_depth, L_list, initial_degrees=None
):
    """Compute misclassification metrics for different node subsets.
    Each subset finds its own best label permutation independently."""
    from itertools import permutations

    metrics = {}
    K = total_nodes // n_per_component
    unique_labels = [lab for lab in np.unique(new_labels) if lab != 0]
    true_assignment = np.array([i // n_per_component for i in range(total_nodes)])

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
        """Find the best permutation and min error for a subset of node indices."""
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

    # First-arriving metrics (node index = arrival order by construction)
    for L in L_list:
        first_L_indices = []
        for k in range(K):
            component_start = k * n_per_component
            component_end = component_start + n_per_component
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

    # Depth metrics: root (depth 0), depth_1, depth_2
    max_depth = max(node_depth.values()) if node_depth else 0
    for d in range(min(max_depth + 1, 3)):
        depth_nodes = [i for i in range(total_nodes) if node_depth.get(i, -1) == d]
        err = best_error_for_subset(depth_nodes)
        if d == 0:
            metrics["root"] = err
        else:
            metrics[f"depth_{d}"] = err

    return metrics


def extract_subgraph_and_compute_depth(forest, n_per_component, K):
    """
    Extract subgraph with first n_per_component nodes from each component.
    Base forest is n=4000, so nodes are arranged as:
    - Component 0: nodes 0-1999 (2000 nodes)
    - Component 1: nodes 2000-3999 (2000 nodes)
    
    Extraction rule for n_per_component nodes per component:
    For K=2, base forest n=4000:
    - n=2000: [0-999], [2000-2999]
    - n=1000: [0-499], [2000-2499]
    - n=500: [0-249], [2000-2249]
    
    Parameters
    ----------
    forest : igraph.Graph
        The full forest (n=4000)
    n_per_component : int
        Number of nodes to extract per component
    K : int
        Number of components
    
    Returns
    -------
    subforest : igraph.Graph
        Extracted subgraph
    sub_components : list of list of int
        Reindexed component lists
    node_depth : dict
        Node depth mapping
    """
    # Extract n_per_component nodes from each component
    selected_nodes = []
    original_components = []
    
    n_full_per_component = forest.vcount() // K  # Size per component in base forest (4000 for n=8000, K=2)
    
    for k in range(K):
        start_idx = k * n_full_per_component
        component_nodes = list(range(start_idx, start_idx + n_per_component))
        selected_nodes.extend(component_nodes)
        original_components.append(component_nodes)
    
    # Create induced subgraph
    subforest = forest.induced_subgraph(selected_nodes)
    
    # After extraction, nodes are reindexed to 0, 1, 2, ...
    # So new components are simple ranges
    sub_components = []
    for k in range(K):
        sub_components.append(list(range(k * n_per_component, (k+1) * n_per_component)))
    
    # Compute node depth in the subgraph
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


def run_single_trial_for_scale(
    forest, components, n_per_component, theta, K, b, loop_bound, 
    min_component_size, select_by, L_list, check_protected_removal, recovery_threshold_divisor=5.0
):
    """
    Run a single trial for a specific scale.
    
    Parameters
    ----------
    forest : igraph.Graph
        The base forest (either full 4000 or subgraph)
    components : list of list of int
        Component node lists
    n_per_component : int
        Nodes per component
    theta : float
        Noise probability
    recovery_threshold_divisor : float
        Divisor for recovery threshold calculation (max_removed_deg / divisor)
    ... (other parameters)
    
    
    Returns
    -------
    cluster_count : int
        Number of clusters found
    metrics : dict or None
        Performance metrics (None if trial failed)
    """
    total_nodes = n_per_component * K
    
    # Add noise to a copy of the forest
    graf_with_noise = forest.copy()
    add_noise_edges(graf_with_noise, theta)
    
    # Compute node depths
    # Check if 'tree' attribute exists
    if "tree" in graf_with_noise.es.attributes():
        tree_edges = [(e.source, e.target) for e in graf_with_noise.es if e["tree"]]
    else:
        # If no 'tree' attribute, assume all edges are tree edges
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
    
    # Handle return values (may include reference_nodes_per_cluster for distance recovery)
    if len(result) == 5:
        cluster_count, new_labels, tmp, component_sizes, reference_nodes_per_cluster = result
    else:
        cluster_count, new_labels, tmp, component_sizes = result
        reference_nodes_per_cluster = None
    
    # Compute metrics if successful
    if cluster_count >= K:
        # Check labels of the K cores BEFORE recovery (from tmp = predicted_labels before recovery)
        # tmp contains the original leaf-stripping output before recovery reassigned unclassified nodes
        core_nodes_before = np.where((tmp > 0) & (tmp <= K))[0]
        core_nodes_before = core_nodes_before[core_nodes_before < total_nodes]
        
        # Print reference nodes used in recovery
        if reference_nodes_per_cluster is not None:
            print(f"\n>>> Reference nodes used in distance recovery:")
            for cluster_idx, ref_nodes in enumerate(reference_nodes_per_cluster):
                print(f"    Cluster {cluster_idx + 1}: {len(ref_nodes)} reference nodes")
                print(f"      Node IDs: {sorted(ref_nodes)}")
        
        if len(core_nodes_before) > 0:
            core_labels_before = tmp[core_nodes_before]
            core_depths_before = np.array([node_depth.get(i, -1) for i in core_nodes_before])
            
            # Map nodes to their original component
            node_to_component = {}
            for comp_idx, comp_nodes in enumerate(components):
                for node in comp_nodes:
                    node_to_component[node] = comp_idx
            
            # Count distribution of core nodes per predicted cluster BEFORE recovery
            print(f"\n>>> Core nodes (K={K}) BEFORE recovery:")
            print(f"    Total core nodes: {len(core_nodes_before)}/{total_nodes}")
            
            # Show depth/layer distribution of core nodes
            print(f"    Depth distribution of core nodes:")
            for d in sorted(np.unique(core_depths_before)):
                count = np.sum(core_depths_before == d)
                print(f"      Depth {d}: {count} nodes")
            
            # Show per cluster breakdown with node and component info
            for pred_label in sorted(np.unique(core_labels_before)):
                nodes_in_cluster = core_nodes_before[core_labels_before == pred_label]
                depths_in_cluster = core_depths_before[core_labels_before == pred_label]
                
                # Show which components these nodes come from
                components_in_cluster = [node_to_component.get(node, -1) for node in nodes_in_cluster]
                unique_comps = sorted(set(components_in_cluster))
                comp_distribution = {c: sum(1 for x in components_in_cluster if x == c) for c in unique_comps}
                
                print(f"    Predicted cluster {pred_label}: {len(nodes_in_cluster)} nodes, depths: {sorted(np.unique(depths_in_cluster))}")
                print(f"      Component distribution: {comp_distribution}")
                print(f"      Node IDs: {sorted(nodes_in_cluster)}")
        
        # Also show AFTER recovery for comparison
        core_nodes_after = np.where((new_labels > 0) & (new_labels <= K))[0]
        core_nodes_after = core_nodes_after[core_nodes_after < total_nodes]
        print(f"\n>>> After recovery: {len(core_nodes_after)} nodes classified (vs {len(core_nodes_before)} before)")
        
        # Compute initial degrees from noisy adjacency for highest-degree metrics
        initial_degrees = {i: int(A_noisy[i].sum()) for i in range(total_nodes)}
        metrics = compute_misclassification_metrics(
            new_labels, n_per_component, total_nodes, node_depth, L_list,
            initial_degrees=initial_degrees
        )
        return cluster_count, metrics
    else:
        return cluster_count, None



def aggregate_trial_results(cluster_counts, metrics_list, K, repeats, strategy_name, n, theta, L_list):
    """
    Aggregate results from multiple trials.
    
    Parameters
    ----------
    cluster_counts : list of int
        Cluster counts from all trials
    metrics_list : list of dict
        Metrics from successful trials
    K : int
        Target number of clusters
    repeats : int
        Total number of trials
    strategy_name : str
        Strategy identifier
    n : int
        Graph size
    theta : float
        Noise probability
    L_list : list of int
        L values for first_L metrics
    
    Returns
    -------
    result : dict
        Aggregated result dictionary
    """
    success_rate = len(metrics_list) / repeats if repeats > 0 else 0.0
    
    # Cluster count distribution
    cluster_count_dist = {}
    for k in range(K):
        cluster_count_dist[f"count_{k}"] = sum(1 for c in cluster_counts if c == k)
    cluster_count_dist[f"count_{K}"] = sum(1 for c in cluster_counts if c == K)
    cluster_count_dist[f"count_{K}plus"] = sum(1 for c in cluster_counts if c > K)
    
    avg_cluster_count = np.mean(cluster_counts) if cluster_counts else 0.0
    
    result = {
        "strategy": strategy_name,
        "n": n,
        "theta": theta,
        "valid_trials": len(metrics_list),
        "success_rate": success_rate,
        "avg_clusters": avg_cluster_count,
    }
    result.update(cluster_count_dist)
    
    # Average metrics across successful trials
    if not metrics_list:
        for L in L_list:
            result[f"first_{L}"] = np.nan
            result[f"highdeg_{L}"] = np.nan
        result["overall_classified"] = np.nan
        result["root"] = np.nan
        for d in range(1, 3):
            result[f"depth_{d}"] = np.nan
    else:
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        for key in sorted(all_keys):
            values = [m.get(key, np.nan) for m in metrics_list]
            result[key] = np.nanmean(values)
    
    return result


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main experiment runner with CSV and TXT output.
    Generates n=4000 forests, then extracts n=2000 and n=1000 subgraphs.
    """
    print("=" * 80)
    print("DISTANCE-BASED RECOVERY EXPERIMENT: PA Model")
    print("Multi-scale approach: Generate n=4000, extract smaller scales")
    print("=" * 80)

    # === Fixed parameters ===
    K = 2
    repeats = 50
    b = 5
    loop_bound = 0
    min_component_size = 50
    select_by = "initial"
    L_list = [5, 10, 50]
    alpha = -0.5
    recovery_threshold_divisor = 1.0

    # === Scale configuration ===
    # For each scale, specify nodes per component and theta
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
        10000: {'n_per_component': 5000}
    }
    
    n_list = [1000, 2000, 3000, 4000, 5000]
    strategies = [True]  # check_protected_removal values
    
    # Automatically use the largest n as base forest
    n_base = max(n_list)
    n_base_per_component = n_base // K

    # === Theta configuration ===
    def get_theta_list(n):
        return [
        1*pow(n,-1/2)
        ]

    # === Storage for all results ===
    all_results = []

    # === Generate theta configs for all scales ===
    theta_configs_by_scale = {}
    for n in n_list:
        theta_configs_by_scale[n] = get_theta_list(n)

    print(f"\n{'=' * 80}")
    print(f"Configuration:")
    print(f"  K = {K} components")
    print(f"  Repeats = {repeats}")
    print(f"  Strategies = {['no_protected', 'with_protected']}")
    print(f"  Scales = {n_list}")
    for n in n_list:
        print(f"    n={n}: theta values = {theta_configs_by_scale[n]}")
    print(f"{'=' * 80}\n")

    # === Run experiments ===
    for check_protected in strategies:
        strategy_name = "with_protected" if check_protected else "no_protected"
        
        print(f"\n{'=' * 80}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'=' * 80}")
        
        # Storage for results across all scales
        # Initialize with (n_size, theta) tuple keys to match usage in trial loop
        results_by_scale = {}
        for n_size in n_list:
            theta_list = theta_configs_by_scale[n_size]
            for theta in theta_list:
                key = (n_size, theta)
                results_by_scale[key] = {'cluster_counts': [], 'metrics_list': []}
        
        # Run all trials
        for trial in trange(repeats, desc=f"{strategy_name} - Multi-scale"):
            # Step 1: Generate base forest with largest n from n_list
            forest_4000, components_4000, assignments_4000 = generate_recursive_forest(
                n=n_base * K, K=K, alpha=alpha  # n_base*K for total nodes
            )
            
            # Step 2: Process each scale
            for n_size in n_list:
                n_per_component = scale_configs[n_size]['n_per_component']
                total_nodes = n_size
                
                # Get appropriate theta for this scale
                theta_list = theta_configs_by_scale[n_size]
                
                # For each theta value at this scale
                for theta in theta_list:
                    # Extract subgraph based on n_per_component
                    # Always extract to ensure correct forest size and node ordering
                    forest, components, _ = extract_subgraph_and_compute_depth(forest_4000, n_per_component, K)
                    
                    # Run single trial
                    cluster_count, metrics = run_single_trial_for_scale(
                        forest=forest,
                        components=components,
                        n_per_component=n_per_component,
                        theta=theta,
                        K=K,
                        b=b,
                        loop_bound=loop_bound,
                        min_component_size=min_component_size,
                        select_by=select_by,
                        L_list=L_list,
                        check_protected_removal=check_protected,
                        recovery_threshold_divisor=recovery_threshold_divisor
                    )
                    
                    # Store results (use tuple as key for scale+theta combination)
                    key = (n_size, theta)
                    if key not in results_by_scale:
                        results_by_scale[key] = {'cluster_counts': [], 'metrics_list': []}
                    
                    results_by_scale[key]['cluster_counts'].append(cluster_count)
                    if metrics is not None:
                        results_by_scale[key]['metrics_list'].append(metrics)
        
        # Aggregate results for each (n, theta) combination
        for n_size in n_list:
            theta_list = theta_configs_by_scale[n_size]
            n_per_component = scale_configs[n_size]['n_per_component']
            
            for theta in theta_list:
                key = (n_size, theta)
                cluster_counts = results_by_scale[key]['cluster_counts']
                metrics_list = results_by_scale[key]['metrics_list']
                
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
                
                all_results.append(result)
                
                print(f"\n{strategy_name} | n={n_size} | θ={theta:.6f}: "
                      f"Valid={result['valid_trials']}/{repeats} "
                      f"(Success={result['success_rate']:.1%}), "
                      f"AvgK={result['avg_clusters']:.2f}, "
                      f"First10={result.get('first_10', np.nan):.4f}")

    # === Create summary DataFrame ===
    df_results = pd.DataFrame(all_results)

    # Reorder columns for clarity
    fixed_cols = ["strategy", "n", "theta", "valid_trials", "success_rate", "avg_clusters"]
    metric_cols = [c for c in df_results.columns if c not in fixed_cols]
    df_results = df_results[fixed_cols + sorted(metric_cols)]

    # === SAVE CSV FILES ===
    output_file = "distance_recovery_results_pa_protected_comparison.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"CSV results saved to: {output_file}")
    
    # Generate comparison dataframe for CSV
    comparison_rows = []
    for n in n_list:
        theta_list = get_theta_list(n)
        for theta in theta_list:
            subset = df_results[(df_results['n'] == n) & (np.isclose(df_results['theta'], theta))]
            if len(subset) == 2:
                no_prot = subset[subset['strategy'] == 'no_protected'].iloc[0]
                with_prot = subset[subset['strategy'] == 'with_protected'].iloc[0]
                comparison_rows.append({
                    'n': n,
                    'theta': theta,
                    'no_prot_success': no_prot['success_rate'],
                    'with_prot_success': with_prot['success_rate'],
                    'success_diff': with_prot['success_rate'] - no_prot['success_rate'],
                    'no_prot_first10': no_prot.get('first_10', np.nan),
                    'with_prot_first10': with_prot.get('first_10', np.nan),
                    'no_prot_overall': no_prot.get('overall_classified', np.nan),
                    'with_prot_overall': with_prot.get('overall_classified', np.nan),
                })
    
    if comparison_rows:
        summary_df = pd.DataFrame(comparison_rows)
        summary_file = "success_rate_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"CSV comparison saved to: {summary_file}")

    # === SAVE COMPREHENSIVE TXT FILE ===
    print(f"{'=' * 80}\n")
    txt_file = save_all_tables_to_txt(
        df_results, 
        n_list, 
        get_theta_list, 
        K, 
        repeats,
        output_prefix="distance_recovery_results"
    )

    # === ALSO PRINT TO CONSOLE ===
    print("\n\n" + "=" * 200)
    print("CONSOLE OUTPUT - FORMATTED TABLES")
    print("=" * 200)
    
    print_results_table(df_results, title="COMPLETE RESULTS TABLE")
    print_success_failure_breakdown(df_results, K, repeats)
    print_comparison_table(df_results, n_list, get_theta_list)
    print_per_strategy_summary(df_results, repeats)


if __name__ == "__main__":
    np.random.seed(17)
    main()

