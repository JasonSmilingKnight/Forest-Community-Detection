"""
Shared utilities for leaf-stripping experiments.
Contains common functions for table printing, metrics computation, and result aggregation.
"""

import numpy as np
import pandas as pd
import networkx as nx
import sys


# ============================================================================
# TABLE PRINTING UTILITIES
# ============================================================================

def save_all_tables_to_txt(df_results, n_list, get_theta_list, K, repeats, output_prefix="results"):
    """Save all formatted tables to a single comprehensive TXT file."""
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
        'overall_classified', 'weighted_overall'
    ]

    # Include theta_rate in display if available
    if 'theta_rate' in df.columns:
        display_cols.insert(display_cols.index('theta') + 1, 'theta_rate')
    display_cols = [col for col in display_cols if col in df.columns]
    df_display = df[display_cols].copy()

    # Sort by theta_rate (group same rate together), then by n
    if 'theta_rate' in df_display.columns:
        df_display = df_display.sort_values(['n', 'theta_rate']).reset_index(drop=True)
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
    for col in ['first_10', 'first_50', 'overall_classified', 'weighted_overall', 'root']:
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
        'weighted_overall': 'WtdOverall',
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
                                 ('overall_classified', 'Overall'), ('weighted_overall', 'WtdOverall'),
                                 ('root', 'Root')]:
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

    if len(comparison_rows) > 0:
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
    else:
        print("No comparison data available (requires both strategies to be run).")
    
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
        for metric in ["first_10", "first_50", "highdeg_10", "highdeg_50", "overall_classified", "weighted_overall", "root"]:
            if metric in strategy_df.columns:
                avg = strategy_df[metric].mean()
                std = strategy_df[metric].std()
                print(f"    {metric:20s}: {avg:.4f} ± {std:.4f}")
        
        print()


# ============================================================================
# RESULT AGGREGATION
# ============================================================================

def aggregate_trial_results(cluster_counts, metrics_list, K, repeats, strategy_name, n, theta, L_list):
    """Aggregate results from multiple trials using mean."""
    success_rate = len(metrics_list) / repeats if repeats > 0 else 0.0

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

    if not metrics_list:
        for L in L_list:
            result[f"first_{L}"] = np.nan
        result["overall_classified"] = np.nan
        for d in range(6):
            result[f"depth_{d}"] = np.nan
    else:
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())

        for key in sorted(all_keys):
            values = [m.get(key, np.nan) for m in metrics_list]
            result[key] = np.nanmean(values)

    return result


def aggregate_trial_results_median(cluster_counts, metrics_list, K, repeats, strategy_name, n, theta, L_list):
    """Aggregate results from multiple trials using median."""
    success_rate = len(metrics_list) / repeats if repeats > 0 else 0.0

    cluster_count_dist = {}
    for k in range(K):
        cluster_count_dist[f"count_{k}"] = sum(1 for c in cluster_counts if c == k)
    cluster_count_dist[f"count_{K}"] = sum(1 for c in cluster_counts if c == K)
    cluster_count_dist[f"count_{K}plus"] = sum(1 for c in cluster_counts if c > K)

    median_cluster_count = np.median(cluster_counts) if cluster_counts else 0.0

    result = {
        "strategy": strategy_name,
        "n": n,
        "theta": theta,
        "valid_trials": len(metrics_list),
        "success_rate": success_rate,
        "median_clusters": median_cluster_count,
    }
    result.update(cluster_count_dist)

    if not metrics_list:
        for L in L_list:
            result[f"first_{L}"] = np.nan
        result["overall_classified"] = np.nan
        for d in range(6):
            result[f"depth_{d}"] = np.nan
    else:
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())

        for key in sorted(all_keys):
            values = [m.get(key, np.nan) for m in metrics_list]
            result[key] = np.nanmedian(values)

    return result
