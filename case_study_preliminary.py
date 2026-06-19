#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case study: co-authorship network community detection (K=2).

Reproduces Figures 7–9 of the paper using the co-authorship dataset of
Ji & Jin (2016).  The pipeline has two stages:

  Stage 1 — Graph pruning (Algorithm 1):
      Leaf stripping with K=2 to extract two community cores.
      Output: Figure 7  (core_component_1.pdf, core_component_2.pdf)

  Stage 2a — Distance-based recovery (Algorithm 2):
      Assigns stripped nodes via distance threshold tau' = tau / 2.
      Output: Figure 8  (dist_recovery_cluster_1.pdf, dist_recovery_cluster_2.pdf)

  Stage 2b — Monte Carlo sample recovery (Algorithm 3):
      Runs a Gibbs sampler (M=1000 samples) and assigns nodes by
      posterior probability over the anchor components from Stage 1.
      Output: Figure 9  (mc_recovery_cluster_1.pdf, mc_recovery_cluster_2.pdf)

Parameters: (K, tau, Q, zeta) = (2, 10th highest degree, n//10, 0.8)
"""

import os
import sys
import pickle
import random

import igraph
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text
from pathlib import Path

from leaf_stripping_recovery import Leaf_stripping, Leaf_stripping_postCT
from gibbsSampling import gibbsFullCT

# ===========================================================================
# Paths
# ===========================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, '..')
out_dir    = Path(script_dir) / "leaf_stripping_K2_results"
out_dir.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Load graph and author names
# ===========================================================================

foo = igraph.read(os.path.join(data_dir, "coauthorThresh2GiantAdj_big.txt"))
foo.to_undirected()
foo.simplify()
foo.vs["myid"] = range(len(foo.vs))

graf = foo.connected_components().giant()
n    = len(graf.vs)
m    = len(graf.es)
print(f"Graph: n={n}, m={m}")

names    = pd.read_table(os.path.join(data_dir, "coauthorThresh2GiantAuthorList_big.txt"), header=None)
name_map = {i: names[0][i] for i in range(names.shape[0])}

# Adjacency matrix (binary, no self-loops)
A = np.array(graf.get_adjacency().data)
A[A > 1] = 1
np.fill_diagonal(A, 0)
print(f"Adjacency matrix: {A.shape}, edges: {int(A.sum()) // 2}")

# ===========================================================================
# Parameters  (K, tau, Q, zeta) = (2, 10th highest degree, n//10, 0.8)
# ===========================================================================

K                          = 2
b                          = 10       # tau = b-th highest degree
min_component_size         = n // 10  # Q
loop_bound                 = 0
select_by                  = "initial"
recovery_threshold_divisor = 2.0      # tau' = tau / 2

np.random.seed(42)
random.seed(10001)

print(f"\nParameters: K={K}, b={b}, Q={min_component_size}, "
      f"recovery_threshold_divisor={recovery_threshold_divisor}")

colors = ['#FF0000', '#0000FF']  # community 1: red, community 2: blue

# ===========================================================================
# Stage 1 — Graph pruning / leaf stripping  (Algorithm 1)
# Produces the two community cores used in both recovery stages.
# ===========================================================================

print(f"\nStage 1: leaf stripping (K={K})...")

result = Leaf_stripping(
    A_input=A,
    K=K,
    b=b,
    d=loop_bound,
    select_by=select_by,
    min_component_size=min_component_size,
    recover=True,
    recover_mode="distance_threshold",
    threshold_by="order",
    mode="True",
    check_protected_removal=True,
    recovery_threshold_divisor=recovery_threshold_divisor,
    beta=0.8,
)

if len(result) == 5:
    cluster_count, new_labels_dist, core_labels, component_sizes, reference_nodes = result
else:
    cluster_count, new_labels_dist, core_labels, component_sizes = result
    reference_nodes = None

# Swap labels 1 <-> 2 so community 1 is the larger / primary group
new_labels_dist_swapped                      = new_labels_dist.copy()
new_labels_dist_swapped[new_labels_dist == 1] = 2
new_labels_dist_swapped[new_labels_dist == 2] = 1
new_labels_dist                              = new_labels_dist_swapped

core_labels_swapped                    = core_labels.copy()
core_labels_swapped[core_labels == 1]  = 2
core_labels_swapped[core_labels == 2]  = 1
core_labels                            = core_labels_swapped

print(f"  Clusters found: {cluster_count},  component sizes: {component_sizes}")

if cluster_count < K:
    print(f"Leaf stripping found only {cluster_count} cluster(s); expected {K}. Exiting.")
    sys.exit(1)

components = []
for k in range(1, K + 1):
    comp_nodes = list(np.where(core_labels == k)[0])
    components.append(comp_nodes)
    print(f"  Core component {k}: {len(comp_nodes)} nodes")

initial_degrees = {i: int(A[i].sum()) for i in range(n)}

# ---------------------------------------------------------------------------
# Figure 7 — Community cores  (core_component_1.pdf, core_component_2.pdf)
# ---------------------------------------------------------------------------

print("\nGenerating Figure 7: community cores...")

core_layouts = []
for k in range(K):
    comp_nodes = components[k]
    sub        = graf.subgraph(comp_nodes)
    layout     = sub.layout_fruchterman_reingold()
    coords     = np.array(layout.coords)
    core_layouts.append(coords)

    fig, ax = plt.subplots(figsize=(8, 8))

    for edge in sub.es:
        src, tgt = edge.source, edge.target
        ax.plot([coords[src, 0], coords[tgt, 0]],
                [coords[src, 1], coords[tgt, 1]],
                color='#a0a0a0', alpha=0.5, linewidth=0.6, zorder=1)

    ax.scatter(coords[:, 0], coords[:, 1], s=20,
               c=colors[k], alpha=0.95, edgecolors='none', linewidths=0, zorder=2)

    # Label top-10 nodes by degree
    comp_deg = sorted(enumerate(comp_nodes), key=lambda x: -initial_degrees[x[1]])
    texts = []
    for idx, orig_node in comp_deg[:10]:
        orig_id = graf.vs[orig_node]["myid"]
        author  = name_map.get(orig_id, f"Node {orig_node}")
        t = ax.text(coords[idx, 0], coords[idx, 1], author,
                    fontsize=20, ha='center', va='bottom',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        texts.append(t)
    adjust_text(texts, ax=ax)

    ax.set_axis_off()
    plt.tight_layout()
    # OUTPUT — Figure 7, subfigure k+1
    fig.savefig(out_dir / f"core_component_{k+1}.pdf", bbox_inches='tight')
    print(f"  Saved: core_component_{k+1}.pdf  [Figure 7, Community Core {k+1}]")
    plt.close(fig)

# ===========================================================================
# Shared full-graph layout (Fruchterman–Reingold, community-nudged start)
# Reused for both Stage 2a and Stage 2b figures.
# ===========================================================================

G_nx = nx.from_numpy_array(A)

init_pos = {}
for i in range(n):
    label = new_labels_dist[i]
    if label == 1:
        init_pos[i] = (-0.3 + np.random.uniform(-0.1, 0.1), np.random.uniform(-0.5, 0.5))
    elif label == 2:
        init_pos[i] = ( 0.3 + np.random.uniform(-0.1, 0.1), np.random.uniform(-0.5, 0.5))
    else:
        init_pos[i] = (np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))

pos = nx.fruchterman_reingold_layout(G_nx, pos=init_pos, seed=42)
xs  = [pos[i][0] for i in range(n)]
ys  = [pos[i][1] for i in range(n)]


def plot_recovery_figures(labels, prefix, fig_number, color_map=None):
    """Save per-cluster PDFs (with top-10 author labels) and a combined full-graph PDF."""
    if color_map is None:
        color_map = {k + 1: colors[k] for k in range(K)}

    unique_labels = sorted(set(labels) - {0})
    gray_nodes    = {i for i in range(n) if labels[i] == 0}

    # Per-cluster PDFs: largest connected component only, top-10 authors labeled
    for label in unique_labels:
        all_cluster = [i for i in range(n) if labels[i] == label]
        sub_g       = G_nx.subgraph(all_cluster)
        ccs         = sorted(nx.connected_components(sub_g), key=len, reverse=True)
        cluster_nodes = sorted(ccs[0]) if ccs else all_cluster
        cluster_set   = set(cluster_nodes)

        fig, ax = plt.subplots(figsize=(8, 8))
        for u, v in G_nx.edges():
            if u in cluster_set and v in cluster_set:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color='#a0a0a0', alpha=0.5, linewidth=0.6, zorder=1)
        ax.scatter([xs[i] for i in cluster_nodes], [ys[i] for i in cluster_nodes],
                   s=20, c=color_map.get(label, '#808080'), alpha=0.95,
                   edgecolors='none', linewidths=0, zorder=2)

        top_nodes = sorted(cluster_nodes, key=lambda i: -initial_degrees[i])
        texts = []
        for i in top_nodes[:10]:
            orig_id = graf.vs[i]["myid"]
            author  = name_map.get(orig_id, f"Node {i}")
            t = ax.text(pos[i][0], pos[i][1], author,
                        fontsize=20, ha='center', va='bottom',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            texts.append(t)
        adjust_text(texts, ax=ax)

        ax.set_axis_off()
        plt.tight_layout()
        # OUTPUT — Figure fig_number, subfigure label
        fig.savefig(out_dir / f"{prefix}_cluster_{label}.pdf", bbox_inches='tight')
        print(f"  Saved: {prefix}_cluster_{label}.pdf  [Figure {fig_number}, Community {label}]")
        plt.close(fig)

    # Combined full-graph PDF (no author labels; unassigned nodes in gray)
    fig, ax = plt.subplots(figsize=(14, 8))
    for u, v in G_nx.edges():
        if u in gray_nodes or v in gray_nodes:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#c0c0c0', alpha=0.4, linewidth=0.4, zorder=1)
        else:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#a0a0a0', alpha=0.5, linewidth=0.5, zorder=1)
    if gray_nodes:
        ax.scatter([xs[i] for i in gray_nodes], [ys[i] for i in gray_nodes],
                   s=20, c='#808080', alpha=0.9, edgecolors='none', linewidths=0, zorder=2)
    for label in unique_labels:
        mask = [i for i in range(n) if labels[i] == label]
        if mask:
            ax.scatter([xs[i] for i in mask], [ys[i] for i in mask],
                       s=20, c=color_map.get(label, '#808080'), alpha=0.95,
                       edgecolors='none', linewidths=0, zorder=3)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_full_graph.pdf", bbox_inches='tight')
    print(f"  Saved: {prefix}_full_graph.pdf")
    plt.close(fig)


# ===========================================================================
# Stage 2a — Distance-based recovery  (Algorithm 2)
# ===========================================================================

print("\nStage 2a: distance-based recovery...")
print("Generating Figure 8: distance-recovery communities...")
# OUTPUT — Figure 8  (dist_recovery_cluster_1.pdf, dist_recovery_cluster_2.pdf)
plot_recovery_figures(new_labels_dist, "dist_recovery", fig_number=8)

# ===========================================================================
# Stage 2b — Monte Carlo sample recovery  (Algorithm 3)
# ===========================================================================

print("\nStage 2b: Gibbs sampler + Monte Carlo recovery...")

np.random.seed(9)
random.seed(9)

res = gibbsFullCT(graf, K=K, alpha=0, beta=1, Burn=50, M=1000, display=False)
print(f"  Samples collected: {len(res.get('samples', []))}")

freq      = res['allfreq']
freq      = freq / freq.sum()
freq_args = np.argsort(-freq)

print(f"\nTop 20 nodes by posterior root probability:")
for i in range(20):
    orig_id = graf.vs[freq_args[i]]["myid"]
    author  = name_map.get(orig_id, f"Node {freq_args[i]}")
    print(f"  {i+1:2d}. {author}: prob={freq[freq_args[i]]:.4f}")

print("\nRunning Monte Carlo recovery...")
postCT_results   = Leaf_stripping_postCT(res, components)
print(f"  Valid samples used: {postCT_results['valid_samples']} / {postCT_results['total_samples']}")

assignment_probs              = postCT_results['assignment_probs']
mc_labels                     = np.argmax(assignment_probs, axis=1) + 1
unassigned_mask               = assignment_probs.sum(axis=1) == 0
mc_labels[unassigned_mask]    = 0

print("\nGenerating Figure 9: Monte Carlo recovery communities...")
mc_color_map = {1: '#FF0000', 2: '#0000FF'}
# OUTPUT — Figure 9  (mc_recovery_cluster_1.pdf, mc_recovery_cluster_2.pdf)
plot_recovery_figures(mc_labels, "mc_recovery", fig_number=9, color_map=mc_color_map)

# ===========================================================================
# Results summary
# ===========================================================================

print(f"\n{'='*60}")
print("MONTE CARLO RECOVERY RESULTS:")
print(f"{'='*60}")
for k in range(1, K + 1):
    cluster_nodes = np.where(mc_labels == k)[0]
    core_nodes    = set(np.where(core_labels == k)[0])
    print(f"\n  Community {k}: {len(cluster_nodes)} nodes ({len(core_nodes)} core)")

    cluster_degs = sorted(cluster_nodes, key=lambda nd: -initial_degrees[nd])
    print("    Top 15 by degree:")
    for i, node in enumerate(cluster_degs[:15]):
        orig_id = graf.vs[node]["myid"]
        author  = name_map.get(orig_id, f"Node {node}")
        marker  = "*" if node in core_nodes else " "
        print(f"      {i+1:2d}. {marker} {author}: degree={initial_degrees[node]}")

    print("    Assignment probabilities for top 5:")
    for node in cluster_degs[:5]:
        orig_id  = graf.vs[node]["myid"]
        author   = name_map.get(orig_id, f"Node {node}")
        probs    = assignment_probs[node]
        prob_str = ", ".join(f"C{j+1}={probs[j]:.3f}" for j in range(K))
        print(f"      {author}: {prob_str}")

print(f"\n  Unclassified: {int(unassigned_mask.sum())} nodes")

print(f"\n{'='*60}")
print("DISTANCE-RECOVERY RESULTS:")
print(f"{'='*60}")
for k in range(1, K + 1):
    cluster_nodes = np.where(new_labels_dist == k)[0]
    core_nodes    = set(np.where(core_labels == k)[0])
    print(f"\n  Community {k}: {len(cluster_nodes)} nodes ({len(core_nodes)} core)")

    cluster_degs = sorted(cluster_nodes, key=lambda nd: -initial_degrees[nd])
    print("    Top 15 by degree:")
    for i, node in enumerate(cluster_degs[:15]):
        orig_id = graf.vs[node]["myid"]
        author  = name_map.get(orig_id, f"Node {node}")
        marker  = "*" if node in core_nodes else " "
        print(f"      {i+1:2d}. {marker} {author}: degree={initial_degrees[node]}")

print(f"\n  Unclassified: {int((new_labels_dist == 0).sum())} nodes")

# Alignment between the two recovery methods
classified_mask = (new_labels_dist > 0) & (mc_labels > 0)
agree_identity  = np.sum((new_labels_dist == mc_labels) & classified_mask)
mc_swap         = mc_labels.copy()
mc_swap[mc_labels == 1] = 2
mc_swap[mc_labels == 2] = 1
agree_swapped = np.sum((new_labels_dist == mc_swap) & classified_mask)
total         = classified_mask.sum()

if agree_swapped > agree_identity:
    mc_labels = mc_swap
    agree     = agree_swapped
    print("\n  (MC labels swapped to best match distance labels)")
else:
    agree = agree_identity

print(f"\n{'='*60}")
print("COMPARISON: distance-threshold vs Monte Carlo recovery:")
print(f"  Agreement on classified nodes: {agree} / {total} "
      f"({agree / total:.4f})" if total > 0 else "  No classified nodes.")

print(f"\n{'='*60}")
print("Top 20 nodes by degree (global):")
degs      = np.array(graf.degree())
degs_args = np.argsort(-degs)
for i in range(20):
    node    = degs_args[i]
    orig_id = graf.vs[node]["myid"]
    author  = name_map.get(orig_id, f"Node {node}")
    label   = new_labels_dist[node]
    print(f"  {i+1:2d}. {author}: degree={degs[node]}, community={f'C{label}' if label > 0 else '?'}")

# ===========================================================================
# Save results to disk
# ===========================================================================

rows = []
for node in range(n):
    orig_id = graf.vs[node]["myid"]
    rows.append({
        'node':            node,
        'original_id':     orig_id,
        'author':          name_map.get(orig_id, f"Node {node}"),
        'degree':          initial_degrees[node],
        'core_cluster':    int(core_labels[node]) if core_labels[node] > 0 else 0,
        'dist_cluster':    int(new_labels_dist[node]),
        'mc_cluster':      int(mc_labels[node]),
    })

df = pd.DataFrame(rows)
df.to_csv(out_dir / "cluster_assignments_K2.csv", index=False)
print(f"\nCluster assignments saved to cluster_assignments_K2.csv")

with open(out_dir / 'results_K2.pkl', 'wb') as f:
    pickle.dump({
        'components':    components,
        'core_labels':   core_labels,
        'dist_labels':   new_labels_dist,
        'mc_labels':     mc_labels,
        'postCT':        postCT_results,
        'gibbs_res':     res,
        'name_map':      name_map,
    }, f)
print("Full results saved to results_K2.pkl")

print(f"\nCluster summary (distance recovery):")
for label in sorted(df['dist_cluster'].unique()):
    subset = df[df['dist_cluster'] == label]
    lbl    = f"Community {label}" if label > 0 else "Unclassified"
    print(f"  {lbl}: {len(subset)} nodes, "
          f"mean degree={subset['degree'].mean():.1f}, "
          f"max degree={subset['degree'].max()}")

print(f"\n{'='*60}")
print("Done.")
