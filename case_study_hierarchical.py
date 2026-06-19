#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
case_study_hierarchical.py — hierarchical community detection via iterative leaf stripping (K=2).

Starting from the full co-authorship network, repeatedly splits any community
whose largest connected component exceeds Q nodes, until no further splits are
possible. Uses distance-based recovery only (no Gibbs sampler).

Pipeline:
  Round 1  — Leaf stripping on the full graph → 2 communities.
  Rounds 2+ — For each community with giant CC > Q, run leaf stripping on its
               subgraph. A split is accepted only when both child communities
               have ≥ Q core nodes.  Stops when no community can be split.

Outputs (in amarel/case_study_iter_core/):
  Figure 10 — community_tree.pdf          (hierarchical dendrogram)
  Figure 11 — final_dist_cluster_1-12.pdf (communities 1–12)
  Figure 12 — final_dist_cluster_13+.pdf  (communities 13–20)
              final_dist_full_graph.pdf    (all communities, no labels)

Parameters: K=2, b=10 (tau = 10th highest degree), Q=10, zeta=0.8
"""

import os
import sys
import pickle
import random
import colorsys

import igraph
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from leaf_stripping_recovery import Leaf_stripping

# ===========================================================================
# Paths
# ===========================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, '..')
out_dir    = Path(script_dir) / "case_study_iter_core"
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
# Parameters
# ===========================================================================

K                          = 2
b                          = 10    # tau = b-th highest degree
Q                          = 10    # stop splitting when giant CC <= Q
size_threshold             = Q
loop_bound                 = 0
select_by                  = "initial"
recovery_threshold_divisor = 2.0   # tau' = tau / 2

np.random.seed(42)
random.seed(10001)

print(f"\nParameters: K={K}, b={b}, Q={Q}, "
      f"recovery_threshold_divisor={recovery_threshold_divisor}")

initial_degrees = {i: int(A[i].sum()) for i in range(n)}
G_nx            = nx.from_numpy_array(A)

# ===========================================================================
# Color palette
# ===========================================================================

def generate_colors(num):
    """Generate `num` distinct colors via HSV spacing."""
    colors = []
    for i in range(num):
        r, g, b_val = colorsys.hsv_to_rgb(i / max(num, 1), 0.8, 0.9)
        colors.append(f'#{int(r*255):02X}{int(g*255):02X}{int(b_val*255):02X}')
    return colors

# ===========================================================================
# Round 1 — leaf stripping on full graph
# ===========================================================================

print(f"\nRound 1: leaf stripping on full graph (n={n})...")

result = Leaf_stripping(
    A_input=A,
    K=K,
    b=b,
    d=loop_bound,
    select_by=select_by,
    recover=True,
    recover_mode="distance_threshold",
    threshold_by="order",
    mode="True",
    check_protected_removal=True,
    recovery_threshold_divisor=recovery_threshold_divisor,
    beta=0.8,
)

if len(result) == 5:
    cluster_count, new_labels_dist, core_labels, component_sizes, _ = result
else:
    cluster_count, new_labels_dist, core_labels, component_sizes = result

if cluster_count < K:
    print(f"Leaf stripping found only {cluster_count} cluster(s); expected {K}. Exiting.")
    sys.exit(1)

# ===========================================================================
# Iterative splitting
# ===========================================================================

all_round_labels      = [new_labels_dist.copy()]
all_round_core_labels = [core_labels.copy()]
current_labels        = new_labels_dist.copy()
current_core_labels   = core_labels.copy()

round_num      = 1
next_label_id  = K + 1

# Track hierarchy: (parent_label, child1_label, child2_label)
split_history  = [("root", 1, 2)]

while True:
    round_num += 1

    # Identify communities large enough to split
    labels_to_split = []
    for lbl in sorted(set(current_labels) - {0}):
        lbl_nodes = list(np.where(current_labels == lbl)[0])
        sub_g     = G_nx.subgraph(lbl_nodes)
        ccs       = sorted(nx.connected_components(sub_g), key=len, reverse=True)
        gcc_size  = len(ccs[0]) if ccs else len(lbl_nodes)
        if gcc_size > size_threshold:
            labels_to_split.append((lbl, lbl_nodes, sorted(ccs[0]) if ccs else lbl_nodes))

    if not labels_to_split:
        break

    new_round_labels      = current_labels.copy()
    new_round_core_labels = current_core_labels.copy()
    any_split             = False

    for split_label, all_split_nodes, split_nodes in labels_to_split:
        A_sub = A[np.ix_(split_nodes, split_nodes)]

        np.random.seed(42 + round_num * 100 + int(split_label))
        random.seed(10001 + round_num * 100 + int(split_label))

        result_sub = Leaf_stripping(
            A_input=A_sub,
            K=K,
            b=b,
            d=loop_bound,
            select_by=select_by,
            recover=True,
            recover_mode="distance_threshold",
            threshold_by="order",
            mode="True",
            check_protected_removal=True,
            recovery_threshold_divisor=recovery_threshold_divisor,
            beta=0.8,
        )

        if len(result_sub) == 5:
            sub_cluster_count, sub_dist_labels, sub_core_labels, sub_comp_sizes, _ = result_sub
        else:
            sub_cluster_count, sub_dist_labels, sub_core_labels, sub_comp_sizes = result_sub

        if sub_cluster_count < K:
            continue

        # Reject split if either child has fewer than Q core nodes
        if any(len(np.where(sub_core_labels == k)[0]) < Q for k in range(1, K + 1)):
            continue

        any_split = True

        # Clear old labels for this cluster
        for node in all_split_nodes:
            new_round_labels[node]      = 0
            new_round_core_labels[node] = 0

        # Assign new child labels
        child_labels = []
        for sub_k in range(1, K + 1):
            new_label = next_label_id
            next_label_id += 1
            child_labels.append(new_label)

            for idx in np.where(sub_dist_labels == sub_k)[0]:
                new_round_labels[split_nodes[idx]] = new_label
            for idx in np.where(sub_core_labels == sub_k)[0]:
                new_round_core_labels[split_nodes[idx]] = new_label

        split_history.append((split_label, child_labels[0], child_labels[1]))

    if not any_split:
        break

    all_round_labels.append(new_round_labels.copy())
    all_round_core_labels.append(new_round_core_labels.copy())
    current_labels      = new_round_labels
    current_core_labels = new_round_core_labels

total_rounds = len(all_round_labels)
print(f"\nIterative splitting complete: {total_rounds} rounds, "
      f"{len(set(current_labels) - {0})} final communities.")

# ===========================================================================
# Build tree ordering of final communities
# ===========================================================================

final_labels      = all_round_labels[-1]
final_core_labels = all_round_core_labels[-1]
final_unique      = sorted(set(final_labels) - {0})

def get_tree_ordered_leaves(split_history, all_leaves):
    """Inorder traversal of split tree → leaf ordering."""
    children = {p: (c1, c2) for p, c1, c2 in split_history}

    def traverse(node):
        if node in children:
            l, r = children[node]
            return traverse(l) + traverse(r)
        return [node] if node in all_leaves else []

    return traverse("root")

all_leaves_set = set(final_unique)
tree_ordered   = get_tree_ordered_leaves(split_history, all_leaves_set)
for lbl in final_unique:
    if lbl not in tree_ordered:
        tree_ordered.append(lbl)

# Finalization-order remapping (leaves that stop splitting first get lower indices)
leaf_order = []
for _, child1, child2 in split_history:
    for c in (child1, child2):
        if c in all_leaves_set and c not in leaf_order:
            leaf_order.append(c)
for lbl in tree_ordered:
    if lbl not in leaf_order:
        leaf_order.append(lbl)
if len(leaf_order) >= 2:
    leaf_order[-1], leaf_order[-2] = leaf_order[-2], leaf_order[-1]

label_remap = {old: new + 1 for new, old in enumerate(leaf_order)}

# Color palette: community 2 gets a distinct orange
palette          = generate_colors(len(tree_ordered))
final_color_map  = {new + 1: palette[new] for new in range(len(tree_ordered))}
final_color_map[2] = '#FF8C00'

# Apply remapping
final_labels_seq      = np.zeros_like(final_labels)
final_core_labels_seq = np.zeros_like(final_core_labels)
for old, new in label_remap.items():
    final_labels_seq[final_labels == old]           = new
    final_core_labels_seq[final_core_labels == old] = new

# ===========================================================================
# Layout: per-community FR, placed in tree-ordered x-positions
# ===========================================================================

def compute_tree_x_ranges(split_history, all_leaves, label_remap, gap=0.4):
    """Assign x-ranges to leaves following tree hierarchy."""
    children = {p: (c1, c2) for p, c1, c2 in split_history}

    def min_remapped(node):
        if node in children:
            c1, c2 = children[node]
            return min(min_remapped(c1), min_remapped(c2))
        return label_remap.get(node, float('inf'))

    leaf_ranges = {}

    def assign_range(node, x_lo, x_hi, depth):
        if node in children:
            l, r = children[node]
            if min_remapped(l) > min_remapped(r):
                l, r = r, l
            mid        = (x_lo + x_hi) / 2
            branch_gap = gap / (depth + 1)
            assign_range(l, x_lo, mid - branch_gap / 2, depth + 1)
            assign_range(r, mid + branch_gap / 2, x_hi, depth + 1)
        elif node in all_leaves:
            leaf_ranges[node] = (x_lo, x_hi)

    assign_range("root", 0, len(all_leaves) * 1.5, 0)
    return leaf_ranges

leaf_x_ranges = compute_tree_x_ranges(split_history, all_leaves_set, label_remap, gap=0.4)

# Compute per-community layout and stitch into global positions
fixed_pos = {}
for old_lbl in tree_ordered:
    comm_nodes = list(np.where(final_labels == old_lbl)[0])
    if not comm_nodes:
        continue
    sub    = graf.subgraph(comm_nodes)
    layout = sub.layout_fruchterman_reingold()
    coords = np.array(layout.coords)

    cmin   = coords.min(axis=0)
    cmax   = coords.max(axis=0)
    span   = np.where(cmax - cmin == 0, 1.0, cmax - cmin)
    normed = (coords - cmin) / span

    x_lo, x_hi = leaf_x_ranges.get(old_lbl, (0, 1))
    x_span      = x_hi - x_lo
    for idx, node in enumerate(comm_nodes):
        fixed_pos[node] = (normed[idx, 0] * x_span + x_lo, normed[idx, 1])

pos = nx.spring_layout(G_nx, pos=fixed_pos, fixed=list(fixed_pos.keys()),
                       iterations=100, seed=42)
xs  = [pos[i][0] for i in range(n)]
ys  = [pos[i][1] for i in range(n)]

# ===========================================================================
# Plotting
# ===========================================================================

def plot_recovery_figures(labels, prefix, core_labels_arr=None):
    """
    Save per-cluster PDFs (largest CC, top-10 authors labeled) and a combined PDF.
    Core-to-core edges within a cluster are drawn in gold; others in gray.
    """
    unique_labels = sorted(set(labels) - {0}, key=lambda lbl: np.sum(labels == lbl))

    # Core sets per label (for edge coloring)
    core_sets = {}
    for label in unique_labels:
        if core_labels_arr is not None:
            core_sets[label] = set(np.where(core_labels_arr == label)[0])
        else:
            core_sets[label] = {i for i in range(n) if labels[i] == label}

    # --- Per-cluster PDFs (Figures 11–12) ---
    for label in unique_labels:
        all_cluster = [i for i in range(n) if labels[i] == label]
        sub_g       = G_nx.subgraph(all_cluster)
        ccs         = sorted(nx.connected_components(sub_g), key=len, reverse=True)
        cluster_nodes = sorted(ccs[0]) if ccs else all_cluster
        cluster_set   = set(cluster_nodes)
        core_here     = core_sets.get(label, set()) & cluster_set

        fig, ax = plt.subplots(figsize=(8, 8))
        for u, v in G_nx.edges():
            if u in cluster_set and v in cluster_set:
                if u in core_here and v in core_here:
                    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                            color='#DAA520', alpha=0.8, linewidth=1.0, zorder=1)
                else:
                    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                            color='#909090', alpha=0.5, linewidth=0.3, zorder=1)

        ax.scatter([xs[i] for i in cluster_nodes], [ys[i] for i in cluster_nodes],
                   s=80, c='#DAA520', alpha=0.95, edgecolors='none', linewidths=0, zorder=2)

        top_nodes = sorted(cluster_nodes, key=lambda i: -initial_degrees[i])
        texts = []
        for i in top_nodes[:10]:
            orig_id = graf.vs[i]["myid"]
            author  = name_map.get(orig_id, f"Node {i}")
            t = ax.text(pos[i][0], pos[i][1], author,
                        fontsize=25, ha='center', va='bottom', color='black',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            texts.append(t)
        adjust_text(texts, ax=ax)

        ax.set_axis_off()
        plt.tight_layout()
        # OUTPUT — Figure 11 (label 1–12) or Figure 12 (label 13–20)
        fig.savefig(out_dir / f"{prefix}_cluster_{label}.pdf", bbox_inches='tight')
        fig_num = 11 if label <= 12 else 12
        print(f"  Saved: {prefix}_cluster_{label}.pdf  [Figure {fig_num}, Community {label}]")
        plt.close(fig)

    # --- Combined full-graph PDF (no labels) ---
    all_core_nodes  = set()
    core_label_map  = {}
    for lbl in unique_labels:
        for node in core_sets.get(lbl, set()):
            all_core_nodes.add(node)
            core_label_map[node] = lbl

    fig, ax = plt.subplots(figsize=(14, 8))
    for u, v in G_nx.edges():
        if (u in all_core_nodes and v in all_core_nodes
                and core_label_map.get(u) == core_label_map.get(v)):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#DAA520', alpha=0.8, linewidth=1.0, zorder=1)
        else:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#909090', alpha=0.4, linewidth=0.3, zorder=1)
    ax.scatter(xs, ys, s=80, c='#DAA520', alpha=0.95,
               edgecolors='none', linewidths=0, zorder=2)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_full_graph.pdf", bbox_inches='tight')
    print(f"  Saved: {prefix}_full_graph.pdf")
    plt.close(fig)


print("\nGenerating Figures 11–12: per-community PDFs...")
# OUTPUT — Figures 11 and 12  (final_dist_cluster_1.pdf … final_dist_cluster_20.pdf)
plot_recovery_figures(final_labels_seq, "final_dist",
                      core_labels_arr=final_core_labels_seq)

# ===========================================================================
# Figure 10 — Hierarchical dendrogram  (community_tree.pdf)
# ===========================================================================

print("\nGenerating Figure 10: community dendrogram...")

try:
    from ete3 import Tree as EteTree, TreeStyle, NodeStyle, TextFace

    children_map = {p: (c1, c2) for p, c1, c2 in split_history}

    def min_leaf_label(node):
        if node in children_map:
            c1, c2 = children_map[node]
            return min(min_leaf_label(c1), min_leaf_label(c2))
        return label_remap[node]

    def to_newick(node, is_root=True):
        if node in children_map:
            l, r = children_map[node]
            # Put larger labels first so smaller labels appear on the left
            if min_leaf_label(l) < min_leaf_label(r):
                l, r = r, l
            sub = f"({to_newick(l, False)},{to_newick(r, False)})"
            return sub if is_root else f"{sub}:1"
        return f"{label_remap[node]}:0.5"

    t  = EteTree(to_newick("root") + ";")
    ts = TreeStyle()
    ts.show_leaf_name          = False
    ts.show_scale              = False
    ts.rotation                = 270
    ts.branch_vertical_margin  = 20

    nstyle = NodeStyle()
    nstyle["size"]          = 0
    nstyle["hz_line_width"] = 3
    nstyle["vt_line_width"] = 3
    for nd in t.traverse():
        nd.set_style(nstyle)
        if nd.is_leaf():
            face          = TextFace(nd.name, fsize=14)
            face.rotation = -90
            nd.add_face(face, column=0, position="branch-right")

    # OUTPUT — Figure 10
    t.render(str(out_dir / "community_tree.pdf"), tree_style=ts)
    print(f"  Saved: community_tree.pdf  [Figure 10]")

except ImportError:
    print("  Skipping dendrogram (install ete3: pip install ete3)")

# ===========================================================================
# Results summary
# ===========================================================================

print(f"\n{'='*60}")
print(f"FINAL RESULTS: {total_rounds} rounds, "
      f"{len(final_unique)} communities")
print(f"{'='*60}")

for old_lbl in tree_ordered:
    new_lbl       = label_remap[old_lbl]
    cluster_nodes = np.where(final_labels == old_lbl)[0]
    core_nodes_k  = np.where(final_core_labels == old_lbl)[0]
    print(f"\n  Community {new_lbl}: {len(cluster_nodes)} nodes "
          f"({len(core_nodes_k)} core)")

    top15 = sorted(cluster_nodes, key=lambda nd: -initial_degrees[nd])[:15]
    print("    Top 15 by degree:")
    for i, node in enumerate(top15):
        orig_id = graf.vs[node]["myid"]
        author  = name_map.get(orig_id, f"Node {node}")
        marker  = "*" if node in set(core_nodes_k) else " "
        print(f"      {i+1:2d}. {marker} {author}: degree={initial_degrees[node]}")

print(f"\n  Unclassified: {int((final_labels == 0).sum())} nodes")

# ===========================================================================
# Save results
# ===========================================================================

rows = []
for node in range(n):
    orig_id = graf.vs[node]["myid"]
    row     = {
        'node':        node,
        'original_id': orig_id,
        'author':      name_map.get(orig_id, f"Node {node}"),
        'degree':      initial_degrees[node],
    }
    for rnd in range(total_rounds):
        row[f'core_cluster_r{rnd+1}'] = int(all_round_core_labels[rnd][node])
        row[f'dist_cluster_r{rnd+1}'] = int(all_round_labels[rnd][node])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(out_dir / "cluster_assignments_iterative.csv", index=False)
print(f"\nCluster assignments saved to cluster_assignments_iterative.csv")

save_dict = {
    'total_rounds':  total_rounds,
    'name_map':      name_map,
    'split_history': split_history,
    'label_remap':   label_remap,
}
for rnd in range(total_rounds):
    save_dict[f'core_labels_r{rnd+1}'] = all_round_core_labels[rnd]
    save_dict[f'dist_labels_r{rnd+1}'] = all_round_labels[rnd]

with open(out_dir / 'results_iterative.pkl', 'wb') as f:
    pickle.dump(save_dict, f)
print("Full results saved to results_iterative.pkl")

print(f"\nCluster sizes (distance recovery):")
for old_lbl in tree_ordered:
    new_lbl = label_remap[old_lbl]
    count   = int(np.sum(final_labels == old_lbl))
    print(f"  Community {new_lbl}: {count} nodes")
print(f"  Unclassified: {int(np.sum(final_labels == 0))} nodes")
print(f"  Total: {n} nodes")

print(f"\n{'='*60}")
print("Done.")
