#!/usr/bin/env python3

import os

import random
from collections import deque
import numpy as np
import igraph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from gibbsSampling import gibbsFullCT
from leaf_stripping_recovery import Leaf_stripping_postCT

_here = os.path.dirname(os.path.abspath(__file__))

def _out(filename):
    return os.path.join(_here, filename)


# ── Shared toy graph (used by both parts) ────────────────────────────────────
# Solid edges = backbone (tree structure); dash-dot edges = noise
backbone_edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 6), (1, 7),
    (2, 8),
    (3, 9),
    (5, 10), (5, 11),
    (8, 22),
    (12, 13), (12, 14), (12, 15), (12, 16),
    (13, 17), (13, 18),
    (14, 19), (14, 20),
    (12, 21),
]

noise_edges = [
    (6, 7),
    (4, 11),
    (17, 18),
    (19, 20),
    (5, 16),
    (8, 19),
    (10, 17),
    (9, 15),
    (8, 3),
]

G = nx.Graph()
G.add_edges_from(backbone_edges + noise_edges)

# ── Shared layout (computed once, used by all figures) ───────────────────────
cluster1 = set(range(0, 12)) | {22}
cluster2 = set(range(12, 22))

pos1 = nx.spring_layout(G.subgraph(cluster1), seed=7,  k=0.55)
pos2 = nx.spring_layout(G.subgraph(cluster2), seed=11, k=0.55)

pos = {nd: (x - 1.8, y - 0.6) for nd, (x, y) in pos1.items()}
pos.update({nd: (x + 1.8, y + 0.6) for nd, (x, y) in pos2.items()})

# ── Shared anchor nodes (used in figure5 and MCMC) ───────────────────────────
anchor_red  = [0, 1, 5]
anchor_blue = [12, 21, 13, 17]

# ── Helper: draw edges, fading those connected to gray (pruned) nodes ────────
def draw_edges(ax, gray_nodes=None):
    gray_nodes = set(gray_nodes) if gray_nodes else set()

    def split(edges):
        active, faded = [], []
        for u, v in edges:
            (faded if u in gray_nodes or v in gray_nodes else active).append((u, v))
        return active, faded

    bb_active, bb_faded = split(backbone_edges)
    ns_active, ns_faded = split(noise_edges)

    if bb_active:
        nx.draw_networkx_edges(G, pos, edgelist=bb_active, ax=ax,
                               width=1.6, edge_color="#1a1a1a", style="solid",   alpha=0.9)
    if bb_faded:
        nx.draw_networkx_edges(G, pos, edgelist=bb_faded,  ax=ax,
                               width=1.6, edge_color="#c0c0c0", style="solid",   alpha=0.6)
    if ns_active:
        nx.draw_networkx_edges(G, pos, edgelist=ns_active, ax=ax,
                               width=1.4, edge_color="#1a1a1a", style="dashdot", alpha=0.9)
    if ns_faded:
        nx.draw_networkx_edges(G, pos, edgelist=ns_faded,  ax=ax,
                               width=1.4, edge_color="#c0c0c0", style="dashdot", alpha=0.6)


# ════════════════════════════════════════════════════════════════════════════
# PART 1 — DETERMINISTIC ALGORITHM FIGURES
# Figures: figure1 (a shared), figure2 (simple b), figure3 (simple c),
#          figure4 (improved b), figure5 (improved c)
# ════════════════════════════════════════════════════════════════════════════

# ── Algorithm parameters (tau1=5, tau2=4) ────────────────────────────────────
tau2 = 4   # degree pruning threshold: remove nodes with degree < tau2
tau1 = 5   # core validation threshold: component must contain degree >= tau1

deg_Gn  = dict(G.degree())
V_tilde = {u for u, d in deg_Gn.items() if d >= tau2}
G_tilde = G.subgraph(V_tilde).copy()

components = [set(c) for c in nx.connected_components(G_tilde)]

def max_deg_in_original(comp):
    return max(deg_Gn[u] for u in comp) if comp else 0

core_components = sorted(
    [c for c in components if max_deg_in_original(c) >= tau1],
    key=lambda c: (-len(c), -max_deg_in_original(c), min(c))
)

core_red      = core_components[0] if len(core_components) >= 1 else set()
core_blue     = core_components[1] if len(core_components) >= 2 else set()
kept_not_core = V_tilde - core_red - core_blue
removed       = set(G.nodes()) - V_tilde


# ── Figure (a) shared: Original graph — figure1_original_graph.pdf ──────────
# Used as subfigure (a) in both paper figures.
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_axis_off()
draw_edges(ax)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(G.nodes()),
                       node_color="#DAA520", node_size=250,
                       linewidths=0.8, edgecolors="none")
plt.tight_layout()
plt.savefig(_out("figure1_original_graph.pdf"))
plt.close()


# ── Figure "simple algorithm" (b): after degree pruning — figure2_algorithm_colored_result.pdf
# gray = pruned (degree < tau2); purple = kept but not a core;
# red/blue = the two identified cores (degree >= tau1).
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_axis_off()
draw_edges(ax, gray_nodes=removed)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(removed),
                       node_color="#808080", node_size=250,
                       linewidths=0.8, edgecolors="none", alpha=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(kept_not_core),
                       node_color="#9932CC", node_size=250,
                       linewidths=0.9, edgecolors="none", alpha=1.0)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(core_red),
                       node_color="#FF0000", node_size=250,
                       linewidths=1.0, edgecolors="none", alpha=1.0)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(core_blue),
                       node_color="#0000FF", node_size=250,
                       linewidths=1.0, edgecolors="none", alpha=1.0)
plt.tight_layout()
plt.savefig(_out("figure2_algorithm_colored_result.pdf"))
plt.close()


# ── Figure "simple algorithm" (c): BFS community assignment — figure3_recovered_communities.pdf
# Every node assigned red or blue by distance to nearest core; ties broken randomly.
def multi_source_bfs(graph, sources):
    dist, q = {}, deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in graph.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

rng          = random.Random(123)
dist_to_red  = multi_source_bfs(G, core_red)  if core_red  else {}
dist_to_blue = multi_source_bfs(G, core_blue) if core_blue else {}

def assign_label(u):
    if u in core_red:       return "red"
    if u in core_blue:      return "blue"
    if u in kept_not_core:  return rng.choice(["red", "blue"])
    dr = dist_to_red.get(u,  float("inf"))
    db = dist_to_blue.get(u, float("inf"))
    if dr < db:  return "red"
    if db < dr:  return "blue"
    return rng.choice(["red", "blue"])

bfs_labels = {u: assign_label(u) for u in G.nodes()}
bfs_red    = [u for u in G.nodes() if bfs_labels[u] == "red"]
bfs_blue   = [u for u in G.nodes() if bfs_labels[u] == "blue"]

plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_axis_off()
draw_edges(ax)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=bfs_red,
                       node_color="#FF0000", node_size=250,
                       linewidths=0.8, edgecolors="none", alpha=0.95)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=bfs_blue,
                       node_color="#0000FF", node_size=250,
                       linewidths=0.8, edgecolors="none", alpha=0.95)
plt.tight_layout()
plt.savefig(_out("figure3_recovered_communities.pdf"))
plt.close()


# ── Figure "improved algorithm" (b): after iteration 1 — figure4_remaining_nodes.pdf
# Nodes with degree != 2 survive (goldenrod); degree-2 nodes are pruned (gray).
pruned_iter1   = sorted([v for v in G.nodes() if G.degree(v) == 2])
survived_iter1 = [v for v in G.nodes() if G.degree(v) != 2]

plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_axis_off()
draw_edges(ax, gray_nodes=pruned_iter1)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=survived_iter1,
                       node_color="#DAA520", node_size=250,
                       edgecolors="none", linewidths=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=pruned_iter1,
                       node_color="#808080", node_size=250,
                       edgecolors="none", linewidths=0.9, alpha=0.95)
plt.tight_layout()
plt.savefig(_out("figure4_remaining_nodes.pdf"))
plt.close()


# ── Figure "improved algorithm" (c): after iteration 2 — figure5_manual_coloring.pdf
# Manually specified anchors (anchor_red, anchor_blue) after second pruning iteration.
# red = anchor community 1, blue = anchor community 2, purple = intermediate nodes.
purple_nodes_f5 = [8, 22]
gray_nodes_f5   = sorted(set(G.nodes()) - set(anchor_red) - set(anchor_blue) - set(purple_nodes_f5))

plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_axis_off()
draw_edges(ax, gray_nodes=gray_nodes_f5)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=gray_nodes_f5,
                       node_color="#808080", node_size=250,
                       edgecolors="none", linewidths=0.9, alpha=0.95)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=purple_nodes_f5,
                       node_color="#9932CC", node_size=250,
                       edgecolors="none", linewidths=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=anchor_red,
                       node_color="#FF0000", node_size=250,
                       edgecolors="none", linewidths=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=anchor_blue,
                       node_color="#0000FF", node_size=250,
                       edgecolors="none", linewidths=0.9)
plt.tight_layout()
plt.savefig(_out("figure5_manual_coloring.pdf"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PART 2 — MCMC CLASSIFICATION FIGURES
# Figures: hard_classification.pdf (a), soft_classification.pdf (b)
# ════════════════════════════════════════════════════════════════════════════

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 7000
np.random.seed(SEED)
random.seed(SEED)

# ── Convert toy graph to igraph ───────────────────────────────────────────────
n_nodes  = G.number_of_nodes()
graf_toy = igraph.Graph()
graf_toy.add_vertices(n_nodes)
graf_toy.add_edges(list(G.edges()))
graf_toy.to_undirected()
graf_toy.simplify()
for edge in graf_toy.es:
    edge["tree"] = True

# ── MCMC: Gibbs sampler (M=500, Burn=30) ─────────────────────────────────────
results = gibbsFullCT(
    graf_toy,
    Burn=30,
    M=500,
    gap=1,
    alpha=0,
    beta=1,
    K=2,
    display=True,
    size_thresh=0.1,
    birth_thresh=0.8,
)

# ── Extract posterior assignment probabilities ───────────────────────────────
# anchor_red / anchor_blue defined in shared section above
posteriors       = Leaf_stripping_postCT(results, [set(anchor_red), set(anchor_blue)])
assignment_probs = posteriors['assignment_probs']   # shape (n_nodes, 2)

print(f"Valid MCMC samples: {posteriors['valid_samples']} / {posteriors['total_samples']}")

# ── Hard labels: argmax of posterior ─────────────────────────────────────────
def hard_label(p):
    return int(p[1] > p[0]) if abs(p[0] - p[1]) >= 1e-6 else random.choice([0, 1])

hard_labels = np.array([hard_label(p) for p in assignment_probs])

color_red  = '#FF0000'
color_blue = '#0000FF'

# ── Figure (a): Hard classification — hard_classification.pdf ────────────────
# Each node assigned to the community with the highest posterior probability.
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_axis_off()
draw_edges(ax)
hard_colors = [color_blue if assignment_probs[nd, 1] > 0.5 else color_red
               for nd in sorted(G.nodes())]
nx.draw_networkx_nodes(G, pos, ax=ax,
                       nodelist=sorted(G.nodes()),
                       node_color=hard_colors,
                       node_size=250, linewidths=0.8, edgecolors="none")
plt.tight_layout()
plt.savefig(_out("hard_classification.pdf"), bbox_inches='tight')
plt.close()
print("Saved: hard_classification.pdf")


# ── Figure (b): Soft classification — soft_classification.pdf ────────────────
# Node color encodes full posterior probability vector via red→blue colormap.
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_axis_off()
draw_edges(ax)
cmap        = mcolors.LinearSegmentedColormap.from_list("", [color_red, color_blue])
soft_colors = [cmap(assignment_probs[nd, 1]) for nd in sorted(G.nodes())]
nx.draw_networkx_nodes(G, pos, ax=ax,
                       nodelist=sorted(G.nodes()),
                       node_color=soft_colors,
                       node_size=250, linewidths=0.8, edgecolors="none")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('P(Community 2)', rotation=270, labelpad=28, fontsize=22)
cbar.set_ticks([0.0, 0.5, 1.0])
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(_out("soft_classification.pdf"), bbox_inches='tight')
plt.close()
print("Saved: soft_classification.pdf")
