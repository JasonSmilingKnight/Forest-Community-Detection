#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import numpy as np
import igraph
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from adjustText import adjust_text

_here = os.path.dirname(os.path.abspath(__file__))

def _out(filename):
    return os.path.join(_here, filename)


# ── Load adjacency matrix and build graph ───────────────────────────────────
_adj_path = os.path.join(_here, "..", "coauthorThresh2GiantAdj_big.txt")
_A = np.loadtxt(_adj_path)
_rows, _cols = np.where(np.triu(_A, k=1) > 0)
foo = igraph.Graph(n=len(_A), edges=list(zip(_rows.tolist(), _cols.tolist())))
foo.vs["myid"] = range(len(foo.vs))

print(f"Loaded graph: {len(foo.vs)} vertices, {len(foo.es)} edges")

# ── Extract giant connected component ────────────────────────────────────────
graf = foo.connected_components().giant()
n    = len(graf.vs)
print(f"Giant component: {n} vertices, {len(graf.es)} edges")

# ── Load author names ────────────────────────────────────────────────────────
names    = pd.read_table(os.path.join(_here, "..", "coauthorThresh2GiantAuthorList_big.txt"), header=None)
name_map = {i: names[0][i] for i in range(names.shape[0])}

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(10001)

# ── Shared precomputation ────────────────────────────────────────────────────
degs            = np.array(graf.degree())
my_layout       = graf.layout_fruchterman_reingold(niter=500)
coords          = np.array(my_layout.coords)
myid_to_grafidx = {graf.vs[j]["myid"]: j for j in range(graf.vcount())}


# ── Figure 1(a): Observed graph topology (tmp_stats.pdf) ────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
edge_segs = [(coords[e.source], coords[e.target]) for e in graf.es]
lc = LineCollection(edge_segs, colors="#5BA3C9", linewidths=0.12, zorder=1)
ax.add_collection(lc)
ax.scatter(coords[:, 0], coords[:, 1], s=1, c="black", zorder=2, linewidths=0)
ax.set_aspect("equal")
ax.axis("off")
ax.autoscale_view()
fig.savefig(_out("tmp_stats.pdf"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: tmp_stats.pdf")


# ── Figure 1(c): Degree distribution (degree_distribution.pdf) ───────────────
degs_filtered = degs[degs <= 15]
bins = np.arange(0.5, 15.5, 1)
fig, ax = plt.subplots(figsize=(8, 8))
ax.hist(degs_filtered, bins=bins, color="#5BA3C9", edgecolor="none", alpha=0.8, rwidth=0.95)
ax.set_xticks(list(range(0, 16, 5)))
ax.tick_params(axis='x', labelsize=18)
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
ax.tick_params(axis='y', labelsize=18)
ax.set_xlabel("Degree", fontsize=18)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-0.3, 15.3)
fig.savefig(_out("degree_distribution.pdf"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: degree_distribution.pdf")


# ── Figure 1(b): Dunson-Brown sub-network (dunson_brown_neighbourhood.pdf) ───

def _find_node(subg, name_map, last, first_variants):
    """Return (vertex_index, matched_name) for the first vertex whose name
    contains `last` and any token in `first_variants` (case-insensitive)."""
    last_l  = last.lower()
    first_l = [f.lower() for f in first_variants]
    for v in subg.vs:
        full = name_map.get(v["myid"], "").lower()
        if last_l in full and any(f in full for f in first_l):
            return v.index, name_map.get(v["myid"], "")
    return None, None


dunson_idx, dunson_name = _find_node(graf, name_map, "dunson", ["david"])
brown_idx,  brown_name  = _find_node(graf, name_map, "brown",  ["lawrence", "larry", "l."])

print("\n── David Dunson ↔ Lawrence Brown distance (full giant component) ───")
if dunson_idx is None:
    print("  David Dunson not found in graph")
elif brown_idx is None:
    print("  Lawrence/Larry Brown not found in graph")
else:
    print(f"  Matched: '{dunson_name}'  ↔  '{brown_name}'")
    print(f"  Degree of {brown_name} (full graph) = {int(degs[brown_idx])}")
    dist = graf.distances(dunson_idx, brown_idx)[0][0]
    print(f"  Shortest path distance = {int(dist)}")

    dunson_nbhd = set(graf.neighborhood(dunson_idx, order=2))
    brown_nbhd  = set(graf.neighborhood(brown_idx,  order=2))
    nbhd_nodes  = sorted(dunson_nbhd | brown_nbhd)

    nbhd_graf    = graf.subgraph(nbhd_nodes)
    nbhd_myids   = [nbhd_graf.vs[i]["myid"] for i in range(nbhd_graf.vcount())]
    nbhd_grafidx = [myid_to_grafidx[mid] for mid in nbhd_myids]

    dunson_sub = next(i for i, gi in enumerate(nbhd_grafidx) if gi == dunson_idx)
    brown_sub  = next(i for i, gi in enumerate(nbhd_grafidx) if gi == brown_idx)

    nbhd_layout = nbhd_graf.layout_fruchterman_reingold(niter=500)
    nc          = np.array(nbhd_layout.coords, dtype=float)

    dunson_only_all = [i for i, _ in enumerate(nbhd_grafidx)
                       if nbhd_nodes[i] in dunson_nbhd and nbhd_nodes[i] not in brown_nbhd]
    brown_only_all  = [i for i, _ in enumerate(nbhd_grafidx)
                       if nbhd_nodes[i] in brown_nbhd  and nbhd_nodes[i] not in dunson_nbhd]
    if dunson_only_all and brown_only_all:
        c_d = nc[dunson_only_all].mean(axis=0)
        c_b = nc[brown_only_all].mean(axis=0)
        axis_vec = c_d - c_b
        span = np.linalg.norm(axis_vec)
        if span > 0:
            axis_vec = axis_vec / span * span * 0.02
            nc[dunson_only_all + [dunson_sub]] += axis_vec
            nc[brown_only_all  + [brown_sub]]  -= axis_vec

    fig, ax = plt.subplots(figsize=(14, 10))
    nbhd_segs = [(nc[e.source], nc[e.target]) for e in nbhd_graf.es]
    ax.add_collection(LineCollection(nbhd_segs, colors="#cccccc", linewidths=0.5, zorder=1))

    nbhd_orig_degs = [degs[gi] for gi in nbhd_grafidx]
    dunson_1hop    = set(graf.neighbors(dunson_idx))
    brown_1hop     = set(graf.neighbors(brown_idx))
    focal_set      = {dunson_sub, brown_sub}

    def _top3_1hop(hop_set, exclude):
        candidates = [i for i, _ in enumerate(nbhd_grafidx)
                      if nbhd_nodes[i] in hop_set and i not in exclude]
        candidates.sort(key=lambda i: nbhd_orig_degs[i], reverse=True)
        return set(candidates[:5])

    top3_dunson = _top3_1hop(dunson_1hop, focal_set)
    top3_brown  = _top3_1hop(brown_1hop,  focal_set)

    extra_names = ["sudipto banerjee", "ming yuan", "peter buhlmann", "carlos m carvalho"]
    extra_set   = {i for i, mid in enumerate(nbhd_myids)
                   if any(en in name_map.get(mid, "").lower() for en in extra_names)}

    label_set = focal_set | top3_dunson | top3_brown | extra_set

    ax.scatter(nc[:, 0], nc[:, 1], s=60, c="#5BA3C9", zorder=2, linewidths=0)

    texts_nbhd = []
    for i in label_set:
        label = name_map.get(nbhd_myids[i], f"Node {i}")
        focal = i in focal_set
        t = ax.text(nc[i, 0], nc[i, 1], label,
                    fontsize=22 if focal else 19,
                    fontweight="bold" if focal else "normal",
                    ha="center", va="bottom", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        texts_nbhd.append(t)
    adjust_text(texts_nbhd, ax=ax,
                expand_text=(2.0, 2.5), expand_points=(1.5, 1.5),
                force_text=(0.8, 1.0),  force_points=(0.5, 0.8),
                lim=1000)

    ax.axis("off")
    ax.autoscale_view()
    ylo, yhi = ax.get_ylim()
    span_y = yhi - ylo
    ax.set_ylim(ylo - 0.05 * span_y, yhi + 0.25 * span_y)
    fig.savefig(_out("dunson_brown_neighbourhood.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dunson_brown_neighbourhood.pdf")
