"""Reproduce the JMLR illustrations and coauthorship analyses locally."""

import argparse
import contextlib
import io
import json
import os
import random
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text

from spar import Leaf_stripping, Leaf_stripping_postCT, spar
from mcmc.gibbsSampling import gibbsFullCT
from mcmc.estimateAlpha import estimateAlphaEM

ROOT = Path(__file__).resolve().parent
COLORS = {0: "#aaaaaa", 1: "#ff0000", 2: "#0000ff"}


def seed(np_seed=42, python_seed=10001):
    np.random.seed(np_seed)
    random.seed(python_seed)
    ig.set_random_number_generator(random)


def load_coauthors(data):
    """Load the binary adjacency matrix; retain original author indices."""
    a = np.loadtxt(data / "coauthorThresh2GiantAdj_big.txt", dtype=np.uint8)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.array_equal(a, a.T):
        raise ValueError("Expected a symmetric square adjacency matrix.")
    a = (a > 0).astype(np.int64)
    np.fill_diagonal(a, 0)
    graph = nx.from_numpy_array(a)
    ids = sorted(max(nx.connected_components(graph), key=len))
    names = pd.read_table(data / "coauthorThresh2GiantAuthorList_big.txt", header=None)[0]
    a = a[np.ix_(ids, ids)]
    return a, np.asarray(ids), names.iloc[ids].astype(str).to_numpy()


def draw_graph(graph, path, labels=None, names=None, positions=None, top=10, degrees=None, recovery_style=False):
    nodes = list(graph)
    if not nodes:
        return
    if positions is None:
        indices = {v: i for i, v in enumerate(nodes)}
        g = ig.Graph(n=len(nodes), edges=[(indices[u], indices[v]) for u, v in graph.edges()])
        layout = np.asarray(g.layout_fruchterman_reingold(niter=500).coords)
        positions = dict(zip(nodes, layout))
    fig, ax = plt.subplots(figsize=(8, 8) if recovery_style else (9, 8))
    ax.add_collection(LineCollection([(positions[u], positions[v]) for u, v in graph.edges()],
                                    colors="#a0a0a0", linewidths=.35 if recovery_style else .4, alpha=.25 if recovery_style else .6))
    xy = np.asarray([positions[v] for v in nodes])
    colors = [COLORS.get(int(labels[v]), plt.cm.tab20((int(labels[v])-1) % 20))
              for v in nodes] if labels is not None else ["#5ba3c9"] * len(nodes)
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=20 if recovery_style else (14 if names is not None else 3), linewidths=0)
    if names is not None:
        degrees = dict(graph.degree()) if degrees is None else degrees
        texts = [ax.text(*positions[v], names[v], fontsize=16 if recovery_style else 10, ha="center",
                        va='bottom' if recovery_style else 'baseline',
                        path_effects=[pe.withStroke(linewidth=1.3, foreground='white')] if recovery_style else [],
                        bbox=None if recovery_style else dict(facecolor="white", alpha=.65, edgecolor="none", pad=.3))
                 for v in sorted(nodes, key=lambda v: (-degrees[v], v))[:top]]
        adjust_text(texts, ax=ax, iter_lim=200)
    ax.autoscale_view()
    ax.set_axis_off()
    if recovery_style:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary(out, record):
    record["outputs"] = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "summary.json")
    (out / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    keys = ("nodes", "edges", "communities", "unclassified", "samples", "valid_samples", "agreement")
    print(json.dumps({"case": out.name, **{k: record[k] for k in keys if k in record}}))


def run_intro(data, out):
    seed()
    a, ids, names = load_coauthors(data)
    graph = nx.from_numpy_array(a)
    draw_graph(graph, out / "tmp_stats.pdf")
    degrees = a.sum(axis=1).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(degrees[degrees <= 15], bins=np.arange(.5, 15.5), color="#5ba3c9", rwidth=.95)
    ax.set(xlabel="Degree", ylabel="Number of vertices", xlim=(-.3, 15.3))
    fig.savefig(out / "degree_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    dunson = next(i for i, name in enumerate(names) if "dunson" in name.lower() and "david" in name.lower())
    brown = next(i for i, name in enumerate(names) if "brown" in name.lower() and "lawrence" in name.lower())
    neighborhood = set(nx.single_source_shortest_path_length(graph, dunson, cutoff=2))
    neighborhood.update(nx.single_source_shortest_path_length(graph, brown, cutoff=2))
    draw_graph(graph.subgraph(sorted(neighborhood)), out / "dunson_brown_neighbourhood.pdf", names=names)
    pd.DataFrame({"node": range(len(a)), "original_id": ids, "author": names, "degree": degrees}).to_csv(out / "vertices.csv", index=False)
    write_summary(out, {"nodes": len(a), "edges": graph.number_of_edges(),
                       "dunson_brown_distance": nx.shortest_path_length(graph, dunson, brown),
                       "neighborhood_nodes": len(neighborhood)})


def prune(a, q, implementation="paper", rank=10):
    with contextlib.redirect_stdout(io.StringIO()):
        result = Leaf_stripping(a, K=2, b=min(rank, len(a)), d=0, min_component_size=q,
                                select_by="initial", recover=True,
                                recover_mode="distance_threshold", threshold_by="order",
                                mode="True", check_protected_removal=True,
                                recovery_threshold_divisor=2., beta=.8, implementation=implementation)
    if result[0] < 2:
        raise RuntimeError("SPAR did not produce two anchors.")
    return np.asarray(result[1]), np.asarray(result[2])


def recovery_layout(a, labels):
    """One full-graph layout, reused by distance and Monte Carlo recovery."""
    rng = np.random.RandomState(42)
    initial = {}
    for v, label in enumerate(labels):
        center = -.3 if label == 1 else .3
        initial[v] = (center + rng.uniform(-.1, .1), rng.uniform(-.5, .5))
    return nx.fruchterman_reingold_layout(nx.from_numpy_array(a), pos=initial, seed=42)


def cluster_plots(a, labels, names, out, prefix, cores=False, positions=None):
    graph = nx.from_numpy_array(a)
    coverage = {}
    for label in sorted(set(labels) - {0}):
        nodes = np.flatnonzero(labels == label).tolist()
        sub = graph.subgraph(nodes)
        selected = nodes if cores else sorted(max(nx.connected_components(sub), key=len))
        coverage[str(int(label))] = {"assigned": len(nodes), "displayed": len(selected)}
        draw_graph(graph.subgraph(selected), out / f"{prefix}_{'component' if cores else 'cluster'}_{label}.pdf",
                   labels, names, positions=positions, degrees=a.sum(axis=1), recovery_style=not cores)
    return coverage


def posterior(graph, anchors, burn, samples, np_seed, python_seed=None, size_thresh=.01):
    seed(np_seed, np_seed if python_seed is None else python_seed)
    with contextlib.redirect_stdout(io.StringIO()):
        result = gibbsFullCT(graph, K=2, alpha=0, beta=1, Burn=burn, M=samples,
                             display=False, size_thresh=size_thresh)
        post = Leaf_stripping_postCT(result, anchors)
    probs = np.asarray(post["assignment_probs"])
    if not post["valid_samples"]:
        raise RuntimeError("No posterior samples meet the anchor condition.")
    assert np.all(np.isfinite(probs)) and np.all(probs >= 0)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    return probs, int(post["valid_samples"])


def posterior_labels(probs, implementation="paper"):
    """Uniform argmax ties in paper mode; hierarchical mode picks the first maximizer."""
    if implementation == "hierarchical":
        return probs.argmax(axis=1)+1
    return np.array([random.choice(np.flatnonzero(row == row.max()).tolist())+1 for row in probs])


def run_coauthorship(data, out, samples=1000, implementation="paper", rank=10, mc_seeds=None):
    seed()
    a, ids, names = load_coauthors(data)
    labels, cores = prune(a, len(a)//10, implementation, rank)
    labels, cores = np.where(labels > 0, 3-labels, 0), np.where(cores > 0, 3-cores, 0)
    anchors = [np.flatnonzero(cores == k).tolist() for k in (1, 2)]
    graph = ig.Graph.Adjacency(a.tolist(), mode="undirected")
    alpha_estimate = float(estimateAlphaEM(graph, K=2, display=False))
    if alpha_estimate != 0:
        raise RuntimeError(f"Paper alpha=0 differs from the reproduced estimate {alpha_estimate}.")
    mc_seeds = [7] if mc_seeds is None else list(mc_seeds)
    if not mc_seeds or len(mc_seeds) != len(set(mc_seeds)):
        raise ValueError("Provide at least one distinct Monte Carlo seed.")
    weighted_probs = np.zeros((len(a), 2))
    valid_by_seed = []
    for mc_seed in mc_seeds:
        try:
            chain_probs, chain_valid = posterior(graph.copy(), anchors, 50, samples, mc_seed)
        except RuntimeError as error:
            if str(error) != "No posterior samples meet the anchor condition.":
                raise
            chain_valid = 0
        valid_by_seed.append(chain_valid)
        if chain_valid:
            weighted_probs += chain_valid * chain_probs
    valid = sum(valid_by_seed)
    if not valid:
        raise RuntimeError("No posterior samples meet the anchor condition across the declared seeds.")
    probs = weighted_probs / valid
    if len(mc_seeds) > 1:
        seed(0, 0)  # Reproducible tie breaking for pooled chains.
    mc = posterior_labels(probs, implementation)
    positions = recovery_layout(a, labels)
    pd.DataFrame({"node": range(len(a)), "x": [positions[v][0] for v in range(len(a))],
                  "y": [positions[v][1] for v in range(len(a))]}).to_csv(out / "layout.csv", index=False)
    coverage = {"cores": cluster_plots(a, cores, names, out, "core", True),
                "distance": cluster_plots(a, labels, names, out, "dist_recovery", positions=positions),
                "monte_carlo": cluster_plots(a, mc, names, out, "mc_recovery", positions=positions)}
    pd.DataFrame({"node": range(len(a)), "original_id": ids, "author": names,
                  "degree": a.sum(axis=1), "core_cluster": cores, "dist_cluster": labels,
                  "mc_cluster": mc, "p1": probs[:, 0], "p2": probs[:, 1]}).to_csv(out / "assignments.csv", index=False)
    write_summary(out, {"nodes": len(a), "edges": int(a.sum()//2), "samples": samples*len(mc_seeds),
                       "valid_samples": valid, "agreement": float(np.mean(labels == mc)),
                       "alpha": 0, "alpha_estimate": alpha_estimate, "alpha_source": "estimateAlphaEM, K=2",
                       "implementation": implementation, "rank": rank,
                       "effective_rank": 2*rank if implementation == "hierarchical" else rank,
                       "core_sizes": [len(c) for c in anchors], "coverage": coverage,
                       "shared_recovery_layout": "layout.csv", "mc_seeds": mc_seeds,
                       "samples_per_chain": samples, "burn_per_chain": 50,
                       "valid_samples_by_seed": dict(zip(map(str, mc_seeds), valid_by_seed)),
                       "seed_mode": "two_node", "tie_seed": 0 if len(mc_seeds) > 1 else None,
                       "seed_selection": "All declared seeds pooled by valid draw count.",
                       "mixing_note": "Finite-chain convergence is not established; acceptance and probabilities vary across seeds."})


def hierarchy_layout(a, labels, history, remap):
    """Original per-community FR layout, positioned along the split tree."""
    children = {p: (left, right) for p, left, right in history}
    def leaves(node):
        return sum((leaves(c) for c in children[node]), []) if node in children else [node]
    def minimum(node):
        return min(remap[c] for c in leaves(node))
    ranges = {}
    def assign(node, lo, hi, depth=0):
        if node not in children:
            ranges[node] = lo, hi
            return
        left, right = sorted(children[node], key=minimum)
        mid, gap = (lo+hi)/2, .4/(depth+1)
        assign(left, lo, mid-gap/2, depth+1)
        assign(right, mid+gap/2, hi, depth+1)
    assign("root", 0, len(remap)*1.5)
    graph = ig.Graph.Adjacency(a.tolist(), mode="undirected")
    fixed = {}
    for old in leaves("root"):
        nodes = np.flatnonzero(labels == old).tolist()
        coords = np.asarray(graph.subgraph(nodes).layout_fruchterman_reingold().coords)
        span = np.ptp(coords, axis=0)
        coords = (coords-coords.min(axis=0))/np.where(span == 0, 1., span)
        lo, hi = ranges[old]
        fixed.update({v: (coords[i, 0]*(hi-lo)+lo, coords[i, 1]) for i, v in enumerate(nodes)})
    return nx.spring_layout(nx.from_numpy_array(a), pos=fixed, fixed=list(fixed), iterations=100, seed=42)


def hierarchy_plots(a, labels, cores, names, positions, out):
    """Original gold vertices, core edges, author labels, and figure sizes."""
    graph, degrees = nx.from_numpy_array(a), a.sum(axis=1)
    coverage = {}
    def panel(nodes, path, full=False):
        selected = set(nodes)
        fig, ax = plt.subplots(figsize=(14, 8) if full else (8, 8))
        for u, v in graph.edges():
            if u not in selected or v not in selected:
                continue
            gold = cores[u] > 0 and cores[u] == cores[v]
            ax.plot([positions[u][0], positions[v][0]], [positions[u][1], positions[v][1]],
                    color="#DAA520" if gold else "#909090",
                    alpha=.8 if gold else (.4 if full else .5),
                    linewidth=1. if gold else .3, zorder=1)
        xy = np.asarray([positions[v] for v in nodes])
        ax.scatter(xy[:, 0], xy[:, 1], s=80, c="#DAA520", alpha=.95,
                   edgecolors="none", linewidths=0, zorder=2)
        if not full:
            texts = [ax.text(*positions[v], names[v], fontsize=25, ha="center", va="bottom",
                             color="black", path_effects=[pe.withStroke(linewidth=2, foreground="white")])
                     for v in sorted(nodes, key=lambda v: -degrees[v])[:10]]
            adjust_text(texts, ax=ax)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    for label in sorted(set(labels)-{0}, key=lambda k: np.sum(labels == k)):
        assigned = np.flatnonzero(labels == label).tolist()
        nodes = sorted(max(nx.connected_components(graph.subgraph(assigned)), key=len))
        coverage[str(int(label))] = {"assigned": len(assigned), "displayed": len(nodes)}
        panel(nodes, out/f"final_dist_cluster_{label}.pdf")
    panel(list(range(len(a))), out/"final_dist_full_graph.pdf", full=True)
    return coverage


def draw_tree(history, remap, path):
    """Original ETE dendrogram, including branch lengths and orientation."""
    if os.name != "nt":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ete3 import Tree, TreeStyle, NodeStyle, TextFace
    children = {p: (left, right) for p, left, right in history}
    def minimum(node):
        return min(minimum(c) for c in children[node]) if node in children else remap[node]
    def newick(node, root=True):
        if node not in children:
            return f"{remap[node]}:0.5"
        left, right = sorted(children[node], key=minimum, reverse=True)
        branch = f"({newick(left, False)},{newick(right, False)})"
        return branch if root else branch+":1"
    tree, style = Tree(newick("root")+";"), TreeStyle()
    style.show_leaf_name = style.show_scale = False
    style.rotation, style.branch_vertical_margin = 270, 28
    node_style = NodeStyle()
    node_style["size"] = 0
    node_style["hz_line_width"] = node_style["vt_line_width"] = 3
    for node in tree.traverse():
        node.set_style(node_style)
        if node.is_leaf():
            face = TextFace(node.name, fsize=24)
            face.rotation = -90
            node.add_face(face, column=0, position="branch-right")
    tree.render(str(path), tree_style=style)


def run_hierarchical(data, out, q=25, implementation="hierarchical", rank=10):
    # The hierarchical routine selects b*K vertices; expose the actual rank here.
    if implementation == "hierarchical" and (rank < 2 or rank % 2):
        raise ValueError("Binary hierarchy requires a positive even degree rank.")
    pruning_rank = rank//2 if implementation == "hierarchical" else rank
    seed()
    a, ids, names = load_coauthors(data)
    graph = nx.from_numpy_array(a)
    current, core = prune(a, q, implementation, pruning_rank)
    history, rounds, core_rounds = [("root", 1, 2)], [current.copy()], [core.copy()]
    next_label, round_number = 3, 1
    while True:
        round_number += 1
        new, new_core = current.copy(), core.copy()
        changed = False
        for label in sorted(set(current)-{0}):
            all_nodes = np.flatnonzero(current == label).tolist()
            nodes = sorted(max(nx.connected_components(graph.subgraph(all_nodes)), key=len))
            if len(nodes) <= 10:
                continue
            seed(42+round_number*100+int(label), 10001+round_number*100+int(label))
            try:
                sub, sub_core = prune(a[np.ix_(nodes, nodes)], q, implementation, pruning_rank)
            except RuntimeError:
                continue
            if any(np.sum(sub_core == k) < 10 for k in (1, 2)):
                continue
            changed = True
            new[all_nodes], new_core[all_nodes] = 0, 0
            for k in (1, 2):
                new[np.asarray(nodes)[sub == k]] = next_label+k-1
                new_core[np.asarray(nodes)[sub_core == k]] = next_label+k-1
            history.append((int(label), next_label, next_label+1))
            next_label += 2
        if not changed:
            break
        current, core = new, new_core
        rounds.append(current.copy())
        core_rounds.append(core.copy())
    leaves = set(current)-{0}
    order = [child for _, left, right in history for child in (left, right) if child in leaves]
    if len(order) >= 2:
        order[-1], order[-2] = order[-2], order[-1]
    remap = {old: i+1 for i, old in enumerate(order)}
    labels = np.array([remap.get(v, 0) for v in current])
    final_core = np.array([remap.get(v, 0) for v in core])
    positions = hierarchy_layout(a, current, history, remap)
    pd.DataFrame({"node": range(len(a)), "x": [positions[v][0] for v in range(len(a))],
                  "y": [positions[v][1] for v in range(len(a))]}).to_csv(out/"layout.csv", index=False)
    coverage = hierarchy_plots(a, labels, final_core, names, positions, out)
    rows = {"node": range(len(a)), "original_id": ids, "author": names,
            "degree": a.sum(axis=1), "cluster": labels, "core_cluster": final_core}
    for i, (dist, anchor) in enumerate(zip(rounds, core_rounds), 1):
        rows[f"dist_cluster_r{i}"], rows[f"core_cluster_r{i}"] = dist, anchor
    pd.DataFrame(rows).to_csv(out / "assignments.csv", index=False)
    draw_tree(history, remap, out / "community_tree.pdf")
    write_summary(out, {"nodes": len(a), "edges": int(a.sum()//2), "pruning_q": q,
                       "implementation": implementation, "rank": rank,
                       "effective_rank": rank, "pruning_argument_b": pruning_rank,
                       "minimum_child_core": 10, "rounds": len(rounds), "communities": len(leaves),
                       "unclassified": int(np.sum(labels == 0)), "split_history": history, "coverage": coverage})


def run_toy(out, samples=500):
    backbone = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,6),(1,7),(2,8),(3,9),(5,10),(5,11),(8,22),
                (12,13),(12,14),(12,15),(12,16),(13,17),(13,18),(14,19),(14,20),(12,21)]
    noise = [(6,7),(4,11),(17,18),(19,20),(5,16),(8,19),(10,17),(9,15),(8,3)]
    graph = nx.Graph(backbone+noise)
    pos = {v: (x-1.8,y-.6) for v,(x,y) in nx.spring_layout(graph.subgraph(set(range(12))|{22}), seed=7,k=.55).items()}
    pos.update({v:(x+1.8,y+.6) for v,(x,y) in nx.spring_layout(graph.subgraph(range(12,22)),seed=11,k=.55).items()})
    anchors = [{0,1,5}, {12,21,13,17}]
    pruning = spar(nx.to_numpy_array(graph, nodelist=range(23), dtype=int), 2,
                   tau=5, Q=3, recover=False)
    actual = [set(np.flatnonzero(pruning.anchor_labels == k)) for k in (1, 2)]
    assert {frozenset(c) for c in actual} == {frozenset(c) for c in anchors}
    kept = {v for v in graph if graph.degree(v) >= 4}
    simple = sorted([c for c in nx.connected_components(graph.subgraph(kept)) if max(graph.degree(v) for v in c)>=5], key=lambda c:(-len(c),-max(graph.degree(v) for v in c),min(c)))
    rng = random.Random(123)
    distances = [nx.multi_source_dijkstra_path_length(graph, c, weight=None) for c in simple]
    simple_labels = {}
    for v in graph:
        if v in simple[0]: label = 1
        elif v in simple[1]: label = 2
        elif v in kept: label = rng.choice([1,2])
        elif distances[0][v] < distances[1][v]: label = 1
        elif distances[1][v] < distances[0][v]: label = 2
        else: label = rng.choice([1,2])
        simple_labels[v] = label
    colors = ["#ff0000", "#0000ff"]
    def panel(filename, node_colors, faded=(), colorbar=False):
        fig, ax = plt.subplots(figsize=(8,8))
        for edges, style in [(backbone,"solid"),(noise,"dashdot")]:
            for u,v in edges:
                nx.draw_networkx_edges(graph,pos,edgelist=[(u,v)],ax=ax,style=style,
                                       edge_color="#cccccc" if u in faded or v in faded else "#222222")
        nx.draw_networkx_nodes(graph,pos,nodelist=sorted(graph),node_color=node_colors,node_size=250,ax=ax)
        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list("posterior",["red","blue"]),norm=plt.Normalize(0,1))
            fig.colorbar(sm,ax=ax,label="P(Community 2)",fraction=.046)
        ax.axis("off")
        fig.savefig(out/filename,bbox_inches="tight")
        plt.close(fig)
    panel("figure1_original_graph.pdf", ["#daa520"]*23)
    panel("figure2_algorithm_colored_result.pdf", [colors[0] if v in simple[0] else colors[1] if v in simple[1] else "#9932cc" if v in kept else "#808080" for v in sorted(graph)], set(graph)-kept)
    panel("figure3_recovered_communities.pdf", [colors[simple_labels[v]-1] for v in sorted(graph)])
    removed = {v for v in graph if graph.degree(v)==2}
    panel("figure4_remaining_nodes.pdf", ["#808080" if v in removed else "#daa520" for v in sorted(graph)], removed)
    faded = set(graph)-anchors[0]-anchors[1]-{8,22}
    panel("figure5_manual_coloring.pdf", [colors[0] if v in anchors[0] else colors[1] if v in anchors[1] else "#9932cc" if v in {8,22} else "#808080" for v in sorted(graph)], faded)
    igraph = ig.Graph(n=23,edges=list(graph.edges()))
    igraph.es["tree"] = True
    probs, valid = posterior(igraph, anchors, 30, samples, 7000, size_thresh=.1)
    hard = posterior_labels(probs)
    panel("hard_classification.pdf", [colors[k-1] for k in hard])
    panel("soft_classification.pdf", [(1-p,0,p) for p in probs[:,1]], colorbar=True)
    pd.DataFrame({"node":range(23),"simple_label":[simple_labels[v] for v in range(23)],"hard_label":hard,"p1":probs[:,0],"p2":probs[:,1]}).to_csv(out/"assignments.csv",index=False)
    write_summary(out,{"nodes":23,"edges":graph.number_of_edges(),"samples":samples,"valid_samples":valid,
                       "seed_mode":"two_node","burn":30,"np_seed":7000,"python_seed":7000,
                       "illustrated_anchors":[sorted(c) for c in anchors],"simple_anchors":[sorted(c) for c in simple],
                       "pruning_trace":pruning.trace})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["intro","toy","coauthorship","hierarchical","all"], default="all")
    parser.add_argument("--data", type=Path, default=ROOT/"data")
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"case_studies")
    parser.add_argument("--hierarchy-q", type=int, help="Initial component-size threshold for each split; default: 25.")
    parser.add_argument("--implementation", choices=["paper","hierarchical"],
                        help="Default: hierarchical for hierarchy, paper for coauthorship.")
    parser.add_argument("--rank", type=int, default=10, help="Degree rank; default: 10. Hierarchy converts to the hierarchical routine's b argument.")
    parser.add_argument("--samples", type=int, help="Override 500 toy / 1000 coauthorship samples for checks.")
    parser.add_argument("--mc-seeds", type=int, nargs='+', help="Coauthorship chains; default: 7.")
    args = parser.parse_args()
    cases = ["intro","toy","coauthorship","hierarchical"] if args.case == "all" else [args.case]
    for case in cases:
        out = args.output/case
        out.mkdir(parents=True,exist_ok=True)
        if case == "intro": run_intro(args.data,out)
        elif case == "toy": run_toy(out,args.samples or 500)
        elif case == "coauthorship": run_coauthorship(args.data,out,args.samples or 1000,args.implementation or "paper",args.rank,args.mc_seeds)
        else:
            implementation = args.implementation or "hierarchical"
            q = args.hierarchy_q if args.hierarchy_q is not None else 25
            run_hierarchical(args.data,out,q,implementation,args.rank)


if __name__ == "__main__":
    main()
