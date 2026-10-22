"""SPAR pruning and recovery (JMLR Algorithms 1–2).

Vertices are indexed 0,...,n-1; community labels are 1,...,K.
Label 0 denotes an unassigned vertex or failed pruning.
"""
from dataclasses import dataclass
import random

import networkx as nx
import numpy as np


@dataclass
class Result:
    success: bool
    anchor_labels: np.ndarray
    labels: np.ndarray
    references: list
    sizes: list
    tau: float
    D_star: float
    Q: int
    trace: list


def _adjacency(A):
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or len(A) == 0:
        raise ValueError('A must be a nonempty square matrix')
    if not np.all(np.isin(A, [0, 1])) or not np.array_equal(A, A.T) or np.any(np.diag(A)):
        raise ValueError('A must be a symmetric binary adjacency matrix with zero diagonal')
    return A


def _qualifying(G, high, Q):
    return [c for c in nx.connected_components(G) if len(c) >= Q and c & high]


def recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels,
                                                degree_threshold):
    """Nearest reference set; break equal distances uniformly at random."""
    A = _adjacency(A_input)
    G = nx.from_numpy_array(A)
    degree = A.sum(axis=1)
    references = [np.flatnonzero((predicted_labels == k) & (degree >= degree_threshold)).tolist()
                  for k in range(1, cluster_count + 1)]
    if not references or not all(references):
        raise ValueError('each community must contain a reference vertex')
    distances = np.full((len(A), cluster_count), np.inf)
    for k, sources in enumerate(references):
        paths = nx.multi_source_dijkstra_path_length(G, sources, weight=None)
        for node, distance in paths.items():
            distances[node, k] = distance
    labels = np.array([random.choice(np.flatnonzero(row == row.min()).tolist()) + 1
                       for row in distances], dtype=int)
    return labels, references


def spar(A, K, rank=10, Q=25, tau=None, beta=0.8, recovery_divisor=2,
         max_steps=10000, recover=True):
    """Prune minimum original-degree candidates, retrying with floor(beta*Q).

    Candidates lie on a cycle or increase the number of qualifying components
    when removed alone. Degree ties at the rank threshold are included.
    The trace records only actual candidate removals, including failed retries.
    """
    A = _adjacency(A)
    n = len(A)
    if not isinstance(K, (int, np.integer)) or not 1 <= K <= n:
        raise ValueError('require integer 1 <= K <= n')
    if not isinstance(Q, (int, np.integer)) or Q < 1 or not 0 < beta < 1:
        raise ValueError('require integer Q >= 1 and 0 < beta < 1')
    if not np.isfinite(recovery_divisor) or recovery_divisor < 1 or max_steps < 1:
        raise ValueError('require recovery_divisor >= 1 and max_steps >= 1')
    degrees = A.sum(axis=1, dtype=np.int64)
    if tau is None:
        if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= n:
            raise ValueError('require integer 1 <= rank <= n')
        tau = float(np.sort(degrees)[-rank])
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('tau must be finite and nonnegative')
    high = set(np.flatnonzero(degrees >= tau))
    original = nx.from_numpy_array(A)
    trace = []
    while True:
        G, D_star = original.copy(), 0.0
        for step in range(max_steps + 1):
            components = _qualifying(G, high, Q)
            if len(components) >= K:
                # Fixed vertex order breaks equal maximum-degree component ties.
                components.sort(key=lambda c: (-max(degrees[list(c)]), min(c)))
                components = components[:K]
                anchors = np.zeros(n, dtype=int)
                for k, component in enumerate(components, 1):
                    anchors[list(component)] = k
                labels, references = anchors.copy(), []
                if recover:
                    threshold = min(D_star, tau / recovery_divisor)
                    labels, references = recover_with_filtered_distance_by_threshold(A, K, anchors, threshold)
                return Result(True, anchors, labels, references, list(map(len, components)),
                              float(tau), D_star, Q, trace)
            if step == max_steps:
                raise RuntimeError('pruning iteration limit reached')
            cycles = set().union(*map(set, nx.cycle_basis(G))) if G else set()
            candidates = set(cycles)
            limit = min(degrees[list(cycles)]) if cycles else np.inf
            for node in sorted(set(nx.articulation_points(G)) - cycles):
                if degrees[node] > limit:
                    continue
                trial = G.copy()
                trial.remove_node(node)
                if len(_qualifying(trial, high, Q)) > len(components):
                    candidates.add(node)
            if not candidates:
                break
            D_star = float(min(degrees[list(candidates)]))
            batch = sorted(node for node in candidates if degrees[node] == D_star)
            trace.append(dict(Q=int(Q), degree=D_star, removed=batch))
            G.remove_nodes_from(batch)
        if Q == 1:
            empty = np.zeros(n, dtype=int)
            return Result(False, empty, empty.copy(), [], [], float(tau), D_star, Q, trace)
        Q = max(1, int(beta * Q))


def Leaf_stripping(A_input, K=3, min_component_size=25, b=3, max_steps=10000,
                   d=0, threshold_by='order', select_by='initial', recover=False,
                   recover_mode='distance_threshold', mode='True', beta=0.8,
                   check_protected_removal=True, recovery_threshold_divisor=2,
                   implementation='paper'):
    """Compatibility entry point for the paper scripts; prefer spar() in new code."""
    if implementation == 'hierarchical':
        from hierarchical_spar import hierarchical_leaf_stripping
        return hierarchical_leaf_stripping(A_input, K, min_component_size, b, max_steps,
            d, threshold_by, select_by, recover, recover_mode, mode, beta,
            check_protected_removal, recovery_threshold_divisor)
    if implementation != 'paper' or threshold_by not in ('order', 'deg'):
        raise ValueError('unknown implementation or threshold mode')
    if d != 0 or select_by != 'initial' or mode != 'True' or not check_protected_removal:
        raise ValueError('these options are outside the paper algorithm')
    if recover and recover_mode != 'distance_threshold':
        raise ValueError('paper recovery uses distance_threshold')
    result = spar(A_input, K, rank=b, Q=min_component_size,
                  tau=b if threshold_by == 'deg' else None, beta=beta,
                  recovery_divisor=recovery_threshold_divisor,
                  max_steps=max_steps, recover=recover)
    count = K if result.success else 0
    if recover:
        return count, result.labels, result.anchor_labels, result.sizes, result.references
    return count, result.anchor_labels, result.sizes


def Leaf_stripping_postCT(gibbs_results, components):
    """Align samples with exactly one root per anchor set, then average."""
    n, K = len(gibbs_results['node_tree_coo']), len(components)
    components = [sorted(c) for c in components]
    flat = [v for c in components for v in c]
    if K == 0 or not all(components) or len(set(flat)) != len(flat) or any(v < 0 or v >= n for v in flat):
        raise ValueError('anchor sets must be nonempty, disjoint and within the graph')
    lookup = {v: k for k, c in enumerate(components) for v in c}
    roots_count, assignments_count = np.zeros(n), np.zeros((n, K))
    valid = 0
    for sample in gibbs_results['samples']:
        roots = sample['roots']
        anchor_ids = [lookup.get(v, -1) for v in roots]
        if len(roots) != K or set(anchor_ids) != set(range(K)):
            continue
        assignments = np.asarray(sample['assignments'])
        if assignments.shape != (n,) or not np.issubdtype(assignments.dtype, np.integer) or np.any((assignments < 0) | (assignments >= K)):
            raise ValueError('invalid posterior sample assignments')
        roots_count[roots] += 1
        assignments_count[np.arange(n), np.asarray(anchor_ids)[assignments]] += 1
        valid += 1
    divisor = max(valid, 1)
    return dict(component_posteriors=[roots_count[c] / divisor for c in components],
                global_posterior=roots_count / (divisor * K), root_probs=roots_count / divisor,
                assignment_probs=assignments_count / divisor, component_purity=None,
                valid_samples=valid, total_samples=len(gibbs_results['samples']))
