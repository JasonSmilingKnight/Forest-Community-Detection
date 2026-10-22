"""Fixed-K Gibbs kernel for disjoint two-vertex founder edges.

The joint target is proportional to prod_v prod_{j=1}^{d(v)-1}(beta*j+alpha)
on forests compatible with the paired-founder ordering and observed graph.
"""
import math
import random
from collections import deque

import networkx as nx
import numpy as np


def _adjacency(graf):
    adj = [set() for _ in range(graf.vcount())]
    for edge in graf.es:
        if edge['tree']:
            u, v = edge.tuple
            adj[u].add(v)
            adj[v].add(u)
    return adj


def _orient(adj, root):
    parent = {root: None}
    order = [root]
    for v in order:
        for w in sorted(adj[v]):
            if w == parent[v]:
                continue
            if w in parent:
                raise ValueError('Tree edges must form an acyclic forest')
            parent[w] = v
            order.append(w)
    sizes = {v: 1 for v in order}
    for v in reversed(order[1:]):
        sizes[parent[v]] += sizes[v]
    return parent, order, sizes


def sample_ordering(graf, K):
    """Draw exactly uniformly from all compatible paired-founder histories."""
    adj = _adjacency(graf)
    unseen = set(range(len(adj)))
    histories = []
    while unseen:
        parent, vertices, sizes = _orient(adj, min(unseen))
        m = len(vertices)
        if m < 2:
            raise ValueError('Two-founder components cannot be singletons')
        # Rerooting identity for rooted linear-extension counts h(r,T).
        log_h = {vertices[0]: 0.0}
        for v in vertices[1:]:
            log_h[v] = log_h[parent[v]] + math.log(sizes[v]) - math.log(m-sizes[v])
        offset = max(log_h.values())
        root = random.choices(vertices, [math.exp(log_h[v]-offset) for v in vertices])[0]
        parent, vertices, sizes = _orient(adj, root)
        frontier = sorted(adj[root])
        history = [root]
        while frontier:
            v = random.choices(frontier, [sizes[w] for w in frontier])[0]
            frontier.remove(v)
            history.append(v)
            frontier.extend(w for w in sorted(adj[v]) if parent.get(w) == v)
        histories.append(history)
        unseen.difference_update(vertices)
    if len(histories) != K:
        raise ValueError('Forest must have exactly K components')
    random.shuffle(histories)
    ordering = [v for history in histories for v in history[:2]]
    positions = [2]*K
    while len(ordering) < len(adj):
        remaining = [len(h)-p for h, p in zip(histories, positions)]
        k = random.choices(range(K), remaining)[0]
        ordering.append(histories[k][positions[k]])
        positions[k] += 1
    return ordering


def _set_parents(graf, pi, K):
    adj = _adjacency(graf)
    parents = [None]*len(pi)
    visited = set()
    rank = {v: i for i, v in enumerate(pi)}
    for k in range(K):
        a, b = pi[2*k:2*k+2]
        if b not in adj[a] or a in visited:
            raise ValueError('Each founder pair must be an edge in a distinct component')
        parent, vertices, _ = _orient(adj, a)
        if len(vertices) < 2 or visited.intersection(vertices):
            raise ValueError('Invalid founder components')
        for v in vertices:
            p = parent[v]
            if p is not None and rank[p] >= rank[v]:
                raise ValueError('Ordering is incompatible with forest parents')
            parents[v] = p
        visited.update(vertices)
    if len(visited) != len(pi):
        raise ValueError('Founder components must cover every vertex')
    graf.vs['pa'] = parents


def _initialize(graf, K):
    observed = nx.Graph()
    observed.add_nodes_from(range(graf.vcount()))
    observed.add_edges_from(graf.get_edgelist())
    components = sorted(nx.connected_components(observed), key=min)
    if len(components) > K or any(len(c) < 2 for c in components):
        raise ValueError('Observed components cannot support K two-founder trees')
    pairs, extra = [], []
    for component in components:
        sub = observed.subgraph(component)
        matching = sorted(tuple(sorted(e)) for e in nx.max_weight_matching(sub, maxcardinality=True))
        pairs.append(matching[0])
        extra.extend(matching[1:])
    random.shuffle(extra)
    if len(pairs)+len(extra) < K:
        raise ValueError('Observed graph has no matching of K founder edges')
    pairs.extend(extra[:K-len(pairs)])
    random.shuffle(pairs)
    pi = [v for pair in pairs for v in pair]
    seen = set(pi)
    queue = deque(pi)
    edges = list(pairs)
    while queue:
        v = queue.popleft()
        neighbors = sorted(observed[v])
        random.shuffle(neighbors)
        for w in neighbors:
            if w not in seen:
                seen.add(w)
                queue.append(w)
                pi.append(w)
                edges.append((v, w))
    graf.es['tree'] = False
    graf.es[graf.get_eids(edges)]['tree'] = True
    _set_parents(graf, pi, K)
    return pi


def sample_parents(graf, pi, K, alpha, beta):
    """One Gibbs sweep preserving F|pi; founder edges remain fixed."""
    rank = {v: i for i, v in enumerate(pi)}
    degrees = np.array([len(a) for a in _adjacency(graf)], dtype=int)
    parents = list(graf.vs['pa'])
    for v in pi[2*K:]:
        degrees[parents[v]] -= 1
        candidates = [w for w in graf.neighbors(v) if rank[w] < rank[v]]
        weights = [beta*degrees[w]+alpha for w in candidates]
        if not candidates or min(weights) <= 0:
            raise ValueError('Invalid two-founder parent conditional')
        parents[v] = random.choices(candidates, weights)[0]
        degrees[parents[v]] += 1
    edges = [(pi[2*k], pi[2*k+1]) for k in range(K)]
    edges.extend((v, parents[v]) for v in pi[2*K:])
    graf.es['tree'] = False
    graf.es[graf.get_eids(edges)]['tree'] = True
    graf.vs['pa'] = parents


def gibbs_two_founder(graf, Burn=40, M=50, gap=1, alpha=0, beta=1,
                      K=1, display=True, size_thresh=.01, birth_thresh=.8, **options):
    """Sample the two-founder posterior; frequencies are empirical root counts.

    node_tree_coo is an empty compatibility placeholder, not an estimator.
    size_thresh and birth_thresh are legacy aggregation options, unused here.
    """
    n = graf.vcount()
    for name, value, minimum in [('K', K, 1), ('Burn', Burn, 0), ('M', M, 0), ('gap', gap, 1)]:
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if n < 2*K or graf.is_directed() or not graf.is_simple():
        raise ValueError('Require a simple undirected graph with n >= 2K')
    if not math.isfinite(alpha) or not math.isfinite(beta) or beta < 0 or beta+alpha <= 0:
        raise ValueError('Require finite beta >= 0 and beta+alpha > 0')
    if 'pi' in options:
        pi = list(options['pi'])
        if sorted(pi) != list(range(n)):
            raise ValueError('pi must be a permutation of all vertices')
        _set_parents(graf, pi, K)
    else:
        pi = _initialize(graf, K)
    freq = {k: np.zeros(n) for k in range(K)}
    for k, values in options.get('freq', {}).items():
        freq[k] = np.array(values, dtype=float, copy=True)
    samples, trace = [], []
    for iteration in range(Burn+M):
        sample_parents(graf, pi, K, alpha, beta)
        pi = sample_ordering(graf, K)
        _set_parents(graf, pi, K)
        if iteration < Burn or (iteration-Burn) % gap:
            continue
        roots = pi[:2*K:2]
        adj = _adjacency(graf)
        assignments = np.empty(n, dtype=int)
        sizes = []
        for k, root in enumerate(roots):
            _, vertices, _ = _orient(adj, root)
            assignments[vertices] = k
            sizes.append(len(vertices))
            freq[k][root] += 1
        trace.append(int(options.get('trace_id', 0) in roots))
        if options.get('store_samples', True):
            samples.append(dict(roots=roots.copy(), assignments=assignments,
                                tree_sizes=sizes, seed_edges=[tuple(pi[2*k:2*k+2]) for k in range(K)]))
    allfreq = sum(freq.values(), np.zeros(n))
    if allfreq.sum():
        allfreq = allfreq/allfreq.sum()
    result = dict(allfreq=allfreq, freq=freq, node_tree_coo=np.zeros((n, 0)),
                  tree2root=pi[:2*K:2], pi=pi, trace=trace, seed_mode='two_node',
                  seed_edges=[tuple(pi[2*k:2*k+2]) for k in range(K)])
    if options.get('store_samples', True):
        result['samples'] = samples
    return result
