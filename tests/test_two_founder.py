"""Exact finite-state checks for the paired-founder Gibbs kernel."""
import importlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools
import random
import unittest
from collections import defaultdict
from unittest.mock import patch

import igraph as ig
import networkx as nx
import numpy as np

MODULES = ['mcmc.two_founder']


def forest_graph(n, edges):
    graph = ig.Graph.Full(n)
    graph.es['tree'] = False
    graph.es[graph.get_eids(edges)]['tree'] = True
    return graph


def forward_weight(n, edges, alpha, beta=1):
    """Independent generative likelihood for pi=range(n), K=2."""
    degree = [1]*4+[0]*(n-4)
    weight = 1.0
    for v in range(4, n):
        p = next(u if w == v else w for u, w in edges if max(u, w) == v)
        weight *= (beta*degree[p]+alpha)/(beta*sum(degree)+alpha*v)
        degree[p] += 1
        degree[v] += 1
    return weight


class Branch(Exception):
    def __init__(self, outcomes):
        self.outcomes = outcomes


def exact_order_distribution(module, graph, K):
    result = defaultdict(float)
    pending = [([], 1.0)]
    while pending:
        decisions, probability = pending.pop()
        index = 0

        def choose(outcomes):
            nonlocal index
            if index == len(decisions):
                raise Branch(outcomes)
            value = decisions[index]
            index += 1
            return value

        def choices(population, weights=None):
            population = list(population)
            weights = [1]*len(population) if weights is None else list(weights)
            total = sum(weights)
            return [choose([(v, w/total) for v, w in zip(population, weights) if w])]

        def shuffle(values):
            permutations = list(itertools.permutations(values))
            values[:] = choose([(p, 1/len(permutations)) for p in permutations])

        with patch.object(module.random, 'choices', choices), patch.object(module.random, 'shuffle', shuffle):
            try:
                pi = module.sample_ordering(graph, K)
            except Branch as branch:
                pending.extend((decisions+[v], probability*p) for v, p in branch.outcomes)
            else:
                result[tuple(pi)] += probability
    return result


class TwoFounderTests(unittest.TestCase):
    def test_exact_order_conditionals(self):
        fixtures = [
            (4, [(0, 1), (2, 3)]),
            (6, [(0, 1), (1, 2), (2, 3), (4, 5)]),
            (6, [(0, 1), (0, 2), (0, 3), (4, 5)]),
            (6, [(0, 1), (1, 2), (3, 4), (4, 5)]),
        ]
        for name in MODULES:
            module = importlib.import_module(name)
            for n, edges in fixtures:
                graph = forest_graph(n, edges)
                forest = nx.Graph()
                forest.add_nodes_from(range(n))
                forest.add_edges_from(edges)
                component = {v: k for k, nodes in enumerate(nx.connected_components(forest)) for v in nodes}
                compatible = set()
                for pi in itertools.permutations(range(n)):
                    if not all(forest.has_edge(pi[2*k], pi[2*k+1]) for k in range(2)):
                        continue
                    if component[pi[0]] == component[pi[2]]:
                        continue
                    rank = {v: i for i, v in enumerate(pi)}
                    if any(sum(rank[w] < rank[v] for w in forest[v]) != 1 for v in pi[4:]):
                        continue
                    compatible.add(pi)
                distribution = exact_order_distribution(module, graph, 2)
                self.assertEqual(set(distribution), compatible)
                for probability in distribution.values():
                    self.assertAlmostEqual(probability, 1/len(compatible), places=13)

    def test_exact_parent_conditionals(self):
        n, pi = 6, list(range(6))
        for name in MODULES:
            module = importlib.import_module(name)
            for alpha in [-.5, 0, 2]:
                for p4, p5 in itertools.product(range(4), range(5)):
                    edges = [(0, 1), (2, 3), (4, p4), (5, p5)]
                    graph = forest_graph(n, edges)
                    module._set_parents(graph, pi, 2)
                    site = 4

                    def verify(population, weights):
                        nonlocal site
                        candidates = list(population)
                        likelihoods = []
                        for p in candidates:
                            trial = [(0, 1), (2, 3), (4, p if site == 4 else p4),
                                     (5, p if site == 5 else p5)]
                            likelihoods.append(forward_weight(n, trial, alpha))
                        np.testing.assert_allclose(np.array(weights)/sum(weights),
                                                   np.array(likelihoods)/sum(likelihoods),
                                                   atol=1e-14, rtol=1e-14)
                        old = p4 if site == 4 else p5
                        site += 1
                        return [old]

                    with patch.object(module.random, 'choices', verify):
                        module.sample_parents(graph, pi, 2, alpha, 1)

    def test_chain_and_continuation(self):
        for name in ['mcmc.gibbsSampling']:
            sampler = importlib.import_module(name).gibbsFullCT
            random.seed(15)
            graph = ig.Graph.Full(8)
            result = sampler(graph, Burn=10, M=21, gap=2, K=3, alpha=-.5, display=False)
            self.assertEqual(len(result['samples']), 11)
            self.assertEqual(sum(graph.es['tree']), 5)
            for sample in result['samples']:
                self.assertEqual(sum(sample['tree_sizes']), 8)
                self.assertGreaterEqual(min(sample['tree_sizes']), 2)
                for k, (a, b) in enumerate(sample['seed_edges']):
                    self.assertEqual(sample['roots'][k], a)
                    self.assertEqual(sample['assignments'][a], k)
                    self.assertEqual(sample['assignments'][b], k)
            continued = sampler(graph, Burn=0, M=3, K=3, alpha=-.5,
                                pi=result['pi'], freq=result['freq'], display=False)
            self.assertEqual(sum(map(np.sum, continued['freq'].values())), 42)
            empty = sampler(graph, Burn=0, M=0, K=3, pi=continued['pi'], display=False)
            self.assertEqual(empty['samples'], [])
            hidden = sampler(graph, Burn=0, M=2, K=3, store_samples=False, display=False)
            self.assertNotIn('samples', hidden)
            self.assertEqual(sum(map(np.sum, hidden['freq'].values())), 6)

    def test_invalid_support(self):
        module = importlib.import_module('mcmc.two_founder')
        for graph, options in [
            (ig.Graph.Star(4), dict(K=2)),
            (ig.Graph.Full(3), dict(K=2)),
            (ig.Graph(n=4, edges=[(0, 1), (1, 2)]), dict(K=1)),
            (ig.Graph.Full(4), dict(K=2, alpha=-1)),
            (ig.Graph.Full(4), dict(K=2, gap=0)),
        ]:
            with self.assertRaises(ValueError):
                module.gibbs_two_founder(graph, Burn=0, M=1, **options)


if __name__ == '__main__':
    unittest.main()
