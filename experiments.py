"""Reproducible JMLR simulations: balanced K=2 and unbalanced K=3."""
import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
from importlib.metadata import version
import json
from multiprocessing import Pool
import os
from pathlib import Path
import random
import sys
import pandas as pd
from tqdm import tqdm
from baselines import score_clustering_python, gao_dcbm_clustering
from spar import Leaf_stripping, Leaf_stripping_postCT
from mcmc import gibbsFullCT_multichain_until_valid
from mcmc.gibbsSampling import InsufficientSamplesError, _sample_is_valid

import numpy as np
import networkx as nx
import igraph as ig

def generate_recursive_forest(n, K, alpha, ratio=None):
    """Generate K recursive trees (Preferential Attachment) without noise."""
    if ratio is None:
        ratio = [1 / K] * K
    assert len(ratio) == K and np.isclose(sum(ratio), 1.0), 'ratio must sum to 1 and have length K'
    sizes = [int(np.floor(r * n)) for r in ratio]
    while sum(sizes) < n:
        sizes[np.argmin(sizes)] += 1
    graf = ig.Graph(n, directed=False)
    true_components = []
    true_assignments = np.zeros(n, dtype=int)
    all_tree_edges = []
    node_offset = 0
    for k in range(K):
        size = sizes[k]
        if size == 0:
            continue
        local_degrees = np.zeros(size, dtype=float)
        component_edges = []
        for t in range(1, size):
            weights = local_degrees[:t] + alpha
            weights_sum = weights.sum()
            if weights_sum > 0:
                probs = weights / weights_sum
            else:
                probs = np.ones(t) / t
            parent_local = np.random.choice(t, p=probs)
            local_degrees[t] += 1
            local_degrees[parent_local] += 1
            source_global = t + node_offset
            target_global = parent_local + node_offset
            component_edges.append((source_global, target_global))
        all_tree_edges.extend(component_edges)
        component = list(range(node_offset, node_offset + size))
        true_components.append(component)
        true_assignments[component] = k
        node_offset += size
    if all_tree_edges:
        graf.add_edges(all_tree_edges)
        graf.es['tree'] = True
    graf.vs['pa'] = None
    graf.vs['subtree_size'] = 1
    graf.vs['name'] = [str(i) for i in range(n)]
    return (graf, true_components, true_assignments)

def add_noise_edges(graf, theta, true_components=None):
    """Add noise edges to an existing graph with probability theta."""
    n = graf.vcount()
    num_tree_edges = graf.ecount()
    i, j = np.triu_indices(n, k=1)
    rand_values = np.random.rand(len(i))
    add_mask = rand_values < theta
    i_add = i[add_mask]
    j_add = j[add_mask]
    noise_edges = list(zip(i_add, j_add))
    if noise_edges:
        graf.add_edges(noise_edges)
        if num_tree_edges > 0 and 'tree' not in graf.es.attributes():
            graf.es['tree'] = False
            graf.es[:num_tree_edges]['tree'] = True
        graf.es[num_tree_edges:]['tree'] = False
    pass
    return graf
ARRIVAL_WEIGHT_SCHEMES = {'weighted_overall': lambda t: 1.0 / t, 'weighted_overall_t1_log': lambda t: 1.0 / (np.maximum(t, 2) * np.log(np.maximum(t, 2))), 'weighted_overall_t3_2': lambda t: 1.0 / t ** 1.5, 'weighted_overall_r2': lambda t: 1.0 / t ** 2}
DEGREE_WEIGHT_SCHEMES = {'weighted_overall_deg2': lambda d: d ** 2, 'weighted_overall_deg2_log': lambda d: d ** 2 * (np.log(np.maximum(d, 1)) + 1.0), 'weighted_overall_deg3': lambda d: d ** 3, 'weighted_overall_deg4': lambda d: d ** 4}

def compute_misclassification_metrics(new_labels, n_per_component, total_nodes, node_depth, L_list, initial_degrees=None):
    """Compute misclassification metrics for different node subsets."""
    from itertools import permutations
    metrics = {}
    if np.isscalar(n_per_component):
        K = total_nodes // n_per_component
        component_sizes = [n_per_component] * K
    else:
        component_sizes = list(n_per_component)
        K = len(component_sizes)
    component_starts = np.cumsum([0] + component_sizes[:-1])
    unique_labels = [lab for lab in np.unique(new_labels) if lab != 0]
    true_assignment = np.concatenate([np.full(size, k) for k, size in enumerate(component_sizes)])
    if len(unique_labels) < K:
        for L in L_list:
            metrics[f'first_{L}'] = np.nan
            metrics[f'highdeg_{L}'] = np.nan
        metrics['overall_classified'] = np.nan
        for name in ARRIVAL_WEIGHT_SCHEMES:
            metrics[name] = np.nan
        for name in DEGREE_WEIGHT_SCHEMES:
            metrics[name] = np.nan
        for d in range(3):
            if d == 0:
                metrics['root'] = np.nan
            else:
                metrics[f'depth_{d}'] = np.nan
        return metrics

    def best_error_for_subset(subset_indices, weights=None):
        """Find the best permutation and min (weighted) error for a subset of node indices."""
        if len(subset_indices) == 0:
            return np.nan
        subset_indices = np.array(subset_indices)
        subset_true = true_assignment[subset_indices]
        subset_pred = new_labels[subset_indices]
        w = np.ones(len(subset_indices)) if weights is None else np.asarray(weights)
        best_err = float('inf')
        for perm in permutations(range(K)):
            expected = np.array([unique_labels[perm[t]] for t in subset_true])
            err = np.sum(w * (subset_pred != expected)) / np.sum(w)
            if err < best_err:
                best_err = err
        return best_err
    for L in L_list:
        first_L_indices = []
        for k in range(K):
            component_start = component_starts[k]
            component_end = component_start + component_sizes[k]
            component_nodes = list(range(component_start, component_end))
            first_L_indices.extend(component_nodes[:min(L, len(component_nodes))])
        metrics[f'first_{L}'] = best_error_for_subset(first_L_indices)
    if initial_degrees is not None:
        for L in L_list:
            all_nodes = list(range(total_nodes))
            highdeg_L_indices = sorted(all_nodes, key=lambda i: initial_degrees.get(i, 0), reverse=True)[:L]
            metrics[f'highdeg_{L}'] = best_error_for_subset(highdeg_L_indices)
    classified = np.where(new_labels != 0)[0]
    classified = classified[classified < total_nodes]
    metrics['overall_classified'] = best_error_for_subset(classified)
    arrival_index = np.empty(total_nodes)
    for k in range(K):
        start = component_starts[k]
        size = component_sizes[k]
        arrival_index[start:start + size] = np.arange(size)
    arrival_time = arrival_index[classified] + 1.0
    for name, weight_fn in ARRIVAL_WEIGHT_SCHEMES.items():
        metrics[name] = best_error_for_subset(classified, weights=weight_fn(arrival_time))
    if initial_degrees is not None:
        degree_vals = np.array([initial_degrees.get(i, 0) for i in classified], dtype=float)
        for name, weight_fn in DEGREE_WEIGHT_SCHEMES.items():
            metrics[name] = best_error_for_subset(classified, weights=weight_fn(degree_vals))
    else:
        for name in DEGREE_WEIGHT_SCHEMES:
            metrics[name] = np.nan
    max_depth = max(node_depth.values()) if node_depth else 0
    for d in range(min(max_depth + 1, 3)):
        depth_nodes = [i for i in range(total_nodes) if node_depth.get(i, -1) == d]
        err = best_error_for_subset(depth_nodes)
        if d == 0:
            metrics['root'] = err
        else:
            metrics[f'depth_{d}'] = err
    return metrics

def extract_subgraph_and_compute_depth(forest, n_per_component, K):
    """Extract subgraph with first n_per_component nodes from each component."""
    selected_nodes = []
    original_components = []
    n_full_per_component = forest.vcount() // K
    for k in range(K):
        start_idx = k * n_full_per_component
        component_nodes = list(range(start_idx, start_idx + n_per_component))
        selected_nodes.extend(component_nodes)
        original_components.append(component_nodes)
    subforest = forest.induced_subgraph(selected_nodes)
    sub_components = []
    for k in range(K):
        sub_components.append(list(range(k * n_per_component, (k + 1) * n_per_component)))
    if 'tree' in subforest.es.attributes():
        tree_edges = [(e.source, e.target) for e in subforest.es if e['tree']]
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
    return (subforest, sub_components, node_depth)

def extract_subgraph_and_compute_depth_unbalanced(forest, n_component_sizes, K):
    """Extract arrival prefixes with specified community sizes."""
    selected_nodes = []
    original_components = []
    n_full_per_component = forest.vcount() // K
    for k in range(K):
        start_idx = k * n_full_per_component
        n_to_extract = n_component_sizes[k]
        component_nodes = list(range(start_idx, start_idx + n_to_extract))
        selected_nodes.extend(component_nodes)
        original_components.append(component_nodes)
    subforest = forest.induced_subgraph(selected_nodes)
    sub_components = []
    cumulative = 0
    for k in range(K):
        sub_components.append(list(range(cumulative, cumulative + n_component_sizes[k])))
        cumulative += n_component_sizes[k]
    if 'tree' in subforest.es.attributes():
        tree_edges = [(e.source, e.target) for e in subforest.es if e['tree']]
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
    return (subforest, sub_components, node_depth)

def build_noisy_graph(forest, components, n_per_component, theta, K):
    """Add independent pairwise noise and record binary adjacency and tree depths."""
    total_nodes = n_per_component * K if np.isscalar(n_per_component) else sum(n_per_component)
    graf_with_noise = forest.copy()
    add_noise_edges(graf_with_noise, theta)
    if 'tree' in graf_with_noise.es.attributes():
        tree_edges = [(e.source, e.target) for e in graf_with_noise.es if e['tree']]
    else:
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
    A_noisy = np.array(graf_with_noise.get_adjacency().data)
    A_noisy[A_noisy > 1] = 1
    initial_degrees = {i: int(A_noisy[i].sum()) for i in range(total_nodes)}
    return (total_nodes, {'A_noisy': A_noisy, 'node_depth': node_depth, 'initial_degrees': initial_degrees, 'graf_with_noise': graf_with_noise})


SIZES = {'balanced': (1000, 2000, 3000, 4000, 5000), 'unbalanced': (1200, 2400, 3600, 4800, 6000)}
RATES = {"rate_1": 1.0, "rate_3_4": 0.75, "rate_1_2": 0.5}
PHASES = {"distance": ("with_protected",), "benchmarks": ("SCORE", "Gao-DCBM"),
          "montecarlo": ("MonteCarlo",)}
METRICS = ["first_10", "first_50", "highdeg_10", "highdeg_50", "root", "depth_1",
           "depth_2", "overall_classified", *ARRIVAL_WEIGHT_SCHEMES, *DEGREE_WEIGHT_SCHEMES]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "raw"


@dataclass(frozen=True)
class Config:
    scenario: str = 'balanced'
    alpha: float = 0
    n_list: tuple = ()
    n_ref: int = 0
    base_n: int = 0
    implementation: str = 'paper'
    allow_distance_fallback: bool = True
    theta_ref: float = 0.01
    repeats: int = 200
    num_cores: int = 25
    seed_base: int = 42
    mc_burn: int = 30
    mc_target_valid: int = 200
    mc_min_valid: int = 50
    mc_samples_per_chain: int = 200
    mc_max_chains: int = 5

    def __post_init__(self):
        if self.scenario not in SIZES:
            raise ValueError('unknown scenario')
        if not self.n_list:
            object.__setattr__(self, 'n_list', SIZES[self.scenario])
        if not self.n_ref:
            object.__setattr__(self, 'n_ref', SIZES[self.scenario][0])
        if not self.base_n:
            object.__setattr__(self, 'base_n', SIZES[self.scenario][-1])

    @property
    def K(self):
        return 2 if self.scenario == 'balanced' else 3

    def validate(self):
        divisor = 2 if self.K == 2 else 6
        if self.base_n < max(self.n_list) or self.base_n % divisor:
            raise ValueError('base_n must cover all sizes and respect the component ratio')
        if self.implementation != 'paper':
            raise ValueError('unknown implementation')
        if self.alpha not in (-0.5, 0, 2):
            raise ValueError("alpha must be -0.5, 0, or 2")
        if not self.n_list or tuple(sorted(set(self.n_list))) != self.n_list:
            raise ValueError("n_list must be nonempty, increasing, and unique")
        if any(n < 12 or n % divisor for n in self.n_list):
            raise ValueError("sizes must be at least 12 and divisible by the component ratio denominator")
        if min(self.n_ref, self.repeats, self.num_cores, self.mc_min_valid,
               self.mc_target_valid, self.mc_samples_per_chain, self.mc_max_chains) <= 0:
            raise ValueError("reference size, counts, and budgets must be positive")
        if self.mc_burn < 0 or self.mc_min_valid > self.mc_target_valid:
            raise ValueError("require burn >= 0 and min_valid <= target_valid")
        if not 0 <= self.seed_base <= 2**32 - self.repeats:
            raise ValueError("trial seeds must fit NumPy's 32-bit seed range")
        if not 0 < self.theta_ref <= 1 or any(
            self.theta_ref * (self.n_ref / n)**rate > 1 for n in self.n_list for rate in RATES.values()
        ):
            raise ValueError("all noise probabilities must lie in (0,1]")


def component_sizes(n, K=3):
    return [n // 2, n // 2] if K == 2 else [n // 2, n // 3, n // 6]


def raw_filename(alpha, phase, scenario='balanced'):
    return f'{scenario}_alpha{alpha:g}_{phase}.csv'


def distance_labels(info, sizes, cfg):
    K = cfg.K
    # Rank 5K, Q=floor(n/10), and tau-prime=min(D*,tau/2).
    result = Leaf_stripping(
        A_input=info["A_noisy"], K=K, b=5*K, d=0, select_by="initial",
        min_component_size=sum(sizes) // 10, recover=True,
        recover_mode="distance_threshold", threshold_by="order", mode="True",
        check_protected_removal=True, recovery_threshold_divisor=2.0, implementation=cfg.implementation,
    )
    count, labels = result[:2]
    anchors = ([np.flatnonzero(result[2] == k).tolist() for k in range(1, K + 1)]
               if cfg.implementation == 'paper' else result[4])
    if count != K:
        raise RuntimeError(f"SPAR found {count} communities, expected {K}")
    return np.asarray(labels), anchors


def run_method(strategy, info, sizes, cfg, trial_id):
    K = cfg.K
    diagnostics = {"result_source": strategy, "mc_valid_samples": None,
                   "mc_total_samples": None, "fallback_reason": ""}
    A = info["A_noisy"]
    if strategy == "SCORE":
        labels = score_clustering_python(A, K, random_state=trial_id) + 1
    elif strategy == "Gao-DCBM":
        labels = gao_dcbm_clustering(A, K, tau_factor=5, n_refinements=1,
                                     random_state=trial_id, verbose=False) + 1
    else:
        labels, references = distance_labels(info, sizes, cfg)
        if strategy == "MonteCarlo":
            diagnostics["result_source"] = "distance_fallback"
            try:
                if references is None or len(references) != K or not all(references):
                    raise RuntimeError("SPAR did not provide K nonempty anchor sets")
                anchors = [set(ref) for ref in references]
                # The observed graph is binary for every method. Noise can
                # duplicate a tree edge in igraph; remove these copies before
                # Gibbs tree traversals, which assume distinct neighbors.
                sampler_graph = info["graf_with_noise"].copy()
                sampler_graph.simplify(multiple=True, loops=True)
                if not sampler_graph.is_connected():
                    raise RuntimeError("Gibbs spanning-tree initialization requires a connected graph")
                samples = gibbsFullCT_multichain_until_valid(
                    sampler_graph, anchors=anchors, K=K,
                    alpha=cfg.alpha, beta=1, target_valid=cfg.mc_target_valid,
                    burn=cfg.mc_burn, samples_per_chain=cfg.mc_samples_per_chain,
                    max_chains=cfg.mc_max_chains, min_valid=cfg.mc_min_valid,
                    size_thresh=0.1, birth_thresh=0.8,
                )
                generated_samples = len(samples['samples'])
                if cfg.implementation == 'paper':
                    lookup = {v: k for k, anchor in enumerate(anchors) for v in anchor}
                    valid = [s for s in samples['samples'] if _sample_is_valid(s, lookup, K)]
                    samples['samples'] = valid[:cfg.mc_target_valid]
                post = Leaf_stripping_postCT(samples, anchors)
                diagnostics.update(mc_valid_samples=post["valid_samples"],
                                   mc_total_samples=generated_samples)
                if post["valid_samples"] < cfg.mc_min_valid:
                    raise RuntimeError("too few anchor-valid posterior samples")
                probabilities = post["assignment_probs"]
                if not np.all(np.isfinite(probabilities)) or np.any(probabilities.sum(axis=1) <= 0):
                    raise RuntimeError("posterior assignment probabilities missing for some nodes")
                mc_labels = (np.asarray([random.choice(np.flatnonzero(p == p.max()).tolist()) + 1
                                         for p in probabilities]) if cfg.implementation == 'paper'
                             else np.argmax(probabilities, axis=1) + 1)
                if len(np.unique(mc_labels)) != K:
                    raise RuntimeError("posterior classification has fewer than K communities")
                labels = mc_labels
                diagnostics["result_source"] = "MonteCarlo"
            except InsufficientSamplesError as exc:
                if not cfg.allow_distance_fallback:
                    raise RuntimeError('Monte Carlo failed: ' + str(exc)) from exc
                diagnostics["fallback_reason"] = str(exc)
                diagnostics.update(mc_valid_samples=exc.valid, mc_total_samples=exc.total)
            except RuntimeError as exc:
                if (not cfg.allow_distance_fallback or
                        str(exc) != "posterior classification has fewer than K communities"):
                    raise
                diagnostics["fallback_reason"] = str(exc)
    metrics = compute_misclassification_metrics(labels, sizes, len(A), info["node_depth"],
                                               [10, 50], initial_degrees=info["initial_degrees"])
    if not np.isfinite(metrics["overall_classified"]):
        raise RuntimeError("no finite overall misclassification result")
    diagnostics.update(cluster_count=len(np.unique(labels[labels != 0])),
                       classified_fraction=float(np.mean(labels != 0)))
    return metrics, diagnostics


def run_trial(trial_id, strategy, cfg):
    """One seed across the complete size/rate grid; all phases share graphs."""
    K = cfg.K
    seed = cfg.seed_base + trial_id
    np.random.seed(seed)
    random.seed(seed)
    rows = []
    # Suppress verbose legacy per-node/Gibbs prints, without buffering them in RAM.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        forest, _, _ = generate_recursive_forest(cfg.base_n // 2 * K, K, cfg.alpha)
        for n in cfg.n_list:
            sizes = component_sizes(n, K)
            subforest, components, _ = extract_subgraph_and_compute_depth_unbalanced(forest, sizes, K)
            for rate_name, exponent in RATES.items():
                theta = cfg.theta_ref * (cfg.n_ref / n)**exponent
                _, info = build_noisy_graph(subforest, components, sizes, theta, K)
                row = dict(strategy=strategy, alpha=cfg.alpha, n=n, theta=theta,
                           theta_rate=rate_name, trial=trial_id + 1, trial_seed=seed,
                           K=K, scenario=cfg.scenario, implementation=cfg.implementation, component_ratio="1:1" if K == 2 else "3:2:1", tree_1=sizes[0], tree_2=sizes[1], tree_3=sizes[2] if K == 3 else 0,
                           graph_hash=hashlib.sha256(info["A_noisy"].astype(np.uint8).tobytes()).hexdigest(),
                           status="failed", cluster_count=0, classified_fraction=None,
                           result_source="failed", mc_valid_samples=None, mc_total_samples=None,
                           fallback_reason="", error="", **{metric: np.nan for metric in METRICS})
                numpy_state, python_state = np.random.get_state(), random.getstate()
                try:
                    metrics, diagnostics = run_method(strategy, info, sizes, cfg, trial_id)
                    row.update(metrics, **diagnostics, status="ok")
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    # Sampling/inference must not change any later generated graph.
                    np.random.set_state(numpy_state)
                    random.setstate(python_state)
                rows.append(row)
    return rows


def summarize(df):
    """Long-form mean/std/count tables; show failures and MC fallback explicitly."""
    rows = []
    for key, cell in df.groupby(["scenario", "strategy", "alpha", "n", "theta_rate"], sort=True):
        for metric in METRICS:
            values = cell.loc[cell.status == "ok", metric].dropna()
            rows.append(dict(zip(("scenario", "strategy", "alpha", "n", "theta_rate"), key),
                theta=cell.theta.iloc[0], metric=metric, mean=values.mean(), std=values.std(),
                valid_trials=len(values), total_trials=len(cell),
                success_rate=float((cell.status == "ok").mean()),
                fallback_trials=int((cell.result_source == "distance_fallback").sum()),
                mean_classified_fraction=cell.classified_fraction.mean()))
    return pd.DataFrame(rows)


def _source_hashes():
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ('spar.py', 'experiments.py', 'baselines.py')]
    paths += sorted((root / 'mcmc').glob('*.py'))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _runtime_versions():
    return dict(python=sys.version.split()[0], **{p: version(p) for p in
        ('numpy', 'pandas', 'scipy', 'networkx', 'igraph')})


def run_experiment(cfg, phase, output_dir=DEFAULT_OUTPUT, resume=False):
    """Write trial records incrementally; resume only matching settings and code."""
    cfg.validate()
    if phase not in PHASES:
        raise ValueError('unknown simulation phase')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / raw_filename(cfg.alpha, phase, cfg.scenario)
    partial_path = destination.with_suffix('.partial.csv')
    config_path = destination.with_suffix('.config.json')
    settings = json.loads(json.dumps(dict(config=asdict(cfg), phase=phase, schema_version=2,
        paper_revision='fd5198d', source_sha256=_source_hashes(), runtime_versions=_runtime_versions(),
        metric_convention='per-subset optimal permutation; weighted errors normalized',
        failure_convention='missing errors for failures; MC low-valid-sample distance fallback explicitly counted',
        mc_target_status='two-founder RC-PF conditional updates; finite-chain convergence not certified')))
    exists = destination.exists() or partial_path.exists() or config_path.exists()
    if exists and not resume:
        raise FileExistsError(f'{destination.name}: use --resume or a new output directory')
    if exists and (not config_path.exists() or json.loads(config_path.read_text()) != settings):
        raise ValueError('cannot resume: settings, dependencies, or source hashes differ')
    if not exists:
        config_path.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    if destination.exists():
        return destination
    completed = set()
    if partial_path.exists() and partial_path.stat().st_size:
        previous = pd.read_csv(partial_path)
        expected = {(n, rate) for n in cfg.n_list for rate in RATES}
        for (strategy, trial), group in previous.groupby(['strategy', 'trial']):
            if len(group) == len(expected) and set(zip(group.n, group.theta_rate)) == expected:
                completed.add((strategy, int(trial)))
        # A stopped write may leave an incomplete trial; rerun that entire trial.
        previous = previous[[ (s, int(t)) in completed for s, t in zip(previous.strategy, previous.trial) ]]
        previous.to_csv(partial_path, index=False)
    has_header = partial_path.exists() and partial_path.stat().st_size > 0
    with partial_path.open('a', newline='', encoding='utf-8') as handle:
        writer = None
        for strategy in PHASES[phase]:
            pending = [i for i in range(cfg.repeats) if (strategy, i + 1) not in completed]
            worker = partial(run_trial, strategy=strategy, cfg=cfg)
            with contextlib.ExitStack() as stack:
                results = map(worker, pending) if cfg.num_cores == 1 else stack.enter_context(Pool(cfg.num_cores)).imap_unordered(worker, pending)
                for rows in tqdm(results, total=len(pending), desc=strategy, unit='trial'):
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        if not has_header:
                            writer.writeheader()
                            has_header = True
                    writer.writerows(rows)
                    handle.flush()
    df = pd.read_csv(partial_path).sort_values(['strategy', 'n', 'theta_rate', 'trial'])
    expected_rows = len(PHASES[phase]) * cfg.repeats * len(cfg.n_list) * len(RATES)
    if len(df) != expected_rows or df.duplicated(['strategy', 'n', 'theta_rate', 'trial']).any():
        raise RuntimeError('incomplete or duplicated simulation grid')
    temporary = destination.with_suffix('.tmp.csv')
    df.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    partial_path.unlink()
    summarize(df).to_csv(destination.with_name(destination.stem + '_means.csv'), index=False)
    print(f'{destination}: {len(df)} rows; failed={(df.status != "ok").sum()}; fallback={(df.result_source == "distance_fallback").sum()}')
    return destination


def _calibration_trial(trial_id, cfg):
    """Weakest community hub rank, counting all ties at the threshold."""
    np.random.seed(cfg.seed_base + trial_id)
    random.seed(cfg.seed_base + trial_id)
    ratios = [.2, .8] if cfg.K == 2 else [.2, .2, .6]
    rows = []
    for n in cfg.n_list:
        forest, components, _ = generate_recursive_forest(n, cfg.K, cfg.alpha, ratio=ratios)
        for rate, exponent in RATES.items():
            theta = cfg.theta_ref * (cfg.n_ref / n)**exponent
            _, info = build_noisy_graph(forest, components, [len(c) for c in components], theta, cfg.K)
            degree = info['A_noisy'].sum(axis=1)
            weakest = min(degree[c].max() for c in components)
            rows.append(dict(scenario=cfg.scenario, K=cfg.K, alpha=cfg.alpha, n=n,
                theta_rate=rate, theta=theta, trial=trial_id + 1,
                rank=int(1 + np.count_nonzero(degree > weakest)), weakest_hub_degree=int(weakest)))
    return rows


def run_calibration(cfg, output_dir):
    cfg.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{cfg.scenario}_alpha{cfg.alpha:g}_calibration.csv'
    if path.exists():
        raise FileExistsError(path)
    rows = []
    worker = partial(_calibration_trial, cfg=cfg)
    with contextlib.ExitStack() as stack:
        results = map(worker, range(cfg.repeats)) if cfg.num_cores == 1 else stack.enter_context(Pool(cfg.num_cores)).imap_unordered(worker, range(cfg.repeats))
        for result in tqdm(results, total=cfg.repeats, desc='rank calibration'):
            rows.extend(result)
    df = pd.DataFrame(rows).sort_values(['n', 'theta_rate', 'trial'])
    df.to_csv(path, index=False)
    df.groupby(['n', 'theta_rate'])['rank'].quantile(.95, interpolation='higher').rename('rank_95').to_csv(path.with_name(path.stem + '_summary.csv'))
    path.with_suffix('.config.json').write_text(json.dumps(dict(config=asdict(cfg), H=5,
        rank='1 + number of degrees strictly greater than weakest community maximum',
        source_sha256=_source_hashes(), runtime_versions=_runtime_versions()), indent=2) + '\n', encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=tuple(SIZES), required=True)
    parser.add_argument('--alpha', type=float, choices=(-.5, 0, 2), required=True)
    parser.add_argument('--phase', choices=(*PHASES, 'calibration'), required=True)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--n-list', type=int, nargs='+', default=())
    parser.add_argument('--implementation', choices=('paper',), default='paper')
    parser.add_argument('--no-distance-fallback', dest='allow_distance_fallback', action='store_false')
    parser.add_argument('--resume', action='store_true')
    for name in ('n_ref', 'base_n', 'repeats', 'num_cores', 'seed_base', 'mc_burn',
                 'mc_target_valid', 'mc_min_valid', 'mc_samples_per_chain', 'mc_max_chains'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(Config(), name) if name not in ('n_ref', 'base_n') else 0)
    parser.add_argument('--theta-ref', type=float, default=.01)
    args = vars(parser.parse_args())
    output, phase, resume = args.pop('output_dir'), args.pop('phase'), args.pop('resume')
    args['n_list'] = tuple(args['n_list'])
    cfg = Config(**args)
    if phase == 'calibration':
        run_calibration(cfg, output)
    else:
        run_experiment(cfg, phase, output, resume=resume)


if __name__ == '__main__':
    main()


