"""Submission checks: fixed-graph regression and hand-checkable invariants."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import spar
import experiments
import baselines
import plots
from mcmc import gibbsFullCT

REFERENCE = ROOT / "tests" / "fixtures" / "reference"


def quiet_call(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((REFERENCE / "manifest.json").read_text())["cases"]


    def test_metrics_match_saved_references(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                data = np.load(REFERENCE / f"{case['name']}.npz", allow_pickle=True)
                metrics = experiments.compute_misclassification_metrics(
                    data["labels"], case["sizes"], len(data["labels"]),
                    dict(enumerate(data["depth"])), [10, 50],
                    initial_degrees=dict(enumerate(data["adjacency"].sum(axis=1))))
                for metric, expected in case["metrics"].items():
                    if expected is None:
                        self.assertTrue(np.isnan(metrics[metric]))
                    else:
                        self.assertAlmostEqual(metrics[metric], expected, places=14)

    def test_baselines_match_saved_references(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                data = np.load(REFERENCE / f"{case['name']}.npz")
                for fn, key in ((baselines.score_clustering_python, "score"),
                                (baselines.gao_dcbm_clustering, "gao")):
                    kwargs = {"verbose": False} if key == "gao" else {}
                    out = quiet_call(fn, data["adjacency"], len(case["sizes"]),
                                     random_state=17, **kwargs)
                    # Eigenvector signs may relabel clusters across BLAS builds.
                    np.testing.assert_array_equal(out[:,None] == out[None,:],
                                                  data[key][:,None] == data[key][None,:])

    def test_graph_generation_matches_saved_edges_and_noise(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                data = np.load(REFERENCE / f"{case['name']}.npz")
                sizes = case["sizes"]
                k = len(sizes)
                np.random.seed(case["seed"])
                random.seed(case["seed"])
                forest, _, _ = experiments.generate_recursive_forest(max(sizes)*k, k, case["alpha"])
                forest, components, _ = experiments.extract_subgraph_and_compute_depth_unbalanced(forest, sizes, k)
                np.testing.assert_array_equal(forest.get_edgelist(), data["forest_edges"])
                _, info = quiet_call(experiments.build_noisy_graph, forest, components, sizes, case["theta"], k)
                np.testing.assert_array_equal(info["A_noisy"], data["adjacency"])

    def test_permuted_perfect_labels_and_hand_weighted_error(self):
        sizes = [4, 4]
        depth = {0: 0, 1: 1, 2: 1, 3: 2, 4: 0, 5: 1, 6: 1, 7: 2}
        degrees = {i: i + 1 for i in range(8)}
        labels = np.array([2, 2, 2, 2, 1, 1, 1, 1])
        metrics = experiments.compute_misclassification_metrics(labels, sizes, 8, depth, [2, 4], degrees)
        self.assertTrue(all(value == 0 for value in metrics.values()))
        labels[3] = 1
        metrics = experiments.compute_misclassification_metrics(labels, sizes, 8, depth, [2, 4], degrees)
        self.assertEqual(metrics["overall_classified"], 1 / 8)
        self.assertEqual(metrics["first_2"], 0)
        self.assertEqual(metrics["depth_2"], 1 / 2)
        self.assertAlmostEqual(metrics["weighted_overall"], (1/4)/(2*sum(1/t for t in range(1, 5))))
        self.assertAlmostEqual(metrics["weighted_overall_deg2"], 4**2/sum(d*d for d in range(1, 9)))

    def test_missing_community_is_not_reported_as_zero_error(self):
        metrics = experiments.compute_misclassification_metrics(np.ones(8), [4, 4], 8,
                   {i: i % 3 for i in range(8)}, [2, 4], {i: 1 for i in range(8)})
        self.assertTrue(all(np.isnan(value) for value in metrics.values()))

    def test_paper_no_pruning_and_threshold_ties(self):
        # Two equal stars: the top-ranked degree is tied across both centers.
        # The paper's degree threshold must include both centers even at rank 1.
        A = np.zeros((8, 8))
        for root, leaves in ((0, [1, 2, 3]), (4, [5, 6, 7])):
            for leaf in leaves:
                A[root, leaf] = A[leaf, root] = 1
        out = quiet_call(spar.Leaf_stripping, A, K=2, min_component_size=4,
                         b=1, d=0, recover=True, recover_mode="distance_threshold",
                         check_protected_removal=True, recovery_threshold_divisor=2,
                         implementation="paper")
        self.assertEqual(out[0], 2)
        self.assertEqual(len(np.unique(out[1])), 2)
        self.assertEqual(len(set(out[1][:4])), 1)
        self.assertEqual(len(set(out[1][4:])), 1)
        self.assertNotEqual(out[1][0], out[1][4])

    def test_real_gibbs_samples_partition_all_vertices(self):
        for k in (1, 2, 3):
            for alpha in (-0.5, 0, 2):
                with self.subTest(k=k, alpha=alpha):
                    np.random.seed(42)
                    random.seed(42)
                    out = quiet_call(gibbsFullCT, ig.Graph.Full(24), Burn=2, M=4,
                                     K=k, alpha=alpha, display=False)
                    self.assertEqual(len(out["samples"]), 4)
                    for sample in out["samples"]:
                        self.assertEqual(len(set(sample["roots"])), k)
                        self.assertEqual(sum(sample["tree_sizes"]), 24)
                        self.assertEqual(np.unique(sample["assignments"]).tolist(), list(range(k)))
                        self.assertEqual(np.bincount(sample["assignments"]).tolist(), sample["tree_sizes"])

    def test_posterior_alignment_and_invalid_root_filter(self):
        out = spar.Leaf_stripping_postCT({"node_tree_coo": np.zeros((4, 4)), "samples": [
            {"roots": [2, 0], "assignments": np.array([1, 1, 0, 0])},
            {"roots": [0, 2], "assignments": np.array([0, 0, 1, 1])},
            {"roots": [0, 1], "assignments": np.array([0, 1, 1, 1])},
        ]}, [[0, 1], [2, 3]])
        self.assertEqual(out["valid_samples"], 2)
        np.testing.assert_array_equal(out["assignment_probs"], [[1, 0], [1, 0], [0, 1], [0, 1]])
        self.assertAlmostEqual(out["global_posterior"].sum(), 1)
        for posterior in out["component_posteriors"]:
            self.assertAlmostEqual(posterior.sum(), 1)

    def test_fractional_recovery_threshold_is_not_rounded(self):
        # Keep integer-degree leaves out when threshold is 1.5.
        A = np.zeros((8, 8))
        for root, leaves in ((0, [1, 2, 3]), (4, [5, 6, 7])):
            for leaf in leaves:
                A[root, leaf] = A[leaf, root] = 1
        _, references = spar.recover_with_filtered_distance_by_threshold(
            A, 2, np.array([1]*4 + [2]*4), 1.5)
        self.assertEqual(references, [[0], [4]])

    def test_invalid_graph_is_rejected(self):
        for A in (np.array([[0,1],[0,0]]), np.eye(2), np.array([[0,2],[2,0]])):
            with self.subTest(A=A.tolist()), self.assertRaises(ValueError):
                spar.spar(A, K=1, rank=1, Q=1)

    def test_toy_trace_matches_independent_paper_calculation(self):
        edges = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,6),(1,7),(2,8),(3,9),
                 (5,10),(5,11),(8,22),(12,13),(12,14),(12,15),(12,16),
                 (13,17),(13,18),(14,19),(14,20),(12,21),
                 (6,7),(4,11),(17,18),(19,20),(5,16),(8,19),(10,17),(9,15),(8,3)]
        A = np.zeros((23, 23))
        for u, v in edges:
            A[u,v] = A[v,u] = 1
        with patch.object(spar, "recover_with_filtered_distance_by_threshold",
                          wraps=spar.recover_with_filtered_distance_by_threshold) as recover:
            result = spar.spar(A, K=2, tau=5, Q=3)
        self.assertEqual(result.trace, [
            {"Q": 3, "degree": 2, "removed": [2,4,6,7,9,10,11,15,16,18,20]},
            {"Q": 3, "degree": 3, "removed": [3,14,19]}])
        self.assertEqual(result.D_star, 3)
        self.assertEqual(recover.call_args.args[-1], 2.5)
        self.assertEqual({frozenset(np.flatnonzero(result.anchor_labels == k)) for k in (1,2)},
                         {frozenset([0,1,5]), frozenset([12,13,17,21])})

    def test_pruning_matches_small_graph_definition(self):
        # Independent oracle: test every vertex and identify cycles by bridges.
        rng = np.random.default_rng(731)
        for case in range(24):
            upper = np.triu(rng.random((9,9)) < (.15 + .02*case), 1)
            A = (upper | upper.T).astype(int)
            degree = A.sum(axis=1)
            high = set(np.flatnonzero(degree >= 2))
            q, trace = 3, []
            while True:
                G = nx.from_numpy_array(A)
                while True:
                    qualifying = lambda g: [set(c) for c in nx.connected_components(g) if len(c) >= q and set(c) & high]
                    components = qualifying(G)
                    if len(components) >= 2:
                        break
                    bridges = {frozenset(e) for e in nx.bridges(G)}
                    candidates = {v for edge in G.edges if frozenset(edge) not in bridges for v in edge}
                    for v in G:
                        reduced = G.copy()
                        reduced.remove_node(v)
                        if len(qualifying(reduced)) > len(components):
                            candidates.add(v)
                    if not candidates:
                        break
                    minimum = min(degree[list(candidates)])
                    removed = sorted(v for v in candidates if degree[v] == minimum)
                    trace.append(dict(Q=q,degree=float(minimum),removed=removed))
                    G.remove_nodes_from(removed)
                if len(components) >= 2 or q == 1:
                    break
                q = max(1, int(.8*q))
            result = spar.spar(A, K=2, tau=2, Q=3, recover=False)
            with self.subTest(case=case):
                self.assertEqual(result.trace, trace)
                self.assertEqual(result.success, len(components) >= 2)
                if result.success:
                    components.sort(key=lambda c: (-max(degree[list(c)]),min(c)))
                    self.assertEqual([set(np.flatnonzero(result.anchor_labels==k)) for k in (1,2)], components[:2])

    def test_method_randomness_and_failures_do_not_change_graphs(self):
        cfg = experiments.Config(n_list=(60,120), base_n=120, n_ref=60, theta_ref=.05,
                                 repeats=2, num_cores=1)
        def method(strategy, info, sizes, cfg, trial_id):
            np.random.random(100)
            random.random()
            if strategy == "SCORE":
                raise RuntimeError("controlled failure")
            return {m: .25 for m in experiments.METRICS}, dict(cluster_count=cfg.K, classified_fraction=1.)
        with patch.object(experiments, "run_method", method):
            successful = experiments.run_trial(1, "with_protected", cfg)
            failed = experiments.run_trial(1, "SCORE", cfg)
        self.assertEqual([r["graph_hash"] for r in successful], [r["graph_hash"] for r in failed])
        self.assertTrue(all(r["status"] == "failed" and np.isnan(r["overall_classified"]) for r in failed))
        self.assertTrue(all("controlled failure" in r["error"] for r in failed))

    def test_monte_carlo_valid_sample_cap_and_narrow_fallback(self):
        cfg = experiments.Config(n_list=(12,), base_n=12, n_ref=12, num_cores=1,
                                 mc_target_valid=2, mc_min_valid=1)
        sizes = [6,6]
        labels = np.repeat([1,2], sizes)
        anchors = [list(range(6)),list(range(6,12))]
        info = dict(A_noisy=np.ones((12,12))-np.eye(12), graf_with_noise=ig.Graph.Full(12),
                    node_depth={i:i%3 for i in range(12)},initial_degrees={i:11 for i in range(12)})
        sample = {"roots":[0,6],"assignments":labels-1}
        samples = {"node_tree_coo":np.zeros((12,12)), "samples":[sample.copy() for _ in range(5)]}
        with patch.object(experiments,"distance_labels",return_value=(labels,anchors)), \
             patch.object(experiments,"gibbsFullCT_multichain_until_valid",return_value=samples) as sampler:
            metrics, diag = experiments.run_method("MonteCarlo",info,sizes,cfg,0)
        self.assertEqual(diag["mc_valid_samples"],2)
        self.assertEqual(diag["mc_total_samples"],5)
        self.assertEqual(metrics["overall_classified"],0)
        self.assertEqual(sampler.call_args.kwargs["anchors"],[set(a) for a in anchors])
        with patch.object(experiments,"distance_labels",return_value=(labels,anchors)), \
             patch.object(experiments,"gibbsFullCT_multichain_until_valid",
                          side_effect=experiments.InsufficientSamplesError(0,5,1)):
            metrics, diag = experiments.run_method("MonteCarlo",info,sizes,cfg,0)
            self.assertEqual(diag["result_source"],"distance_fallback")
            self.assertEqual(diag["mc_total_samples"],5)
            self.assertEqual(metrics["overall_classified"],0)
            with self.assertRaises(RuntimeError):
                experiments.run_method("MonteCarlo",info,sizes,replace(cfg,allow_distance_fallback=False),0)
        with patch.object(experiments,"distance_labels",return_value=(labels,anchors)), \
             patch.object(experiments,"gibbsFullCT_multichain_until_valid",return_value=samples), \
             patch.object(experiments,"Leaf_stripping_postCT",return_value={
                 "valid_samples":2,"assignment_probs":np.tile([1.,0.],(12,1))}):
            metrics, diag = experiments.run_method("MonteCarlo",info,sizes,cfg,0)
            self.assertEqual(diag["result_source"],"distance_fallback")
            self.assertEqual(diag["mc_valid_samples"],2)
            self.assertEqual(metrics["overall_classified"],0)
            with self.assertRaisesRegex(RuntimeError,"fewer than K communities"):
                experiments.run_method("MonteCarlo",info,sizes,replace(cfg,allow_distance_fallback=False),0)
        with patch.object(experiments,"distance_labels",return_value=(labels,anchors)), \
             patch.object(experiments,"gibbsFullCT_multichain_until_valid",side_effect=RuntimeError("sampler defect")):
            with self.assertRaisesRegex(RuntimeError,"sampler defect"):
                experiments.run_method("MonteCarlo",info,sizes,cfg,0)

    def test_parallel_execution_resume_and_no_overwrite(self):
        cfg = experiments.Config(n_list=(60,120), base_n=120, n_ref=60, theta_ref=.05,
                                 repeats=2, num_cores=1)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            single = quiet_call(experiments.run_experiment, cfg, "distance", base / "single")
            parallel = quiet_call(experiments.run_experiment, replace(cfg,num_cores=2), "distance", base / "parallel")
            pd.testing.assert_frame_equal(pd.read_csv(single), pd.read_csv(parallel), check_exact=True)
            before = single.read_bytes()
            with self.assertRaises(FileExistsError):
                experiments.run_experiment(cfg, "distance", base / "single")
            self.assertEqual(experiments.run_experiment(cfg, "distance", base / "single", resume=True), single)
            self.assertEqual(single.read_bytes(), before)
            with self.assertRaises(ValueError):
                experiments.run_experiment(replace(cfg,theta_ref=.06), "distance", base / "single", resume=True)
            # Emulate a stopped write: one complete trial and part of another.
            full = pd.read_csv(single)
            partial_rows = pd.concat([full[full.trial == 1], full[full.trial == 2].head(2)])
            single.unlink()
            partial_rows.to_csv(single.with_suffix(".partial.csv"), index=False)
            quiet_call(experiments.run_experiment, cfg, "distance", base / "single", resume=True)
            pd.testing.assert_frame_equal(pd.read_csv(single), full, check_exact=True)

    def test_plot_loader_rejects_duplicate_and_unpaired_records(self):
        source = ROOT / "tests/fixtures/pilot_balanced.csv"
        frame = pd.read_csv(source)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frame.to_csv(base / "distance.csv",index=False)
            duplicate = base / "duplicate.csv"
            frame.to_csv(duplicate,index=False)
            with self.assertRaisesRegex(ValueError,"duplicate trial"):
                plots.load_trials(base)
            duplicate.unlink()
            other = frame.copy()
            other["strategy"] = "SCORE"
            other.loc[0,"graph_hash"] = "different"
            other.to_csv(base / "score.csv",index=False)
            with self.assertRaisesRegex(ValueError,"different graphs"):
                plots.load_trials(base)

    def test_plot_loader_rejects_invalid_metrics_and_success(self):
        source = ROOT / "tests/fixtures/pilot_balanced.csv"
        frame = pd.read_csv(source)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "distance.csv"
            invalid = frame.copy()
            invalid.loc[0,"first_50"] = 1.1
            invalid.to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError,"out-of-range"):
                plots.load_trials(file.parent)
            invalid = frame.copy()
            invalid.loc[0,"cluster_count"] = 1
            invalid.to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError,"inconsistent successful"):
                plots.load_trials(file.parent)



if __name__ == "__main__":
    unittest.main(verbosity=2)
