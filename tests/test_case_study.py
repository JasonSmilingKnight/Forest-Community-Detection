"""Case-study data, tie handling, and saved-result checks."""
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from case_study import ROOT, load_coauthors, posterior_labels, seed
import case_study


class CaseStudyTests(unittest.TestCase):
    def test_hierarchy_uses_tenth_degree_rank_and_q25_at_each_split(self):
        a = np.ones((40, 40), dtype=int) - np.eye(40, dtype=int)
        def split(graph, q, implementation, rank):
            labels = np.repeat([1, 2], len(graph)//2)
            return labels, labels.copy()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(case_study, 'load_coauthors', return_value=(a, np.arange(40), np.array([str(v) for v in range(40)]))), \
                patch.object(case_study, 'prune', side_effect=split) as pruning, \
                patch.object(case_study, 'hierarchy_layout', return_value={v: (0., 0.) for v in range(40)}), \
                patch.object(case_study, 'hierarchy_plots', return_value={}), \
                patch.object(case_study, 'draw_tree'):
            case_study.run_hierarchical(Path(directory), Path(directory))
            self.assertEqual(len(pruning.call_args_list), 3)
            for call in pruning.call_args_list:
                self.assertEqual(call.args[1:], (25, 'hierarchical', 5))
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(summary['effective_rank'], 10)
            self.assertEqual(summary['pruning_q'], 25)
            self.assertEqual(summary['communities'], 4)

    def test_coauthorship_pools_by_valid_sample_count_and_keeps_zero_chains(self):
        a = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)
        labels = np.array([1, 1, 2, 2])
        first = np.tile([1., 0.], (4, 1))
        second = np.tile([0., 1.], (4, 1))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(case_study, 'load_coauthors', return_value=(a, np.arange(4), np.array(list('abcd')))), \
                patch.object(case_study, 'prune', return_value=(labels, labels)), \
                patch.object(case_study, 'estimateAlphaEM', return_value=0), \
                patch.object(case_study, 'posterior', side_effect=[(first, 2), RuntimeError('No posterior samples meet the anchor condition.'), (second, 6)]), \
                patch.object(case_study, 'cluster_plots', return_value={}):
            case_study.run_coauthorship(Path(directory), Path(directory), samples=10, mc_seeds=[1, 2, 3])
            saved = pd.read_csv(Path(directory) / 'assignments.csv')
            np.testing.assert_allclose(saved.p1, .25)
            self.assertTrue((saved.mc_cluster == 2).all())
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(summary['samples'], 30)
            self.assertEqual(summary['valid_samples'], 8)
            self.assertEqual(summary['valid_samples_by_seed'], {'1': 2, '2': 0, '3': 6})

    def test_recovery_figures_share_coordinates(self):
        a = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)
        labels = np.array([1, 1, 2, 2])
        probs = np.array([[.9, .1], [.4, .6], [.1, .9], [.2, .8]])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(case_study, 'load_coauthors', return_value=(a, np.arange(4), np.array(list('abcd')))), \
                patch.object(case_study, 'prune', return_value=(labels, labels)), \
                patch.object(case_study, 'estimateAlphaEM', return_value=0), \
                patch.object(case_study, 'posterior', return_value=(probs, 5)), \
                patch.object(case_study, 'cluster_plots', return_value={}) as plots:
            case_study.run_coauthorship(Path(directory), Path(directory), samples=5)
            distance_positions = plots.call_args_list[1].kwargs['positions']
            mc_positions = plots.call_args_list[2].kwargs['positions']
            self.assertIs(distance_positions, mc_positions)
            self.assertEqual(set(distance_positions), set(range(4)))
            saved = pd.read_csv(Path(directory) / 'layout.csv')
            np.testing.assert_allclose(saved[['x', 'y']], [distance_positions[v] for v in range(4)])

    def test_data(self):
        a, ids, names = load_coauthors(ROOT / "data")
        self.assertEqual(a.shape, (2263, 2263))
        self.assertEqual(int(a.sum()//2), 4388)
        self.assertTrue(np.issubdtype(a.dtype, np.signedinteger))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(names), 2263)

    def test_posterior_ties(self):
        probabilities = np.tile([.5, .5], (100, 1))
        seed()
        labels = posterior_labels(probabilities)
        self.assertEqual(set(labels), {1, 2})
        np.testing.assert_array_equal(posterior_labels(probabilities, "hierarchical"), 1)
        self.assertEqual(posterior_labels(np.array([[.8, .2]]))[0], 1)

    def test_toy_outputs(self):
        path = ROOT / "results/case_studies/toy"
        if not (path / "summary.json").exists():
            self.skipTest("Run --case toy first")
        summary = json.loads((path / "summary.json").read_text())
        self.assertEqual(summary["nodes"], 23)
        self.assertEqual(summary["edges"], 30)
        result = pd.read_csv(path / "assignments.csv")
        np.testing.assert_allclose(result[["p1", "p2"]].sum(axis=1), 1)
        self.assertTrue(result[["p1", "p2"]].ge(0).all().all())

    def test_saved_coverage(self):
        for path in (ROOT / "results/case_studies").rglob("summary.json"):
            summary = json.loads(path.read_text())
            for filename in summary["outputs"]:
                self.assertTrue((path.parent / filename).is_file(), str(path / filename))
            if "communities" in summary:
                result = pd.read_csv(path.parent / "assignments.csv")
                self.assertEqual(len(result), summary["nodes"])
                self.assertEqual(int((result.cluster == 0).sum()), summary["unclassified"])
                self.assertEqual(len(set(result.cluster)-{0}), summary["communities"])
                for coverage in summary["coverage"].values():
                    self.assertGreaterEqual(coverage["assigned"], coverage["displayed"])


if __name__ == "__main__":
    unittest.main()
