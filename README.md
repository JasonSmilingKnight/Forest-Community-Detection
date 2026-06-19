# Community Detection on a Randomly Growing Network

This repository contains the code for examples in the paper

**Community Detection on a Randomly Growing Network**, by Jianxiang Wang and Min Xu. ([https://arxiv.org/abs/2606.07447](https://arxiv.org/abs/2606.07447))


We ask you to kindly include the reference to the above paper if you use any part of the code for publications.

---


## Files

**Main scripts** (run directly):

- `introduction_code.py`: Figure 1 — co-authorship network visualization.
- `SPAR_code.py`: Figures 2–3, 13 — SPAR algorithm on a toy graph with hard and soft community assignments.
- `case_study_preliminary.py`: Figures 7–9 — K=2 community detection on the co-authorship network.
- `case_study_hierarchical.py`: Figures 10–12 — hierarchical detection yielding 20 communities.
- `experiment-0.5_scaled_theta.py`, `experiment0_scaled_theta.py`, `experiment2_scaled_theta.py`: Simulations for balanced K=2 (Section 5), at α = −0.5, 0, 2 respectively. Outputs raw CSVs.
- `analyze_results_scaled_theta.py`: Tables 1–4, Figures 4–6 — aggregates the K=2 simulation CSVs.
- `experiment-0.5_3_scaled_theta.py`, `experiment0_3_scaled_theta.py`, `experiment2_3_scaled_theta.py`: Simulations for unbalanced K=3 (Appendix 8.3). Outputs raw CSVs.
- `analyze_results_3_scaled_theta.py`: Tables 5–7, Figures 14–16 — aggregates the K=3 simulation CSVs.

**Helper modules** :

- `leaf_stripping_recovery.py`, `gibbsSampling.py`: leaf-stripping recovery and Gibbs sampling routines used by `SPAR_code.py`.
- `experiment_distance_recovery_recursive_new.py`: core simulation loop used by all experiment scripts.
- `utilities.py`: shared metrics, aggregation, and table-printing utilities.
- `treesize.py`, `treesize_order.py`, `treesize_order_scaled_theta.py`: tree size diagnostic utilities.

**HPC scripts**: `run-experiment-*.txt` are SLURM submission scripts for the Amarel cluster, one per experiment script.

**Data**: `coauthorThresh2GiantAdj_big.txt` and `coauthorThresh2GiantAuthorList_big.txt` are the co-authorship network edge list and author names from Ji and Jin (2016).

---

## Requirements

Python packages: `numpy`, `pandas`, `matplotlib`, `networkx`, `python-igraph`, `adjustText`, `tqdm`, `ete3` (optional, for dendrogram).

---

## References

- J. Wang and M. Xu. Community Detection on a Randomly Growing Network. arXiv:2606.07447, 2026.
- P. Ji and J. Jin. Coauthorship and citation networks for statisticians. *The Annals of Applied Statistics*, 10(4):1779–1812, 2016.
