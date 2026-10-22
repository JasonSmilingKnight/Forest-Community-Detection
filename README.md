# Community Detection on a Randomly Growing Network

Code accompanying [*Community Detection on a Randomly Growing Network*](https://arxiv.org/abs/2606.07447), by Jianxiang Wang and Min Xu.

The code implements SPAR for community detection in preferential-attachment forests with Erdős–Rényi noise, including graph pruning, distance-based recovery, and Monte Carlo recovery.

This code was prepared for an earlier tailored version of the manuscript and includes some experiments not reported in the main paper.

The main files are:

- `spar.py`: SPAR pruning and distance recovery.
- `mcmc/`: Gibbs sampler for Monte Carlo recovery.
- `experiments.py` and `baselines.py`: simulation experiments and benchmark methods.
- `case_study.py` and `hierarchical_spar.py`: coauthorship analyses and hierarchical community detection.
- `paper_plots/`: plotting functions for the numerical results.
- `data/` and `results/`: input data and selected results.

Python dependencies are listed in `requirements.txt`. The coauthorship data are from Ji and Jin (2016).

Please cite the paper if you use this code in your work.
