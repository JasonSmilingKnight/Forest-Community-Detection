# Community Detection on a Randomly Growing Network

Implementation of SPAR for community detection in random networks formed by preferential-attachment trees and Erdős–Rényi noise. The code includes graph pruning, distance-based and Monte Carlo recovery, simulation studies, and coauthorship examples.

See details in the [paper](https://arxiv.org/abs/2606.07447) by Jianxiang Wang and Min Xu.

This code was prepared for an earlier tailored version of the manuscript and includes some experiments not reported in the main paper.

## Installation

Use Python 3.11 or later. From the repository root:

~~~bash
python -m pip install -r requirements.txt
~~~

## Usage

Apply SPAR to the supplied coauthorship network:

~~~python
from pathlib import Path
from case_study import load_coauthors
from spar import spar

A, author_ids, author_names = load_coauthors(Path("data"))
result = spar(A, K=2, rank=10, Q=25)
labels = result.labels
~~~

`result.success` indicates whether pruning succeeded. `result.anchor_labels` contains the anchor-community assignments; `result.labels` contains the assignments after distance recovery.

## Notes

- Input to `spar` must be a nonempty symmetric binary adjacency matrix with a zero diagonal. The example uses the largest connected component of the supplied network.
- Community labels are `1, ..., K`; `0` denotes an unassigned vertex or failed pruning.
- [`experiments.py`](experiments.py) contains the simulation studies, and [`case_study.py`](case_study.py) contains the network examples.
- [`mcmc/`](mcmc/) contains the two-founder Monte Carlo sampler. See the [sampler notes](docs/TWO_FOUNDER_SAMPLER.md) for its assumptions and limitations.

Please cite the paper when using this code. The coauthorship data are from Ji and Jin (2016); Min Xu's original sampler attribution is retained in the source.
