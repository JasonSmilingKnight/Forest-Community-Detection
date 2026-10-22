# Community Detection on a Randomly Growing Network

Code, data, and selected results for the [paper](https://arxiv.org/abs/2606.07447) by Jianxiang Wang and Min Xu.

The code was prepared for an earlier tailored version of the manuscript and includes some experiments that are not reported in the main paper.

## Setup

Use Python 3.11 or later. Run from the repository root:

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

## Run

~~~bash
python reproduce.py                          # Rebuild paper simulation tables and figures
python reproduce.py --compile                # Also compile the JMLR manuscript
python case_study.py --case all               # Run the paper case studies
python experiments.py --scenario balanced --alpha 0 --phase distance --num-cores 25
bash jobs/submit.sh                           # Submit simulations on Amarel
~~~

Simulation options are `--scenario balanced` or `unbalanced`, `--alpha -0.5`, `0`, or `2`, and `--phase distance`, `benchmarks`, or `montecarlo`. Use `--resume` for compatible interrupted runs. New simulations save to `results/raw/`; case studies save to `results/case_studies/`.

`reproduce.py` uses completed inputs under `results/amarel/rerun_20261015/`, preferring matching new inputs under `results/raw/` or `--data-dir`. The selected inputs and their hashes are recorded in `results/reproduced/input_manifest.json`. The script updates the JMLR manuscript tables and figures. Compilation requires `pdflatex` and BibTeX.

Monte Carlo simulations record distance fallback when the valid-sample budget is insufficient. Saved configuration files describe the runs when recorded. The sampler target, defaults, and limitations are described in [the sampler notes](docs/TWO_FOUNDER_SAMPLER.md).

## Files

| File or folder | Contents |
|---|---|
| `experiments.py` | Balanced and unbalanced paper simulations |
| `spar.py`, `baselines.py`, `mcmc/` | SPAR, SCORE, GMZZ, and Monte Carlo methods |
| `case_study.py`, `hierarchical_spar.py` | Toy and coauthorship case studies |
| `reproduce.py`, `paper_plots/` | Paper tables and figures |
| `plots.py` | Simulation CSV loading, aggregation, and plotting |
| `sync_case_figures.py` | Copy regenerated case-study figures into the manuscript |
| `data/` | Coauthorship inputs from Ji and Jin (2016) |
| `results/` | Selected simulation inputs, case-study results, and table templates |
| `manuscript/` | JMLR source, PDF, bibliography, and required figures |
| `jobs/`, `tests/` | Simulation submission scripts and core tests |

Please cite Jianxiang Wang and Min Xu, *Community Detection on a Randomly Growing Network*, arXiv:2606.07447, 2026, when using this code. Min Xu's original sampler attribution is retained in the source.
