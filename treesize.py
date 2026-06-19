import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from tqdm import trange
from collections import defaultdict
import sys
from leaf_stripping_recovery import (
    generate_ua_mixture_graph, generate_pa_mixture_graph
)
# Path and import
code_path = "/mnt/data/code_leaf_stripping"
sys.path.append(code_path)
# --- your function is assumed to be defined above this cell ---
# def generate_pa_mixture_graph(...):  # (omitted here for brevity)
#     ...

def component_max_degrees(graf: ig.Graph, components: list[list[int]]) -> np.ndarray:
    """
    Return an array of shape (K,) where each entry is the max degree
    within that component (degree taken in the FULL graph).
    """
    deg = graf.degree()
    comp_max = []
    for comp in components:
        comp_max.append(max(deg[v] for v in comp) if len(comp) > 0 else 0)
    return np.array(comp_max, dtype=int)

def min_of_component_max_degrees_once(
    n: int = 5000,
    K: int = 2,
    theta: float = 5e-4,
    ratio: tuple | list | None = None,
    seed: int | None = None,
) -> tuple[int, np.ndarray]:
    """
    Build a K-component PA mixture and compute:
      - comp_max: vector of per-component max degrees
      - min_max:  min(comp_max)
    Returns (min_max, comp_max).
    """
    if seed is not None:
        np.random.seed(seed)

    graf, true_components, _ = generate_pa_mixture_graph(
        n=n, K=K, theta=theta, ratio=list(ratio) if ratio is not None else None
    )
    comp_max = component_max_degrees(graf, true_components)
    return int(comp_max.min()), comp_max

def sample_min_of_component_max(
    n: int = 5000,
    K: int = 2,
    theta: float = 5e-4,
    ratio: tuple | list | None = None,
    trials: int = 300,
    base_seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Repeat `trials` times and collect:
      - mins:  array of length trials with min(comp_max) each run
      - all_max: array of shape (trials, K) with per-component maxima per run
    """
    rng = np.random.default_rng(base_seed)
    mins = np.empty(trials, dtype=int)
    all_max = np.empty((trials, K), dtype=int)

    for t in range(trials):
        seed = int(rng.integers(0, 2**31 - 1))
        min_max, comp_max = min_of_component_max_degrees_once(
            n=n, K=K, theta=theta, ratio=ratio, seed=seed
        )
        mins[t] = min_max
        all_max[t, :] = comp_max
    return mins, all_max

# ===== Example usage =====
if __name__ == "__main__":
    # Choose K and (optionally) a ratio of sizes that sums to 1
    K =2
    ratio = (0.2,0.8)  # or None for equal sizes
    n = 2200
    theta = 1/n
    trials = 1000

    mins, all_max = sample_min_of_component_max(
        n=n, K=K, theta=theta, ratio=ratio, trials=trials, base_seed=2025
    )

    print(f"Trials: {trials}, n={n}, K={K}, theta={theta}, ratio={ratio}")
    print(f"Mean of min(component max degree): {mins.mean():.2f}")
    print(f"Median: {np.median(mins)}")
    print(f"Std: {mins.std(ddof=1):.2f}")
    for q in [0.05, 0.25, 0.5, 0.75, 0.9]:
        print(f"{int(100*q)}th pct: {np.quantile(mins, q)}")

    # Optional: quick peek at per-component maxima for the last run
    print("Per-component max degrees (last run):", all_max[-1].tolist())