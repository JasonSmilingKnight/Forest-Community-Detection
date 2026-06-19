# --- imports ---
import numpy as np
import igraph as ig
import sys
from leaf_stripping_recovery import generate_pa_mixture_graph, generate_recursive_mixture_graph

# Path and import (if needed)
code_path = "/mnt/data/code_leaf_stripping"
if code_path not in sys.path:
    sys.path.append(code_path)

# ---------- helpers ----------

def top_node_and_degree_per_component(graf: ig.Graph, components: list[list[int]]):
    """
    For each component, return the node id with the largest degree (in the FULL graph)
    and its degree. If a component is empty, returns (None, -1) for that component.
    """
    deg = graf.degree()
    top_nodes = []
    top_degrees = []
    for comp in components:
        if len(comp) == 0:
            top_nodes.append(None)
            top_degrees.append(-1)
        else:
            # pick any argmax if ties
            leader = max(comp, key=lambda v: deg[v])
            top_nodes.append(leader)
            top_degrees.append(deg[leader])
    return np.array(top_nodes, dtype=object), np.array(top_degrees, dtype=int)

def dense_descending_ranks(values: np.ndarray) -> np.ndarray:
    """
    Dense ranks, 1 = highest value. Ties share the same rank; ranks are consecutive.
    Example: values [10, 9, 9, 7] -> ranks [1, 2, 2, 3].
    """
    order = np.argsort(-values)  # indices sorted by descending values
    ranks = np.empty_like(order)
    rank = 1
    ranks[order[0]] = 1
    for i in range(1, len(values)):
        if values[order[i]] < values[order[i - 1]]:
            rank += 1
        ranks[order[i]] = rank
    return ranks

# If you prefer "competition" ranking (1,2,2,4), use this instead:
# def competition_descending_ranks(values: np.ndarray) -> np.ndarray:
#     order = np.argsort(-values)
#     ranks = np.empty_like(order)
#     ranks[order[0]] = 1
#     for i in range(1, len(values)):
#         if values[order[i]] == values[order[i - 1]]:
#             ranks[order[i]] = ranks[order[i - 1]]
#         else:
#             ranks[order[i]] = i + 1
#     return ranks

# ---------- one-shot + sampler ----------

def order_strength_once(
    n: int = 5000,
    K: int = 2,
    theta: float = 5e-4,
    alpha = 0,
    ratio: tuple | list | None = None,
    seed: int | None = None,
):
    """
    Build a K-component PA mixture, compute each component's top-degree node.
    Let 'weakest hub' = the component top node with the smallest degree among components.
    Return:
      - weak_rank: global dense rank (1 = largest degree) of that weakest hub
      - weak_deg: its degree
      - top_nodes: array of top node ids per component
      - top_degs: array of top degrees per component
      - global_ranks: dense ranks for all nodes (optional diagnostic)
    """
    if seed is not None:
        np.random.seed(seed)

    graf, true_components, _ = generate_recursive_mixture_graph(
        n=n, K=K, theta=theta, alpha=alpha, ratio=list(ratio) if ratio is not None else None
    )

    # per-component top nodes
    top_nodes, top_degs = top_node_and_degree_per_component(graf, true_components)

    # global ranks of ALL nodes by degree (descending, dense)
    deg = np.array(graf.degree(), dtype=int)
    global_ranks = dense_descending_ranks(deg)

    # identify 'weakest hub' among component leaders
    # (ignore None components, but those shouldn't occur in your generator)
    valid = [i for i, nd in enumerate(top_nodes) if nd is not None]
    if len(valid) == 0:
        # degenerate case: return NaNs
        return np.nan, np.nan, top_nodes, top_degs, global_ranks

    weakest_idx = min(valid, key=lambda i: top_degs[i])  # argmin over component leaders
    weakest_node = int(top_nodes[weakest_idx])
    weak_deg = int(top_degs[weakest_idx])
    weak_rank = int(global_ranks[weakest_node])

    return weak_rank, weak_deg, top_nodes, top_degs, global_ranks

def sample_order_strength(
    n: int = 5000,
    K: int = 2,
    theta: float = 5e-4,
    alpha = 20,
    ratio: tuple | list | None = None,
    trials: int = 300,
    base_seed: int = 1234,
):
    """
    Repeat `trials` times:
      - weak_ranks[t] = global rank (dense) of 'weakest hub' (1 = strongest possible)
      - weak_degs[t]  = that hub's degree
    Also collect the per-component top degrees for each run: top_degs_all shape (trials, K).
    """
    rng = np.random.default_rng(base_seed)
    weak_ranks = np.empty(trials, dtype=int)
    weak_degs = np.empty(trials, dtype=int)
    top_degs_all = np.empty((trials, K), dtype=int)

    for t in range(trials):
        seed = int(rng.integers(0, 2**31 - 1))
        wr, wd, _, td, _ = order_strength_once(
            n=n, K=K, theta=theta,alpha=alpha, ratio=ratio, seed=seed
        )
        weak_ranks[t] = wr
        weak_degs[t] = wd
        # td may include -1 if a component is empty; your generator shouldn't do that.
        # If you want to guard, replace negatives with 0 or np.nan here.
        # td = np.maximum(td, 0)
        # ensure length K
        if len(td) != K:
            raise ValueError(f"Expected {K} components, got {len(td)}.")
        top_degs_all[t, :] = td

    return weak_ranks, weak_degs, top_degs_all

# ---------- example usage ----------
if __name__ == "__main__":
    import pandas as pd

    K = 2
    ratio = (0.2, 0.8)
    trials = 1000
    alpha = -0.5

    # Define 5 n values and 3 theta values
    n_list = [1000, 2000, 3000, 4000, 5000]
    theta_types = [
        ('1/n', lambda n: 1.0 / n),
        ('1/n^(3/4)', lambda n: pow(n, -3/4)),
        ('1/n^(1/2)', lambda n: pow(n, -1/2))
    ]

    # Store results
    results = []

    print(f"Running experiments: K={K}, ratio={ratio}, trials={trials}, alpha={alpha}")
    print("=" * 80)

    # Run for each combination of n and theta
    for n in n_list:
        for theta_label, theta_func in theta_types:
            theta = theta_func(n)

            print(f"\nRunning: n={n}, θ={theta:.6f} ({theta_label})...")

            weak_ranks, weak_degs, top_degs_all = sample_order_strength(
                n=n, K=K, theta=theta, alpha=alpha, ratio=ratio, trials=trials, base_seed=2025
            )

            # Collect statistics
            results.append({
                'n': n,
                'theta_type': theta_label,
                'theta_value': theta,
                'mean_rank': weak_ranks.mean(),
                'median_rank': np.median(weak_ranks),
                'std_rank': weak_ranks.std(ddof=1),
                'q05_rank': np.quantile(weak_ranks, 0.05),
                'q25_rank': np.quantile(weak_ranks, 0.25),
                'q75_rank': np.quantile(weak_ranks, 0.75),
                'q95_rank': np.quantile(weak_ranks, 0.95),
                'mean_deg': weak_degs.mean(),
                'median_deg': np.median(weak_degs),
            })

    # Create DataFrame and display table
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)

    # Display full table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    print(df.to_string(index=False))

    # Save to CSV
    output_file = "treesize_order_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
