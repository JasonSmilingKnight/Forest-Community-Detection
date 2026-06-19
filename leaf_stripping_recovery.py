import networkx as nx
import numpy as np
from itertools import combinations
import igraph as ig
import random
from collections import deque
import matplotlib.pyplot as plt
import os
from typing import Iterable, Dict, Any, List, Tuple, Union


def Protected_node(A, mode ="True"):
    """
    Returns nodes that are NOT in any cycle (i.e., nodes that should be protected from removal).
    Removable nodes are the complement: nodes that ARE in cycles.
    """
    if mode != "True":
        return set()
    G = nx.from_numpy_array(A)
    # Find all nodes that are part of any cycle
    cycle_nodes = set()
    for cycle in nx.cycle_basis(G):
        cycle_nodes.update(cycle)
    # Protected nodes are those NOT in any cycle
    n = A.shape[0]
    protected = set(range(n)) - cycle_nodes
    return protected


def Removable_node(A, mode="True"):
    """
    Directly returns nodes that ARE in any cycle (i.e., nodes that can be removed).
    This is the complement of Protected_node.
    """
    if mode != "True":
        return set(range(A.shape[0]))  # All nodes removable if mode is off
    G = nx.from_numpy_array(A)
    # Find and return all nodes that are part of any cycle
    cycle_nodes = set()
    for cycle in nx.cycle_basis(G):
        cycle_nodes.update(cycle)
    return cycle_nodes


def Find_removable_protected_nodes(A, live_nodes, protected_nodes, top_b_nodes,
                                   min_size_cur, cutoff, initial_degrees, K, min_removable_degree=None):
    """
    Find protected nodes whose single removal would create at least one additional valid component.

    This function checks if removing any protected node (typically nodes not in cycles)
    would immediately achieve the goal of increasing valid components by at least 1.

    Parameters
    ----------
    A : np.ndarray
        Current adjacency matrix
    live_nodes : list
        Currently active nodes
    protected_nodes : set
        Nodes that are protected (not in cycles)
    top_b_nodes : set
        High-degree reference nodes
    min_size_cur : int
        Minimum qualified component size
    cutoff : int
        Degree threshold for qualified nodes
    initial_degrees : dict
        Initial degree of each node
    K : int
        Target number of components (kept for signature compatibility)
    min_removable_degree : int, optional
        Only consider protected nodes with degree <= this value (optimization)

    Returns
    -------
    set
        Protected nodes whose removal creates at least one additional valid component
    """
    if not protected_nodes:
        return set()

    G = nx.from_numpy_array(A).subgraph(live_nodes)
    qsize = lambda comp: len(comp)

    # Optimization: filter protected nodes to only those with degree <= min_removable_degree
    if min_removable_degree is not None:
        protected_nodes = {node for node in protected_nodes if initial_degrees[node] <= min_removable_degree}
    
    if not protected_nodes:
        return set()

    # Compute baseline number of valid components before removing any protected node
    base_components = [
        comp for comp in nx.connected_components(G)
        if qsize(comp) >= min_size_cur
    ]
    base_components = [
        comp for comp in base_components
        if any(n in top_b_nodes for n in comp)
    ]
    base_count = len(base_components)

    valid_protected_removals = set()

    # Optimization: only check articulation points among protected nodes
    # (only these can create new components when removed)
    try:
        articulation_points = set(nx.articulation_points(G))
        candidates = protected_nodes.intersection(articulation_points)
    except:
        # If graph is disconnected or other issues, check all protected nodes
        candidates = protected_nodes

    # If no articulation points found but few protected nodes, check all
    if not candidates and len(protected_nodes) <= 10:
        candidates = protected_nodes

    for node in candidates:
        # Temporarily remove node and check components
        G_temp = G.copy()
        G_temp.remove_node(node)

        # Check if this creates at least one additional valid component
        components = [comp for comp in nx.connected_components(G_temp)
                     if qsize(comp) >= min_size_cur]
        components = [comp for comp in components
                     if any(n in top_b_nodes for n in comp)]

        new_count = len(components)
        if new_count >= base_count + 1:
            valid_protected_removals.add(node)

    return valid_protected_removals


def recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels, degree_threshold):
    n = A_input.shape[0]
    degrees = A_input.sum(axis=1)
    G = nx.from_numpy_array(A_input)

    # Guard: if cluster_count is 0, return empty labels
    if cluster_count == 0:
        return np.zeros(n, dtype=int), []

    # Step 1: Identify high-degree reference nodes for each cluster (still based on predicted labels 1..K)
    reference_nodes_lists = []
    for k in range(cluster_count):
        target_label = k + 1
        indices = np.where((predicted_labels == target_label) & (degrees > degree_threshold))[0]
        if len(indices) > 0:
            reference_nodes_lists.append(indices.tolist())
        else:
            fallback = np.where(predicted_labels == target_label)[0]
            reference_nodes_lists.append([int(fallback[0])] if len(fallback) > 0 else [])

    # Build a length-n vector in the same style as labels: 0 = not reference, k+1 = reference of cluster k+1
    reference_nodes_vector = np.zeros(n, dtype=int)
    for k, nodes in enumerate(reference_nodes_lists):
        if nodes:
            reference_nodes_vector[nodes] = k + 1

    # Step 2: Assign each node to the nearest reference cluster (with random tie-breaking), keep labels 1..K
    new_labels = np.zeros(n, dtype=int)
    for i in range(n):
        dist_list = []
        for k in range(cluster_count):
            refs = reference_nodes_lists[k]
            min_dist_to_cluster = np.inf
            for r in refs:
                if nx.has_path(G, i, r):
                    d = nx.shortest_path_length(G, i, r)
                    if d < min_dist_to_cluster:
                        min_dist_to_cluster = d
            dist_list.append(min_dist_to_cluster)

        # Guard: ensure dist_list is not empty before calling min()
        if len(dist_list) > 0:
            min_dist = min(dist_list)
            candidates = [k for k, d in enumerate(dist_list) if d == min_dist]
            chosen_cluster = random.choice(candidates)
            new_labels[i] = chosen_cluster + 1  # keep 1..K

    # Return labels (1..K) and the reference vector in the same format (0 for non-ref, 1..K for refs)
    return new_labels, reference_nodes_lists






def Leaf_stripping(
    A_input,
    K=3,
    min_component_size=25,
    b=3,
    max_steps=10000,
    d=5,
    threshold_by="order",
    select_by="initial",
    recover=False,
    recover_mode="batch",
    mode="True",
    beta: float = 0.8,          # shrink factor
    check_protected_removal: bool = False,  # <--- NEW: check protected nodes for removal
    recovery_threshold_divisor: float = 5.0,  # <--- NEW: divisor for recovery threshold calculation
):
    # Precompute once (doesn't change across retries)
    n = A_input.shape[0]
    initial_degrees = {i: A_input[i].sum() for i in range(n)}
    # Top-b selection (same as your code)
    if threshold_by == "deg":
        top_b_nodes = {node for node, deg in initial_degrees.items() if deg > b}
        core_deg_threshold = b  # For "deg" mode, core threshold is the degree parameter b
    else:
        sorted_nodes_by_degree = sorted(initial_degrees.items(), key=lambda x: -x[1])
        top_b_nodes = {node for node, _ in sorted_nodes_by_degree[:b]}
        # For "order" mode, core threshold is the degree of the b-th highest degree node
        if len(sorted_nodes_by_degree) >= b:
            core_deg_threshold = sorted_nodes_by_degree[b - 1][1]
        else:
            core_deg_threshold = sorted_nodes_by_degree[-1][1] if sorted_nodes_by_degree else 0

    # Start at the user-specified size threshold; shrink by beta until success
    min_size_cur = int(min_component_size)
    while True:
        # -------- one attempt with current size threshold --------
        A = A_input.copy()
        removed_batches = []
        predicted_labels = np.zeros(n, dtype=int)
        final_components = []
        last_removed_degrees = []
        removal_levels = []
        cutoff = 0
        
        for _ in range(max_steps):
            if removed_batches:
                cutoff = removal_levels[-1] if removal_levels else 0

            remaining_nodes = [i for i in range(n) if not any(i in batch for batch in removed_batches)]
            g_current = nx.from_numpy_array(A).subgraph(remaining_nodes)
            # qualified size: count all nodes in component
            qsize = lambda comp: len(comp)

            # --- use the current threshold min_size_cur here ---
            components_all = [comp for comp in nx.connected_components(g_current) if qsize(comp) >= min_size_cur]
            components = [comp for comp in components_all if any(node in top_b_nodes for node in comp)]

            if len(components) >= K:
                final_components = components
                break

            live_nodes = remaining_nodes
            if len(live_nodes) <= 1:
                break

            # Directly get removable nodes (nodes in cycles)
            removable_set = Removable_node(A, mode)
            removable_nodes = [i for i in live_nodes if i in removable_set]

            # Optionally check if any protected nodes can be removed to create K components
            if check_protected_removal and mode == "True":
                protected_set = set(live_nodes) - removable_set
                if protected_set:
                    # Optimization: only search protected nodes with degree <= min degree of removable nodes
                    if removable_nodes:
                        deg_array = np.array([initial_degrees[i] for i in range(n)])
                        min_removable_deg = min(deg_array[i] for i in removable_nodes)
                        extra_removable = Find_removable_protected_nodes(
                            A, live_nodes, protected_set, top_b_nodes,
                            min_size_cur, cutoff, initial_degrees, K, min_removable_degree=min_removable_deg
                        )
                    else:
                        extra_removable = Find_removable_protected_nodes(
                            A, live_nodes, protected_set, top_b_nodes,
                            min_size_cur, cutoff, initial_degrees, K
                        )
                    if extra_removable:
                        print(f"INFO: Found {len(extra_removable)} protected node(s) whose removal creates K components: {sorted(extra_removable)}")
                        # Add to removable set
                        removable_set = removable_set.union(extra_removable)
                        removable_nodes = [i for i in live_nodes if i in removable_set]

            if not removable_nodes:
                # When check_protected_removal is enabled, skip manual removal and let beta-shrinking retry
                if check_protected_removal:
                    print("INFO: No removable nodes found. Relying on beta-shrinking to retry with smaller threshold.")
                    break

                # Standard mode: attempt manual removal
                print("WARNING: No removable nodes found (all remaining nodes are protected).")
                print("         Attempting manual removal to break graph into components...")
                G = nx.from_numpy_array(A).subgraph(live_nodes).copy()

                # (You can also change these two len(...) to qsize(...) if you want consistency)
                components_before_removal = [comp for comp in nx.connected_components(G) if len(comp) >= min_size_cur]
                components_before_removal = [comp for comp in components_before_removal if any(node in top_b_nodes for node in comp)]
                if len(components_before_removal) >= K:
                    final_components = components_before_removal
                    break

                degrees = {node: G.degree(node) for node in G.nodes()}
                sorted_nodes = [node for node, deg in sorted(degrees.items(), key=lambda x: x[1])]
                for r in range(1, min(K, len(sorted_nodes))):
                    for nodes_to_remove in combinations(sorted_nodes, r):
                        G_temp = G.copy()
                        G_temp.remove_nodes_from(nodes_to_remove)
                        new_components = [comp for comp in nx.connected_components(G_temp) if qsize(comp) >= min_size_cur]
                        new_components = [comp for comp in new_components if any(node in top_b_nodes for node in comp)]
                        if len(new_components) >= K:
                            print(f"         Successfully removed {len(nodes_to_remove)} protected node(s) manually: {sorted(nodes_to_remove)}")
                            for node in nodes_to_remove:
                                A[node, :] = 0
                                A[:, node] = 0
                            removed_batches.append(set(nodes_to_remove))
                            final_components = new_components
                            break
                    if final_components:
                        break
                if not final_components:
                    final_components = components_before_removal
                break

            G_current = nx.from_numpy_array(A).subgraph(live_nodes)
            loop_count = len(nx.cycle_basis(G_current))

            if loop_count < d:
                deg_array = np.array([initial_degrees[i] for i in range(n)])
                sorted_nodes = sorted(removable_nodes, key=lambda x: deg_array[x])
                if sorted_nodes:
                    node = sorted_nodes[0]
                    # record current removal level if you want to base cutoff on CURRENT degrees:
                    # removal_levels.append(G_current.degree(node))
                    A[node, :] = 0
                    A[:, node] = 0
                    removed_batches.append({node})
                    continue
                else:
                    break

            deg_array = np.array([initial_degrees[i] for i in range(n)])
            min_deg = min(deg_array[i] for i in removable_nodes)
            leaves = {i for i in removable_nodes if deg_array[i] == min_deg}
            removal_levels.append(min_deg)
            if not leaves:
                break

            for leaf in leaves:
                A[leaf, :] = 0
                A[:, leaf] = 0
                last_removed_degrees.append(initial_degrees[leaf])

            degrees_now = A.sum(axis=1)
            deg0_now = {i for i in live_nodes if degrees_now[i] == 0 and all(i not in batch for batch in removed_batches)}
            all_removed = leaves.union(deg0_now)
            removed_batches.append(all_removed)

            for node in deg0_now:
                A[node, :] = 0
                A[:, node] = 0
                last_removed_degrees.append(initial_degrees[node])

        # -------- post-processing for this attempt --------
        component_sizes = []
        
        # Select K clusters with highest maximum degree in their cores
        # Core nodes are those with initial degree >= cutoff
        component_max_degrees = []
        for comp in final_components:
            # Find maximum degree among core nodes in this component
            core_nodes = [node for node in comp if initial_degrees[node] >= cutoff]
            if core_nodes:
                max_degree_in_core = max(initial_degrees[node] for node in core_nodes)
            else:
                # Fallback: use max degree in component if no core nodes found
                max_degree_in_core = max(initial_degrees[node] for node in comp) if comp else 0
            component_max_degrees.append((max_degree_in_core, comp))
        
        # Sort by max degree (descending) and take top K
        component_max_degrees_sorted = sorted(component_max_degrees, key=lambda x: -x[0])
        top_K_components = [comp for _, comp in component_max_degrees_sorted[:K]]
        
        for label, comp in enumerate(top_K_components, start=1):
            for node in comp:
                predicted_labels[node] = label
            component_sizes.append(len(comp))

        cluster_count = len(top_K_components)

        # If success, return results from this attempt
        if cluster_count >= K:
            print("last removed degree is", max(last_removed_degrees))
            if not recover:
                return cluster_count, predicted_labels, component_sizes
            if recover_mode == "distance_threshold":
                if not last_removed_degrees:
                    raise ValueError("No node was removed; cannot determine threshold.")
                max_removed_deg = max(last_removed_degrees)
                threshold = int(min(core_deg_threshold / recovery_threshold_divisor, cutoff))
                print(f"Recovery threshold calculation:")
                print(f"  core_deg_threshold = {core_deg_threshold}")
                print(f"  cutoff = {cutoff}")
                print(f"  max(last_removed_degrees) = {max_removed_deg}")
                print(f"  Recovery threshold = min(core_deg_threshold / {recovery_threshold_divisor}, cutoff) = {threshold}")
                final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels, threshold)
                return cluster_count, final_labels, predicted_labels, component_sizes, reference_nodes_per_cluster

            # Default recover
            predicted_vectors = np.zeros((n, K))
            for i in range(n):
                if predicted_labels[i] > 0:
                    predicted_vectors[i, predicted_labels[i] - 1] = 1

            for batch_idx, batch in enumerate(reversed(removed_batches)):
                batch = set(batch)
                batch_vec = predicted_vectors.copy()
                for node in batch:
                    neighbors = np.where(A_input[node] > 0)[0]
                    neighbor_vecs = batch_vec[neighbors]
                    mask = neighbor_vecs.sum(axis=1) > 0
                    if mask.sum() > 0:
                        avg_vec = neighbor_vecs[mask].mean(axis=0)
                        predicted_vectors[node] = avg_vec / avg_vec.sum()

            leftover_nodes = np.where(predicted_vectors.sum(axis=1) == 0)[0]
            G_leftover = nx.from_numpy_array(A_input).subgraph(leftover_nodes)
            components_rest = list(nx.connected_components(G_leftover))

            for comp in components_rest:
                comp = list(comp)
                all_neighbors = set()
                for node in comp:
                    neighbors = np.where(A_input[node] > 0)[0]
                    all_neighbors.update(neighbors)
                neighbor_vecs = predicted_vectors[list(all_neighbors)]
                mask = neighbor_vecs.sum(axis=1) > 0
                if mask.sum() > 0:
                    avg_vec = neighbor_vecs[mask].mean(axis=0)
                    avg_vec = avg_vec / avg_vec.sum()
                    for node in comp:
                        predicted_vectors[node] = avg_vec

            return cluster_count, predicted_vectors, component_sizes

        # -------- shrink step: not enough components, try again with beta*size --------
        new_min_size = max(1, int(np.floor(beta * min_size_cur)))
        if new_min_size == min_size_cur:   # prevent infinite loop when beta≈1 and size already 1
            # return the best-so-far attempt (no shrink possible)
            if not recover:
                return cluster_count, predicted_labels, component_sizes
            # same recovery block as above (re-run for completeness)
            if recover_mode == "distance_threshold":
                if not last_removed_degrees:
                    raise ValueError("No node was removed; cannot determine threshold.")
                threshold = max(last_removed_degrees)  # Use the maximum degree of last removed nodes
                final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels, threshold)
                return cluster_count, final_labels, predicted_labels, component_sizes
            predicted_vectors = np.zeros((n, K))
            for i in range(n):
                if predicted_labels[i] > 0:
                    predicted_vectors[i, predicted_labels[i] - 1] = 1
            for batch_idx, batch in enumerate(reversed(removed_batches)):
                batch = set(batch)
                batch_vec = predicted_vectors.copy()
                for node in batch:
                    neighbors = np.where(A_input[node] > 0)[0]
                    neighbor_vecs = batch_vec[neighbors]
                    mask = neighbor_vecs.sum(axis=1) > 0
                    if mask.sum() > 0:
                        avg_vec = neighbor_vecs[mask].mean(axis=0)
                        predicted_vectors[node] = avg_vec / avg_vec.sum()
            leftover_nodes = np.where(predicted_vectors.sum(axis=1) == 0)[0]
            G_leftover = nx.from_numpy_array(A_input).subgraph(leftover_nodes)
            components_rest = list(nx.connected_components(G_leftover))
            for comp in components_rest:
                comp = list(comp)
                all_neighbors = set()
                for node in comp:
                    neighbors = np.where(A_input[node] > 0)[0]
                    all_neighbors.update(neighbors)
                neighbor_vecs = predicted_vectors[list(all_neighbors)]
                mask = neighbor_vecs.sum(axis=1) > 0
                if mask.sum() > 0:
                    avg_vec = neighbor_vecs[mask].mean(axis=0)
                    avg_vec = avg_vec / avg_vec.sum()
                    for node in comp:
                        predicted_vectors[node] = avg_vec
            return cluster_count, predicted_vectors, component_sizes

        # update and retry
        min_size_cur = new_min_size
        print(min_size_cur)


def Leaf_stripping_adaptive(
    A_input,
    Q=10,
    b=10,
    max_steps=10000,
    d=0,
    threshold_by="order",
    select_by="initial",
    mode="True",
    check_protected_removal: bool = True,
    recovery_threshold_divisor: float = 2.0,
    recover: bool = True,
    recover_mode: str = "distance_threshold",
    beta: float = 0.8,
):
    """
    Adaptive leaf stripping: same as Leaf_stripping but with no fixed K.

    Strips nodes exactly like the original (lowest degree in cycles first),
    but the qualified-component threshold is Q (default 10 core nodes).
    If removing a node would collapse an existing qualified core (a component
    that currently has >= Q core nodes would drop below Q), block that removal.
    Otherwise proceed normally. Communities emerge as the graph splits apart.

    Parameters
    ----------
    A_input : np.ndarray
        Adjacency matrix
    Q : int
        Minimum number of core nodes required per component (default 10)
    b : int
        Number of top reference nodes (for threshold_by="order") or degree threshold
    max_steps : int
        Maximum stripping iterations
    d : int
        Cycle count threshold: <d = single removal, >=d = batch removal
    threshold_by : str
        "order" (top-b by degree) or "deg" (degree > b)
    select_by : str
        "initial" to use initial degrees for removal ordering
    mode : str
        "True" to use cycle-based removal
    check_protected_removal : bool
        If True, also check protected nodes for removal
    recovery_threshold_divisor : float
        Divisor for distance recovery threshold
    recover : bool
        Whether to run distance recovery
    recover_mode : str
        Recovery mode ("distance_threshold" supported)
    beta : float
        Not used for shrinking here, kept for interface compatibility

    Returns
    -------
    (cluster_count, distance_labels, core_labels, component_sizes, reference_nodes)
    """
    n = A_input.shape[0]
    initial_degrees = {i: int(A_input[i].sum()) for i in range(n)}

    # Top-b selection (same as Leaf_stripping)
    if threshold_by == "deg":
        top_b_nodes = {node for node, deg in initial_degrees.items() if deg > b}
        core_deg_threshold = b
    else:
        sorted_nodes_by_degree = sorted(initial_degrees.items(), key=lambda x: -x[1])
        top_b_nodes = {node for node, _ in sorted_nodes_by_degree[:b]}
        if len(sorted_nodes_by_degree) >= b:
            core_deg_threshold = sorted_nodes_by_degree[b - 1][1]
        else:
            core_deg_threshold = sorted_nodes_by_degree[-1][1] if sorted_nodes_by_degree else 0

    def count_core_nodes(node_set):
        """Count nodes in node_set with initial_degree >= core_deg_threshold."""
        return sum(1 for nd in node_set if initial_degrees[nd] >= core_deg_threshold)

    def is_qualified(node_set):
        """A component is qualified if it has >= Q nodes AND at least 1 core node."""
        return len(node_set) >= Q and count_core_nodes(node_set) >= 1

    def would_reduce_clusters(node, G_graph):
        """Check if removing node would reduce the number of qualified components.
        Returns True if removal should be BLOCKED."""
        comp = nx.node_connected_component(G_graph, node)
        if not is_qualified(comp):
            # This node's component is not qualified, removal can't reduce count
            return False

        # Simulate removal
        comp_sub = G_graph.subgraph(comp).copy()
        comp_sub.remove_node(node)
        sub_components = list(nx.connected_components(comp_sub))

        if len(sub_components) <= 1:
            # No split — check if removing this node destroys qualification
            remaining = comp - {node}
            if not is_qualified(remaining):
                return True
            return False

        # Split occurred — check if any piece that had core nodes lost qualification
        for sc in sub_components:
            if count_core_nodes(sc) >= 1 and len(sc) < Q:
                # A piece has a core node but is too small
                return True
            if count_core_nodes(sc) == 0 and len(sc) >= Q:
                # A large piece lost all its core nodes — that's a reduction
                pass  # This piece just becomes unqualified, not a collapse
        # Check if total qualified count decreased
        before_count = 1  # the original comp was qualified
        after_count = sum(1 for sc in sub_components if is_qualified(sc))
        if after_count < before_count:
            return True

        return False

    A = A_input.copy()
    removed_batches = []
    last_removed_degrees = []
    removal_levels = []
    blocked_nodes = set()

    for step in range(max_steps):
        remaining_nodes = [i for i in range(n) if not any(i in batch for batch in removed_batches)]
        if len(remaining_nodes) <= 1:
            break

        cutoff = removal_levels[-1] if removal_levels else 0
        G_current = nx.from_numpy_array(A).subgraph(remaining_nodes)

        # Find removable nodes (in cycles) — "creates one more cluster" set
        removable_set = Removable_node(A, mode)
        removable_nodes = [i for i in remaining_nodes if i in removable_set and i not in blocked_nodes]

        # Optionally check protected nodes that would create one more cluster
        if check_protected_removal and mode == "True":
            protected_set = set(remaining_nodes) - removable_set
            if protected_set:
                if removable_nodes:
                    deg_array = np.array([initial_degrees[i] for i in range(n)])
                    min_removable_deg = min(deg_array[i] for i in removable_nodes)
                    extra_removable = Find_removable_protected_nodes(
                        A, remaining_nodes, protected_set, top_b_nodes,
                        Q, cutoff, initial_degrees, 2, min_removable_degree=min_removable_deg
                    )
                else:
                    extra_removable = Find_removable_protected_nodes(
                        A, remaining_nodes, protected_set, top_b_nodes,
                        Q, cutoff, initial_degrees, 2
                    )
                if extra_removable:
                    removable_set = removable_set.union(extra_removable)
                    removable_nodes = [i for i in remaining_nodes if i in removable_set and i not in blocked_nodes]

        if not removable_nodes:
            break

        # Filter target set: remove nodes that would REDUCE qualified component count.
        # Only articulation points and core nodes can reduce clusters, so only check those.
        removable_set_current = set(removable_nodes)
        try:
            art_points = set(nx.articulation_points(G_current))
        except:
            art_points = set()
        core_in_target = {i for i in removable_nodes if initial_degrees[i] >= core_deg_threshold}
        candidates_to_check = (art_points | core_in_target) & removable_set_current

        if candidates_to_check:
            nodes_to_exclude = set()
            for node in candidates_to_check:
                if would_reduce_clusters(node, G_current):
                    nodes_to_exclude.add(node)
            if nodes_to_exclude:
                removable_nodes = [i for i in removable_nodes if i not in nodes_to_exclude]
                blocked_nodes.update(nodes_to_exclude)

        if not removable_nodes:
            break

        loop_count = len(nx.cycle_basis(G_current))

        if loop_count < d:
            # Single node removal mode
            deg_array = np.array([initial_degrees[i] for i in range(n)])
            sorted_nodes = sorted(removable_nodes, key=lambda x: deg_array[x])
            removed_any = False
            for node in sorted_nodes:
                A[node, :] = 0
                A[:, node] = 0
                removed_batches.append({node})
                last_removed_degrees.append(initial_degrees[node])
                removed_any = True
                break
            if not removed_any:
                break
            continue

        # Batch removal mode: remove all nodes with minimum degree
        # Process nodes one at a time within the batch, re-checking against
        # updated graph state to catch collective fragmentation effects.
        deg_array = np.array([initial_degrees[i] for i in range(n)])
        min_deg = min(deg_array[i] for i in removable_nodes)
        leaves = [i for i in removable_nodes if deg_array[i] == min_deg]
        removal_levels.append(min_deg)

        if not leaves:
            break

        # Build a mutable working graph for sequential processing
        G_working = G_current.copy()

        accepted_leaves = []
        newly_isolated = []

        for leaf in leaves:
            if leaf not in G_working or G_working.degree(leaf) == 0:
                continue

            # Re-check against current working graph state
            if would_reduce_clusters(leaf, G_working):
                blocked_nodes.add(leaf)
                continue

            # Safe to remove — update working graph
            neighbors_before = list(G_working.neighbors(leaf))
            G_working.remove_node(leaf)
            accepted_leaves.append(leaf)

            # Remove any neighbors that became isolated (degree 0)
            for nb in neighbors_before:
                if nb in G_working and G_working.degree(nb) == 0:
                    newly_isolated.append(nb)
                    G_working.remove_node(nb)

        if not accepted_leaves:
            break

        # Update A matrix for all removed nodes
        for leaf in accepted_leaves:
            A[leaf, :] = 0
            A[:, leaf] = 0
            last_removed_degrees.append(initial_degrees[leaf])

        for node in newly_isolated:
            A[node, :] = 0
            A[:, node] = 0
            last_removed_degrees.append(initial_degrees[node])

        all_removed = set(accepted_leaves).union(set(newly_isolated))
        removed_batches.append(all_removed)

        if step % 50 == 0:
            remaining = len(remaining_nodes) - len(all_removed)
            n_comps = nx.number_connected_components(G_working)
            print(f"  Step {step}: {remaining} nodes remaining, {len(blocked_nodes)} blocked, {n_comps} components")

    # Collect final components: >= Q nodes AND at least 1 core node
    live_nodes = [i for i in range(n) if not any(i in batch for batch in removed_batches)]
    g_final = nx.from_numpy_array(A).subgraph(live_nodes)

    print(f"\n  core_deg_threshold = {core_deg_threshold}")
    total_core = count_core_nodes(range(n))
    print(f"  Total core nodes (degree >= {core_deg_threshold}): {total_core}")
    print(f"  Live nodes after stripping: {len(live_nodes)}")

    all_components = list(nx.connected_components(g_final))
    print(f"  Components in final graph: {len(all_components)}")
    for ci, comp in enumerate(sorted(all_components, key=len, reverse=True)[:10]):
        cc = count_core_nodes(comp)
        q = is_qualified(comp)
        print(f"    Component {ci+1}: {len(comp)} nodes, {cc} core nodes {'✓' if q else ''}")

    final_components = []
    for comp in all_components:
        if is_qualified(comp):
            final_components.append(comp)

    # Sort by max degree in component (descending)
    final_components.sort(key=lambda c: max(initial_degrees[nd] for nd in c), reverse=True)

    cluster_count = len(final_components)
    print(f"\nAdaptive leaf stripping found {cluster_count} communities")
    for i, comp in enumerate(final_components):
        print(f"  Core {i+1}: {len(comp)} nodes, {count_core_nodes(comp)} core nodes")

    # Assign core labels
    core_labels = np.zeros(n, dtype=int)
    component_sizes = []
    for label, comp in enumerate(final_components, start=1):
        for node in comp:
            core_labels[node] = label
        component_sizes.append(len(comp))

    if not recover:
        return cluster_count, core_labels, component_sizes

    if recover_mode == "distance_threshold" and cluster_count > 0:
        threshold = int(min(core_deg_threshold / recovery_threshold_divisor, cutoff if cutoff > 0 else core_deg_threshold))
        print(f"Recovery threshold = {threshold}")
        final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(
            A_input, cluster_count, core_labels, threshold
        )
        return cluster_count, final_labels, core_labels, component_sizes, reference_nodes_per_cluster

    return cluster_count, core_labels, core_labels, component_sizes, None


def Leaf_stripping_autoK(
    A_input,
    min_component_size=25,
    b=3,
    max_steps=10000,
    d=5,                                 # int or iterable[int]
    threshold_by="order",                # "order" -> top-b by degree; "deg" -> degree > b
    select_by="initial",                 # kept for compatibility
    recover=False,
    recover_mode="batch",                # "batch" or "distance_threshold"
    mode="True",
    beta: float = 0.8,                   # kept for signature compatibility (unused here)
):
    """
    Leaf-stripping WITHOUT a target K (and no loop_bound).
    - Precomputes the full peeling sequence once (independent of d).
    - For each cutoff d, stops at the last batch whose *initial-degree level* <= d,
      then counts qualified components to get K_found (including K=0 or K=1).

    Returns
    -------
    If d is an int:
        (K_found, labels_or_soft, component_sizes, meta)
    If d is iterable:
        {d_value: {"K": int, "labels": np.ndarray, "component_sizes": List[int], "meta": dict}, ...}
    """
    # --- Setup ---
    A_input = np.asarray(A_input)
    n = A_input.shape[0]
    A_work = A_input.copy()
    initial_degrees = {i: int(A_input[i].sum()) for i in range(n)}

    # --- Top node rule (independent of K) ---
    if threshold_by == "deg":
        top_nodes = {v for v, dv in initial_degrees.items() if dv > b}
        top_rule_desc = f"deg> {b}"
    else:
        # "order": pick exactly top b nodes by initial degree (global)
        sorted_deg = sorted(initial_degrees.items(), key=lambda x: -x[1])
        top_nodes = {v for v, _ in sorted_deg[:max(0, int(b))]}
        top_rule_desc = f"top-{b} by degree"

    # Provide a no-op Protected_node if not defined globally
    try:
        _ = Protected_node  # type: ignore[name-defined]
    except NameError:
        def Protected_node(_A, _mode="True"):
            return set()

    # --- Precompute full peeling sequence (independent of d) ---
    removed_batches: List[set] = []
    removal_levels: List[int] = []   # min *initial* degree of nodes removed in that batch
    steps = 0

    while steps < max_steps:
        steps += 1
        # live nodes
        if removed_batches:
            live = set(range(n)).difference(*removed_batches)
        else:
            live = set(range(n))

        if len(live) <= 1:
            break

        # Directly get removable nodes (nodes in cycles)
        removable_set = Removable_node(A_work, mode)
        removable = list(live.intersection(removable_set))
        if not removable:
            break

        # batch = all removable nodes with minimum INITIAL degree
        min_deg = min(initial_degrees[i] for i in removable)
        batch = {i for i in removable if initial_degrees[i] == min_deg}
        if not batch:
            break

        # remove them in A_work
        for u in batch:
            A_work[u, :] = 0
            A_work[:, u] = 0

        removed_batches.append(batch)
        removal_levels.append(min_deg)

    # Prefix unions for fast slicing by index
    prefix_removed: List[set] = []
    acc = set()
    for bset in removed_batches:
        acc = acc.union(bset)
        prefix_removed.append(acc.copy())

    def evaluate_cutoff(d_cut: int) -> Tuple[int, np.ndarray, List[int], Dict[str, Any]]:
        # find last batch index with level <= d_cut
        t = -1
        for i, lv in enumerate(removal_levels):
            if lv <= d_cut:
                t = i
            else:
                break

        removed_set = prefix_removed[t] if t >= 0 else set()
        cutoff = (removal_levels[t] if t >= 0 else 0)
        remaining_nodes = [i for i in range(n) if i not in removed_set]

        # components on ORIGINAL graph but restricted to remaining nodes
        G_final = nx.from_numpy_array(A_input).subgraph(remaining_nodes).copy()

        def qsize(comp) -> int:
            # qualified size by initial-degree threshold
            return sum(initial_degrees[v] >= cutoff for v in comp)

        comps_all = [c for c in nx.connected_components(G_final) if qsize(c) >= int(min_component_size)]
        comps = [c for c in comps_all if any(u in top_nodes for u in c)]
        comps_sorted = sorted(comps, key=lambda x: -len(x))

        K_found = len(comps_sorted)

        # hard labels first
        labels = np.zeros(n, dtype=int)
        component_sizes: List[int] = []
        for lab, comp in enumerate(comps_sorted, start=1):
            for u in comp:
                labels[u] = lab
            component_sizes.append(len(comp))

        meta = {
            "cutoff": cutoff,
            "removed_count": len(removed_set),
            "last_removed_degree_max": (cutoff if t >= 0 else None),
            "min_component_size_used": int(min_component_size),
            "top_rule": top_rule_desc,
        }

        if not recover:
            return K_found, labels, component_sizes, meta

        # ---- Recovery paths ----
        if recover_mode == "distance_threshold":
            # use your existing helper (global)
            try:
                _ = recover_with_filtered_distance_by_threshold  # type: ignore[name-defined]
            except NameError:
                raise ValueError("recover_mode='distance_threshold' requires recover_with_filtered_distance_by_threshold.")

            last_batch = removed_batches[t] if t >= 0 else set()
            if len(last_batch) == 0:
                # nothing removed -> return hard labels
                return K_found, labels, component_sizes, meta

            last_removed_degs = [initial_degrees[u] for u in last_batch]
            thr = min(max(last_removed_degs), b)
            final_labels, _ref = recover_with_filtered_distance_by_threshold(
                A_input, K_found, labels, thr
            )
            return K_found, final_labels, component_sizes, {**meta, "distance_threshold": thr}

        # default: batch backfill soft recovery
        if K_found == 0:
            return K_found, np.zeros((n, 0)), component_sizes, meta

        pred_vecs = np.zeros((n, K_found))
        for i in range(n):
            if labels[i] > 0:
                pred_vecs[i, labels[i] - 1] = 1.0

        # backfill only through batches up to index t
        for idx in range(t, -1, -1):
            batch = removed_batches[idx]
            for node in batch:
                nbrs = np.where(A_input[node] > 0)[0]
                nbr_vecs = pred_vecs[nbrs]
                mask = (nbr_vecs.sum(axis=1) > 0)
                if mask.any():
                    avg = nbr_vecs[mask].mean(axis=0)
                    s = float(avg.sum())
                    if s > 0:
                        pred_vecs[node] = avg / s

        # fill any leftover zeros by neighbor-avg within their CC
        leftover = np.where(pred_vecs.sum(axis=1) == 0)[0]
        if len(leftover) > 0:
            G_left = nx.from_numpy_array(A_input).subgraph(leftover)
            for comp in nx.connected_components(G_left):
                comp = list(comp)
                all_nbrs = set()
                for u in comp:
                    all_nbrs.update(np.where(A_input[u] > 0)[0])
                nbr_vecs = pred_vecs[list(all_nbrs)]
                m = (nbr_vecs.sum(axis=1) > 0)
                if m.any():
                    avg = nbr_vecs[m].mean(axis=0)
                    s = float(avg.sum())
                    if s > 0:
                        pred_vecs[comp, :] = avg / s

        return K_found, pred_vecs, component_sizes, meta

    # --- Run for single or multiple d ---
    if isinstance(d, (int, np.integer)):
        return evaluate_cutoff(int(d))

    # iterable of d's
    results: Dict[int, Dict[str, Any]] = {}
    for d_cut in d:  # preserve input order
        d_int = int(d_cut)
        K_found, labels_or_vecs, sizes, meta = evaluate_cutoff(d_int)
        results[d_int] = {
            "K": K_found,
            "labels": labels_or_vecs,
            "component_sizes": sizes,
            "meta": meta,
        }
    return results

def Leaf_stripping_general(
    A_input,
    K=3,
    min_component_size=25,
    b=3,
    max_steps=10000,
    d=5,
    threshold_by="order",
    select_by="initial",
    recover=False,
    recover_mode="batch",
    mode="True",
    beta: float = 0.8,          # shrink factor
    check_protected_removal: bool = False,  # <--- NEW: check protected nodes for removal
):
    n = A_input.shape[0]
    if A_input.shape[0] != A_input.shape[1]:
        raise ValueError("A_input must be square.")
    if not np.all(A_input == A_input.T):
        raise ValueError("A_input must be symmetric (undirected).")

    G_full = nx.from_numpy_array(A_input)
    initial_degrees = A_input.sum(axis=1).astype(float).ravel()   # shape (n,)

    # Build raw centrality c
    if isinstance(select_by, (list, np.ndarray)):
        c = np.asarray(select_by, dtype=float)
        if c.shape != (n,): raise ValueError(f"`select_by` vector must have shape ({n},)")
        if np.any(c < 0) or np.isnan(c).any(): raise ValueError("`select_by` vector must be non-negative and finite.")
        # ---- discretize to dense ranks: 0..(M-1), monotone in c ----
        # This preserves order, batches ties, and gives integer thresholds compatible with your code.
        uniq, inv = np.unique(c, return_inverse=True)
        disc = inv.astype(int)   # 0 for smallest c, larger integers for larger c

        # ---- ONLY CHANGE: replace initial_degrees by the discrete centrality score ----
        initial_degrees = {i: int(disc[i]) for i in range(n)}
    else:
        c = initial_degrees

    # ---- discretize to dense ranks: 0..(M-1), monotone in c ----
    # This preserves order, batches ties, and gives integer thresholds compatible with your code.
    uniq, inv = np.unique(c, return_inverse=True)
    disc = inv.astype(int)   # 0 for smallest c, larger integers for larger c

    # ---- ONLY CHANGE: replace initial_degrees by the discrete centrality score ----
    initial_degrees = {i: int(disc[i]) for i in range(n)}

    # Top-b selection (same as your code)
    if threshold_by == "deg":
        top_b_nodes = {node for node, deg in initial_degrees.items() if deg > b}
    else:
        sorted_nodes_by_degree = sorted(initial_degrees.items(), key=lambda x: -x[1])
        top_b_nodes = {node for node, _ in sorted_nodes_by_degree[:b]}

    # Start at the user-specified size threshold; shrink by beta until success
    min_size_cur = int(min_component_size)

    while True:
        # -------- one attempt with current size threshold --------
        A = A_input.copy()
        removed_batches = []
        predicted_labels = np.zeros(n, dtype=int)
        final_components = []
        last_removed_degrees = []
        removal_levels = []
        cutoff = 0
        for _ in range(max_steps):
            if removed_batches:
                cutoff = removal_levels[-1] if removal_levels else 0

            remaining_nodes = [i for i in range(n) if not any(i in batch for batch in removed_batches)]
            g_current = nx.from_numpy_array(A).subgraph(remaining_nodes)
            cutoff=0
            # qualified size: count all nodes in component
            qsize = lambda comp: len(comp)
            # --- use the current threshold min_size_cur here ---
            components_all = [comp for comp in nx.connected_components(g_current) if qsize(comp) >= min_size_cur]
            components = [comp for comp in components_all if any(node in top_b_nodes for node in comp)]

            if len(components) >= K:
                final_components = components
                break

            live_nodes = remaining_nodes
            if len(live_nodes) <= 1:
                break

            # Directly get removable nodes (nodes in cycles)
            removable_set = Removable_node(A, mode)
            removable_nodes = [i for i in live_nodes if i in removable_set]

            # Optionally check if any protected nodes can be removed to create K components
            if check_protected_removal and mode == "True":
                protected_set = set(live_nodes) - removable_set
                if protected_set:
                    # Optimization: only search protected nodes with degree <= min degree of removable nodes
                    if removable_nodes:
                        deg_array = np.array([initial_degrees[i] for i in range(n)])
                        min_removable_deg = min(deg_array[i] for i in removable_nodes)
                        extra_removable = Find_removable_protected_nodes(
                            A, live_nodes, protected_set, top_b_nodes,
                            min_size_cur, cutoff, initial_degrees, K, min_removable_degree=min_removable_deg
                        )
                    else:
                        extra_removable = Find_removable_protected_nodes(
                            A, live_nodes, protected_set, top_b_nodes,
                            min_size_cur, cutoff, initial_degrees, K
                        )
                    if extra_removable:
                        print(f"INFO: Found {len(extra_removable)} protected node(s) whose removal creates K components: {sorted(extra_removable)}")
                        # Add to removable set
                        removable_set = removable_set.union(extra_removable)
                        removable_nodes = [i for i in live_nodes if i in removable_set]

            if not removable_nodes:
                print("WARNING: No removable nodes found (all remaining nodes are protected).")
                print("         Attempting manual removal to break graph into components...")
                G = nx.from_numpy_array(A).subgraph(live_nodes).copy()

                # (You can also change these two len(...) to qsize(...) if you want consistency)
                components_before_removal = [comp for comp in nx.connected_components(G) if len(comp) >= min_size_cur]
                components_before_removal = [comp for comp in components_before_removal if any(node in top_b_nodes for node in comp)]
                if len(components_before_removal) >= K:
                    final_components = components_before_removal
                    break

                degrees = {node: G.degree(node) for node in G.nodes()}
                sorted_nodes = [node for node, deg in sorted(degrees.items(), key=lambda x: x[1])]
                for r in range(1, min(K, len(sorted_nodes))):
                    for nodes_to_remove in combinations(sorted_nodes, r):
                        G_temp = G.copy()
                        G_temp.remove_nodes_from(nodes_to_remove)
                        new_components = [comp for comp in nx.connected_components(G_temp) if qsize(comp) >= min_size_cur]
                        new_components = [comp for comp in new_components if any(node in top_b_nodes for node in comp)]
                        if len(new_components) >= K:
                            print(f"         Successfully removed {len(nodes_to_remove)} protected node(s) manually: {sorted(nodes_to_remove)}")
                            for node in nodes_to_remove:
                                A[node, :] = 0
                                A[:, node] = 0
                            removed_batches.append(set(nodes_to_remove))
                            final_components = new_components
                            break
                    if final_components:
                        break
                if not final_components:
                    final_components = components_before_removal
                break

            G_current = nx.from_numpy_array(A).subgraph(live_nodes)
            loop_count = len(nx.cycle_basis(G_current))

            if loop_count < d:
                deg_array = np.array([initial_degrees[i] for i in range(n)])
                sorted_nodes = sorted(removable_nodes, key=lambda x: deg_array[x])
                if sorted_nodes:
                    node = sorted_nodes[0]
                    # record current removal level if you want to base cutoff on CURRENT degrees:
                    # removal_levels.append(G_current.degree(node))
                    A[node, :] = 0
                    A[:, node] = 0
                    removed_batches.append({node})
                    continue
                else:
                    break

            deg_array = np.array([initial_degrees[i] for i in range(n)])
            min_deg = min(deg_array[i] for i in removable_nodes)
            leaves = {i for i in removable_nodes if deg_array[i] == min_deg}
            removal_levels.append(min_deg)
            if not leaves:
                break

            for leaf in leaves:
                A[leaf, :] = 0
                A[:, leaf] = 0
                last_removed_degrees.append(initial_degrees[leaf])

            degrees_now = A.sum(axis=1)
            deg0_now = {i for i in live_nodes if degrees_now[i] == 0 and all(i not in batch for batch in removed_batches)}
            all_removed = leaves.union(deg0_now)
            removed_batches.append(all_removed)

            for node in deg0_now:
                A[node, :] = 0
                A[:, node] = 0
                last_removed_degrees.append(initial_degrees[node])

        # -------- post-processing for this attempt --------
        component_sizes = []
        final_components_sorted = sorted(final_components, key=lambda x: -len(x))
        top_K_components = final_components_sorted[:K]

        for label, comp in enumerate(top_K_components, start=1):
            for node in comp:
                predicted_labels[node] = label
            component_sizes.append(len(comp))

        cluster_count = len(top_K_components)

        # If success, return results from this attempt
        if cluster_count >= K:
            
            if not recover:
                return cluster_count, predicted_labels, component_sizes
            if recover_mode == "distance_threshold":
                if not last_removed_degrees:
                    raise ValueError("No node was removed; cannot determine threshold.")
                threshold = max(last_removed_degrees)
                print("last removed degree is", max(last_removed_degrees))
                final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels, threshold)
                return cluster_count, final_labels, predicted_labels, component_sizes

            # Default recover
            predicted_vectors = np.zeros((n, K))
            for i in range(n):
                if predicted_labels[i] > 0:
                    predicted_vectors[i, predicted_labels[i] - 1] = 1

            for batch_idx, batch in enumerate(reversed(removed_batches)):
                batch = set(batch)
                batch_vec = predicted_vectors.copy()
                for node in batch:
                    neighbors = np.where(A_input[node] > 0)[0]
                    neighbor_vecs = batch_vec[neighbors]
                    mask = neighbor_vecs.sum(axis=1) > 0
                    if mask.sum() > 0:
                        avg_vec = neighbor_vecs[mask].mean(axis=0)
                        predicted_vectors[node] = avg_vec / avg_vec.sum()

            leftover_nodes = np.where(predicted_vectors.sum(axis=1) == 0)[0]
            G_leftover = nx.from_numpy_array(A_input).subgraph(leftover_nodes)
            components_rest = list(nx.connected_components(G_leftover))

            for comp in components_rest:
                comp = list(comp)
                all_neighbors = set()
                for node in comp:
                    neighbors = np.where(A_input[node] > 0)[0]
                    all_neighbors.update(neighbors)
                neighbor_vecs = predicted_vectors[list(all_neighbors)]
                mask = neighbor_vecs.sum(axis=1) > 0
                if mask.sum() > 0:
                    avg_vec = neighbor_vecs[mask].mean(axis=0)
                    avg_vec = avg_vec / avg_vec.sum()
                    for node in comp:
                        predicted_vectors[node] = avg_vec

            return cluster_count, predicted_vectors, component_sizes

        # -------- shrink step: not enough components, try again with beta*size --------
        new_min_size = max(1, int(np.floor(beta * min_size_cur)))
        print(new_min_size)
        print("last removed degree is", max(last_removed_degrees))
        if new_min_size == min_size_cur:   # prevent infinite loop when beta≈1 and size already 1
            # return the best-so-far attempt (no shrink possible)
            if not recover:
                return cluster_count, predicted_labels, component_sizes
            # same recovery block as above (re-run for completeness)
            if recover_mode == "distance_threshold":
                if not last_removed_degrees:
                    raise ValueError("No node was removed; cannot determine threshold.")
                threshold = max(last_removed_degrees)
                final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels, threshold)
                return cluster_count, final_labels, predicted_labels, component_sizes
            predicted_vectors = np.zeros((n, K))
            for i in range(n):
                if predicted_labels[i] > 0:
                    predicted_vectors[i, predicted_labels[i] - 1] = 1
            for batch_idx, batch in enumerate(reversed(removed_batches)):
                batch = set(batch)
                batch_vec = predicted_vectors.copy()
                for node in batch:
                    neighbors = np.where(A_input[node] > 0)[0]
                    neighbor_vecs = batch_vec[neighbors]
                    mask = neighbor_vecs.sum(axis=1) > 0
                    if mask.sum() > 0:
                        avg_vec = neighbor_vecs[mask].mean(axis=0)
                        predicted_vectors[node] = avg_vec / avg_vec.sum()
            leftover_nodes = np.where(predicted_vectors.sum(axis=1) == 0)[0]
            G_leftover = nx.from_numpy_array(A_input).subgraph(leftover_nodes)
            components_rest = list(nx.connected_components(G_leftover))
            for comp in components_rest:
                comp = list(comp)
                all_neighbors = set()
                for node in comp:
                    neighbors = np.where(A_input[node] > 0)[0]
                    all_neighbors.update(neighbors)
                neighbor_vecs = predicted_vectors[list(all_neighbors)]
                mask = neighbor_vecs.sum(axis=1) > 0
                if mask.sum() > 0:
                    avg_vec = neighbor_vecs[mask].mean(axis=0)
                    avg_vec = avg_vec / avg_vec.sum()
                    for node in comp:
                        predicted_vectors[node] = avg_vec
            return cluster_count, predicted_vectors, component_sizes

        # update and retry
        min_size_cur = new_min_size




def Leaf_stripping_general_image(
    A_input,
    K=3,
    min_component_size=25,
    b=3,
    max_steps=10000,
    d=5,
    threshold_by="order",
    select_by="initial",
    recover=False,
    recover_mode="batch",
    mode="True",
    beta: float = 0.8,
    check_protected_removal: bool = False,  # <--- NEW: check protected nodes for removal
    # --------------------- snapshot controls ---------------------  # <<< CHANGED
    return_snapshots: bool = False,                                 # <<< CHANGED
    snapshot_dir: str | None = None,                                # <<< CHANGED
    max_snapshots: int = 20,                                        # <<< CHANGED
    layout_seed: int = 0,                                           # <<< CHANGED
    dpi: int = 140,                                                 # <<< CHANGED
):
    """
    Identical behavior to your original function, but can optionally return
    the last `max_snapshots` graph snapshots of the pruning process.

    When `return_snapshots=True`, the function returns one extra item at the end:
        (... original returns ..., snapshot_paths)
    where `snapshot_paths` is a list[str] of up to `max_snapshots` PNG files,
    ordered from oldest to newest (i.e., **last 20** steps kept).
    """
    n = A_input.shape[0]
    if A_input.shape[0] != A_input.shape[1]:
        raise ValueError("A_input must be square.")
    if not np.all(A_input == A_input.T):
        raise ValueError("A_input must be symmetric (undirected).")

    G_full = nx.from_numpy_array(A_input)

    # ---- centrality / degree preprocessing (same as your latest logic) ----
    initial_degrees_vec = A_input.sum(axis=1).astype(float).ravel()  # shape (n,)
    if isinstance(select_by, (list, np.ndarray)):
        c = np.asarray(select_by, dtype=float)
        if c.shape != (n,):
            raise ValueError(f"`select_by` vector must have shape ({n},)")
        if np.any(c < 0) or np.isnan(c).any():
            raise ValueError("`select_by` vector must be non-negative and finite.")
    else:
        c = initial_degrees_vec

    uniq, inv = np.unique(c, return_inverse=True)
    disc = inv.astype(int)   # 0 for smallest c, larger ints for larger c
    initial_degrees = {i: int(disc[i]) for i in range(n)}

    # Top-b selection
    if threshold_by == "deg":
        top_b_nodes = {node for node, deg in initial_degrees.items() if deg > b}
    else:
        sorted_nodes_by_degree = sorted(initial_degrees.items(), key=lambda x: -x[1])
        top_b_nodes = {node for node, _ in sorted_nodes_by_degree[:b]}

    # --------------------- snapshot helpers ---------------------             # <<< CHANGED
    snapshots = deque(maxlen=max_snapshots)                                     # <<< CHANGED
    if return_snapshots:                                                        # <<< CHANGED
        if snapshot_dir is None:                                                # <<< CHANGED
            snapshot_dir = os.path.join(os.getcwd(), "prune_snapshots")         # <<< CHANGED
        os.makedirs(snapshot_dir, exist_ok=True)                                # <<< CHANGED
        # fixed layout for consistency across frames                             # <<< CHANGED
        pos = nx.spring_layout(G_full, seed=layout_seed)                        # <<< CHANGED

        def _save_snapshot(A_mat, removed_batches, step_tag):                   # <<< CHANGED
            """Draw current live graph; removed nodes in light gray."""         # <<< CHANGED
            removed = set()                                                     # <<< CHANGED
            for bch in removed_batches:                                         # <<< CHANGED
                removed.update(bch)                                             # <<< CHANGED
            live_nodes = [i for i in range(n) if i not in removed]              # <<< CHANGED
            G_now = nx.from_numpy_array(A_mat)                                  # <<< CHANGED
            fig = plt.figure(figsize=(6, 6), dpi=dpi)                           # <<< CHANGED
            ax = plt.gca(); ax.axis("off")                                      # <<< CHANGED
            # edges and live nodes                                               # <<< CHANGED
            nx.draw_networkx_edges(G_now, pos, edgelist=G_now.edges(),          # <<< CHANGED
                                    width=0.5, alpha=0.25, ax=ax)               # <<< CHANGED
            nx.draw_networkx_nodes(G_now, pos, nodelist=live_nodes,             # <<< CHANGED
                                    node_size=12, linewidths=0,                 # <<< CHANGED
                                    node_color="#1f77b4", alpha=0.9, ax=ax)     # <<< CHANGED
            # removed nodes (faded)                                              # <<< CHANGED
            if removed:                                                         # <<< CHANGED
                nx.draw_networkx_nodes(G_full, pos, nodelist=list(removed),     # <<< CHANGED
                                        node_size=10, linewidths=0,             # <<< CHANGED
                                        node_color="#d9d9d9", alpha=0.6, ax=ax) # <<< CHANGED
            # optional: highlight top-b nodes that are still live                # <<< CHANGED
            tb_live = list(top_b_nodes.intersection(live_nodes))                # <<< CHANGED
            if tb_live:                                                         # <<< CHANGED
                nx.draw_networkx_nodes(G_now, pos, nodelist=tb_live,            # <<< CHANGED
                                        node_size=22, linewidths=0.6,           # <<< CHANGED
                                        edgecolors="#222222",                   # <<< CHANGED
                                        node_color="#ffcc66", alpha=0.95, ax=ax)# <<< CHANGED
            fname = os.path.join(snapshot_dir, f"prune_{step_tag}.png")         # <<< CHANGED
            fig.savefig(fname, bbox_inches="tight"); plt.close(fig)             # <<< CHANGED
            snapshots.append(fname)                                             # <<< CHANGED
    # ------------------------------------------------------------------------

    # Start at the user-specified size threshold; shrink by beta until success
    min_size_cur = int(min_component_size)
    attempt_idx = 0  # <<< CHANGED

    while True:
        # -------- one attempt with current size threshold --------
        A = A_input.copy()
        removed_batches = []
        predicted_labels = np.zeros(n, dtype=int)
        final_components = []
        last_removed_degrees = []
        removal_levels = []
        cutoff = 0

        if return_snapshots:                             # <<< CHANGED
            _save_snapshot(A, removed_batches,           # <<< CHANGED
                           f"attempt{attempt_idx}_step0_min{min_size_cur}")  # <<< CHANGED

        for step in range(max_steps):
            if removed_batches:
                cutoff = removal_levels[-1] if removal_levels else 0

            remaining_nodes = [i for i in range(n) if not any(i in batch for batch in removed_batches)]
            g_current = nx.from_numpy_array(A).subgraph(remaining_nodes)

            qsize = lambda comp: len(comp)

            components_all = [comp for comp in nx.connected_components(g_current) if qsize(comp) >= min_size_cur]
            components = [comp for comp in components_all if any(node in top_b_nodes for node in comp)]

            if len(components) >= K:
                final_components = components
                break

            live_nodes = remaining_nodes
            if len(live_nodes) <= 1:
                break

            # Directly get removable nodes (nodes in cycles)
            removable_set = Removable_node(A, mode)
            removable_nodes = [i for i in live_nodes if i in removable_set]

            # Optionally check if any protected nodes can be removed to create K components
            if check_protected_removal and mode == "True":
                protected_set = set(live_nodes) - removable_set
                if protected_set:
                    # Optimization: only search protected nodes with degree <= min degree of removable nodes
                    if removable_nodes:
                        deg_array = np.array([initial_degrees[i] for i in range(n)])
                        min_removable_deg = min(deg_array[i] for i in removable_nodes)
                        extra_removable = Find_removable_protected_nodes(
                            A, live_nodes, protected_set, top_b_nodes,
                            min_size_cur, cutoff, initial_degrees, K, min_removable_degree=min_removable_deg
                        )
                    else:
                        extra_removable = Find_removable_protected_nodes(
                            A, live_nodes, protected_set, top_b_nodes,
                            min_size_cur, cutoff, initial_degrees, K
                        )
                    if extra_removable:
                        print(f"INFO: Found {len(extra_removable)} protected node(s) whose removal creates K components: {sorted(extra_removable)}")
                        # Add to removable set
                        removable_set = removable_set.union(extra_removable)
                        removable_nodes = [i for i in live_nodes if i in removable_set]

            if not removable_nodes:
                print("WARNING: No removable nodes found (all remaining nodes are protected).")
                print("         Attempting manual removal to break graph into components...")
                G = nx.from_numpy_array(A).subgraph(live_nodes).copy()

                components_before_removal = [comp for comp in nx.connected_components(G) if len(comp) >= min_size_cur]
                components_before_removal = [comp for comp in components_before_removal if any(node in top_b_nodes for node in comp)]
                if len(components_before_removal) >= K:
                    final_components = components_before_removal
                    break

                degrees = {node: G.degree(node) for node in G.nodes()}
                sorted_nodes = [node for node, deg in sorted(degrees.items(), key=lambda x: x[1])]
                found = False
                for r in range(1, min(K, len(sorted_nodes))):
                    for nodes_to_remove in combinations(sorted_nodes, r):
                        G_temp = G.copy()
                        G_temp.remove_nodes_from(nodes_to_remove)
                        new_components = [comp for comp in nx.connected_components(G_temp) if qsize(comp) >= min_size_cur]
                        new_components = [comp for comp in new_components if any(node in top_b_nodes for node in comp)]
                        if len(new_components) >= K:
                            print(f"         Successfully removed {len(nodes_to_remove)} protected node(s) manually: {sorted(nodes_to_remove)}")
                            for node in nodes_to_remove:
                                A[node, :] = 0
                                A[:, node] = 0
                            removed_batches.append(set(nodes_to_remove))
                            final_components = new_components
                            found = True
                            if return_snapshots:  # <<< CHANGED
                                _save_snapshot(A, removed_batches, f"attempt{attempt_idx}_step{step+1}_heur")  # <<< CHANGED
                            break
                    if found:
                        break
                if not found and not final_components:
                    final_components = components_before_removal
                break

            G_current = nx.from_numpy_array(A).subgraph(live_nodes)
            loop_count = len(nx.cycle_basis(G_current))

            if loop_count < d:
                deg_array = np.array([initial_degrees[i] for i in range(n)])
                sorted_nodes = sorted(removable_nodes, key=lambda x: deg_array[x])
                if sorted_nodes:
                    node = sorted_nodes[0]
                    A[node, :] = 0
                    A[:, node] = 0
                    removed_batches.append({node})
                    if return_snapshots:  # <<< CHANGED
                        _save_snapshot(A, removed_batches, f"attempt{attempt_idx}_step{step+1}_leaf1")  # <<< CHANGED
                    continue
                else:
                    break

            deg_array = np.array([initial_degrees[i] for i in range(n)])
            min_deg = min(deg_array[i] for i in removable_nodes)
            leaves = {i for i in removable_nodes if deg_array[i] == min_deg}
            removal_levels.append(min_deg)
            if not leaves:
                break

            for leaf in leaves:
                A[leaf, :] = 0
                A[:, leaf] = 0
                last_removed_degrees.append(initial_degrees[leaf])

            degrees_now = A.sum(axis=1)
            deg0_now = {i for i in live_nodes if degrees_now[i] == 0 and all(i not in batch for batch in removed_batches)}
            all_removed = leaves.union(deg0_now)
            removed_batches.append(all_removed)

            for node in deg0_now:
                A[node, :] = 0
                A[:, node] = 0
                last_removed_degrees.append(initial_degrees[node])

            if return_snapshots:  # <<< CHANGED
                _save_snapshot(A, removed_batches, f"attempt{attempt_idx}_step{step+1}_leafk")  # <<< CHANGED

        # -------- post-processing for this attempt --------
        component_sizes = []
        final_components_sorted = sorted(final_components, key=lambda x: -len(x))
        top_K_components = final_components_sorted[:K]

        for label, comp in enumerate(top_K_components, start=1):
            for node in comp:
                predicted_labels[node] = label
            component_sizes.append(len(comp))

        cluster_count = len(top_K_components)

        # If success, return results from this attempt
        if cluster_count >= K:
            if not recover:
                if return_snapshots:                         # <<< CHANGED
                    return cluster_count, predicted_labels, component_sizes, list(snapshots)  # <<< CHANGED
                return cluster_count, predicted_labels, component_sizes

            if recover_mode == "distance_threshold":
                if not last_removed_degrees:
                    raise ValueError("No node was removed; cannot determine threshold.")
                threshold = min(max(last_removed_degrees), b)
                final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(
                    A_input, cluster_count, predicted_labels, threshold
                )
                if return_snapshots:                         # <<< CHANGED
                    return cluster_count, final_labels, component_sizes, list(snapshots)  # <<< CHANGED
                return cluster_count, final_labels, component_sizes

            # Default recover (unchanged except for snapshots return)
            predicted_vectors = np.zeros((n, K))
            for i in range(n):
                if predicted_labels[i] > 0:
                    predicted_vectors[i, predicted_labels[i] - 1] = 1

            for batch_idx, batch in enumerate(reversed(removed_batches)):
                batch = set(batch)
                batch_vec = predicted_vectors.copy()
                for node in batch:
                    neighbors = np.where(A_input[node] > 0)[0]
                    neighbor_vecs = batch_vec[neighbors]
                    mask = neighbor_vecs.sum(axis=1) > 0
                    if mask.sum() > 0:
                        avg_vec = neighbor_vecs[mask].mean(axis=0)
                        predicted_vectors[node] = avg_vec / avg_vec.sum()

            leftover_nodes = np.where(predicted_vectors.sum(axis=1) == 0)[0]
            G_leftover = nx.from_numpy_array(A_input).subgraph(leftover_nodes)
            components_rest = list(nx.connected_components(G_leftover))
            for comp in components_rest:
                comp = list(comp)
                all_neighbors = set()
                for node in comp:
                    neighbors = np.where(A_input[node] > 0)[0]
                    all_neighbors.update(neighbors)
                neighbor_vecs = predicted_vectors[list(all_neighbors)]
                mask = neighbor_vecs.sum(axis=1) > 0
                if mask.sum() > 0:
                    avg_vec = neighbor_vecs[mask].mean(axis=0)
                    avg_vec = avg_vec / avg_vec.sum()
                    for node in comp:
                        predicted_vectors[node] = avg_vec

            if return_snapshots:                             # <<< CHANGED
                return cluster_count, predicted_vectors, component_sizes, list(snapshots)  # <<< CHANGED
            return cluster_count, predicted_vectors, component_sizes

        # -------- shrink step: not enough components, try again with beta*size --------
        new_min_size = max(1, int(np.floor(beta * min_size_cur)))
        if new_min_size == min_size_cur:
            if not recover:
                if return_snapshots:                         # <<< CHANGED
                    return cluster_count, predicted_labels, component_sizes, list(snapshots)  # <<< CHANGED
                return cluster_count, predicted_labels, component_sizes

            if recover_mode == "distance_threshold":
                if not last_removed_degrees:
                    raise ValueError("No node was removed; cannot determine threshold.")
                threshold = max(last_removed_degrees)
                final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(
                    A_input, cluster_count, predicted_labels, threshold
                )
                if return_snapshots:                         # <<< CHANGED
                    return cluster_count, final_labels, component_sizes, list(snapshots)  # <<< CHANGED
                return cluster_count, final_labels, component_sizes

            predicted_vectors = np.zeros((n, K))
            for i in range(n):
                if predicted_labels[i] > 0:
                    predicted_vectors[i, predicted_labels[i] - 1] = 1
            for batch_idx, batch in enumerate(reversed(removed_batches)):
                batch = set(batch)
                batch_vec = predicted_vectors.copy()
                for node in batch:
                    neighbors = np.where(A_input[node] > 0)[0]
                    neighbor_vecs = batch_vec[neighbors]
                    mask = neighbor_vecs.sum(axis=1) > 0
                    if mask.sum() > 0:
                        avg_vec = neighbor_vecs[mask].mean(axis=0)
                        predicted_vectors[node] = avg_vec / avg_vec.sum()
            leftover_nodes = np.where(predicted_vectors.sum(axis=1) == 0)[0]
            G_leftover = nx.from_numpy_array(A_input).subgraph(leftover_nodes)
            components_rest = list(nx.connected_components(G_leftover))
            for comp in components_rest:
                comp = list(comp)
                all_neighbors = set()
                for node in comp:
                    neighbors = np.where(A_input[node] > 0)[0]
                    all_neighbors.update(neighbors)
                neighbor_vecs = predicted_vectors[list(all_neighbors)]
                mask = neighbor_vecs.sum(axis=1) > 0
                if mask.sum() > 0:
                    avg_vec = neighbor_vecs[mask].mean(axis=0)
                    avg_vec = avg_vec / avg_vec.sum()
                    for node in comp:
                        predicted_vectors[node] = avg_vec

            if return_snapshots:                             # <<< CHANGED
                return cluster_count, predicted_vectors, component_sizes, list(snapshots)  # <<< CHANGED
            return cluster_count, predicted_vectors, component_sizes

        # update and retry
        min_size_cur = new_min_size
        attempt_idx += 1  # <<< CHANGED



def Leaf_stripping_postCT(gibbs_results, components):
    """
    Robust version that handles missing nodes in components
    """
    n = len(gibbs_results['node_tree_coo'])
    K = len(components)
    
    # Initialize storage
    component_posteriors = [np.zeros(len(comp)) for comp in components]
    root_counts = np.zeros(n)
    assignment_counts = np.zeros((n, K))
    
    # Create component lookup dictionary with existence check
    node_to_component = {}
    component_nodes = set()  # Track all nodes that should be in components
    for k, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = k
            component_nodes.add(node)
    
    valid_samples = 0
    
    # Process each sample
    for sample in gibbs_results['samples']:
        # Get roots and verify they exist in components
        sample_roots = sample['roots']
        
        # Skip sample if any roots are missing from components
        missing_roots = [root for root in sample_roots if root not in node_to_component]
        if missing_roots:
            continue
            
        root_components = [node_to_component[root] for root in sample_roots]
        
        # Verify this sample has exactly one root per component
        if len(set(root_components)) != len(sample_roots):
            continue
            
        valid_samples += 1
        
        # Count root occurrences
        for root in sample_roots:
            root_counts[root] += 1
        
        # Process assignments
        for node, comp_idx in enumerate(sample['assignments']):
            if node >= n:  # Skip if node index is out of bounds
                continue
                
            if comp_idx < K:  # Only count valid components
                # Find which root's component this assignment belongs to
                for root, root_comp in zip(sample_roots, root_components):
                    if comp_idx == root_comp:
                        assignment_counts[node, comp_idx] += 1
                        break
    
    # Compute probabilities
    total_samples = valid_samples if valid_samples > 0 else 1
    
    # 1. Component-wise posteriors
    for k, comp in enumerate(components):
        comp_nodes = list(comp)
        comp_root_sum = root_counts[comp_nodes].sum()
        if comp_root_sum > 0:
            comp_root_probs = root_counts[comp_nodes] / comp_root_sum
        else:
            comp_root_probs = np.zeros(len(comp_nodes))
        component_posteriors[k] = comp_root_probs
    
    # 2. Global root probabilities
    global_root_sum = root_counts.sum()
    if global_root_sum > 0:
        global_posterior = root_counts / global_root_sum
    else:
        global_posterior = np.zeros(n)
    
    # 3. Assignment probabilities
    assignment_sums = assignment_counts.sum(axis=1, keepdims=True)
    assignment_sums[assignment_sums == 0] = 1  # Avoid division by zero
    assignment_probs = assignment_counts / assignment_sums
    
    # 4. Component purity
    purity_scores = None
    if 'node_tree_coo' in gibbs_results:
        purity_scores = []
        for k in range(K):
            comp_nodes = list(components[k])
            if len(comp_nodes) > 0 and gibbs_results['node_tree_coo'].shape[1] > k:
                purity = np.mean(np.argmax(gibbs_results['node_tree_coo'][comp_nodes], axis=1) == k)
                purity_scores.append(purity)
    
    return {
        'component_posteriors': component_posteriors,
        'global_posterior': global_posterior,
        'root_probs': root_counts / total_samples,
        'assignment_probs': assignment_probs,
        'component_purity': purity_scores if 'node_tree_coo' in gibbs_results else None,
        'valid_samples': valid_samples,
        'total_samples': len(gibbs_results['samples'])
    }




def generate_pa_mixture_graph(n, K, theta, ratio=None):
    """
    Generate K preferential attachment trees with noise edges added between any node pairs.

    Parameters
    ----------
    n : int
        Total number of nodes.
    K : int
        Number of clusters (components).
    theta : float
        Probability of adding a noise edge between any pair of nodes.
    ratio : list of float, optional
        List of proportions for node count in each cluster. Must sum to 1. If None, use equal sizes.

    Returns
    -------
    graf : igraph.Graph
        The generated graph with PA trees and noise edges.
    true_components : list of list of int
        Node indices for each true component.
    true_assignments : np.ndarray
        Cluster assignment for each node (values in {0, ..., K-1}).
    """

    if ratio is None:
        ratio = [1 / K] * K
    assert len(ratio) == K and np.isclose(sum(ratio), 1.0), "ratio must sum to 1 and have length K"

    # Step 1: Determine sizes per cluster
    sizes = [int(np.floor(r * n)) for r in ratio]
    while sum(sizes) < n:
        sizes[np.argmin(sizes)] += 1

    graf = ig.Graph()
    true_components = []
    true_assignments = np.zeros(n, dtype=int)

    node_offset = 0
    for k in range(K):
        size = sizes[k]
        if size == 0:
            continue

        ba = ig.Graph.Barabasi(n=size, m=1, directed=False)
        ba.vs["name"] = [i + node_offset for i in range(size)]

        if k == 0:
            graf = ba
        else:
            graf = graf.union(ba)

        component = list(range(node_offset, node_offset + size))
        true_components.append(component)
        true_assignments[component] = k
        node_offset += size

    # Step 2: Add noise edges between any two nodes with probability theta
    i, j = np.triu_indices(n, k=1)
    rand_values = np.random.rand(len(i))
    add_mask = rand_values < theta
    i_add = i[add_mask]
    j_add = j[add_mask]

    graf.add_edges(list(zip(i_add, j_add)))
    print(f"Added {len(i_add)} noise edges between all node pairs")

    # Step 3: Mark edges that are part of the original PA trees
    graf.es["tree"] = False
    for component in true_components:
        comp_set = set(component)
        for e in graf.es:
            if e.source in comp_set and e.target in comp_set:
                e["tree"] = True

    # Step 4: Initialize vertex attributes
    graf.vs["pa"] = None
    graf.vs["subtree_size"] = 1

    return graf, true_components, true_assignments




def generate_recursive_mixture_graph(n, K, theta, alpha, ratio=None):
    """
    Generate K recursive trees (Preferential Attachment) with noise edges added.
    
    The probability of a new node 't' attaching to an existing node 'i' is proportional 
    to degree(i) + alpha.

    Parameters
    ----------
    n : int
        Total number of nodes.
    K : int
        Number of clusters (components).
    theta : float
        Probability of adding a noise edge between any pair of nodes.
    alpha : float
        Preferential attachment parameter. 
        alpha=0 approaches standard PA (Barabasi-Albert-like).
        alpha->infinity approaches Uniform Attachment.
    ratio : list of float, optional
        List of proportions for node count in each cluster. Must sum to 1. 

    Returns
    -------
    graf : igraph.Graph
        The generated graph with recursive trees and noise edges.
    true_components : list of list of int
        Node indices for each true component.
    true_assignments : np.ndarray
        Cluster assignment for each node (values in {0, ..., K-1}).
    """

    if ratio is None:
        ratio = [1 / K] * K
    assert len(ratio) == K and np.isclose(sum(ratio), 1.0), "ratio must sum to 1 and have length K"

    # Step 1: Determine sizes per cluster
    sizes = [int(np.floor(r * n)) for r in ratio]
    while sum(sizes) < n:
        sizes[np.argmin(sizes)] += 1

    # Initialize the full graph with n vertices immediately to prevent indexing errors
    graf = ig.Graph(n, directed=False)
    
    true_components = []
    true_assignments = np.zeros(n, dtype=int)
    all_tree_edges = []

    node_offset = 0
    
    for k in range(K):
        size = sizes[k]
        if size == 0:
            continue
            
        # Recursive Tree Generation with alpha
        # We track local degrees for probability calculation
        local_degrees = np.zeros(size, dtype=float)
        
        # Edges for this specific component
        component_edges = []
        
        for t in range(1, size):
            # t is the local index of the new node being added
            
            # Calculate weights for existing nodes 0..t-1
            # Weight = Degree + alpha
            weights = local_degrees[:t] + alpha
            
            weights_sum = weights.sum()
            
            if weights_sum > 0:
                probs = weights / weights_sum
            else:
                # Fallback for the very first edge or if alpha cancels out degrees perfectly
                probs = np.ones(t) / t
            
            # Select parent based on probability
            parent_local = np.random.choice(t, p=probs)
            
            # Update degrees (undirected: +1 for both nodes)
            local_degrees[t] += 1
            local_degrees[parent_local] += 1
            
            # Add edge with global offsets
            source_global = t + node_offset
            target_global = parent_local + node_offset
            component_edges.append((source_global, target_global))

        # Add these edges to our main list
        all_tree_edges.extend(component_edges)

        # Track component indices
        component = list(range(node_offset, node_offset + size))
        true_components.append(component)
        true_assignments[component] = k
        node_offset += size

    # Add all tree edges to the graph at once
    graf.add_edges(all_tree_edges)
    
    # Mark strictly the edges we just added as tree edges
    # (We do this before noise to avoid confusion if noise duplicates an edge)
    graf.es["tree"] = True 

    # Step 2: Add noise edges between any two nodes with probability theta
    # We generate all potential pairs, select subset, and add them
    # Note: simple graphs usually don't want self-loops or multi-edges.
    # igraph handles adding edges fine, but we might create multi-edges if 
    # a noise edge lands exactly on a tree edge.
    
    i, j = np.triu_indices(n, k=1)
    
    # Randomly select pairs based on theta
    # Using float32 for memory efficiency if n is large
    rand_values = np.random.rand(len(i))
    add_mask = rand_values < theta
    
    i_add = i[add_mask]
    j_add = j[add_mask]
    
    noise_edges = list(zip(i_add, j_add))
    
    if noise_edges:
        graf.add_edges(noise_edges)
        # The newly added edges will have attributes initialized to None/default
        # We need to ensure the 'tree' attribute for these new edges is False
        
        # The indices of the new edges start after the last existing edge
        num_tree_edges = len(all_tree_edges)
        total_edges = graf.ecount()
        
        # Set 'tree' = False for the noise edges
        # We can slice the edge sequence
        graf.es[num_tree_edges:]["tree"] = False

    print(f"Added {len(noise_edges)} noise edges between all node pairs")

    # Step 4: Initialize vertex attributes (matching your request)
    graf.vs["pa"] = None
    graf.vs["subtree_size"] = 1
    
    # Give nodes names matching their index to ensure compatibility with downstream code
    graf.vs["name"] = [str(i) for i in range(n)]

    return graf, true_components, true_assignments




def generate_recursive_forest(n, K, alpha, ratio=None):
    """
    Generate K recursive trees (Preferential Attachment) without noise.
    
    The probability of a new node 't' attaching to an existing node 'i' is proportional 
    to degree(i) + alpha.

    Parameters
    ----------
    n : int
        Total number of nodes.
    K : int
        Number of clusters (components).
    alpha : float
        Preferential attachment parameter. 
        alpha=0 approaches standard PA (Barabasi-Albert-like).
        alpha->infinity approaches Uniform Attachment.
    ratio : list of float, optional
        List of proportions for node count in each cluster. Must sum to 1. 

    Returns
    -------
    graf : igraph.Graph
        The generated forest with K recursive trees (no noise edges).
    true_components : list of list of int
        Node indices for each true component.
    true_assignments : np.ndarray
        Cluster assignment for each node (values in {0, ..., K-1}).
    """
    
    if ratio is None:
        ratio = [1 / K] * K
    assert len(ratio) == K and np.isclose(sum(ratio), 1.0), "ratio must sum to 1 and have length K"

    # Step 1: Determine sizes per cluster
    sizes = [int(np.floor(r * n)) for r in ratio]
    while sum(sizes) < n:
        sizes[np.argmin(sizes)] += 1

    # Initialize the full graph with n vertices
    graf = ig.Graph(n, directed=False)
    
    true_components = []
    true_assignments = np.zeros(n, dtype=int)
    all_tree_edges = []

    node_offset = 0
    
    for k in range(K):
        size = sizes[k]
        if size == 0:
            continue
            
        # Recursive Tree Generation with alpha
        local_degrees = np.zeros(size, dtype=float)
        component_edges = []
        
        for t in range(1, size):
            # Calculate weights for existing nodes 0..t-1
            weights = local_degrees[:t] + alpha
            weights_sum = weights.sum()
            
            if weights_sum > 0:
                probs = weights / weights_sum
            else:
                probs = np.ones(t) / t
            
            # Select parent based on probability
            parent_local = np.random.choice(t, p=probs)
            
            # Update degrees (undirected: +1 for both nodes)
            local_degrees[t] += 1
            local_degrees[parent_local] += 1
            
            # Add edge with global offsets
            source_global = t + node_offset
            target_global = parent_local + node_offset
            component_edges.append((source_global, target_global))

        # Add these edges to our main list
        all_tree_edges.extend(component_edges)

        # Track component indices
        component = list(range(node_offset, node_offset + size))
        true_components.append(component)
        true_assignments[component] = k
        node_offset += size

    # Add all tree edges to the graph
    if all_tree_edges:
        graf.add_edges(all_tree_edges)
        graf.es["tree"] = True 

    # Initialize vertex attributes
    graf.vs["pa"] = None
    graf.vs["subtree_size"] = 1
    graf.vs["name"] = [str(i) for i in range(n)]

    return graf, true_components, true_assignments


def add_noise_edges(graf, theta, true_components=None):
    """
    Add noise edges to an existing graph with probability theta.
    
    Parameters
    ----------
    graf : igraph.Graph
        The input graph (typically a forest of trees).
    theta : float
        Probability of adding a noise edge between any pair of nodes.
    true_components : list of list of int, optional
        Node indices for each true component (for informational purposes).

    Returns
    -------
    graf : igraph.Graph
        The graph with noise edges added (modified in place).
    """
    
    n = graf.vcount()
    num_tree_edges = graf.ecount()
    
    # Generate all potential pairs
    i, j = np.triu_indices(n, k=1)
    
    # Randomly select pairs based on theta
    rand_values = np.random.rand(len(i))
    add_mask = rand_values < theta
    
    i_add = i[add_mask]
    j_add = j[add_mask]
    
    noise_edges = list(zip(i_add, j_add))
    
    if noise_edges:
        graf.add_edges(noise_edges)
        
        # Mark tree edges and noise edges appropriately
        # Ensure all edges have the 'tree' attribute
        if num_tree_edges > 0 and "tree" not in graf.es.attributes():
            graf.es["tree"] = False
            graf.es[:num_tree_edges]["tree"] = True
        
        # Set 'tree' = False for the newly added noise edges
        graf.es[num_tree_edges:]["tree"] = False

    print(f"Added {len(noise_edges)} noise edges between all node pairs")
    
    return graf



def generate_ua_mixture_graph(n, K, theta, ratio=None):
    """
    Generate K uniform attachment trees with noise edges added between any node pairs.

    Parameters
    ----------
    n : int
        Total number of nodes.
    K : int
        Number of clusters (components).
    theta : float
        Probability of adding a noise edge between any pair of nodes.
    ratio : list of float, optional
        List of proportions for node count in each cluster. Must sum to 1. If None, use equal sizes.

    Returns
    -------
    graf : igraph.Graph
        The generated graph with UA trees and noise edges.
    true_components : list of list of int
        Node indices for each true component.
    true_assignments : np.ndarray
        Cluster assignment for each node (values in {0, ..., K-1}).
    """

    if ratio is None:
        ratio = [1 / K] * K
    assert len(ratio) == K and np.isclose(sum(ratio), 1.0), "ratio must sum to 1 and have length K"

    # Step 1: Determine sizes per cluster
    sizes = [int(np.floor(r * n)) for r in ratio]
    while sum(sizes) < n:
        sizes[np.argmin(sizes)] += 1

    graf = ig.Graph()
    true_components = []
    true_assignments = np.zeros(n, dtype=int)

    node_offset = 0
    for k in range(K):
        size = sizes[k]
        if size == 0:
            continue

        # Uniform Attachment: node i connects to a uniformly randomly chosen earlier node
        edges = []
        for i in range(1, size):
            j = np.random.randint(0, i)
            edges.append((i, j))
        ua = ig.Graph(edges=edges, directed=False)
        ua.add_vertices(size - ua.vcount())  # Ensure all vertices exist
        ua.vs["name"] = [i + node_offset for i in range(size)]

        if k == 0:
            graf = ua
        else:
            graf = graf.union(ua)

        component = list(range(node_offset, node_offset + size))
        true_components.append(component)
        true_assignments[component] = k
        node_offset += size

    # Step 2: Add noise edges between any two nodes with probability theta
    i, j = np.triu_indices(n, k=1)
    rand_values = np.random.rand(len(i))
    add_mask = rand_values < theta
    i_add = i[add_mask]
    j_add = j[add_mask]

    graf.add_edges(list(zip(i_add, j_add)))
    print(f"Added {len(i_add)} noise edges between all node pairs")

    # Step 3: Mark edges that are part of the original UA trees
    graf.es["tree"] = False
    for component in true_components:
        comp_set = set(component)
        for e in graf.es:
            if e.source in comp_set and e.target in comp_set:
                e["tree"] = True

    # Step 4: Initialize vertex attributes
    graf.vs["pa"] = None
    graf.vs["subtree_size"] = 1

    return graf, true_components, true_assignments
