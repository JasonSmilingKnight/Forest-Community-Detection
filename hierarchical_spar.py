"""SPAR pruning and recovery variant for recursive hierarchical case studies.
Function bodies are preserved from the existing hierarchy implementation.
Source SHA256: 5ff892fe01f4bb4ee8bfb1faa9a1159e2b589a3f62a9c5b3271299edaf6f350c
"""
import numpy as np
import networkx as nx
import random
from itertools import combinations

def Protected_node(A, mode='True'):
    """
    Returns nodes that are NOT in any cycle (i.e., nodes that should be protected from removal).
    Removable nodes are the complement: nodes that ARE in cycles.
    """
    if mode != 'True':
        return set()
    G = nx.from_numpy_array(A)
    cycle_nodes = set()
    for cycle in nx.cycle_basis(G):
        cycle_nodes.update(cycle)
    n = A.shape[0]
    protected = set(range(n)) - cycle_nodes
    return protected

def Removable_node(A, mode='True'):
    """
    Directly returns nodes that ARE in any cycle (i.e., nodes that can be removed).
    This is the complement of Protected_node.
    """
    if mode != 'True':
        return set(range(A.shape[0]))
    G = nx.from_numpy_array(A)
    cycle_nodes = set()
    for cycle in nx.cycle_basis(G):
        cycle_nodes.update(cycle)
    return cycle_nodes

def Find_removable_protected_nodes(A, live_nodes, protected_nodes, top_b_nodes, min_size_cur, cutoff, initial_degrees, K, min_removable_degree=None):
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
    if min_removable_degree is not None:
        protected_nodes = {node for node in protected_nodes if initial_degrees[node] <= min_removable_degree}
    if not protected_nodes:
        return set()
    base_components = [comp for comp in nx.connected_components(G) if qsize(comp) >= min_size_cur]
    base_components = [comp for comp in base_components if any((n in top_b_nodes for n in comp))]
    base_count = len(base_components)
    valid_protected_removals = set()
    try:
        articulation_points = set(nx.articulation_points(G))
        candidates = protected_nodes.intersection(articulation_points)
    except:
        candidates = protected_nodes
    if not candidates and len(protected_nodes) <= 10:
        candidates = protected_nodes
    for node in candidates:
        G_temp = G.copy()
        G_temp.remove_node(node)
        components = [comp for comp in nx.connected_components(G_temp) if qsize(comp) >= min_size_cur]
        components = [comp for comp in components if any((n in top_b_nodes for n in comp))]
        new_count = len(components)
        if new_count >= base_count + 1:
            valid_protected_removals.add(node)
    return valid_protected_removals

def recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels, degree_threshold):
    n = A_input.shape[0]
    degrees = A_input.sum(axis=1)
    G = nx.from_numpy_array(A_input)
    if cluster_count == 0:
        return (np.zeros(n, dtype=int), [])
    reference_nodes_lists = []
    for k in range(cluster_count):
        target_label = k + 1
        indices = np.where((predicted_labels == target_label) & (degrees > degree_threshold))[0]
        if len(indices) > 0:
            reference_nodes_lists.append(indices.tolist())
        else:
            fallback = np.where(predicted_labels == target_label)[0]
            reference_nodes_lists.append([int(fallback[0])] if len(fallback) > 0 else [])
    reference_nodes_vector = np.zeros(n, dtype=int)
    for k, nodes in enumerate(reference_nodes_lists):
        if nodes:
            reference_nodes_vector[nodes] = k + 1
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
        if len(dist_list) > 0:
            min_dist = min(dist_list)
            candidates = [k for k, d in enumerate(dist_list) if d == min_dist]
            chosen_cluster = random.choice(candidates)
            new_labels[i] = chosen_cluster + 1
    return (new_labels, reference_nodes_lists)

def hierarchical_leaf_stripping(A_input, K=3, min_component_size=25, b=3, max_steps=10000, d=5, threshold_by='order', select_by='initial', recover=False, recover_mode='batch', mode='True', beta: float=0.5, check_protected_removal: bool=False, recovery_threshold_divisor: float=5.0):
    n = A_input.shape[0]
    initial_degrees = {i: A_input[i].sum() for i in range(n)}
    if threshold_by == 'deg':
        top_b_nodes = {node for node, deg in initial_degrees.items() if deg > b}
        core_deg_threshold = b
    else:
        sorted_nodes_by_degree = sorted(initial_degrees.items(), key=lambda x: -x[1])
        top_b_nodes = {node for node, _ in sorted_nodes_by_degree[:b * K]}
        if len(sorted_nodes_by_degree) >= b * 2:
            core_deg_threshold = sorted_nodes_by_degree[b * 2 - 1][1]
        else:
            core_deg_threshold = sorted_nodes_by_degree[-1][1] if sorted_nodes_by_degree else 0
    min_size_cur = int(min_component_size)
    while True:
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
            remaining_nodes = [i for i in range(n) if not any((i in batch for batch in removed_batches))]
            g_current = nx.from_numpy_array(A).subgraph(remaining_nodes)
            qsize = lambda comp: len(comp)
            components_all = [comp for comp in nx.connected_components(g_current) if qsize(comp) >= min_size_cur]
            components = [comp for comp in components_all if any((node in top_b_nodes for node in comp))]
            if len(components) >= K:
                final_components = components
                break
            live_nodes = remaining_nodes
            if len(live_nodes) <= 1:
                break
            removable_set = Removable_node(A, mode)
            removable_nodes = [i for i in live_nodes if i in removable_set]
            if check_protected_removal and mode == 'True':
                protected_set = set(live_nodes) - removable_set
                if protected_set:
                    if removable_nodes:
                        deg_array = np.array([initial_degrees[i] for i in range(n)])
                        min_removable_deg = min((deg_array[i] for i in removable_nodes))
                        extra_removable = Find_removable_protected_nodes(A, live_nodes, protected_set, top_b_nodes, min_size_cur, cutoff, initial_degrees, K, min_removable_degree=min_removable_deg)
                    else:
                        extra_removable = Find_removable_protected_nodes(A, live_nodes, protected_set, top_b_nodes, min_size_cur, cutoff, initial_degrees, K)
                    if extra_removable:
                        print(f'INFO: Found {len(extra_removable)} protected node(s) whose removal creates K components: {sorted(extra_removable)}')
                        removable_set = removable_set.union(extra_removable)
                        removable_nodes = [i for i in live_nodes if i in removable_set]
            if not removable_nodes:
                if check_protected_removal:
                    print('INFO: No removable nodes found. Relying on beta-shrinking to retry with smaller threshold.')
                    break
                print('WARNING: No removable nodes found (all remaining nodes are protected).')
                print('         Attempting manual removal to break graph into components...')
                G = nx.from_numpy_array(A).subgraph(live_nodes).copy()
                components_before_removal = [comp for comp in nx.connected_components(G) if len(comp) >= min_size_cur]
                components_before_removal = [comp for comp in components_before_removal if any((node in top_b_nodes for node in comp))]
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
                        new_components = [comp for comp in new_components if any((node in top_b_nodes for node in comp))]
                        if len(new_components) >= K:
                            print(f'         Successfully removed {len(nodes_to_remove)} protected node(s) manually: {sorted(nodes_to_remove)}')
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
                    A[node, :] = 0
                    A[:, node] = 0
                    removed_batches.append({node})
                    continue
                else:
                    break
            deg_array = np.array([initial_degrees[i] for i in range(n)])
            min_deg = min((deg_array[i] for i in removable_nodes))
            leaves = {i for i in removable_nodes if deg_array[i] == min_deg}
            removal_levels.append(min_deg)
            if not leaves:
                break
            for leaf in leaves:
                A[leaf, :] = 0
                A[:, leaf] = 0
                last_removed_degrees.append(initial_degrees[leaf])
            degrees_now = A.sum(axis=1)
            deg0_now = {i for i in live_nodes if degrees_now[i] == 0 and all((i not in batch for batch in removed_batches))}
            all_removed = leaves.union(deg0_now)
            removed_batches.append(all_removed)
            for node in deg0_now:
                A[node, :] = 0
                A[:, node] = 0
                last_removed_degrees.append(initial_degrees[node])
        component_sizes = []
        component_max_degrees = []
        for comp in final_components:
            core_nodes = [node for node in comp if initial_degrees[node] >= cutoff]
            if core_nodes:
                max_degree_in_core = max((initial_degrees[node] for node in core_nodes))
            else:
                max_degree_in_core = max((initial_degrees[node] for node in comp)) if comp else 0
            component_max_degrees.append((max_degree_in_core, comp))
        component_max_degrees_sorted = sorted(component_max_degrees, key=lambda x: -x[0])
        top_K_components = [comp for _, comp in component_max_degrees_sorted[:K]]
        for label, comp in enumerate(top_K_components, start=1):
            for node in comp:
                predicted_labels[node] = label
            component_sizes.append(len(comp))
        cluster_count = len(top_K_components)
        if cluster_count >= K:
            print('last removed degree is', max(last_removed_degrees))
            if not recover:
                return (cluster_count, predicted_labels, component_sizes)
            if recover_mode == 'distance_threshold':
                if not last_removed_degrees:
                    raise ValueError('No node was removed; cannot determine threshold.')
                max_removed_deg = max(last_removed_degrees)
                threshold = int(min(core_deg_threshold / 2, cutoff))
                print(f'Recovery threshold calculation:')
                print(f'  core_deg_threshold = {core_deg_threshold}')
                print(f'  cutoff = {cutoff}')
                print(f'  max(last_removed_degrees) = {max_removed_deg}')
                print(f'  Recovery threshold = min(core_deg_threshold / 2, cutoff) = {threshold}')
                final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels, threshold)
                return (cluster_count, final_labels, predicted_labels, component_sizes, reference_nodes_per_cluster)
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
            return (cluster_count, predicted_vectors, component_sizes)
        new_min_size = max(1, int(np.floor(beta * min_size_cur)))
        if new_min_size == min_size_cur:
            if not recover:
                return (cluster_count, predicted_labels, component_sizes)
            if recover_mode == 'distance_threshold':
                if not last_removed_degrees:
                    raise ValueError('No node was removed; cannot determine threshold.')
                threshold = max(last_removed_degrees)
                final_labels, reference_nodes_per_cluster = recover_with_filtered_distance_by_threshold(A_input, cluster_count, predicted_labels, threshold)
                return (cluster_count, final_labels, predicted_labels, component_sizes)
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
            return (cluster_count, predicted_vectors, component_sizes)
        min_size_cur = new_min_size
        print(min_size_cur)
