"""SCORE and the existing Gao-style local K-medians benchmark."""
import numpy as np

def _weighted_median_rows(X_cluster, weights_cluster):
    """Coordinatewise weighted medians."""
    order = np.argsort(X_cluster, axis=0)
    sorted_vals = np.take_along_axis(X_cluster, order, axis=0)
    sorted_weights = weights_cluster[order]
    cum_weights = np.cumsum(sorted_weights, axis=0)
    half = cum_weights[-1] / 2.0
    idx = np.argmax(cum_weights >= half, axis=0)
    return sorted_vals[idx, np.arange(X_cluster.shape[1])]

def _weighted_kmeans_init(X, sample_weight, K, n_init=10, max_iter=100, random_state=0):
    """Initialize labels using weighted K-means with multiple restarts."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    best_labels, best_inertia = (None, np.inf)
    for _ in range(n_init):
        centers = X[rng.choice(n, size=K, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.empty((n, K))
            for k in range(K):
                diff = X - centers[k]
                dists[:, k] = np.einsum('ij,ij->i', diff, diff)
            new_labels = np.argmin(dists, axis=1)
            new_centers = centers.copy()
            for k in range(K):
                mask = new_labels == k
                if np.any(mask):
                    new_centers[k] = np.average(X[mask], axis=0, weights=sample_weight[mask])
            converged = np.array_equal(new_labels, labels) and np.allclose(new_centers, centers)
            labels, centers = (new_labels, new_centers)
            if converged:
                break
        dists = np.empty((n, K))
        for k in range(K):
            diff = X - centers[k]
            dists[:, k] = np.einsum('ij,ij->i', diff, diff)
        inertia = np.sum(sample_weight * np.min(dists, axis=1))
        if inertia < best_inertia:
            best_inertia, best_labels = (inertia, labels)
    return best_labels

def score_clustering_python(A, K, threshold=None, n_init=10, random_state=0):
    """SCORE: clipped eigenvector ratios followed by K-means."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('score_clustering_python: A must be square')
    if not np.allclose(A, A.T):
        raise ValueError('score_clustering_python: A must be symmetric')
    if not np.allclose(np.diag(A), 0):
        raise ValueError('score_clustering_python: A must have zero diagonal')
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError('score_clustering_python: A must be binary')
    if not 2 <= K <= n:
        raise ValueError('score_clustering_python: require 2 <= K <= n')
    if threshold is None:
        threshold = np.log(n)
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(-np.abs(eigvals))[:K]
    top_vecs = eigvecs[:, order]
    leading = top_vecs[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = top_vecs[:, 1:] / leading[:, None]
    ratios = np.nan_to_num(ratios, nan=0.0, posinf=threshold, neginf=-threshold)
    ratios = np.clip(ratios, -threshold, threshold)
    labels = _weighted_kmeans_init(ratios, np.ones(n), K, n_init=n_init, random_state=random_state)
    if len(np.unique(labels)) < K:
        raise RuntimeError('score_clustering_python: k-means produced fewer than K nonempty clusters')
    return labels

def weighted_kmedians_l1(X, sample_weight, K, random_state=0, max_iter=20, tol=1e-06):
    """Local weighted L1 K-medians; no approximation guarantee is claimed."""
    n, d = X.shape
    sample_weight = np.asarray(sample_weight, dtype=float)
    labels = _weighted_kmeans_init(X, sample_weight, K, random_state=random_state)

    def cluster_centers(labels):
        centers = np.zeros((K, d))
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                centers[k] = _weighted_median_rows(X[mask], sample_weight[mask])
        return centers

    def repair_empty_clusters(labels, centers):
        counts = np.bincount(labels, minlength=K)
        for k in range(K):
            if counts[k] > 0:
                continue
            own_centers = centers[labels]
            loss = np.sum(np.abs(X - own_centers), axis=1) * sample_weight
            eligible = counts[labels] >= 2
            if not np.any(eligible):
                raise RuntimeError('weighted_kmedians_l1: cannot repair an empty cluster without emptying another (too few active points for K)')
            donor = np.argmax(np.where(eligible, loss, -np.inf))
            counts[labels[donor]] -= 1
            counts[k] += 1
            labels[donor] = k
        return labels

    def compute_dists(centers):
        dists = np.empty((n, K))
        for k in range(K):
            dists[:, k] = np.sum(np.abs(X - centers[k]), axis=1)
        return dists
    centers = cluster_centers(labels)
    labels = repair_empty_clusters(labels, centers)
    centers = cluster_centers(labels)
    dists = compute_dists(centers)
    prev_obj = np.sum(sample_weight * dists[np.arange(n), labels])
    for _ in range(max_iter):
        new_labels = np.argmin(dists, axis=1)
        new_labels = repair_empty_clusters(new_labels, centers)
        new_centers = cluster_centers(new_labels)
        new_dists = compute_dists(new_centers)
        obj = np.sum(sample_weight * new_dists[np.arange(n), new_labels])
        converged = np.array_equal(new_labels, labels)
        rel_improvement = (prev_obj - obj) / prev_obj if np.isfinite(prev_obj) and prev_obj > 0 else np.inf
        labels, centers, dists, prev_obj = (new_labels, new_centers, new_dists, obj)
        if converged or 0 <= rel_improvement <= tol:
            break
    if len(np.unique(labels)) < K:
        raise RuntimeError('weighted_kmedians_l1: could not maintain K nonempty clusters')
    return labels

def gao_dcbm_clustering(A, K, tau_factor=5.0, n_refinements=1, random_state=0, max_iter=20, tol=1e-06, verbose=True):
    """Trim, embed, cluster by local K-medians, then refine labels."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('gao_dcbm_clustering: A must be square')
    if not np.allclose(A, A.T):
        raise ValueError('gao_dcbm_clustering: A must be symmetric')
    if not np.allclose(np.diag(A), 0):
        raise ValueError('gao_dcbm_clustering: A must have zero diagonal')
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError('gao_dcbm_clustering: A must be binary')
    if not 2 <= K <= n:
        raise ValueError('gao_dcbm_clustering: require 2 <= K <= n')
    if tau_factor <= 0:
        raise ValueError('gao_dcbm_clustering: tau_factor must be positive')
    if n_refinements < 1:
        raise ValueError('gao_dcbm_clustering: n_refinements must be >= 1')
    A = A.copy()
    degrees = A.sum(axis=1)
    average_degree = degrees.mean()
    tau = tau_factor * average_degree
    trimmed = degrees > tau
    n_trimmed = int(trimmed.sum())
    A_trim = A.copy()
    A_trim[trimmed, :] = 0.0
    A_trim[:, trimmed] = 0.0
    eigvals_full, eigvecs_full = np.linalg.eigh(A_trim)
    order = np.argsort(-np.abs(eigvals_full))[:K]
    eigvals, eigvecs = (eigvals_full[order], eigvecs_full[:, order])
    P_hat = eigvecs @ np.diag(eigvals) @ eigvecs.T
    numerical_tolerance = 1e-08
    w = np.sum(np.abs(P_hat), axis=1)
    is_s0 = w <= numerical_tolerance
    nonzero = ~is_s0
    n_active = int(nonzero.sum())
    if n_active < K:
        raise RuntimeError(f'gao_dcbm_clustering: only {n_active} rows have positive L1 norm, need at least K={K}')
    X = np.zeros_like(P_hat)
    X[nonzero] = P_hat[nonzero] / w[nonzero, None]
    init_labels_active = weighted_kmedians_l1(X[nonzero], w[nonzero], K, random_state=random_state, max_iter=max_iter, tol=tol)
    labels = np.full(n, -1, dtype=int)
    labels[nonzero] = init_labels_active
    prev_group_indices, prev_group_sizes = (None, None)
    for round_idx in range(n_refinements):
        group_indices = [np.where(labels == u)[0] for u in range(K)]
        group_sizes = np.array([len(idx) for idx in group_indices], dtype=float)
        if np.any(group_sizes == 0):
            if round_idx == 0:
                raise RuntimeError('gao_dcbm_clustering: initialization produced an empty community before any refinement')
            group_indices, group_sizes = (prev_group_indices, prev_group_sizes)
        s = np.zeros((n, K))
        for u in range(K):
            if group_sizes[u] == 0:
                continue
            s[:, u] = A[:, group_indices[u]].sum(axis=1) / group_sizes[u]
        labels = np.argmax(s, axis=1)
        prev_group_indices, prev_group_sizes = (group_indices, group_sizes)
    labels = labels.astype(int)
    if len(np.unique(labels)) < K:
        raise RuntimeError('gao_dcbm_clustering: refinement produced fewer than K communities')
    return labels
