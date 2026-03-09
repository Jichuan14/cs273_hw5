"""CS 273P HW5 - Problem 1 starter code.

Implement k-means clustering from scratch.
"""

from __future__ import annotations

import numpy as np


def init_centroids(
    X: np.ndarray,
    K: int,
    seed: int,
    method: str = "kmeans++",
) -> np.ndarray:
    """Initialize K centroids.

    Args:
        X: Data matrix of shape (N, d).
        K: Number of clusters.
        seed: Random seed for deterministic behavior.
        method: "random" or "kmeans++".

    Returns:
        Centroid matrix C of shape (K, d).
    """
    N,d = X.shape
    rng=np.random.default_rng(seed)
    if method == "random":
        indices = rng.choice(N, K, replace=False)
        return X[indices].copy()
    elif method == "kmeans++":
        C = np.zeros((K, d))
        C[0] = X[rng.choice(N)]
        for k in range(1, K):
            D = np.min(np.sum((X - C[:k])**2, axis=1), axis=1)
            probs = D / np.sum(D)
            C[k] = X[rng.choice(N, p=probs)]
        return C


def assign_clusters(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Assign each point to nearest centroid (Euclidean distance).

    Args:
        X: Data matrix of shape (N, d).
        C: Centroids of shape (K, d).

    Returns:
        Integer labels z of shape (N,), values in {0, ..., K-1}.
    """
    D = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    return np.argmin(D, axis=1)


def update_centroids(
    X: np.ndarray,
    z: np.ndarray,
    K: int,
    seed: int = 0,
) -> np.ndarray:
    """Update centroids as cluster means.

    If a cluster is empty, reinitialize its centroid by sampling one data point
    uniformly at random from X, using the provided seed for determinism.

    Args:
        X: Data matrix of shape (N, d).
        z: Cluster labels of shape (N,).
        K: Number of clusters.
        seed: Random seed for deterministic empty-cluster reinitialization.

    Returns:
        Updated centroids C_new of shape (K, d).
    """
    N,d = X.shape
    rng=np.random.default_rng(seed)
    C_new = np.zeros((K, d))
    for k in range(K):
        indices = np.where(z == k)[0]
        if len(indices) > 0:
            C_new[k] = np.mean(X[indices], axis=0)
        else:
            C_new[k] = X[rng.choice(N)]
    return C_new


def kmeans(
    X: np.ndarray,
    K: int,
    seed: int,
    max_iters: int = 100,
    tol: float = 1e-6,
    method: str = "kmeans++",
) -> dict:
    """Run Lloyd's algorithm for k-means.

    Must return a dict with at least:
    {
        "centroids": C,
        "labels": z,
        "inertia": J,
        "history": [...],
    }
    """
    C = init_centroids(X, K, seed, method)
    history = []
    for i in range(max_iters):
        z = assign_clusters(X, C)
        C_new = update_centroids(X, z, K, seed)
        history.append(C_new)
        if np.linalg.norm(C_new - C) < tol:
            break
        C = C_new
    return {
        "centroids": C,
        "labels": z,
        "inertia": np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - C, axis=2), axis=1)),
        "history": history,
    }


def cluster_purity(y_true: np.ndarray, z_pred: np.ndarray) -> float:
    """Compute cluster purity.

    Args:
        y_true: Ground-truth labels, shape (N,).
        z_pred: Predicted cluster labels, shape (N,).

    Returns:
        Purity score as float in [0, 1].
    """
    raise NotImplementedError
    matched_pairs = 0
    cluster_counts = np.unique(z_pred)
    for cluster in cluster_counts:
        true_labels = y_true[z_pred == cluster]
        true_counts = np.bincount(true_labels)
        matched_pairs += np.max(true_counts)
    return float(matched_pairs / len(y_true))
