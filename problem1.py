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
    raise NotImplementedError


def assign_clusters(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Assign each point to nearest centroid (Euclidean distance).

    Args:
        X: Data matrix of shape (N, d).
        C: Centroids of shape (K, d).

    Returns:
        Integer labels z of shape (N,), values in {0, ..., K-1}.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def cluster_purity(y_true: np.ndarray, z_pred: np.ndarray) -> float:
    """Compute cluster purity.

    Args:
        y_true: Ground-truth labels, shape (N,).
        z_pred: Predicted cluster labels, shape (N,).

    Returns:
        Purity score as float in [0, 1].
    """
    raise NotImplementedError
