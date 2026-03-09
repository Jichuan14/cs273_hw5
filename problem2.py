"""CS 273P HW5 - Problem 2 starter code.

Implement PCA-based compression and reconstruction from scratch.
"""

from __future__ import annotations

import numpy as np


def pca_fit(X: np.ndarray, k: int) -> dict:
    """Fit PCA using SVD and return model dict.

    Args:
        X: Data matrix of shape (N, d).
        k: Number of principal components.

    Returns:
        {
            "mu": mean vector of shape (d,),
            "components": top-k principal directions of shape (k, d),
            "explained_var": length-k array of explained variances,
        }
    """
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:k]
    explained_var = S[:k]**2 / (X.shape[0] - 1)
    return {
        "mu": mu,
        "components": components,
        "explained_var": explained_var,
    }


def pca_transform(X: np.ndarray, model: dict) -> np.ndarray:
    """Project centered data onto principal components.

    Args:
        X: Data matrix of shape (N, d).
        model: Output of pca_fit.

    Returns:
        Latent representation Z of shape (N, k).
    """
    mu = model["mu"]
    components = model["components"]
    X_centered = X - mu
    return X_centered @ components


def pca_inverse(Z: np.ndarray, model: dict) -> np.ndarray:
    """Reconstruct data from PCA latent representation.

    Args:
        Z: Latent matrix of shape (N, k).
        model: Output of pca_fit.

    Returns:
        Reconstructed data Xhat of shape (N, d).
    """
    mu = model["mu"]
    components = model["components"]
    return Z @ components.T + mu


def min_k_for_variance(X: np.ndarray, var_target: float = 0.9) -> int:
    """Return the smallest k with cumulative explained variance ratio >= var_target.

    Args:
        X: Data matrix of shape (N, d).
        var_target: Target cumulative variance ratio in (0, 1].

    Returns:
        Integer k.
    """
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    S = np.linalg.svd(X_centered, full_matrices=False)[1]
    explained_var = S**2 / (X.shape[0] - 1)
    var_ratio = np.cumsum(explained_var) / np.sum(explained_var)
    return np.argmax(var_ratio >= var_target) + 1


def reconstruction_mse(X: np.ndarray, Xhat: np.ndarray) -> float:
    """Compute mean squared reconstruction error per entry."""
    return np.mean((X - Xhat)**2)
