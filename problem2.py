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
    raise NotImplementedError


def pca_transform(X: np.ndarray, model: dict) -> np.ndarray:
    """Project centered data onto principal components.

    Args:
        X: Data matrix of shape (N, d).
        model: Output of pca_fit.

    Returns:
        Latent representation Z of shape (N, k).
    """
    raise NotImplementedError


def pca_inverse(Z: np.ndarray, model: dict) -> np.ndarray:
    """Reconstruct data from PCA latent representation.

    Args:
        Z: Latent matrix of shape (N, k).
        model: Output of pca_fit.

    Returns:
        Reconstructed data Xhat of shape (N, d).
    """
    raise NotImplementedError


def min_k_for_variance(X: np.ndarray, var_target: float = 0.9) -> int:
    """Return the smallest k with cumulative explained variance ratio >= var_target.

    Args:
        X: Data matrix of shape (N, d).
        var_target: Target cumulative variance ratio in (0, 1].

    Returns:
        Integer k.
    """
    raise NotImplementedError


def reconstruction_mse(X: np.ndarray, Xhat: np.ndarray) -> float:
    """Compute mean squared reconstruction error per entry."""
    raise NotImplementedError
