"""CS 273P HW5 - Problem 3 starter code.

Implement a denoising autoencoder in PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def add_noise(
    x: torch.Tensor,
    kind: str,
    severity: float,
    seed: int,
) -> torch.Tensor:
    """Add deterministic noise to input tensor in [0, 1].

    Args:
        x: Input tensor of shape (B, d), values in [0, 1].
        kind: "gaussian" or "mask".
        severity: Noise magnitude (std for gaussian, mask prob for mask).
        seed: Random seed.

    Returns:
        Noisy tensor with values clipped to [0, 1].
    """
    raise NotImplementedError


class DenoisingAutoencoder(nn.Module):
    """Simple fully-connected denoising autoencoder.

    Encoder: Linear -> ReLU -> Linear(latent)
    Decoder: Linear -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstruction with shape equal to input shape."""
        raise NotImplementedError


def train_dae(
    model: nn.Module,
    X_train: np.ndarray,
    noise_kind: str,
    severity: float,
    seed: int,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> dict:
    """Train DAE to reconstruct clean inputs from noisy inputs.

    Returns:
        {
            "loss_history": [...],
            "final_loss": float,
        }
    """
    raise NotImplementedError


@torch.no_grad()
def denoise_and_score(
    model: nn.Module,
    X_test: np.ndarray,
    noise_kind: str,
    severity: float,
    seed: int,
) -> float:
    """Add noise to X_test, denoise with model, and return reconstruction MSE."""
    raise NotImplementedError
