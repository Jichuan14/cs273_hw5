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
    generator = torch.Generator().manual_seed(seed)
    if kind == "gaussian":
        noise = torch.randn(x.shape, generator=generator) * severity
        return torch.clamp(x + noise, 0, 1)
    elif kind == "mask":
        mask = torch.rand(x.shape, generator=generator) > severity
        return torch.clamp(x * mask, 0, 1)


class DenoisingAutoencoder(nn.Module):
    """Simple fully-connected denoising autoencoder.

    Encoder: Linear -> ReLU -> Linear(latent)
    Decoder: Linear -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstruction with shape equal to input shape."""
        return self.decoder(self.encoder(x))


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
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        noisy_X = add_noise(X_tensor, noise_kind, severity, seed)
        noisy_X_hat = model(noisy_X)
        loss = loss_fn(noisy_X_hat, X_tensor)

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
    }

@torch.no_grad()
def denoise_and_score(
    model: nn.Module,
    X_test: np.ndarray,
    noise_kind: str,
    severity: float,
    seed: int,
) -> float:
    """Add noise to X_test, denoise with model, and return reconstruction MSE."""
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    noisy_X = add_noise(X_tensor, noise_kind, severity, seed)
    noisy_X_hat = model(noisy_X)
    return torch.nn.functional.mse_loss(noisy_X_hat, X_tensor).item()
