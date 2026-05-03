"""
Variational Autoencoder (VAE) for anomaly detection.
Trained on benign traffic only; reconstruction error serves as anomaly score.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import structlog

logger = structlog.get_logger(__name__)

FEATURE_DIM = 46


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, latent_dim)
        self.log_var_head = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_head(h), self.log_var_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list) -> None:
        super().__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VariationalAutoencoder(nn.Module):
    """
    VAE for network flow anomaly detection.
    During inference, samples with high reconstruction loss are flagged
    as anomalous (i.e., dissimilar to the benign distribution).
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        latent_dim: int = 16,
        hidden_dims: Optional[list] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    @staticmethod
    def loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(x_hat, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return (recon_loss + kl_loss) / x.size(0)


class AutoencoderDetector:
    """Inference wrapper for the VAE anomaly detector."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = VariationalAutoencoder().to(self.device)
        self.model.eval()
        self._threshold: Optional[float] = None
        self._threshold_percentile = threshold_percentile

        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self._threshold = state.get("threshold")
            logger.info("autoencoder_loaded", path=model_path, threshold=self._threshold)
        else:
            logger.warning("autoencoder_no_checkpoint", path=model_path)

    def calibrate_threshold(self, X_benign: np.ndarray) -> float:
        """Fit anomaly threshold from reconstruction errors on benign data."""
        errors = self._reconstruction_errors(X_benign)
        self._threshold = float(np.percentile(errors, self._threshold_percentile))
        logger.info("autoencoder_threshold_calibrated", threshold=self._threshold)
        return self._threshold

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        tensor = torch.FloatTensor(X).to(self.device)
        errors = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = tensor[i : i + batch_size]
                x_hat, _, _ = self.model(batch)
                err = F.mse_loss(x_hat, batch, reduction="none").mean(dim=1)
                errors.extend(err.cpu().numpy().tolist())
        return np.array(errors, dtype=np.float32)

    def anomaly_score(self, x: np.ndarray) -> float:
        """Returns anomaly score in [0, 1]."""
        err = self._reconstruction_errors(x.reshape(1, -1))[0]
        if self._threshold is None:
            return float(np.tanh(err / 100.0))
        return float(min(err / self._threshold, 1.0))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_errors(X)
        if self._threshold is None:
            return np.tanh(errors / 100.0)
        return np.clip(errors / self._threshold, 0.0, 1.0)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "threshold": self._threshold,
            },
            path,
        )
        logger.info("autoencoder_saved", path=path)
