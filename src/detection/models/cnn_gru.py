"""
CNN-GRU hybrid sequence model for temporal attack pattern detection.

Architecture:
  Input sequence of flow feature vectors (seq_len, feature_dim)
    → 1D-CNN feature map (local pattern extraction)
    → GRU layers         (temporal dependency modelling)
    → Attention pooling  (focus on most suspicious timesteps)
    → Fully-connected head → attack probability
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import structlog

logger = structlog.get_logger(__name__)

INPUT_DIM = 46    # flow (29) + statistical (17) features
SEQ_LEN = 20      # packets per sequence
NUM_CLASSES = 5   # benign, DoS, probe, R2L, U2R


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden)
        scores = self.attn(x).squeeze(-1)          # (batch, seq)
        weights = F.softmax(scores, dim=-1)         # (batch, seq)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)


class CNNGRUDetector(nn.Module):
    """
    Hybrid CNN-GRU model that extracts local patterns via 1D convolutions
    before modelling temporal dependencies with a bidirectional GRU.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        cnn_channels: int = 64,
        cnn_kernel: int = 3,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        gru_input_dim = cnn_channels * 2
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.attention = AttentionPooling(gru_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        cnn_in = x.permute(0, 2, 1)           # (batch, input_dim, seq_len)
        cnn_out = self.cnn(cnn_in)             # (batch, channels, seq_len)
        gru_in = cnn_out.permute(0, 2, 1)     # (batch, seq_len, channels)

        gru_out, _ = self.gru(gru_in)         # (batch, seq_len, hidden*2)
        pooled = self.attention(gru_out)       # (batch, hidden*2)
        logits = self.classifier(pooled)       # (batch, num_classes)

        probs = F.softmax(logits, dim=-1)
        return logits, probs


class CNNGRUInference:
    """Wraps CNNGRUDetector for single-sample and batch inference."""

    CLASS_LABELS = ["benign", "DoS", "probe", "R2L", "U2R"]

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = CNNGRUDetector().to(self.device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            self._load(model_path)
        else:
            logger.warning("cnn_gru_no_checkpoint", path=model_path)

    def _load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info("cnn_gru_loaded", path=path)

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Args:
            sequence: (seq_len, feature_dim) numpy array

        Returns:
            (predicted_class, attack_probability)
        """
        if sequence.shape[0] < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - sequence.shape[0], sequence.shape[1]))
            sequence = np.vstack([sequence, pad])
        elif sequence.shape[0] > SEQ_LEN:
            sequence = sequence[-SEQ_LEN:]

        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, probs = self.model(tensor)

        probs_np = probs.squeeze(0).cpu().numpy()
        predicted_idx = int(probs_np.argmax())
        attack_prob = float(1.0 - probs_np[0])  # 1 - P(benign)

        return self.CLASS_LABELS[predicted_idx], attack_prob

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Args:
            sequences: (batch, seq_len, feature_dim)

        Returns:
            attack_probabilities: (batch,)
        """
        tensor = torch.FloatTensor(sequences).to(self.device)
        with torch.no_grad():
            _, probs = self.model(tensor)
        attack_probs = 1.0 - probs[:, 0].cpu().numpy()
        return attack_probs


def train_epoch(
    model: CNNGRUDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
