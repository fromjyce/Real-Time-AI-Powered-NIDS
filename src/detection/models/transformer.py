"""
Transformer-based NIDS model (replaces the GRU with self-attention).
Uses positional encoding + multi-head self-attention + feed-forward blocks.
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import structlog

logger = structlog.get_logger(__name__)

INPUT_DIM = 46
SEQ_LEN = 20
NUM_CLASSES = 5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TransformerDetector(nn.Module):
    """
    Transformer encoder stack for sequence-level intrusion detection.
    Uses the [CLS] token representation for classification.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        batch = x.size(0)

        x = self.input_projection(x)

        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (batch, seq+1, d_model)
        x = self.pos_encoding(x)

        if padding_mask is not None:
            cls_mask = torch.zeros(batch, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        cls_out = x[:, 0, :]                     # (batch, d_model)
        logits = self.classifier(cls_out)
        probs = F.softmax(logits, dim=-1)

        return logits, probs


class TransformerInference:
    CLASS_LABELS = ["benign", "DoS", "probe", "R2L", "U2R"]

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = TransformerDetector().to(self.device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            logger.info("transformer_loaded", path=model_path)
        else:
            logger.warning("transformer_no_checkpoint", path=model_path)

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        if sequence.shape[0] < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - sequence.shape[0], sequence.shape[1]))
            sequence = np.vstack([sequence, pad])
        else:
            sequence = sequence[-SEQ_LEN:]

        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, probs = self.model(tensor)

        probs_np = probs.squeeze(0).cpu().numpy()
        predicted_idx = int(probs_np.argmax())
        attack_prob = float(1.0 - probs_np[0])

        return self.CLASS_LABELS[predicted_idx], attack_prob

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        tensor = torch.FloatTensor(sequences).to(self.device)
        with torch.no_grad():
            _, probs = self.model(tensor)
        return (1.0 - probs[:, 0].cpu().numpy())
