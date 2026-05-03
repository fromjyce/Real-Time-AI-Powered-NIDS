"""
Model training script.
Downloads the CICIDS-2017 / NSL-KDD dataset, preprocesses it,
trains the CNN-GRU sequence model and Isolation Forest,
then saves them to the models/ directory.

Usage:
  python scripts/train_models.py --dataset cicids --epochs 20 --batch-size 256
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import structlog

from src.config import settings
from src.detection.models.cnn_gru import CNNGRUDetector, train_epoch
from src.detection.models.isolation_forest import IsolationForestDetector
from src.detection.models.autoencoder import VariationalAutoencoder, AutoencoderDetector
from src.features.flow_features import FEATURE_NAMES
from src.features.statistical_features import STAT_FEATURE_NAMES

logger = structlog.get_logger(__name__)

ALL_FEATURES = FEATURE_NAMES + STAT_FEATURE_NAMES
SEQ_LEN = 20

CICIDS_LABEL_MAP = {
    "BENIGN": "benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DoS",
    "PortScan": "probe",
    "FTP-Patator": "R2L",
    "SSH-Patator": "R2L",
    "Web Attack – Brute Force": "R2L",
    "Web Attack – XSS": "R2L",
    "Web Attack – Sql Injection": "R2L",
    "Infiltration": "U2R",
    "Heartbleed": "U2R",
    "Bot": "DoS",
}

CLASS_TO_IDX = {"benign": 0, "DoS": 1, "probe": 2, "R2L": 3, "U2R": 4}


def load_cicids(data_dir: str) -> pd.DataFrame:
    csv_files = list(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("cicids_loaded", rows=len(combined), files=len(csv_files))
    return combined


def preprocess_cicids(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    label_col = " Label" if " Label" in df.columns else "Label"
    df["_label"] = df[label_col].map(CICIDS_LABEL_MAP).fillna("benign")
    df["_class"] = df["_label"].map(CLASS_TO_IDX).fillna(0).astype(int)

    # Map CICIDS column names to our feature names (best-effort)
    col_map = {
        " Total Fwd Packets": "packet_count",
        " Total Length of Fwd Packets": "total_bytes",
        " Flow Duration": "flow_duration",
        " Flow Bytes/s": "bytes_per_second",
        " Flow Packets/s": "packets_per_second",
        " Fwd Packet Length Mean": "mean_packet_size",
        " Fwd Packet Length Std": "std_packet_size",
        " SYN Flag Count": "syn_count",
        " ACK Flag Count": "ack_count",
        " FIN Flag Count": "fin_count",
        " RST Flag Count": "rst_count",
        " PSH Flag Count": "psh_count",
        " URG Flag Count": "urg_count",
        " Avg Fwd Segment Size": "fwd_bytes",
        " Avg Bwd Segment Size": "bwd_bytes",
    }
    for src, dst in col_map.items():
        if src in df.columns and dst in ALL_FEATURES:
            df[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0)

    for feat in ALL_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    X = df[ALL_FEATURES].values.astype(np.float32)
    y = df["_class"].values.astype(np.int64)

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=0.0)
    X = np.clip(X, -1e9, 1e9)

    return X, y


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    X_seq = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
    for i in range(n):
        start = max(0, i - seq_len + 1)
        seq = X[start : i + 1]
        X_seq[i, -len(seq):] = seq
    return X_seq, y


def train_cnn_gru(
    X_seq_train: np.ndarray,
    y_train: np.ndarray,
    X_seq_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    save_path: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("cnn_gru_training_start", device=str(device), epochs=epochs)

    model = CNNGRUDetector(input_dim=len(ALL_FEATURES)).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(
        torch.FloatTensor(X_seq_train),
        torch.LongTensor(y_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_seq_val),
        torch.LongTensor(y_val),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx)
                val_loss += criterion(logits, by).item()
                preds = logits.argmax(dim=1)
                correct += (preds == by).sum().item()
                total += len(by)

        val_loss /= max(len(val_loader), 1)
        val_acc = correct / max(total, 1)
        logger.info(
            "cnn_gru_epoch",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                save_path,
            )

    logger.info("cnn_gru_training_done", best_val_loss=round(best_val_loss, 4))


def train_anomaly_models(X_benign: np.ndarray, X_val: np.ndarray, model_dir: str) -> None:
    logger.info("anomaly_models_training_start", benign_samples=len(X_benign))

    iso = IsolationForestDetector(n_estimators=200, contamination=0.05)
    iso.fit(X_benign)
    iso.save(os.path.join(model_dir, "isolation_forest.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = AutoencoderDetector(device=str(device))
    ae.calibrate_threshold(X_benign)
    ae.save(os.path.join(model_dir, "autoencoder.pt"))

    logger.info("anomaly_models_training_done")


@click.command()
@click.option("--dataset", default="cicids", type=click.Choice(["cicids", "nsl-kdd"]))
@click.option("--data-dir", default="./data", help="Directory with raw CSVs")
@click.option("--model-dir", default=settings.MODEL_PATH)
@click.option("--epochs", default=20, type=int)
@click.option("--batch-size", default=256, type=int)
@click.option("--test-size", default=0.2, type=float)
@click.option("--skip-anomaly", is_flag=True)
def main(dataset, data_dir, model_dir, epochs, batch_size, test_size, skip_anomaly):
    logger.info("training_start", dataset=dataset, data_dir=data_dir)

    X, y = preprocess_cicids(load_cicids(data_dir))

    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    X_seq_train, y_seq_train = create_sequences(X_train, y_train)
    X_seq_test, y_seq_test = create_sequences(X_test, y_test)

    train_cnn_gru(
        X_seq_train, y_seq_train,
        X_seq_test, y_seq_test,
        epochs=epochs,
        batch_size=batch_size,
        save_path=os.path.join(model_dir, "cnn_gru.pt"),
    )

    if not skip_anomaly:
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        train_anomaly_models(X_benign, X_test, model_dir)

    logger.info("training_complete", model_dir=model_dir)


if __name__ == "__main__":
    main()
