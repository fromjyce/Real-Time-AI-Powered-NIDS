"""
Isolation Forest anomaly detector for zero-day / unknown attack detection.
Wraps scikit-learn's IsolationForest with a calibrated anomaly score in [0, 1].
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

logger = structlog.get_logger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest with RobustScaler preprocessing.
    The raw decision_function output is remapped to a [0, 1] anomaly score
    where 1.0 = certain anomaly and 0.0 = certain normal.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: str = "auto",
        contamination: float = 0.05,
        max_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self._trained = False
        self._score_min: float = -1.0
        self._score_max: float = 1.0

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        raw_scores = self.model.decision_function(X_scaled)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())

        self._trained = True
        logger.info(
            "isolation_forest_fitted",
            samples=len(X),
            score_range=(round(self._score_min, 4), round(self._score_max, 4)),
        )
        return self

    def _normalize_score(self, raw: np.ndarray) -> np.ndarray:
        """Map decision_function scores to [0, 1] anomaly probability."""
        clipped = np.clip(raw, self._score_min, self._score_max)
        normalized = (clipped - self._score_min) / max(
            self._score_max - self._score_min, 1e-9
        )
        return 1.0 - normalized  # invert: high score = anomalous

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            labels: -1 (anomaly) or 1 (normal) per scikit-learn convention
            anomaly_scores: float in [0, 1], higher = more anomalous
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before calling predict()")

        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        raw_scores = self.model.decision_function(X_scaled)
        anomaly_scores = self._normalize_score(raw_scores)
        return labels, anomaly_scores

    def anomaly_score(self, x: np.ndarray) -> float:
        """Single-sample convenience wrapper."""
        _, scores = self.predict(x.reshape(1, -1))
        return float(scores[0])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "score_min": self._score_min,
            "score_max": self._score_max,
            "trained": self._trained,
        }
        joblib.dump(state, path)
        logger.info("isolation_forest_saved", path=path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = state["model"]
        obj.scaler = state["scaler"]
        obj._score_min = state["score_min"]
        obj._score_max = state["score_max"]
        obj._trained = state["trained"]
        logger.info("isolation_forest_loaded", path=path)
        return obj
