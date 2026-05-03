"""
Online (incremental) learning module.
Uses the River library for concept-drift-aware incremental updates
to lightweight classifiers that run alongside the heavyweight PyTorch models.

The online model acts as a fast-adapting complement: it is updated on every
labelled sample (from analyst feedback or high-confidence detections) and
can generalize to new patterns within minutes rather than hours.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import structlog
from river import ensemble, metrics, preprocessing
from river.drift import ADWIN

from src.config import settings
from src.features.flow_features import FEATURE_NAMES
from src.features.statistical_features import STAT_FEATURE_NAMES

logger = structlog.get_logger(__name__)

ALL_FEATURES = FEATURE_NAMES + STAT_FEATURE_NAMES


@dataclass
class LearnerStats:
    total_samples: int = 0
    recent_accuracy: float = 0.0
    recent_kappa: float = 0.0
    last_updated: float = field(default_factory=time.time)


class OnlineNIDSClassifier:
    """
    Adaptive Random Forest classifier that learns incrementally from every
    new labelled observation. Wraps River's ARF implementation.
    """

    CLASS_MAP = {
        "benign": 0,
        "DoS": 1,
        "probe": 2,
        "R2L": 3,
        "U2R": 4,
    }
    REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._scaler = preprocessing.StandardScaler()
        self._model = ensemble.AdaptiveRandomForestClassifier(
            n_models=10,
            drift_detector=ADWIN(delta=0.001),
            warning_detector=ADWIN(delta=0.01),
            seed=42,
        )
        self._accuracy = metrics.Accuracy()
        self._kappa = metrics.CohenKappa()
        self._stats = LearnerStats()
        self._save_path = model_path or (settings.MODEL_PATH + "/online_arf.joblib")

        if Path(self._save_path).exists():
            self._load()

    def _features_to_dict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(ALL_FEATURES, feature_vector)}

    def learn(self, feature_vector: np.ndarray, label: str) -> None:
        """Update the model with a single labelled observation."""
        if label not in self.CLASS_MAP:
            raise ValueError(f"Unknown label: {label!r}")

        x = self._features_to_dict(feature_vector)
        y = self.CLASS_MAP[label]

        # Predict before learning (prequential evaluation)
        y_hat = self._model.predict_one(x)
        if y_hat is not None:
            self._accuracy.update(y, y_hat)
            self._kappa.update(y, y_hat)

        self._scaler.learn_one(x)
        x_scaled = self._scaler.transform_one(x)
        self._model.learn_one(x_scaled, y)

        self._stats.total_samples += 1
        self._stats.recent_accuracy = self._accuracy.get()
        self._stats.recent_kappa = self._kappa.get()
        self._stats.last_updated = time.time()

        if self._stats.total_samples % 1000 == 0:
            logger.info(
                "online_learner_stats",
                samples=self._stats.total_samples,
                accuracy=round(self._stats.recent_accuracy, 4),
                kappa=round(self._stats.recent_kappa, 4),
            )
            self._save()

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Returns:
            predicted_label, probability_dict
        """
        x = self._features_to_dict(feature_vector)
        x_scaled = self._scaler.transform_one(x)

        probs = self._model.predict_proba_one(x_scaled)
        if not probs:
            return "benign", {k: 0.0 for k in self.CLASS_MAP}

        predicted_class = max(probs, key=probs.get)
        label = self.REVERSE_CLASS_MAP.get(predicted_class, "benign")

        prob_map = {
            self.REVERSE_CLASS_MAP.get(k, str(k)): v
            for k, v in probs.items()
        }
        return label, prob_map

    def attack_probability(self, feature_vector: np.ndarray) -> float:
        """1 - P(benign)"""
        _, probs = self.predict(feature_vector)
        return 1.0 - probs.get("benign", 0.0)

    def bulk_learn(self, X: np.ndarray, labels: List[str]) -> None:
        for vec, label in zip(X, labels):
            self.learn(vec, label)

    def _save(self) -> None:
        Path(self._save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "scaler": self._scaler,
                "stats": self._stats,
            },
            self._save_path,
        )
        logger.debug("online_learner_saved", path=self._save_path)

    def _load(self) -> None:
        state = joblib.load(self._save_path)
        self._model = state["model"]
        self._scaler = state["scaler"]
        self._stats = state["stats"]
        logger.info(
            "online_learner_loaded",
            path=self._save_path,
            samples=self._stats.total_samples,
        )

    @property
    def stats(self) -> LearnerStats:
        return self._stats
