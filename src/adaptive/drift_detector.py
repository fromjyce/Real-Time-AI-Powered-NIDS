"""
Concept drift detection using the ADWIN (Adaptive Windowing) algorithm
and Page-Hinkley test. Monitors model accuracy over time and triggers
retraining when a statistically significant distribution shift is detected.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DriftEvent:
    timestamp: float
    detector: str
    metric: str
    pre_drift_value: float
    post_drift_value: float
    n_samples: int
    description: str


class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) algorithm for concept drift detection.
    Maintains a dynamically-sized window and detects when two sub-windows
    have statistically different means (indicating drift).

    Reference: Bifet & Gavalda (2007).
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._window: Deque[float] = deque()
        self._total = 0.0
        self._n = 0
        self._drift_detected = False
        self._last_drift_at: Optional[int] = None

    def add(self, value: float) -> bool:
        self._window.append(value)
        self._total += value
        self._n += 1

        if self._n > 5:
            drift = self._check_drift()
            if drift:
                self._drift_detected = True
                self._last_drift_at = self._n
                return True

        self._drift_detected = False
        return False

    def _check_drift(self) -> bool:
        n = len(self._window)
        if n < 10:
            return False

        total = sum(self._window)
        mean_all = total / n
        epsilon_cut = math.sqrt((1.0 / (2 * n)) * math.log(4 * n / self.delta))

        cumsum = 0.0
        for i, val in enumerate(self._window):
            cumsum += val
            n0 = i + 1
            n1 = n - n0
            if n1 == 0:
                continue

            mean0 = cumsum / n0
            mean1 = (total - cumsum) / n1

            if abs(mean0 - mean1) > epsilon_cut:
                # Cut the window at this point (drop older sub-window)
                for _ in range(n0):
                    dropped = self._window.popleft()
                    self._total -= dropped
                self._n = len(self._window)
                return True

        return False

    @property
    def current_mean(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def reset(self) -> None:
        self._window.clear()
        self._total = 0.0
        self._n = 0


class PageHinkleyDetector:
    """
    Page-Hinkley test for change detection in sequential data.
    Sensitive to gradual drift; complements ADWIN for slow shifts.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._cumsum = 0.0
        self._min_cumsum = 0.0
        self._n = 0
        self._mean = 0.0

    def add(self, value: float) -> bool:
        self._n += 1
        self._mean = self.alpha * self._mean + (1 - self.alpha) * value

        self._cumsum += value - self._mean - self.delta
        self._min_cumsum = min(self._min_cumsum, self._cumsum)

        if self._cumsum - self._min_cumsum > self.threshold:
            return True
        return False

    def reset(self) -> None:
        self._cumsum = 0.0
        self._min_cumsum = 0.0
        self._n = 0
        self._mean = 0.0


class ModelDriftMonitor:
    """
    Monitors model performance metrics (accuracy, false positive rate, F1)
    using ADWIN and Page-Hinkley detectors per metric.
    Triggers a retraining callback when drift is confirmed.
    """

    def __init__(
        self,
        on_drift: Optional[Callable[[DriftEvent], None]] = None,
        adwin_delta: float = 0.002,
        ph_threshold: float = 50.0,
    ) -> None:
        self.on_drift = on_drift
        self._detectors: Dict[str, Tuple[ADWINDetector, PageHinkleyDetector]] = {}
        self._adwin_delta = adwin_delta
        self._ph_threshold = ph_threshold
        self._drift_log: List[DriftEvent] = []
        self._sample_counts: Dict[str, int] = {}
        self._pre_drift_values: Dict[str, float] = {}

    def _get_or_create(self, metric: str) -> Tuple[ADWINDetector, PageHinkleyDetector]:
        if metric not in self._detectors:
            self._detectors[metric] = (
                ADWINDetector(delta=self._adwin_delta),
                PageHinkleyDetector(threshold=self._ph_threshold),
            )
            self._sample_counts[metric] = 0
            self._pre_drift_values[metric] = 0.0
        return self._detectors[metric]

    def record(self, metric: str, value: float) -> bool:
        adwin, ph = self._get_or_create(metric)
        n = self._sample_counts[metric]

        if n == 0:
            self._pre_drift_values[metric] = value

        self._sample_counts[metric] += 1
        adwin_drift = adwin.add(value)
        ph_drift = ph.add(value)

        if adwin_drift or ph_drift:
            detected_by = []
            if adwin_drift:
                detected_by.append("ADWIN")
            if ph_drift:
                detected_by.append("PageHinkley")

            event = DriftEvent(
                timestamp=time.time(),
                detector="+".join(detected_by),
                metric=metric,
                pre_drift_value=self._pre_drift_values[metric],
                post_drift_value=adwin.current_mean,
                n_samples=self._sample_counts[metric],
                description=f"Drift in {metric}: {self._pre_drift_values[metric]:.4f} → {adwin.current_mean:.4f}",
            )
            self._drift_log.append(event)
            logger.warning(
                "concept_drift_detected",
                metric=metric,
                detector=event.detector,
                pre=event.pre_drift_value,
                post=event.post_drift_value,
                samples=event.n_samples,
            )

            self._pre_drift_values[metric] = adwin.current_mean
            adwin.reset()
            ph.reset()

            if self.on_drift:
                self.on_drift(event)

            return True

        return False

    def record_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        accuracy = float(np.mean(y_true == y_pred))
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))
        fn = float(np.sum((y_pred == 0) & (y_true == 1)))

        fpr = fp / max(fp + float(np.sum(y_true == 0)), 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        precision = tp / max(tp + fp, 1e-9)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        self.record("accuracy", accuracy)
        self.record("false_positive_rate", fpr)
        self.record("f1_score", f1)

    @property
    def drift_history(self) -> List[DriftEvent]:
        return list(self._drift_log)
