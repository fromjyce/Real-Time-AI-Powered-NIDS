"""
Statistical feature extraction over a window of packets in a flow.
Computes inter-arrival time statistics and higher-order moments.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


STAT_FEATURE_NAMES = [
    "iat_mean",
    "iat_std",
    "iat_min",
    "iat_max",
    "iat_skewness",
    "iat_kurtosis",
    "iat_cv",             # coefficient of variation
    "pkt_size_entropy",
    "pkt_size_skewness",
    "pkt_size_kurtosis",
    "payload_entropy",
    "burst_count",
    "burst_mean_duration",
    "burst_mean_size",
    "active_time",
    "idle_time",
    "active_idle_ratio",
]

BURST_THRESHOLD_SECONDS = 0.1


class StatisticalFeatureExtractor:
    """
    Derives statistical features from the timing and size distributions
    of packets within a network flow.
    """

    def extract(self, packets: List[Dict]) -> Dict[str, float]:
        if len(packets) < 2:
            return {name: 0.0 for name in STAT_FEATURE_NAMES}

        timestamps = sorted(p.get("timestamp", 0.0) for p in packets)
        ip_lens = np.array([p.get("ip_len", 0) or 0 for p in packets], dtype=np.float32)
        payload_lens = np.array([p.get("payload_len", 0) or 0 for p in packets], dtype=np.float32)

        # Inter-arrival times
        iats = np.diff(timestamps).astype(np.float64)
        iat_mean = float(np.mean(iats))
        iat_std = float(np.std(iats))
        iat_min = float(np.min(iats))
        iat_max = float(np.max(iats))
        iat_skew = float(scipy_stats.skew(iats)) if len(iats) > 2 else 0.0
        iat_kurt = float(scipy_stats.kurtosis(iats)) if len(iats) > 2 else 0.0
        iat_cv = iat_std / max(iat_mean, 1e-9)

        # Packet size distribution
        pkt_size_entropy = self._entropy(ip_lens)
        pkt_skew = float(scipy_stats.skew(ip_lens)) if len(ip_lens) > 2 else 0.0
        pkt_kurt = float(scipy_stats.kurtosis(ip_lens)) if len(ip_lens) > 2 else 0.0

        payload_entropy = self._entropy(payload_lens)

        # Burst detection (packets arriving within BURST_THRESHOLD_SECONDS of each other)
        bursts = self._detect_bursts(timestamps)
        burst_count = len(bursts)
        burst_mean_dur = float(np.mean([b[1] - b[0] for b in bursts])) if bursts else 0.0
        burst_mean_size = float(np.mean([b[2] for b in bursts])) if bursts else 0.0

        # Active / idle periods
        active_time, idle_time = self._active_idle(timestamps)
        active_idle_ratio = active_time / max(idle_time, 1e-9)

        return {
            "iat_mean": iat_mean,
            "iat_std": iat_std,
            "iat_min": iat_min,
            "iat_max": iat_max,
            "iat_skewness": iat_skew,
            "iat_kurtosis": iat_kurt,
            "iat_cv": iat_cv,
            "pkt_size_entropy": pkt_size_entropy,
            "pkt_size_skewness": pkt_skew,
            "pkt_size_kurtosis": pkt_kurt,
            "payload_entropy": payload_entropy,
            "burst_count": float(burst_count),
            "burst_mean_duration": burst_mean_dur,
            "burst_mean_size": burst_mean_size,
            "active_time": active_time,
            "idle_time": idle_time,
            "active_idle_ratio": active_idle_ratio,
        }

    @staticmethod
    def _entropy(values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        unique, counts = np.unique(values, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    @staticmethod
    def _detect_bursts(timestamps: List[float]) -> List[Tuple[float, float, int]]:
        """Returns list of (start, end, packet_count) for each burst."""
        if not timestamps:
            return []
        bursts = []
        burst_start = timestamps[0]
        burst_count = 1

        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            if gap <= BURST_THRESHOLD_SECONDS:
                burst_count += 1
            else:
                if burst_count > 1:
                    bursts.append((burst_start, timestamps[i - 1], burst_count))
                burst_start = timestamps[i]
                burst_count = 1

        if burst_count > 1:
            bursts.append((burst_start, timestamps[-1], burst_count))

        return bursts

    @staticmethod
    def _active_idle(timestamps: List[float], idle_threshold: float = 1.0) -> Tuple[float, float]:
        if len(timestamps) < 2:
            return 0.0, 0.0

        active = 0.0
        idle = 0.0
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            if gap < idle_threshold:
                active += gap
            else:
                idle += gap

        return active, idle

    def to_vector(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features.get(name, 0.0) for name in STAT_FEATURE_NAMES], dtype=np.float32)
