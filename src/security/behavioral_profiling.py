"""
Behavioral profiling — maintains a per-IP baseline of normal traffic
and flags deviations using a statistical z-score model.

Profiles are persisted to Redis for cross-process sharing and survive restarts.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from src.storage.redis_client import RedisClient
from src.config import settings

logger = structlog.get_logger(__name__)

PROFILE_KEY_PREFIX = "nids:profile:"
MIN_SAMPLES_FOR_BASELINE = 50
DEVIATION_THRESHOLD_Z = 3.5


@dataclass
class IPProfile:
    ip: str
    sample_count: int
    feature_means: List[float]
    feature_m2: List[float]          # Welford's online M2
    last_updated: float
    alert_count: int = 0

    def feature_stds(self) -> List[float]:
        if self.sample_count < 2:
            return [0.0] * len(self.feature_means)
        return [
            math.sqrt(m2 / (self.sample_count - 1)) if self.sample_count > 1 else 0.0
            for m2 in self.feature_m2
        ]

    def is_established(self) -> bool:
        return self.sample_count >= MIN_SAMPLES_FOR_BASELINE


class BehavioralProfiler:
    """
    Maintains Welford online statistics per source IP.
    Scores new observations against the established baseline.
    """

    PROFILE_FEATURE_KEYS = [
        "packets_per_second",
        "bytes_per_second",
        "mean_packet_size",
        "unique_dst_ports_5m",
        "conn_rate_5m",
        "failed_conn_ratio",
        "syn_ratio",
    ]

    def __init__(self, redis: Optional[RedisClient] = None) -> None:
        self._redis = redis or RedisClient()
        self._local_cache: Dict[str, IPProfile] = {}

    def _profile_key(self, ip: str) -> str:
        return f"{PROFILE_KEY_PREFIX}{ip}"

    def _load_profile(self, ip: str) -> Optional[IPProfile]:
        if ip in self._local_cache:
            return self._local_cache[ip]

        raw = self._redis.get(self._profile_key(ip))
        if raw:
            data = json.loads(raw)
            profile = IPProfile(**data)
            self._local_cache[ip] = profile
            return profile
        return None

    def _save_profile(self, profile: IPProfile) -> None:
        self._local_cache[profile.ip] = profile
        self._redis.set(
            self._profile_key(profile.ip),
            json.dumps(asdict(profile)),
            ttl=settings.REDIS_PROFILE_TTL,
        )

    def _extract_feature_vector(self, flow: Dict) -> List[float]:
        return [float(flow.get(k, 0.0)) for k in self.PROFILE_FEATURE_KEYS]

    def update(self, src_ip: str, flow: Dict) -> None:
        """Update per-IP Welford statistics with a new flow observation."""
        features = self._extract_feature_vector(flow)
        dim = len(features)

        profile = self._load_profile(src_ip)
        if profile is None:
            profile = IPProfile(
                ip=src_ip,
                sample_count=0,
                feature_means=[0.0] * dim,
                feature_m2=[0.0] * dim,
                last_updated=time.time(),
            )

        profile.sample_count += 1
        n = profile.sample_count
        for i, x in enumerate(features):
            delta = x - profile.feature_means[i]
            profile.feature_means[i] += delta / n
            delta2 = x - profile.feature_means[i]
            profile.feature_m2[i] += delta * delta2

        profile.last_updated = time.time()
        self._save_profile(profile)

    def score_deviation(self, src_ip: str, flow: Dict) -> Tuple[float, bool]:
        """
        Returns:
            (deviation_score, is_anomalous)
            deviation_score: max z-score across features
            is_anomalous: True if score > DEVIATION_THRESHOLD_Z and profile established
        """
        profile = self._load_profile(src_ip)
        if profile is None or not profile.is_established():
            return 0.0, False

        features = self._extract_feature_vector(flow)
        stds = profile.feature_stds()
        z_scores = []

        for i, x in enumerate(features):
            std = stds[i]
            if std < 1e-9:
                continue
            z = abs(x - profile.feature_means[i]) / std
            z_scores.append(z)

        if not z_scores:
            return 0.0, False

        max_z = float(max(z_scores))
        mean_z = float(np.mean(z_scores))
        deviation_score = 0.7 * max_z + 0.3 * mean_z

        is_anomalous = deviation_score > DEVIATION_THRESHOLD_Z
        if is_anomalous:
            profile.alert_count += 1
            self._save_profile(profile)
            logger.info(
                "behavioral_anomaly",
                ip=src_ip,
                deviation=round(deviation_score, 2),
                max_z=round(max_z, 2),
            )

        return deviation_score, is_anomalous

    def get_profile_summary(self, src_ip: str) -> Optional[Dict]:
        profile = self._load_profile(src_ip)
        if profile is None:
            return None
        return {
            "ip": profile.ip,
            "sample_count": profile.sample_count,
            "is_established": profile.is_established(),
            "feature_means": dict(zip(self.PROFILE_FEATURE_KEYS, profile.feature_means)),
            "feature_stds": dict(zip(self.PROFILE_FEATURE_KEYS, profile.feature_stds())),
            "alert_count": profile.alert_count,
            "last_updated": profile.last_updated,
        }
