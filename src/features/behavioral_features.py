"""
Behavioral feature extraction.
Tracks per-source-IP connection patterns over a rolling time window
to detect anomalous behaviour such as port scans, brute-force attempts,
and high connection rates.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

BEHAVIORAL_FEATURE_NAMES = [
    "conn_rate_1m",
    "conn_rate_5m",
    "unique_dst_ips_1m",
    "unique_dst_ports_1m",
    "unique_dst_ports_5m",
    "failed_conn_ratio",
    "syn_only_ratio",
    "avg_flow_duration",
    "repeated_dst_ratio",
    "new_dst_ratio",
    "icmp_ratio",
    "udp_ratio",
    "tcp_ratio",
    "service_diversity",
    "failed_auth_count",
]

ONE_MINUTE = 60.0
FIVE_MINUTES = 300.0


@dataclass
class ConnectionRecord:
    timestamp: float
    dst_ip: str
    dst_port: int
    protocol: str
    conn_state: str
    duration: float
    is_failed: bool
    is_syn_only: bool


class BehavioralFeatureExtractor:
    """
    Maintains per-source-IP connection histories and computes
    behavioral deviation features for anomaly detection.
    """

    def __init__(self, max_history: int = 10_000) -> None:
        self.max_history = max_history
        self._histories: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._known_destinations: Dict[str, Set[str]] = defaultdict(set)

    def record(self, src_ip: str, record: ConnectionRecord) -> None:
        self._histories[src_ip].append(record)
        self._known_destinations[src_ip].add(record.dst_ip)

    def extract(self, src_ip: str, now: Optional[float] = None) -> Dict[str, float]:
        ts = now or time.time()
        history = list(self._histories.get(src_ip, []))

        if not history:
            return {name: 0.0 for name in BEHAVIORAL_FEATURE_NAMES}

        in_1m = [r for r in history if ts - r.timestamp <= ONE_MINUTE]
        in_5m = [r for r in history if ts - r.timestamp <= FIVE_MINUTES]

        conn_rate_1m = len(in_1m) / ONE_MINUTE
        conn_rate_5m = len(in_5m) / FIVE_MINUTES

        unique_dst_ips_1m = len({r.dst_ip for r in in_1m})
        unique_dst_ports_1m = len({r.dst_port for r in in_1m})
        unique_dst_ports_5m = len({r.dst_port for r in in_5m})

        failed = [r for r in in_5m if r.is_failed]
        failed_ratio = len(failed) / max(len(in_5m), 1)

        syn_only = [r for r in in_5m if r.is_syn_only]
        syn_ratio = len(syn_only) / max(len(in_5m), 1)

        durations = [r.duration for r in in_5m if r.duration > 0]
        avg_duration = float(np.mean(durations)) if durations else 0.0

        all_known = self._known_destinations.get(src_ip, set())
        recent_dsts = {r.dst_ip for r in in_5m}
        repeated = recent_dsts & all_known
        repeated_ratio = len(repeated) / max(len(recent_dsts), 1)
        new_ratio = 1.0 - repeated_ratio

        protocols = [r.protocol for r in in_5m]
        total = max(len(protocols), 1)
        icmp_ratio = protocols.count("ICMP") / total
        udp_ratio = protocols.count("UDP") / total
        tcp_ratio = protocols.count("TCP") / total

        services = {r.dst_port for r in in_5m}
        service_diversity = len(services) / max(len(in_5m), 1)

        # Count failed auth (SSH:22, FTP:21, Telnet:23, RDP:3389)
        auth_ports = {22, 21, 23, 3389}
        failed_auth = sum(
            1 for r in in_5m if r.dst_port in auth_ports and r.is_failed
        )

        return {
            "conn_rate_1m": conn_rate_1m,
            "conn_rate_5m": conn_rate_5m,
            "unique_dst_ips_1m": float(unique_dst_ips_1m),
            "unique_dst_ports_1m": float(unique_dst_ports_1m),
            "unique_dst_ports_5m": float(unique_dst_ports_5m),
            "failed_conn_ratio": failed_ratio,
            "syn_only_ratio": syn_ratio,
            "avg_flow_duration": avg_duration,
            "repeated_dst_ratio": repeated_ratio,
            "new_dst_ratio": new_ratio,
            "icmp_ratio": icmp_ratio,
            "udp_ratio": udp_ratio,
            "tcp_ratio": tcp_ratio,
            "service_diversity": service_diversity,
            "failed_auth_count": float(failed_auth),
        }

    def to_vector(self, features: Dict[str, float]) -> np.ndarray:
        return np.array(
            [features.get(name, 0.0) for name in BEHAVIORAL_FEATURE_NAMES],
            dtype=np.float32,
        )

    def cleanup(self, now: Optional[float] = None, ttl: float = FIVE_MINUTES * 2) -> int:
        """Remove stale per-IP histories to bound memory usage."""
        ts = now or time.time()
        stale_ips = []
        for ip, history in self._histories.items():
            if history and ts - history[-1].timestamp > ttl:
                stale_ips.append(ip)
        for ip in stale_ips:
            del self._histories[ip]
        return len(stale_ips)
