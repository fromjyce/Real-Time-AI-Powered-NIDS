"""
Adversarial attack detection.
Detects traffic obfuscation and model evasion attempts including:
  - Low-and-slow attacks (artificially low rate to dodge thresholds)
  - Packet fragmentation abuse
  - Padding-based payload obfuscation
  - Mimicry attacks (matching benign traffic profiles)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AdversarialSignal:
    technique: str
    confidence: float
    evidence: str


class AdversarialDetector:
    """
    Heuristic checks for evasion behaviours that deliberately manipulate
    traffic patterns to avoid detection thresholds.
    """

    # Threshold constants
    MIN_SUSPICIOUS_PACKETS = 10
    LOW_AND_SLOW_PPS_UPPER = 0.5       # packets/sec
    LOW_AND_SLOW_BPS_UPPER = 200       # bytes/sec
    LOW_AND_SLOW_DURATION_MIN = 60     # seconds
    FRAGMENTATION_RATIO_THRESHOLD = 0.4
    PAYLOAD_ENTROPY_MIMICRY_RANGE = (3.5, 4.5)
    MIMICRY_STD_THRESHOLD = 0.05       # unnaturally consistent packet sizes

    def analyze(self, flow: Dict) -> List[AdversarialSignal]:
        signals: List[AdversarialSignal] = []

        signals.extend(self._check_low_and_slow(flow))
        signals.extend(self._check_fragmentation_abuse(flow))
        signals.extend(self._check_payload_mimicry(flow))
        signals.extend(self._check_size_padding(flow))
        signals.extend(self._check_protocol_tunneling(flow))

        return signals

    def aggregate_score(self, signals: List[AdversarialSignal]) -> float:
        """Returns maximum confidence among detected signals."""
        if not signals:
            return 0.0
        return float(max(s.confidence for s in signals))

    def _check_low_and_slow(self, flow: Dict) -> List[AdversarialSignal]:
        signals = []
        pps = flow.get("packets_per_second", 0)
        bps = flow.get("bytes_per_second", 0)
        duration = flow.get("flow_duration", 0)
        pkt_count = flow.get("packet_count", 0)

        if (
            pkt_count > self.MIN_SUSPICIOUS_PACKETS
            and pps < self.LOW_AND_SLOW_PPS_UPPER
            and bps < self.LOW_AND_SLOW_BPS_UPPER
            and duration > self.LOW_AND_SLOW_DURATION_MIN
        ):
            confidence = min(
                0.9,
                0.5 + (duration / 600) * 0.4,
            )
            signals.append(AdversarialSignal(
                technique="low_and_slow",
                confidence=round(confidence, 3),
                evidence=(
                    f"pps={pps:.3f}, bps={bps:.1f}, "
                    f"duration={duration:.1f}s, pkts={pkt_count}"
                ),
            ))
        return signals

    def _check_fragmentation_abuse(self, flow: Dict) -> List[AdversarialSignal]:
        signals = []
        mf_count = flow.get("ip_flags_mf_count", 0)
        pkt_count = flow.get("packet_count", 1)
        frag_ratio = mf_count / max(pkt_count, 1)

        if frag_ratio > self.FRAGMENTATION_RATIO_THRESHOLD and pkt_count > 20:
            signals.append(AdversarialSignal(
                technique="fragmentation_abuse",
                confidence=min(0.85, 0.5 + frag_ratio),
                evidence=f"frag_ratio={frag_ratio:.2f}, mf_count={mf_count}",
            ))
        return signals

    def _check_payload_mimicry(self, flow: Dict) -> List[AdversarialSignal]:
        """
        Mimicry attacks craft payloads with entropy matching benign HTTP/DNS traffic.
        We flag when entropy falls in the typical benign range but other flow
        characteristics are anomalous.
        """
        signals = []
        entropy = flow.get("payload_entropy", -1)
        anomaly_score = flow.get("anomaly_score", 0)
        lo, hi = self.PAYLOAD_ENTROPY_MIMICRY_RANGE

        if lo <= entropy <= hi and anomaly_score > 0.6:
            signals.append(AdversarialSignal(
                technique="payload_mimicry",
                confidence=min(0.75, 0.4 + anomaly_score * 0.4),
                evidence=(
                    f"entropy={entropy:.2f} in benign range [{lo},{hi}] "
                    f"but anomaly_score={anomaly_score:.2f}"
                ),
            ))
        return signals

    def _check_size_padding(self, flow: Dict) -> List[AdversarialSignal]:
        """
        Attackers sometimes pad packets to a fixed size to evade size-based rules.
        Unnaturally low std_dev of packet sizes is a tell.
        """
        signals = []
        std = flow.get("std_packet_size", -1)
        pkt_count = flow.get("packet_count", 0)
        mean = flow.get("mean_packet_size", 0)

        if (
            pkt_count > 50
            and 0 <= std < self.MIMICRY_STD_THRESHOLD * mean
            and mean > 100
        ):
            signals.append(AdversarialSignal(
                technique="size_padding",
                confidence=0.65,
                evidence=f"std={std:.2f}, mean={mean:.1f}, count={pkt_count}",
            ))
        return signals

    def _check_protocol_tunneling(self, flow: Dict) -> List[AdversarialSignal]:
        """
        Detect data exfiltration via DNS or ICMP tunneling (large payloads on those protos).
        """
        signals = []
        protocol = flow.get("protocol", "")
        dst_port = flow.get("dst_port", 0) or 0
        mean_size = flow.get("mean_packet_size", 0)

        # DNS tunneling: large DNS payloads
        if dst_port == 53 and mean_size > 512:
            signals.append(AdversarialSignal(
                technique="dns_tunneling",
                confidence=0.80,
                evidence=f"mean_dns_pkt_size={mean_size:.0f} bytes (>512)",
            ))

        # ICMP tunneling: ICMP with large unusual payloads
        if protocol == "ICMP" and mean_size > 1000:
            signals.append(AdversarialSignal(
                technique="icmp_tunneling",
                confidence=0.72,
                evidence=f"mean_icmp_pkt_size={mean_size:.0f} bytes (>1000)",
            ))

        return signals
