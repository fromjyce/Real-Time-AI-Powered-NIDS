"""
Signature-based detection engine (Snort-like rule system).
Rules are defined as Python dataclasses and evaluated against flow feature dicts.
Supports threshold rules, port-based rules, flag combination rules, and rate rules.
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


class RuleAction(str, Enum):
    ALERT = "alert"
    DROP = "drop"
    LOG = "log"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RuleMatch:
    rule_id: str
    rule_name: str
    severity: Severity
    action: RuleAction
    description: str
    confidence: float


@dataclass
class SignatureRule:
    rule_id: str
    name: str
    severity: Severity
    action: RuleAction
    description: str
    predicate: Callable[[Dict], bool]
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)


def _tcp_flags_match(flow: Dict, required: str) -> bool:
    flags = flow.get("tcp_flags", "") or ""
    return all(f in flags for f in required)


def _port_in(flow: Dict, field: str, ports: Set[int]) -> bool:
    return flow.get(field) in ports


BUILTIN_RULES: List[SignatureRule] = [
    # ── SYN Flood ────────────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-001",
        name="SYN Flood Detection",
        severity=Severity.HIGH,
        action=RuleAction.ALERT,
        description="High ratio of SYN-only packets indicates SYN flood DoS attempt",
        predicate=lambda f: (
            f.get("syn_ratio", 0) > 0.8
            and f.get("packet_count", 0) > 100
            and f.get("protocol") == "TCP"
        ),
        confidence=0.90,
        tags=["dos", "syn-flood"],
    ),
    # ── Port Scan ────────────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-002",
        name="Horizontal Port Scan",
        severity=Severity.MEDIUM,
        action=RuleAction.ALERT,
        description="Single source scanning many destination ports",
        predicate=lambda f: f.get("unique_dst_ports_1m", 0) > 50,
        confidence=0.85,
        tags=["probe", "portscan"],
    ),
    # ── UDP Flood ─────────────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-003",
        name="UDP Flood Detection",
        severity=Severity.HIGH,
        action=RuleAction.ALERT,
        description="Extremely high UDP packet rate to single destination",
        predicate=lambda f: (
            f.get("protocol") == "UDP"
            and f.get("packets_per_second", 0) > 5000
        ),
        confidence=0.88,
        tags=["dos", "udp-flood"],
    ),
    # ── ICMP Flood ───────────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-004",
        name="ICMP Flood / Ping of Death",
        severity=Severity.MEDIUM,
        action=RuleAction.ALERT,
        description="Abnormally high ICMP traffic volume",
        predicate=lambda f: (
            f.get("protocol") == "ICMP"
            and f.get("packets_per_second", 0) > 500
        ),
        confidence=0.80,
        tags=["dos", "icmp"],
    ),
    # ── SSH Brute Force ───────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-005",
        name="SSH Brute Force",
        severity=Severity.HIGH,
        action=RuleAction.ALERT,
        description="Repeated failed connections to SSH port",
        predicate=lambda f: (
            f.get("dst_port") == 22
            and f.get("failed_auth_count", 0) > 10
            and f.get("conn_rate_1m", 0) > 5
        ),
        confidence=0.92,
        tags=["brute-force", "ssh", "R2L"],
    ),
    # ── DNS Amplification ─────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-006",
        name="DNS Amplification Attack",
        severity=Severity.HIGH,
        action=RuleAction.ALERT,
        description="High response-to-request byte ratio on DNS port (amplification)",
        predicate=lambda f: (
            f.get("dst_port") == 53
            and f.get("bwd_bytes", 0) > f.get("fwd_bytes", 1) * 20
            and f.get("packet_count", 0) > 50
        ),
        confidence=0.87,
        tags=["dos", "amplification", "dns"],
    ),
    # ── Null Scan ─────────────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-007",
        name="TCP Null Scan",
        severity=Severity.MEDIUM,
        action=RuleAction.ALERT,
        description="TCP packets with no flags set (null scan)",
        predicate=lambda f: (
            f.get("tcp_flags", "") == ""
            and f.get("protocol") == "TCP"
            and f.get("packet_count", 0) > 5
        ),
        confidence=0.75,
        tags=["probe", "null-scan"],
    ),
    # ── FTP Brute Force ───────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-008",
        name="FTP Brute Force",
        severity=Severity.MEDIUM,
        action=RuleAction.ALERT,
        description="Repeated failed connections to FTP",
        predicate=lambda f: (
            f.get("dst_port") == 21
            and f.get("failed_auth_count", 0) > 5
        ),
        confidence=0.88,
        tags=["brute-force", "ftp", "R2L"],
    ),
    # ── Slowloris ─────────────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-009",
        name="Slowloris DoS",
        severity=Severity.HIGH,
        action=RuleAction.ALERT,
        description="Long-lived low-bandwidth HTTP connections (Slowloris pattern)",
        predicate=lambda f: (
            f.get("dst_port") in {80, 443, 8080}
            and f.get("flow_duration", 0) > 120
            and f.get("packets_per_second", 999) < 0.5
            and f.get("packet_count", 0) > 5
        ),
        confidence=0.82,
        tags=["dos", "slowloris", "http"],
    ),
    # ── Internal Recon ───────────────────────────────────────────────────────
    SignatureRule(
        rule_id="SIG-010",
        name="Network Reconnaissance",
        severity=Severity.LOW,
        action=RuleAction.LOG,
        description="Source scanning many unique destination IPs",
        predicate=lambda f: f.get("unique_dst_ips_1m", 0) > 30,
        confidence=0.70,
        tags=["probe", "recon"],
    ),
]


class SignatureEngine:
    """
    Evaluates a flow feature dict against all loaded signature rules
    and returns a list of matched rules along with an aggregate score.
    """

    def __init__(self, rules: Optional[List[SignatureRule]] = None) -> None:
        self.rules = rules or BUILTIN_RULES
        logger.info("signature_engine_loaded", rule_count=len(self.rules))

    def evaluate(self, flow: Dict) -> Tuple[List[RuleMatch], float]:
        """
        Returns:
            matches: list of matched RuleMatch objects
            score:   float in [0, 1] — highest confidence among matches
        """
        matches = []
        for rule in self.rules:
            try:
                if rule.predicate(flow):
                    matches.append(
                        RuleMatch(
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            severity=rule.severity,
                            action=rule.action,
                            description=rule.description,
                            confidence=rule.confidence,
                        )
                    )
            except Exception:
                logger.warning("rule_evaluation_error", rule_id=rule.rule_id)

        score = max((m.confidence for m in matches), default=0.0)
        return matches, score

    def add_rule(self, rule: SignatureRule) -> None:
        self.rules.append(rule)
        logger.info("signature_rule_added", rule_id=rule.rule_id)

    def remove_rule(self, rule_id: str) -> bool:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < before
