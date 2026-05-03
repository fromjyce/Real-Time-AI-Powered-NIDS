"""
Alert manager — deduplicates, enriches, and routes alerts to configured
notifiers (Slack, email, webhooks). Also persists alerts to Elasticsearch.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import structlog

from src.config import settings
from src.storage.elasticsearch_client import ElasticsearchClient
from src.storage.redis_client import RedisClient
from src.alerting.notifiers.slack_notifier import SlackNotifier
from src.alerting.notifiers.email_notifier import EmailNotifier
from src.alerting.notifiers.webhook_notifier import WebhookNotifier

logger = structlog.get_logger(__name__)

DEDUP_WINDOW_SECONDS = 300        # suppress duplicate alerts for 5 minutes
DEDUP_KEY_PREFIX = "nids:alert:dedup:"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    id: str
    timestamp: float
    src_ip: str
    dst_ip: str
    dst_port: Optional[int]
    protocol: str
    threat_label: str           # benign / suspicious / malicious
    attack_class: str
    fusion_score: float
    ml_score: float
    anomaly_score: float
    signature_score: float
    matched_rules: List[str]
    behavioral_deviation: float
    threat_intel_score: float
    adversarial_signals: List[str]
    severity: AlertSeverity
    confidence: float
    raw_features: Dict = field(default_factory=dict)


def _compute_severity(fusion_score: float) -> AlertSeverity:
    if fusion_score >= 0.90:
        return AlertSeverity.CRITICAL
    if fusion_score >= 0.70:
        return AlertSeverity.HIGH
    if fusion_score >= 0.50:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


def _make_fingerprint(src_ip: str, dst_ip: str, dst_port: Optional[int], attack_class: str) -> str:
    raw = f"{src_ip}:{dst_ip}:{dst_port}:{attack_class}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class AlertManager:
    """
    Central hub for all NIDS alerts.
    Deduplicates, stores, and fans out to all configured notifiers.
    """

    def __init__(
        self,
        es: Optional[ElasticsearchClient] = None,
        redis: Optional[RedisClient] = None,
    ) -> None:
        self._es = es or ElasticsearchClient()
        self._redis = redis or RedisClient()
        self._slack = SlackNotifier() if settings.SLACK_BOT_TOKEN else None
        self._email = EmailNotifier() if settings.SMTP_USER else None
        self._webhook = WebhookNotifier() if settings.WEBHOOK_URL else None
        self._alert_count = 0

    def _is_duplicate(self, fingerprint: str) -> bool:
        key = f"{DEDUP_KEY_PREFIX}{fingerprint}"
        result = self._redis.get(key)
        if result:
            return True
        self._redis.set(key, "1", ttl=DEDUP_WINDOW_SECONDS)
        return False

    def build_alert(self, detection_result: Dict, enrichment: Dict = {}) -> Alert:
        src_ip = detection_result.get("src_ip", "")
        dst_ip = detection_result.get("dst_ip", "")
        dst_port = detection_result.get("dst_port")
        attack_class = detection_result.get("attack_class", "")
        fusion_score = detection_result.get("fusion_score", 0.0)

        return Alert(
            id=_make_fingerprint(src_ip, dst_ip, dst_port, attack_class),
            timestamp=time.time(),
            src_ip=src_ip,
            dst_ip=dst_ip,
            dst_port=dst_port,
            protocol=detection_result.get("protocol", ""),
            threat_label=detection_result.get("threat_label", ""),
            attack_class=attack_class,
            fusion_score=fusion_score,
            ml_score=detection_result.get("ml_score", 0.0),
            anomaly_score=detection_result.get("anomaly_score", 0.0),
            signature_score=detection_result.get("signature_score", 0.0),
            matched_rules=detection_result.get("matched_rules", []),
            behavioral_deviation=enrichment.get("behavioral_deviation", 0.0),
            threat_intel_score=enrichment.get("threat_intel_score", 0.0),
            adversarial_signals=enrichment.get("adversarial_signals", []),
            severity=_compute_severity(fusion_score),
            confidence=detection_result.get("confidence", 0.0),
            raw_features=detection_result.get("raw_features", {}),
        )

    async def dispatch(self, alert: Alert) -> bool:
        fingerprint = _make_fingerprint(alert.src_ip, alert.dst_ip, alert.dst_port, alert.attack_class)
        if self._is_duplicate(fingerprint):
            logger.debug("alert_deduplicated", fingerprint=fingerprint)
            return False

        self._alert_count += 1
        logger.info(
            "alert_dispatching",
            id=alert.id,
            severity=alert.severity,
            src=alert.src_ip,
            dst=alert.dst_ip,
            score=round(alert.fusion_score, 3),
        )

        self._es.index_alert(alert)

        tasks = []
        if self._slack and alert.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
            tasks.append(self._slack.send(alert))
        if self._email and alert.severity == AlertSeverity.CRITICAL:
            tasks.append(self._email.send(alert))
        if self._webhook:
            tasks.append(self._webhook.send(alert))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("alert_notifier_error", error=str(r))

        return True

    def dispatch_sync(self, alert: Alert) -> bool:
        return asyncio.run(self.dispatch(alert))

    @property
    def total_alerts(self) -> int:
        return self._alert_count
