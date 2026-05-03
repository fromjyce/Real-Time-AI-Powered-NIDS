"""Webhook notifier — posts JSON alert payloads to a configurable HTTP endpoint."""

import time
from dataclasses import asdict
from typing import TYPE_CHECKING

import httpx
import structlog

from src.config import settings

if TYPE_CHECKING:
    from src.alerting.alert_manager import Alert

logger = structlog.get_logger(__name__)


class WebhookNotifier:
    def __init__(self, url: str = "") -> None:
        self._url = url or settings.WEBHOOK_URL
        self._client = httpx.AsyncClient(timeout=10.0)

    async def send(self, alert: "Alert") -> None:
        payload = {
            "id": alert.id,
            "timestamp": alert.timestamp,
            "severity": alert.severity,
            "src_ip": alert.src_ip,
            "dst_ip": alert.dst_ip,
            "dst_port": alert.dst_port,
            "protocol": alert.protocol,
            "threat_label": alert.threat_label,
            "attack_class": alert.attack_class,
            "fusion_score": alert.fusion_score,
            "ml_score": alert.ml_score,
            "anomaly_score": alert.anomaly_score,
            "signature_score": alert.signature_score,
            "matched_rules": alert.matched_rules,
            "behavioral_deviation": alert.behavioral_deviation,
            "threat_intel_score": alert.threat_intel_score,
            "adversarial_signals": alert.adversarial_signals,
            "confidence": alert.confidence,
        }

        try:
            resp = await self._client.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json", "X-NIDS-Source": "nids-pipeline"},
            )
            resp.raise_for_status()
            logger.info("webhook_sent", alert_id=alert.id, status=resp.status_code)
        except httpx.HTTPStatusError as e:
            logger.error("webhook_http_error", status=e.response.status_code, alert_id=alert.id)
            raise
        except httpx.RequestError as e:
            logger.error("webhook_request_error", error=str(e), alert_id=alert.id)
            raise

    async def close(self) -> None:
        await self._client.aclose()
