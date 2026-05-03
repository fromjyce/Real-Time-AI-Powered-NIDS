"""Slack notifier using the slack-sdk library."""

from typing import TYPE_CHECKING

import structlog
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from src.config import settings

if TYPE_CHECKING:
    from src.alerting.alert_manager import Alert

logger = structlog.get_logger(__name__)

SEVERITY_COLORS = {
    "low": "#36a64f",       # green
    "medium": "#ffcc00",    # yellow
    "high": "#ff6600",      # orange
    "critical": "#ff0000",  # red
}

SEVERITY_EMOJI = {
    "low": ":white_circle:",
    "medium": ":large_yellow_circle:",
    "high": ":large_orange_circle:",
    "critical": ":red_circle:",
}


class SlackNotifier:
    def __init__(self) -> None:
        self._client = AsyncWebClient(token=settings.SLACK_BOT_TOKEN)
        self._channel = settings.SLACK_CHANNEL_ID

    async def send(self, alert: "Alert") -> None:
        color = SEVERITY_COLORS.get(alert.severity, "#cccccc")
        emoji = SEVERITY_EMOJI.get(alert.severity, ":white_circle:")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji}  NIDS Alert — {alert.severity.upper()} [{alert.attack_class}]",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Source IP:*\n`{alert.src_ip}`"},
                    {"type": "mrkdwn", "text": f"*Destination:*\n`{alert.dst_ip}:{alert.dst_port}`"},
                    {"type": "mrkdwn", "text": f"*Protocol:*\n{alert.protocol}"},
                    {"type": "mrkdwn", "text": f"*Fusion Score:*\n{alert.fusion_score:.3f}"},
                    {"type": "mrkdwn", "text": f"*ML Score:*\n{alert.ml_score:.3f}"},
                    {"type": "mrkdwn", "text": f"*Anomaly Score:*\n{alert.anomaly_score:.3f}"},
                ],
            },
        ]

        if alert.matched_rules:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Matched Rules:* {', '.join(alert.matched_rules)}",
                },
            })

        if alert.adversarial_signals:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":warning: *Adversarial signals:* {', '.join(alert.adversarial_signals)}",
                },
            })

        try:
            await self._client.chat_postMessage(
                channel=self._channel,
                attachments=[{"color": color, "blocks": blocks}],
                text=f"NIDS {alert.severity.upper()} Alert: {alert.attack_class} from {alert.src_ip}",
            )
            logger.info("slack_alert_sent", alert_id=alert.id)
        except SlackApiError as e:
            logger.error("slack_send_failed", error=str(e), alert_id=alert.id)
            raise
