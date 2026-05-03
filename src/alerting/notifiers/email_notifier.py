"""Email notifier using aiosmtplib for async SMTP delivery."""

import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

import aiosmtplib
import structlog

from src.config import settings

if TYPE_CHECKING:
    from src.alerting.alert_manager import Alert

logger = structlog.get_logger(__name__)


class EmailNotifier:
    def __init__(self) -> None:
        self._host = settings.SMTP_HOST
        self._port = settings.SMTP_PORT
        self._user = settings.SMTP_USER
        self._password = settings.SMTP_PASSWORD
        self._to = settings.ALERT_EMAIL_TO

    def _build_html(self, alert: "Alert") -> str:
        rules = ", ".join(alert.matched_rules) if alert.matched_rules else "None"
        adversarial = ", ".join(alert.adversarial_signals) if alert.adversarial_signals else "None"
        color_map = {
            "low": "#28a745",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545",
        }
        badge_color = color_map.get(alert.severity, "#6c757d")

        return textwrap.dedent(f"""
        <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <div style="background:{badge_color}; color:white; padding:12px 20px; border-radius:6px 6px 0 0;">
            <h2 style="margin:0;">NIDS {alert.severity.upper()} Alert — {alert.attack_class}</h2>
          </div>
          <div style="border:1px solid #dee2e6; padding:20px; border-radius:0 0 6px 6px;">
            <table style="width:100%; border-collapse:collapse;">
              <tr><td style="padding:6px; font-weight:bold;">Alert ID</td><td>{alert.id}</td></tr>
              <tr style="background:#f8f9fa;"><td style="padding:6px; font-weight:bold;">Source IP</td><td>{alert.src_ip}</td></tr>
              <tr><td style="padding:6px; font-weight:bold;">Destination</td><td>{alert.dst_ip}:{alert.dst_port}</td></tr>
              <tr style="background:#f8f9fa;"><td style="padding:6px; font-weight:bold;">Protocol</td><td>{alert.protocol}</td></tr>
              <tr><td style="padding:6px; font-weight:bold;">Threat Label</td><td>{alert.threat_label}</td></tr>
              <tr style="background:#f8f9fa;"><td style="padding:6px; font-weight:bold;">Fusion Score</td><td>{alert.fusion_score:.4f}</td></tr>
              <tr><td style="padding:6px; font-weight:bold;">ML Score</td><td>{alert.ml_score:.4f}</td></tr>
              <tr style="background:#f8f9fa;"><td style="padding:6px; font-weight:bold;">Anomaly Score</td><td>{alert.anomaly_score:.4f}</td></tr>
              <tr><td style="padding:6px; font-weight:bold;">Signature Score</td><td>{alert.signature_score:.4f}</td></tr>
              <tr style="background:#f8f9fa;"><td style="padding:6px; font-weight:bold;">Matched Rules</td><td>{rules}</td></tr>
              <tr><td style="padding:6px; font-weight:bold;">Adversarial Signals</td><td>{adversarial}</td></tr>
              <tr style="background:#f8f9fa;"><td style="padding:6px; font-weight:bold;">Behavioral Deviation</td><td>{alert.behavioral_deviation:.4f}</td></tr>
              <tr><td style="padding:6px; font-weight:bold;">Threat Intel Score</td><td>{alert.threat_intel_score:.4f}</td></tr>
            </table>
            <p style="margin-top:20px; font-size:12px; color:#6c757d;">
              This alert was generated automatically by the NIDS pipeline.
            </p>
          </div>
        </body></html>
        """).strip()

    async def send(self, alert: "Alert") -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (
            f"[NIDS] {alert.severity.upper()} Alert — {alert.attack_class} from {alert.src_ip}"
        )
        msg["From"] = self._user
        msg["To"] = self._to

        text_body = (
            f"NIDS Alert\n"
            f"Severity: {alert.severity}\n"
            f"Attack: {alert.attack_class}\n"
            f"Source: {alert.src_ip} → {alert.dst_ip}:{alert.dst_port}\n"
            f"Fusion Score: {alert.fusion_score:.4f}\n"
        )
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(self._build_html(alert), "html"))

        try:
            await aiosmtplib.send(
                msg,
                hostname=self._host,
                port=self._port,
                username=self._user,
                password=self._password,
                start_tls=True,
            )
            logger.info("email_alert_sent", alert_id=alert.id, to=self._to)
        except Exception as e:
            logger.error("email_send_failed", error=str(e), alert_id=alert.id)
            raise
