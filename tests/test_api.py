"""Tests for the FastAPI REST endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external I/O so tests don't need Kafka/ES/Redis."""
    with (
        patch("src.api.routes.alerts.ElasticsearchClient") as mock_es_cls,
        patch("src.api.routes.metrics.ElasticsearchClient") as mock_es_metrics_cls,
        patch("src.api.routes.metrics.RedisClient") as mock_redis_cls,
        patch("src.api.routes.alerts.OnlineNIDSClassifier") as mock_learner_cls,
    ):
        mock_es = MagicMock()
        mock_es.query_alerts.return_value = [
            {
                "id": "abc123",
                "timestamp": 1706000000.0,
                "src_ip": "192.168.1.10",
                "dst_ip": "1.2.3.4",
                "dst_port": 443,
                "protocol": "TCP",
                "threat_label": "malicious",
                "attack_class": "DoS",
                "severity": "high",
                "fusion_score": 0.88,
                "ml_score": 0.90,
                "anomaly_score": 0.75,
                "signature_score": 0.95,
                "matched_rules": ["SIG-001"],
                "behavioral_deviation": 3.5,
                "threat_intel_score": 0.6,
                "adversarial_signals": [],
                "confidence": 0.82,
            }
        ]
        mock_es.alert_counts_by_class.return_value = {"DoS": 42, "probe": 10}
        mock_es.top_source_ips.return_value = [{"ip": "1.2.3.4", "count": 5}]
        mock_es_cls.return_value = mock_es
        mock_es_metrics_cls.return_value = mock_es

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.ping.return_value = True
        mock_redis_cls.return_value = mock_redis

        mock_learner = MagicMock()
        mock_learner.stats.total_samples = 500
        mock_learner.stats.recent_accuracy = 0.94
        mock_learner_cls.return_value = mock_learner

        yield


class TestRootEndpoints:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "running"

    def test_healthz(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestAlertsEndpoints:
    def test_list_alerts(self):
        r = client.get("/api/v1/alerts/")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["attack_class"] == "DoS"

    def test_list_alerts_with_filter(self):
        r = client.get("/api/v1/alerts/?severity=high&size=50")
        assert r.status_code == 200

    def test_alert_counts(self):
        r = client.get("/api/v1/alerts/counts?hours=24")
        assert r.status_code == 200
        data = r.json()
        assert "DoS" in data

    def test_top_sources(self):
        r = client.get("/api/v1/alerts/top-sources?hours=1&limit=5")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_feedback_accepted(self):
        r = client.post(
            "/api/v1/alerts/feedback",
            json={"alert_id": "abc123", "analyst_label": "DoS"},
        )
        assert r.status_code == 202
        assert r.json()["status"] == "accepted"

    def test_feedback_invalid_label(self):
        r = client.post(
            "/api/v1/alerts/feedback",
            json={"alert_id": "abc123", "analyst_label": "notaclass"},
        )
        assert r.status_code == 422


class TestMetricsEndpoints:
    def test_health(self):
        r = client.get("/api/v1/metrics/health")
        assert r.status_code == 200
        assert r.json()["redis"] is True

    def test_summary(self):
        r = client.get("/api/v1/metrics/summary")
        assert r.status_code == 200
        data = r.json()
        assert "total_alerts_24h" in data
        assert "model_accuracy" in data


class TestModelsEndpoints:
    def test_list_rules(self):
        r = client.get("/api/v1/models/rules")
        assert r.status_code == 200
        rules = r.json()
        assert isinstance(rules, list)
        assert len(rules) > 0
        rule_ids = [r["rule_id"] for r in rules]
        assert "SIG-001" in rule_ids

    def test_delete_nonexistent_rule(self):
        r = client.delete("/api/v1/models/rules/NONEXISTENT")
        assert r.status_code == 404
