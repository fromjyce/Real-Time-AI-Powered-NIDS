"""Tests for detection models and signature engine."""

import numpy as np
import pytest
import torch

from src.detection.models.cnn_gru import CNNGRUDetector
from src.detection.models.transformer import TransformerDetector
from src.detection.models.autoencoder import VariationalAutoencoder
from src.detection.models.isolation_forest import IsolationForestDetector
from src.detection.signature_engine import SignatureEngine
from src.detection.fusion_layer import FusionLayer
from src.adaptive.drift_detector import ADWINDetector, PageHinkleyDetector, ModelDriftMonitor

INPUT_DIM = 46
SEQ_LEN = 20
BATCH = 4


class TestCNNGRU:
    def setup_method(self):
        self.model = CNNGRUDetector(input_dim=INPUT_DIM)
        self.model.eval()

    def test_forward_shape(self):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        logits, probs = self.model(x)
        assert logits.shape == (BATCH, 5)
        assert probs.shape == (BATCH, 5)

    def test_probs_sum_to_one(self):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        _, probs = self.model(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_no_nan_in_output(self):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        logits, probs = self.model(x)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(probs).any()

    def test_single_sample(self):
        x = torch.randn(1, SEQ_LEN, INPUT_DIM)
        _, probs = self.model(x)
        assert probs.shape == (1, 5)


class TestTransformer:
    def setup_method(self):
        self.model = TransformerDetector(input_dim=INPUT_DIM, d_model=64, num_heads=4, num_layers=2)
        self.model.eval()

    def test_forward_shape(self):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        logits, probs = self.model(x)
        assert logits.shape == (BATCH, 5)

    def test_probs_are_normalized(self):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        _, probs = self.model(x)
        assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH), atol=1e-5)


class TestVariationalAutoencoder:
    def setup_method(self):
        self.model = VariationalAutoencoder(input_dim=INPUT_DIM, latent_dim=8, hidden_dims=[32, 16])
        self.model.eval()

    def test_reconstruction_shape(self):
        x = torch.randn(BATCH, INPUT_DIM)
        x_hat, mu, log_var = self.model(x)
        assert x_hat.shape == x.shape
        assert mu.shape == (BATCH, 8)

    def test_loss_is_positive(self):
        x = torch.randn(BATCH, INPUT_DIM)
        x_hat, mu, log_var = self.model(x)
        loss = self.model.loss(x, x_hat, mu, log_var)
        assert loss.item() > 0


class TestIsolationForest:
    def setup_method(self):
        self.detector = IsolationForestDetector(n_estimators=50, contamination=0.1)
        X_train = np.random.randn(500, INPUT_DIM).astype(np.float32)
        self.detector.fit(X_train)

    def test_predict_returns_correct_shape(self):
        X = np.random.randn(20, INPUT_DIM).astype(np.float32)
        labels, scores = self.detector.predict(X)
        assert labels.shape == (20,)
        assert scores.shape == (20,)

    def test_scores_in_valid_range(self):
        X = np.random.randn(50, INPUT_DIM).astype(np.float32)
        _, scores = self.detector.predict(X)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_anomaly_score_single(self):
        x = np.random.randn(INPUT_DIM).astype(np.float32)
        score = self.detector.anomaly_score(x)
        assert 0.0 <= score <= 1.0

    def test_extreme_outlier_has_high_score(self):
        normal = np.zeros((200, INPUT_DIM), dtype=np.float32)
        self.detector.fit(normal)

        outlier = np.ones((1, INPUT_DIM), dtype=np.float32) * 1000
        _, scores = self.detector.predict(outlier)
        assert scores[0] > 0.5


class TestSignatureEngine:
    def setup_method(self):
        self.engine = SignatureEngine()

    def _flow(self, **kwargs):
        defaults = {
            "protocol": "TCP",
            "packet_count": 10,
            "packets_per_second": 5.0,
            "bytes_per_second": 1000.0,
            "syn_ratio": 0.1,
            "rst_ratio": 0.0,
            "flow_duration": 2.0,
            "dst_port": 443,
            "bwd_bytes": 100,
            "fwd_bytes": 100,
            "unique_dst_ports_1m": 2,
            "failed_auth_count": 0,
            "conn_rate_1m": 1.0,
        }
        return {**defaults, **kwargs}

    def test_syn_flood_triggers(self):
        flow = self._flow(
            syn_ratio=0.95,
            packet_count=500,
            packets_per_second=800,
        )
        matches, score = self.engine.evaluate(flow)
        rule_ids = [m.rule_id for m in matches]
        assert "SIG-001" in rule_ids
        assert score > 0.5

    def test_benign_no_match(self):
        flow = self._flow()
        matches, score = self.engine.evaluate(flow)
        assert len(matches) == 0
        assert score == 0.0

    def test_port_scan_triggers(self):
        flow = self._flow(unique_dst_ports_1m=75)
        matches, _ = self.engine.evaluate(flow)
        assert any(m.rule_id == "SIG-002" for m in matches)

    def test_ssh_brute_triggers(self):
        flow = self._flow(dst_port=22, failed_auth_count=20, conn_rate_1m=10)
        matches, _ = self.engine.evaluate(flow)
        assert any(m.rule_id == "SIG-005" for m in matches)

    def test_add_remove_rule(self):
        from src.detection.signature_engine import SignatureRule, Severity, RuleAction
        rule = SignatureRule(
            rule_id="TEST-001",
            name="Test Rule",
            severity=Severity.LOW,
            action=RuleAction.LOG,
            description="Always matches",
            predicate=lambda f: True,
        )
        self.engine.add_rule(rule)
        matches, _ = self.engine.evaluate({})
        assert any(m.rule_id == "TEST-001" for m in matches)

        removed = self.engine.remove_rule("TEST-001")
        assert removed
        matches2, _ = self.engine.evaluate({})
        assert all(m.rule_id != "TEST-001" for m in matches2)


class TestDriftDetectors:
    def test_adwin_detects_drift(self):
        detector = ADWINDetector(delta=0.01)
        drift_found = False
        for _ in range(500):
            detector.add(0.95)
        for i in range(200):
            if detector.add(0.50):
                drift_found = True
                break
        assert drift_found

    def test_adwin_stable_no_drift(self):
        detector = ADWINDetector(delta=0.002)
        for _ in range(1000):
            detected = detector.add(0.90 + np.random.normal(0, 0.01))
            assert not detected

    def test_page_hinkley_detects_upshift(self):
        ph = PageHinkleyDetector(delta=0.005, threshold=20.0)
        drift_found = False
        for _ in range(200):
            ph.add(0.95)
        for _ in range(100):
            if ph.add(0.50):
                drift_found = True
                break
        assert drift_found

    def test_drift_monitor_fires_callback(self):
        fired = []
        monitor = ModelDriftMonitor(on_drift=fired.append, adwin_delta=0.05, ph_threshold=5.0)

        for _ in range(200):
            monitor.record("accuracy", 0.98)
        for _ in range(200):
            monitor.record("accuracy", 0.50)

        assert len(fired) >= 1
        assert fired[0].metric == "accuracy"
