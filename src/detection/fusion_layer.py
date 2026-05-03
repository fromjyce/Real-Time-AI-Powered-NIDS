"""
Decision fusion layer.
Combines ML model score, anomaly score, and signature score into a single
weighted threat score and maps it to a human-readable threat label.

Final Score = w1*ML + w2*Anomaly + w3*Signature
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from src.config import settings
from src.detection.models.cnn_gru import CNNGRUInference
from src.detection.models.isolation_forest import IsolationForestDetector
from src.detection.models.autoencoder import AutoencoderDetector
from src.detection.signature_engine import SignatureEngine, RuleMatch
from src.features.flow_features import FlowFeatureExtractor, FEATURE_NAMES
from src.features.statistical_features import StatisticalFeatureExtractor, STAT_FEATURE_NAMES

logger = structlog.get_logger(__name__)


@dataclass
class DetectionResult:
    src_ip: str
    dst_ip: str
    dst_port: Optional[int]
    protocol: str
    ml_score: float
    anomaly_score: float
    signature_score: float
    fusion_score: float
    threat_label: str          # benign / suspicious / malicious
    attack_class: str          # benign / DoS / probe / R2L / U2R
    matched_rules: List[str] = field(default_factory=list)
    confidence: float = 0.0


class FusionLayer:
    """
    Loads all detection models and orchestrates the three-stage pipeline:
      1. ML sequence model (CNN-GRU)
      2. Anomaly model (Isolation Forest + VAE)
      3. Signature engine (rule-based)
    then fuses outputs into a single verdict.
    """

    LABEL_BENIGN = "benign"
    LABEL_SUSPICIOUS = "suspicious"
    LABEL_MALICIOUS = "malicious"

    def __init__(
        self,
        w_ml: float = settings.FUSION_WEIGHT_ML,
        w_anomaly: float = settings.FUSION_WEIGHT_ANOMALY,
        w_signature: float = settings.FUSION_WEIGHT_SIGNATURE,
    ) -> None:
        self.w_ml = w_ml
        self.w_anomaly = w_anomaly
        self.w_signature = w_signature

        total = w_ml + w_anomaly + w_signature
        if abs(total - 1.0) > 0.01:
            logger.warning("fusion_weights_not_normalized", total=total)

        self.cnn_gru = CNNGRUInference(model_path=settings.SEQUENCE_MODEL_PATH)
        self.iso_forest = self._load_iso_forest()
        self.autoencoder = AutoencoderDetector(model_path=settings.AUTOENCODER_PATH)
        self.signature_engine = SignatureEngine()

        self.flow_extractor = FlowFeatureExtractor()
        self.stat_extractor = StatisticalFeatureExtractor()

        logger.info(
            "fusion_layer_initialized",
            w_ml=w_ml,
            w_anomaly=w_anomaly,
            w_signature=w_signature,
        )

    def _load_iso_forest(self) -> Optional[IsolationForestDetector]:
        path = settings.ANOMALY_MODEL_PATH
        if Path(path).exists():
            return IsolationForestDetector.load(path)
        logger.warning("iso_forest_not_found", path=path)
        return None

    def _build_feature_vector(self, flow_row: Dict[str, Any]) -> np.ndarray:
        flow_feats = np.array(
            [flow_row.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32
        )
        stat_feats = np.array(
            [flow_row.get(name, 0.0) for name in STAT_FEATURE_NAMES], dtype=np.float32
        )
        return np.concatenate([flow_feats, stat_feats])

    def score_flow(self, flow: Dict[str, Any], packet_sequence: Optional[np.ndarray] = None) -> DetectionResult:
        feature_vec = self._build_feature_vector(flow)

        # ── ML Score ────────────────────────────────────────────────────────
        if packet_sequence is not None:
            attack_class, ml_score = self.cnn_gru.predict(packet_sequence)
        else:
            seq = feature_vec.reshape(1, -1).repeat(20, axis=0)
            attack_class, ml_score = self.cnn_gru.predict(seq)

        # ── Anomaly Score ────────────────────────────────────────────────────
        iso_score = 0.0
        ae_score = 0.0
        if self.iso_forest is not None:
            try:
                iso_score = self.iso_forest.anomaly_score(feature_vec)
            except Exception:
                logger.warning("iso_forest_score_failed")

        try:
            ae_score = self.autoencoder.anomaly_score(feature_vec)
        except Exception:
            logger.warning("autoencoder_score_failed")

        anomaly_score = 0.6 * iso_score + 0.4 * ae_score

        # ── Signature Score ───────────────────────────────────────────────────
        combined_flow = {**flow}
        matches, sig_score = self.signature_engine.evaluate(combined_flow)
        matched_rule_ids = [m.rule_id for m in matches]

        # ── Fusion ────────────────────────────────────────────────────────────
        fusion_score = (
            self.w_ml * ml_score
            + self.w_anomaly * anomaly_score
            + self.w_signature * sig_score
        )
        fusion_score = float(np.clip(fusion_score, 0.0, 1.0))

        if fusion_score >= settings.THRESHOLD_MALICIOUS:
            threat_label = self.LABEL_MALICIOUS
        elif fusion_score >= settings.THRESHOLD_SUSPICIOUS:
            threat_label = self.LABEL_SUSPICIOUS
        else:
            threat_label = self.LABEL_BENIGN
            attack_class = "benign"

        confidence = min(
            abs(fusion_score - settings.THRESHOLD_SUSPICIOUS)
            / (settings.THRESHOLD_MALICIOUS - settings.THRESHOLD_SUSPICIOUS + 1e-9),
            1.0,
        )

        return DetectionResult(
            src_ip=flow.get("src_ip", ""),
            dst_ip=flow.get("dst_ip", ""),
            dst_port=flow.get("dst_port"),
            protocol=flow.get("protocol", ""),
            ml_score=ml_score,
            anomaly_score=anomaly_score,
            signature_score=sig_score,
            fusion_score=fusion_score,
            threat_label=threat_label,
            attack_class=attack_class,
            matched_rules=matched_rule_ids,
            confidence=confidence,
        )

    def score_batch(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of aggregated flow rows."""
        results = []
        for _, row in flows.iterrows():
            try:
                result = self.score_flow(row.to_dict())
                results.append({
                    "src_ip": result.src_ip,
                    "dst_ip": result.dst_ip,
                    "dst_port": result.dst_port,
                    "protocol": result.protocol,
                    "ml_score": result.ml_score,
                    "anomaly_score": result.anomaly_score,
                    "signature_score": result.signature_score,
                    "fusion_score": result.fusion_score,
                    "threat_label": result.threat_label,
                    "attack_class": result.attack_class,
                    "matched_rules": result.matched_rules,
                    "confidence": result.confidence,
                })
            except Exception:
                logger.exception("fusion_score_flow_error")

        return pd.DataFrame(results) if results else pd.DataFrame()
