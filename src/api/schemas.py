"""Pydantic request/response schemas for the NIDS API."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, IPvAnyAddress


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLabel(str, Enum):
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


class AlertResponse(BaseModel):
    id: str
    timestamp: float
    src_ip: str
    dst_ip: str
    dst_port: Optional[int]
    protocol: str
    threat_label: ThreatLabel
    attack_class: str
    severity: AlertSeverity
    fusion_score: float = Field(..., ge=0.0, le=1.0)
    ml_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    signature_score: float = Field(..., ge=0.0, le=1.0)
    matched_rules: List[str]
    behavioral_deviation: float
    threat_intel_score: float
    adversarial_signals: List[str]
    confidence: float


class AlertQuery(BaseModel):
    severity: Optional[AlertSeverity] = None
    attack_class: Optional[str] = None
    src_ip: Optional[str] = None
    from_ts: Optional[float] = None
    to_ts: Optional[float] = None
    size: int = Field(default=100, ge=1, le=1000)


class MetricsSummary(BaseModel):
    total_alerts_24h: int
    malicious_count: int
    suspicious_count: int
    top_attack_classes: Dict[str, int]
    top_source_ips: List[Dict]
    throughput_pps: float
    active_flows: int
    model_accuracy: float
    drift_detected: bool


class IPProfileResponse(BaseModel):
    ip: str
    sample_count: int
    is_established: bool
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]
    alert_count: int
    last_updated: float


class FeedbackRequest(BaseModel):
    alert_id: str
    analyst_label: str = Field(..., pattern="^(benign|DoS|probe|R2L|U2R)$")
    analyst_note: Optional[str] = None


class ModelStatusResponse(BaseModel):
    cnn_gru_loaded: bool
    transformer_loaded: bool
    isolation_forest_loaded: bool
    autoencoder_loaded: bool
    online_learner_samples: int
    online_learner_accuracy: float
    drift_events_24h: int
    last_retrain: Optional[float]


class RuleResponse(BaseModel):
    rule_id: str
    name: str
    severity: str
    action: str
    description: str
    tags: List[str]
    confidence: float


class WebSocketMessage(BaseModel):
    type: str           # "alert" | "metric" | "heartbeat"
    payload: Dict
