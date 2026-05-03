"""Model management and signature rule endpoints."""

from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import ModelStatusResponse, RuleResponse
from src.adaptive.online_learner import OnlineNIDSClassifier
from src.detection.signature_engine import SignatureEngine, SignatureRule, Severity, RuleAction
from src.config import settings

router = APIRouter(prefix="/models", tags=["models"])


def get_learner() -> OnlineNIDSClassifier:
    return OnlineNIDSClassifier()


def get_signature_engine() -> SignatureEngine:
    return SignatureEngine()


@router.get("/status", response_model=ModelStatusResponse)
async def model_status(
    learner: OnlineNIDSClassifier = Depends(get_learner),
) -> dict:
    return {
        "cnn_gru_loaded": Path(settings.SEQUENCE_MODEL_PATH).exists(),
        "transformer_loaded": Path(settings.TRANSFORMER_MODEL_PATH).exists(),
        "isolation_forest_loaded": Path(settings.ANOMALY_MODEL_PATH).exists(),
        "autoencoder_loaded": Path(settings.AUTOENCODER_PATH).exists(),
        "online_learner_samples": learner.stats.total_samples,
        "online_learner_accuracy": learner.stats.recent_accuracy,
        "drift_events_24h": 0,
        "last_retrain": None,
    }


@router.get("/rules", response_model=List[RuleResponse])
async def list_rules(
    engine: SignatureEngine = Depends(get_signature_engine),
) -> list:
    return [
        {
            "rule_id": r.rule_id,
            "name": r.name,
            "severity": r.severity,
            "action": r.action,
            "description": r.description,
            "tags": r.tags,
            "confidence": r.confidence,
        }
        for r in engine.rules
    ]


@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: str,
    engine: SignatureEngine = Depends(get_signature_engine),
) -> dict:
    removed = engine.remove_rule(rule_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id!r} not found")
    return {"status": "removed", "rule_id": rule_id}
