"""Alert query and feedback endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import AlertQuery, AlertResponse, FeedbackRequest
from src.storage.elasticsearch_client import ElasticsearchClient
from src.adaptive.online_learner import OnlineNIDSClassifier
from src.features.flow_features import FlowFeatureExtractor, FEATURE_NAMES
from src.features.statistical_features import STAT_FEATURE_NAMES

import numpy as np

router = APIRouter(prefix="/alerts", tags=["alerts"])


def get_es() -> ElasticsearchClient:
    return ElasticsearchClient()


def get_learner() -> OnlineNIDSClassifier:
    return OnlineNIDSClassifier()


@router.get("/", response_model=List[AlertResponse])
async def list_alerts(
    severity: Optional[str] = Query(None),
    attack_class: Optional[str] = Query(None),
    src_ip: Optional[str] = Query(None),
    from_ts: Optional[float] = Query(None),
    to_ts: Optional[float] = Query(None),
    size: int = Query(default=100, ge=1, le=1000),
    es: ElasticsearchClient = Depends(get_es),
) -> List[dict]:
    docs = es.query_alerts(
        severity=severity,
        attack_class=attack_class,
        src_ip=src_ip,
        from_ts=from_ts,
        to_ts=to_ts,
        size=size,
    )
    return docs


@router.get("/counts")
async def alert_counts(
    hours: int = Query(default=24, ge=1, le=168),
    es: ElasticsearchClient = Depends(get_es),
) -> dict:
    return es.alert_counts_by_class(hours=hours)


@router.get("/top-sources")
async def top_sources(
    hours: int = Query(default=1, ge=1, le=24),
    limit: int = Query(default=10, ge=1, le=50),
    es: ElasticsearchClient = Depends(get_es),
) -> list:
    return es.top_source_ips(hours=hours, limit=limit)


@router.get("/{alert_id}")
async def get_alert(
    alert_id: str,
    es: ElasticsearchClient = Depends(get_es),
) -> dict:
    results = es.query_alerts(size=1)
    hits = [r for r in results if r.get("id") == alert_id]
    if not hits:
        raise HTTPException(status_code=404, detail="Alert not found")
    return hits[0]


@router.post("/feedback", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(
    feedback: FeedbackRequest,
    learner: OnlineNIDSClassifier = Depends(get_learner),
) -> dict:
    """
    Submit analyst feedback to update the online learning model.
    """
    zero_vec = np.zeros(len(FEATURE_NAMES) + len(STAT_FEATURE_NAMES), dtype=np.float32)
    learner.learn(zero_vec, feedback.analyst_label)

    return {
        "status": "accepted",
        "alert_id": feedback.alert_id,
        "label": feedback.analyst_label,
        "total_feedback_samples": learner.stats.total_samples,
    }
