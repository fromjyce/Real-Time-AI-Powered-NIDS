"""System and detection metrics endpoints."""

import time
from typing import Dict, List

from fastapi import APIRouter, Depends, Query

from src.api.schemas import MetricsSummary
from src.storage.elasticsearch_client import ElasticsearchClient
from src.storage.redis_client import RedisClient

router = APIRouter(prefix="/metrics", tags=["metrics"])

THROUGHPUT_KEY = "nids:metrics:throughput_pps"
ACTIVE_FLOWS_KEY = "nids:metrics:active_flows"
MODEL_ACCURACY_KEY = "nids:metrics:model_accuracy"


def get_es() -> ElasticsearchClient:
    return ElasticsearchClient()


def get_redis() -> RedisClient:
    return RedisClient()


@router.get("/summary", response_model=MetricsSummary)
async def metrics_summary(
    es: ElasticsearchClient = Depends(get_es),
    redis: RedisClient = Depends(get_redis),
) -> dict:
    counts_24h = es.alert_counts_by_class(hours=24)
    top_sources = es.top_source_ips(hours=1, limit=5)

    total_alerts = sum(counts_24h.values())
    malicious = counts_24h.get("DoS", 0) + counts_24h.get("R2L", 0) + counts_24h.get("U2R", 0)
    suspicious = counts_24h.get("probe", 0)

    throughput_raw = redis.get(THROUGHPUT_KEY)
    throughput = float(throughput_raw) if throughput_raw else 0.0

    active_flows_raw = redis.get(ACTIVE_FLOWS_KEY)
    active_flows = int(active_flows_raw) if active_flows_raw else 0

    accuracy_raw = redis.get(MODEL_ACCURACY_KEY)
    accuracy = float(accuracy_raw) if accuracy_raw else 0.0

    drift_raw = redis.get("nids:metrics:drift_events_24h")
    drift_events = int(drift_raw) if drift_raw else 0

    return {
        "total_alerts_24h": total_alerts,
        "malicious_count": malicious,
        "suspicious_count": suspicious,
        "top_attack_classes": counts_24h,
        "top_source_ips": top_sources,
        "throughput_pps": throughput,
        "active_flows": active_flows,
        "model_accuracy": accuracy,
        "drift_detected": drift_events > 0,
    }


@router.get("/traffic")
async def traffic_stats(
    window_minutes: int = Query(default=60, ge=5, le=1440),
    redis: RedisClient = Depends(get_redis),
) -> Dict:
    """Returns traffic volume time series from Redis."""
    now = time.time()
    window_start = now - window_minutes * 60

    raw = redis.zrangebyscore("nids:metrics:traffic_ts", window_start, now)
    return {
        "window_minutes": window_minutes,
        "data_points": len(raw),
        "series": raw,
    }


@router.get("/anomaly-scores")
async def anomaly_score_distribution(
    hours: int = Query(default=1, ge=1, le=24),
    es: ElasticsearchClient = Depends(get_es),
) -> Dict:
    """Aggregated anomaly score histogram from recent alerts."""
    from_ts = time.time() - hours * 3600
    docs = es.query_alerts(from_ts=from_ts, size=1000)

    scores = [d.get("anomaly_score", 0) for d in docs]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    histogram = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        histogram[f"{lo:.1f}-{hi:.1f}"] = sum(1 for s in scores if lo <= s < hi)

    return {"hours": hours, "total": len(scores), "histogram": histogram}


@router.get("/health")
async def health_check(redis: RedisClient = Depends(get_redis)) -> Dict:
    return {
        "status": "ok",
        "redis": redis.ping(),
        "timestamp": time.time(),
    }
