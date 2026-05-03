"""
Elasticsearch client wrapper.
Manages index templates, bulk ingestion, and alert querying.
"""

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog
from elasticsearch import Elasticsearch, helpers, NotFoundError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

if TYPE_CHECKING:
    from src.alerting.alert_manager import Alert

logger = structlog.get_logger(__name__)

ALERT_INDEX_TEMPLATE = {
    "mappings": {
        "properties": {
            "timestamp": {"type": "date", "format": "epoch_second"},
            "src_ip": {"type": "ip"},
            "dst_ip": {"type": "ip"},
            "dst_port": {"type": "integer"},
            "protocol": {"type": "keyword"},
            "threat_label": {"type": "keyword"},
            "attack_class": {"type": "keyword"},
            "severity": {"type": "keyword"},
            "fusion_score": {"type": "float"},
            "ml_score": {"type": "float"},
            "anomaly_score": {"type": "float"},
            "signature_score": {"type": "float"},
            "behavioral_deviation": {"type": "float"},
            "threat_intel_score": {"type": "float"},
            "matched_rules": {"type": "keyword"},
            "adversarial_signals": {"type": "keyword"},
            "confidence": {"type": "float"},
        }
    },
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 0,
        "refresh_interval": "5s",
    },
}

FLOW_INDEX_TEMPLATE = {
    "mappings": {
        "properties": {
            "src_ip": {"type": "ip"},
            "dst_ip": {"type": "ip"},
            "dst_port": {"type": "integer"},
            "protocol": {"type": "keyword"},
            "packet_count": {"type": "long"},
            "total_bytes": {"type": "long"},
            "flow_duration": {"type": "float"},
            "bytes_per_second": {"type": "float"},
            "packets_per_second": {"type": "float"},
            "window_start": {"type": "date"},
            "window_end": {"type": "date"},
        }
    },
    "settings": {
        "number_of_shards": 4,
        "number_of_replicas": 0,
        "refresh_interval": "10s",
    },
}


class ElasticsearchClient:
    def __init__(self, host: Optional[str] = None) -> None:
        self._es = Elasticsearch(
            host or settings.ELASTICSEARCH_HOST,
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        self._ensure_indices()

    def _ensure_indices(self) -> None:
        for index, template in [
            (settings.ELASTICSEARCH_INDEX_ALERTS, ALERT_INDEX_TEMPLATE),
            (settings.ELASTICSEARCH_INDEX_FLOWS, FLOW_INDEX_TEMPLATE),
        ]:
            try:
                if not self._es.indices.exists(index=index):
                    self._es.indices.create(index=index, body=template)
                    logger.info("es_index_created", index=index)
            except Exception as e:
                logger.warning("es_index_create_failed", index=index, error=str(e))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def index_alert(self, alert: "Alert") -> None:
        doc = {
            "id": alert.id,
            "timestamp": alert.timestamp,
            "src_ip": alert.src_ip,
            "dst_ip": alert.dst_ip,
            "dst_port": alert.dst_port,
            "protocol": alert.protocol,
            "threat_label": alert.threat_label,
            "attack_class": alert.attack_class,
            "severity": alert.severity,
            "fusion_score": alert.fusion_score,
            "ml_score": alert.ml_score,
            "anomaly_score": alert.anomaly_score,
            "signature_score": alert.signature_score,
            "behavioral_deviation": alert.behavioral_deviation,
            "threat_intel_score": alert.threat_intel_score,
            "matched_rules": alert.matched_rules,
            "adversarial_signals": alert.adversarial_signals,
            "confidence": alert.confidence,
        }
        self._es.index(
            index=settings.ELASTICSEARCH_INDEX_ALERTS,
            id=alert.id,
            document=doc,
        )

    def bulk_index(self, index: str, docs: List[Dict[str, Any]]) -> None:
        actions = [
            {"_index": index, "_source": doc}
            for doc in docs
        ]
        success, errors = helpers.bulk(self._es, actions, raise_on_error=False)
        if errors:
            logger.warning("es_bulk_errors", count=len(errors))

    def query_alerts(
        self,
        severity: Optional[str] = None,
        attack_class: Optional[str] = None,
        src_ip: Optional[str] = None,
        from_ts: Optional[float] = None,
        to_ts: Optional[float] = None,
        size: int = 100,
    ) -> List[Dict]:
        must = []
        if severity:
            must.append({"term": {"severity": severity}})
        if attack_class:
            must.append({"term": {"attack_class": attack_class}})
        if src_ip:
            must.append({"term": {"src_ip": src_ip}})
        if from_ts or to_ts:
            rng: Dict[str, Any] = {}
            if from_ts:
                rng["gte"] = from_ts
            if to_ts:
                rng["lte"] = to_ts
            must.append({"range": {"timestamp": rng}})

        query = {"bool": {"must": must}} if must else {"match_all": {}}
        resp = self._es.search(
            index=settings.ELASTICSEARCH_INDEX_ALERTS,
            query=query,
            sort=[{"timestamp": {"order": "desc"}}],
            size=size,
        )
        return [hit["_source"] for hit in resp["hits"]["hits"]]

    def alert_counts_by_class(self, hours: int = 24) -> Dict[str, int]:
        from_ts = time.time() - hours * 3600
        resp = self._es.search(
            index=settings.ELASTICSEARCH_INDEX_ALERTS,
            query={"range": {"timestamp": {"gte": from_ts}}},
            aggs={"by_class": {"terms": {"field": "attack_class", "size": 20}}},
            size=0,
        )
        buckets = resp.get("aggregations", {}).get("by_class", {}).get("buckets", [])
        return {b["key"]: b["doc_count"] for b in buckets}

    def top_source_ips(self, hours: int = 1, limit: int = 10) -> List[Dict]:
        from_ts = time.time() - hours * 3600
        resp = self._es.search(
            index=settings.ELASTICSEARCH_INDEX_ALERTS,
            query={"range": {"timestamp": {"gte": from_ts}}},
            aggs={"top_ips": {"terms": {"field": "src_ip", "size": limit}}},
            size=0,
        )
        buckets = resp.get("aggregations", {}).get("top_ips", {}).get("buckets", [])
        return [{"ip": b["key"], "count": b["doc_count"]} for b in buckets]
