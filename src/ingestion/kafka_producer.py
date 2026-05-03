"""
Kafka producer wrapper with JSON serialisation, retry logic,
and per-topic delivery callbacks.
"""

import json
import threading
from typing import Any, Dict, Optional

import structlog
from confluent_kafka import Producer, KafkaError, KafkaException
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = structlog.get_logger(__name__)


class NIDSProducer:
    """
    Thread-safe Kafka producer.  Call `start()` before producing and
    `flush()` / `close()` before shutdown.
    """

    def __init__(self, bootstrap_servers: Optional[str] = None) -> None:
        self._bootstrap = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS
        self._producer: Optional[Producer] = None
        self._lock = threading.Lock()
        self._delivered = 0
        self._failed = 0

    def start(self) -> None:
        conf = {
            "bootstrap.servers": self._bootstrap,
            "acks": "all",
            "retries": 5,
            "retry.backoff.ms": 500,
            "compression.type": "lz4",
            "linger.ms": 5,
            "batch.size": 65536,
            "enable.idempotence": True,
        }
        self._producer = Producer(conf)
        logger.info("kafka_producer_started", brokers=self._bootstrap)

    def _delivery_callback(self, err: Optional[KafkaError], msg) -> None:
        if err:
            self._failed += 1
            logger.warning(
                "kafka_delivery_failed",
                topic=msg.topic(),
                error=str(err),
                total_failed=self._failed,
            )
        else:
            self._delivered += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        reraise=True,
    )
    def produce(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if self._producer is None:
            raise RuntimeError("Producer not started. Call start() first.")

        encoded_value = json.dumps(value, default=str).encode("utf-8")
        encoded_key = key.encode("utf-8") if key else None
        kafka_headers = list((headers or {}).items())

        with self._lock:
            self._producer.produce(
                topic=topic,
                key=encoded_key,
                value=encoded_value,
                headers=kafka_headers,
                on_delivery=self._delivery_callback,
            )
            # Poll to trigger delivery callbacks without blocking
            self._producer.poll(0)

    def flush(self, timeout: float = 30.0) -> None:
        if self._producer:
            remaining = self._producer.flush(timeout=timeout)
            if remaining > 0:
                logger.warning("kafka_flush_incomplete", remaining=remaining)

    def close(self) -> None:
        self.flush()
        logger.info(
            "kafka_producer_closed",
            delivered=self._delivered,
            failed=self._failed,
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {"delivered": self._delivered, "failed": self._failed}
