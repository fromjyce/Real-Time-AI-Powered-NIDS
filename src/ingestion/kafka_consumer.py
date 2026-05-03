"""
Kafka consumer wrapper with automatic offset management,
graceful shutdown, and per-record deserialization.
"""

import json
import signal
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional

import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException, Message, TopicPartition

from src.config import settings

logger = structlog.get_logger(__name__)


class NIDSConsumer:
    """
    Wraps a Confluent Kafka consumer for a single consumer group.
    Supports both a manual poll loop and a convenience `stream()` generator.
    """

    def __init__(
        self,
        topics: List[str],
        group_id: Optional[str] = None,
        bootstrap_servers: Optional[str] = None,
        auto_offset_reset: str = "latest",
    ) -> None:
        self.topics = topics
        self._bootstrap = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS
        self._group_id = group_id or settings.KAFKA_GROUP_ID
        self._auto_offset_reset = auto_offset_reset
        self._consumer: Optional[Consumer] = None
        self._running = threading.Event()
        self._consumed = 0
        self._errors = 0

    def start(self) -> None:
        conf = {
            "bootstrap.servers": self._bootstrap,
            "group.id": self._group_id,
            "auto.offset.reset": self._auto_offset_reset,
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 45000,
            "heartbeat.interval.ms": 3000,
        }
        self._consumer = Consumer(conf)
        self._consumer.subscribe(self.topics, on_assign=self._on_assign)
        self._running.set()
        logger.info(
            "kafka_consumer_started",
            topics=self.topics,
            group=self._group_id,
        )

    def _on_assign(self, consumer, partitions: List[TopicPartition]) -> None:
        logger.info(
            "kafka_partitions_assigned",
            partitions=[(p.topic, p.partition) for p in partitions],
        )

    def stream(self, timeout: float = 1.0) -> Iterator[Dict[str, Any]]:
        """Yields deserialized records one at a time. Commits offsets after each."""
        if self._consumer is None:
            raise RuntimeError("Consumer not started. Call start() first.")

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        while self._running.is_set():
            msg: Optional[Message] = self._consumer.poll(timeout=timeout)

            if msg is None:
                continue

            if msg.error():
                err = msg.error()
                if err.code() == KafkaError._PARTITION_EOF:
                    continue
                self._errors += 1
                logger.warning("kafka_consumer_error", error=str(err))
                continue

            try:
                value = json.loads(msg.value().decode("utf-8"))
                self._consumed += 1
                yield value
                self._consumer.commit(message=msg, asynchronous=True)
            except json.JSONDecodeError:
                logger.warning("kafka_bad_json", offset=msg.offset(), partition=msg.partition())
            except Exception:
                logger.exception("kafka_record_processing_error")

    def _signal_handler(self, *_) -> None:
        logger.info("kafka_consumer_stopping")
        self._running.clear()

    def close(self) -> None:
        self._running.clear()
        if self._consumer:
            self._consumer.close()
        logger.info(
            "kafka_consumer_closed",
            consumed=self._consumed,
            errors=self._errors,
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {"consumed": self._consumed, "errors": self._errors}
