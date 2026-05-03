"""
Zeek log parser. Tails Zeek conn.log, dns.log, and http.log files
and feeds parsed records into Kafka for stream processing.

Zeek log format reference:
  https://docs.zeek.org/en/master/logs/conn.html
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import structlog

from src.ingestion.kafka_producer import NIDSProducer
from src.config import settings

logger = structlog.get_logger(__name__)

ZEEK_SEPARATOR = "\t"
UNSET_FIELD = "-"
EMPTY_FIELD = "(empty)"


@dataclass
class ZeekConnRecord:
    ts: float
    uid: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: str
    service: Optional[str]
    duration: Optional[float]
    orig_bytes: Optional[int]
    resp_bytes: Optional[int]
    conn_state: str
    missed_bytes: int
    orig_pkts: Optional[int]
    orig_ip_bytes: Optional[int]
    resp_pkts: Optional[int]
    resp_ip_bytes: Optional[int]
    history: Optional[str]
    tunnel_parents: List[str] = field(default_factory=list)


class ZeekLogParser:
    """
    Parses Zeek log files and publishes structured records to Kafka.
    Supports live tail (follows the file as Zeek appends to it).
    """

    CONN_FIELDS = [
        "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
        "proto", "service", "duration", "orig_bytes", "resp_bytes",
        "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
        "tunnel_parents",
    ]

    def __init__(
        self,
        log_dir: str = "/opt/zeek/logs/current",
        producer: Optional[NIDSProducer] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.producer = producer or NIDSProducer()

    @staticmethod
    def _parse_value(value: str, cast=str):
        if value in (UNSET_FIELD, EMPTY_FIELD, ""):
            return None
        try:
            return cast(value)
        except (ValueError, TypeError):
            return None

    def _parse_conn_line(self, line: str, field_map: Dict[str, int]) -> Optional[ZeekConnRecord]:
        parts = line.rstrip("\n").split(ZEEK_SEPARATOR)
        if len(parts) < len(field_map):
            return None

        def get(name):
            idx = field_map.get(name)
            return parts[idx] if idx is not None and idx < len(parts) else UNSET_FIELD

        tunnel_raw = get("tunnel_parents")
        tunnel_parents = (
            [] if tunnel_raw in (UNSET_FIELD, EMPTY_FIELD)
            else tunnel_raw.split(",")
        )

        return ZeekConnRecord(
            ts=self._parse_value(get("ts"), float) or 0.0,
            uid=get("uid"),
            src_ip=get("id.orig_h"),
            src_port=self._parse_value(get("id.orig_p"), int) or 0,
            dst_ip=get("id.resp_h"),
            dst_port=self._parse_value(get("id.resp_p"), int) or 0,
            proto=get("proto"),
            service=self._parse_value(get("service")),
            duration=self._parse_value(get("duration"), float),
            orig_bytes=self._parse_value(get("orig_bytes"), int),
            resp_bytes=self._parse_value(get("resp_bytes"), int),
            conn_state=get("conn_state"),
            missed_bytes=self._parse_value(get("missed_bytes"), int) or 0,
            orig_pkts=self._parse_value(get("orig_pkts"), int),
            orig_ip_bytes=self._parse_value(get("orig_ip_bytes"), int),
            resp_pkts=self._parse_value(get("resp_pkts"), int),
            resp_ip_bytes=self._parse_value(get("resp_ip_bytes"), int),
            history=self._parse_value(get("history")),
            tunnel_parents=tunnel_parents,
        )

    async def _tail_file(self, path: Path) -> AsyncIterator[str]:
        with open(path, "r") as fh:
            fh.seek(0, 2)  # seek to end for live tail
            while True:
                line = fh.readline()
                if not line:
                    await asyncio.sleep(0.1)
                    continue
                yield line

    async def tail_conn_log(self) -> None:
        conn_log = self.log_dir / "conn.log"
        logger.info("zeek_tail_start", path=str(conn_log))

        field_map: Dict[str, int] = {}
        header_parsed = False
        records_sent = 0

        async for line in self._tail_file(conn_log):
            if line.startswith("#fields"):
                headers = line.lstrip("#fields").strip().split(ZEEK_SEPARATOR)
                field_map = {name: idx for idx, name in enumerate(headers)}
                header_parsed = True
                continue
            if line.startswith("#"):
                continue
            if not header_parsed:
                continue

            record = self._parse_conn_line(line, field_map)
            if record is None:
                continue

            payload = {
                "ts": record.ts,
                "uid": record.uid,
                "src_ip": record.src_ip,
                "src_port": record.src_port,
                "dst_ip": record.dst_ip,
                "dst_port": record.dst_port,
                "proto": record.proto,
                "service": record.service,
                "duration": record.duration,
                "orig_bytes": record.orig_bytes,
                "resp_bytes": record.resp_bytes,
                "conn_state": record.conn_state,
                "missed_bytes": record.missed_bytes,
                "orig_pkts": record.orig_pkts,
                "orig_ip_bytes": record.orig_ip_bytes,
                "resp_pkts": record.resp_pkts,
                "resp_ip_bytes": record.resp_ip_bytes,
                "history": record.history,
                "source": "zeek",
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }

            self.producer.produce(
                topic=settings.KAFKA_TOPIC_RAW_PACKETS,
                key=record.uid,
                value=payload,
            )
            records_sent += 1

            if records_sent % 5_000 == 0:
                logger.info("zeek_records_sent", count=records_sent)

    def run(self) -> None:
        self.producer.start()
        asyncio.run(self.tail_conn_log())
