"""
Packet capture layer using scapy. Captures raw packets from a network interface
and publishes them as structured JSON to the Kafka `raw_packets` topic.
"""

import json
import signal
import socket
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog
from scapy.all import (
    IP,
    TCP,
    UDP,
    ICMP,
    ARP,
    Ether,
    Raw,
    conf as scapy_conf,
    sniff,
)

from src.ingestion.kafka_producer import NIDSProducer
from src.config import settings

logger = structlog.get_logger(__name__)

scapy_conf.verb = 0  # suppress scapy output


@dataclass
class RawPacket:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    ip_flags: int
    ttl: int
    ip_len: int
    payload_len: int
    tcp_flags: Optional[str]
    tcp_seq: Optional[int]
    tcp_ack: Optional[int]
    tcp_window: Optional[int]
    icmp_type: Optional[int]
    icmp_code: Optional[int]
    capture_interface: str


class PacketCapture:
    """
    Captures network packets from a specified interface and publishes them
    to Kafka for downstream stream processing.
    """

    def __init__(
        self,
        interface: str = settings.CAPTURE_INTERFACE,
        bpf_filter: str = settings.CAPTURE_BPF_FILTER,
        producer: Optional[NIDSProducer] = None,
    ) -> None:
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.producer = producer or NIDSProducer()
        self._stop_event = threading.Event()
        self._packet_count = 0
        self._error_count = 0
        self._start_time: Optional[float] = None

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _extract_packet(self, pkt) -> Optional[RawPacket]:
        if not pkt.haslayer(IP):
            return None

        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        protocol = "OTHER"
        src_port = dst_port = None
        tcp_flags = tcp_seq = tcp_ack = tcp_window = None
        icmp_type = icmp_code = None

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            protocol = "TCP"
            src_port = tcp.sport
            dst_port = tcp.dport
            tcp_flags = str(tcp.flags)
            tcp_seq = tcp.seq
            tcp_ack = tcp.ack
            tcp_window = tcp.window
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            protocol = "UDP"
            src_port = udp.sport
            dst_port = udp.dport
        elif pkt.haslayer(ICMP):
            icmp = pkt[ICMP]
            protocol = "ICMP"
            icmp_type = icmp.type
            icmp_code = icmp.code

        payload_len = len(pkt[Raw].load) if pkt.haslayer(Raw) else 0

        return RawPacket(
            timestamp=float(pkt.time),
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            ip_flags=int(ip.flags),
            ttl=ip.ttl,
            ip_len=ip.len,
            payload_len=payload_len,
            tcp_flags=tcp_flags,
            tcp_seq=tcp_seq,
            tcp_ack=tcp_ack,
            tcp_window=tcp_window,
            icmp_type=icmp_type,
            icmp_code=icmp_code,
            capture_interface=self.interface,
        )

    def _on_packet(self, pkt) -> None:
        try:
            raw = self._extract_packet(pkt)
            if raw is None:
                return

            record = asdict(raw)
            record["captured_at"] = datetime.now(timezone.utc).isoformat()

            key = f"{raw.src_ip}:{raw.src_port}->{raw.dst_ip}:{raw.dst_port}"
            self.producer.produce(
                topic=settings.KAFKA_TOPIC_RAW_PACKETS,
                key=key,
                value=record,
            )
            self._packet_count += 1

            if self._packet_count % 10_000 == 0:
                elapsed = time.monotonic() - self._start_time
                pps = self._packet_count / max(elapsed, 1)
                logger.info(
                    "capture_stats",
                    packets=self._packet_count,
                    errors=self._error_count,
                    pps=round(pps, 1),
                    interface=self.interface,
                )
        except Exception:
            self._error_count += 1
            logger.exception("packet_processing_error")

    def start(self) -> None:
        logger.info(
            "capture_start",
            interface=self.interface,
            bpf_filter=self.bpf_filter,
        )
        self._start_time = time.monotonic()
        self.producer.start()

        sniff(
            iface=self.interface,
            filter=self.bpf_filter,
            prn=self._on_packet,
            store=False,
            stop_filter=lambda _: self._stop_event.is_set(),
        )

    def _shutdown(self, *_) -> None:
        logger.info("capture_shutdown", packets=self._packet_count)
        self._stop_event.set()
        self.producer.flush()
        self.producer.close()


if __name__ == "__main__":
    capture = PacketCapture()
    capture.start()
