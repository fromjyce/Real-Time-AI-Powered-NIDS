"""Tests for feature extraction modules."""

import time
from typing import Dict, List

import numpy as np
import pytest

from src.features.flow_features import FlowFeatureExtractor, FEATURE_NAMES
from src.features.statistical_features import StatisticalFeatureExtractor, STAT_FEATURE_NAMES
from src.features.behavioral_features import (
    BehavioralFeatureExtractor,
    ConnectionRecord,
    BEHAVIORAL_FEATURE_NAMES,
)
from src.processing.windowing import SlidingWindow, TumblingWindow, FlowTracker


def make_packet(
    src_ip: str = "192.168.1.10",
    dst_ip: str = "1.2.3.4",
    protocol: str = "TCP",
    ip_len: int = 500,
    payload_len: int = 400,
    tcp_flags: str = "SA",
    timestamp: float = None,
    ttl: int = 64,
) -> Dict:
    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": 12345,
        "dst_port": 443,
        "protocol": protocol,
        "ip_len": ip_len,
        "payload_len": payload_len,
        "tcp_flags": tcp_flags,
        "timestamp": timestamp or time.time(),
        "ttl": ttl,
        "ip_flags": 0x02,
    }


class TestFlowFeatureExtractor:
    def setup_method(self):
        self.extractor = FlowFeatureExtractor()

    def test_returns_all_features(self):
        packets = [make_packet() for _ in range(10)]
        features = self.extractor.extract(packets, src_ip="192.168.1.10")
        assert set(features.keys()) == set(FEATURE_NAMES)

    def test_empty_packets_returns_zeros(self):
        features = self.extractor.extract([], src_ip="192.168.1.10")
        assert all(v == 0.0 for v in features.values())

    def test_syn_ratio_calculation(self):
        packets = [
            make_packet(tcp_flags="S"),
            make_packet(tcp_flags="S"),
            make_packet(tcp_flags="SA"),
            make_packet(tcp_flags="A"),
        ]
        features = self.extractor.extract(packets, src_ip="192.168.1.10")
        assert abs(features["syn_ratio"] - 0.5) < 0.01

    def test_packet_count_matches(self):
        n = 42
        packets = [make_packet() for _ in range(n)]
        features = self.extractor.extract(packets, src_ip="192.168.1.10")
        assert features["packet_count"] == n

    def test_fwd_bwd_bytes_split(self):
        src = "192.168.1.10"
        fwd = [make_packet(src_ip=src, ip_len=100) for _ in range(5)]
        bwd = [make_packet(src_ip="1.2.3.4", dst_ip=src, ip_len=200) for _ in range(5)]
        features = self.extractor.extract(fwd + bwd, src_ip=src)
        assert features["fwd_bytes"] == 500.0
        assert features["bwd_bytes"] == 1000.0

    def test_to_vector_length(self):
        features = self.extractor.extract([make_packet()], src_ip="192.168.1.10")
        vec = self.extractor.to_vector(features)
        assert len(vec) == len(FEATURE_NAMES)
        assert vec.dtype == np.float32


class TestStatisticalFeatureExtractor:
    def setup_method(self):
        self.extractor = StatisticalFeatureExtractor()

    def test_returns_all_features(self):
        now = time.time()
        pkts = []
        for i in range(20):
            p = make_packet()
            p["timestamp"] = now + i * 0.1
            pkts.append(p)

        features = self.extractor.extract(pkts)
        assert set(features.keys()) == set(STAT_FEATURE_NAMES)

    def test_iat_mean_is_approx_correct(self):
        now = time.time()
        packets = []
        for i in range(10):
            p = make_packet()
            p["timestamp"] = now + i * 0.5  # exactly 0.5s apart
            packets.append(p)

        features = self.extractor.extract(packets)
        assert abs(features["iat_mean"] - 0.5) < 0.01

    def test_single_packet_returns_zeros(self):
        features = self.extractor.extract([make_packet()])
        assert features["iat_mean"] == 0.0

    def test_burst_detection(self):
        now = time.time()
        # 5 packets in rapid succession (burst), then 1 after a gap
        packets = []
        for i in range(5):
            p = make_packet()
            p["timestamp"] = now + i * 0.01
            packets.append(p)
        after = make_packet()
        after["timestamp"] = now + 10.0
        packets.append(after)

        features = self.extractor.extract(packets)
        assert features["burst_count"] >= 1


class TestBehavioralFeatureExtractor:
    def setup_method(self):
        self.extractor = BehavioralFeatureExtractor()

    def _make_record(self, dst_port: int = 443, failed: bool = False, syn_only: bool = False) -> ConnectionRecord:
        return ConnectionRecord(
            timestamp=time.time(),
            dst_ip="1.2.3.4",
            dst_port=dst_port,
            protocol="TCP",
            conn_state="SF" if not failed else "RSTO",
            duration=random_duration(),
            is_failed=failed,
            is_syn_only=syn_only,
        )

    def test_returns_all_features(self):
        ip = "192.168.1.100"
        for _ in range(10):
            self.extractor.record(ip, self._make_record())
        features = self.extractor.extract(ip)
        assert set(features.keys()) == set(BEHAVIORAL_FEATURE_NAMES)

    def test_unknown_ip_returns_zeros(self):
        features = self.extractor.extract("10.99.99.99")
        assert all(v == 0.0 for v in features.values())

    def test_failed_auth_count(self):
        ip = "192.168.1.50"
        for _ in range(15):
            self.extractor.record(ip, self._make_record(dst_port=22, failed=True))
        features = self.extractor.extract(ip)
        assert features["failed_auth_count"] >= 10

    def test_port_scan_signature(self):
        ip = "192.168.1.77"
        for port in range(1, 80):
            r = ConnectionRecord(
                timestamp=time.time(),
                dst_ip="10.0.0.1",
                dst_port=port,
                protocol="TCP",
                conn_state="RSTO",
                duration=0.001,
                is_failed=True,
                is_syn_only=True,
            )
            self.extractor.record(ip, r)
        features = self.extractor.extract(ip)
        assert features["unique_dst_ports_1m"] >= 50


def random_duration() -> float:
    import random
    return random.uniform(0.1, 5.0)


class TestSlidingWindow:
    def test_evicts_old_records(self):
        win = SlidingWindow[int](window_size=1.0, slide_size=0.5)
        now = time.monotonic()
        win.add("key", 1, ts=now - 2.0)  # expired
        win.add("key", 2, ts=now - 0.5)  # active
        result = win.get_window("key", now=now)
        assert 1 not in result
        assert 2 in result

    def test_slide_due(self):
        win = SlidingWindow[int](window_size=10.0, slide_size=1.0)
        now = time.monotonic()
        win.add("k", 1, ts=now)
        assert win.is_slide_due("k", now=now + 1.1)
        assert not win.is_slide_due("k", now=now + 1.2)


class TestFlowTracker:
    def test_flow_state_accumulates(self):
        from src.processing.windowing import FlowKey
        tracker = FlowTracker()
        fk = FlowKey("192.168.1.1", "10.0.0.1", 12345, 80, "TCP")
        for i in range(5):
            tracker.update(fk, {"ip_len": 100, "timestamp": time.time(), "tcp_flags": "SA"})
        flow = tracker._flows[str(fk)]
        assert flow["packet_count"] == 5
        assert flow["total_bytes"] == 500

    def test_expired_flows_evicted(self):
        from src.processing.windowing import FlowKey
        tracker = FlowTracker(flow_timeout=0.01)
        fk = FlowKey("10.0.0.1", "10.0.0.2", 1111, 80, "TCP")
        tracker.update(fk, {"ip_len": 50, "timestamp": time.time() - 1, "tcp_flags": ""})
        evicted = tracker.evict_expired(now=time.time())
        assert len(evicted) == 1
