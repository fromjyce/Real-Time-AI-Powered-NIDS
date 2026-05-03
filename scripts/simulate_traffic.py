"""
Traffic simulation script.
Generates synthetic benign and attack flow records and publishes them
to Kafka to validate the detection pipeline end-to-end.

Usage:
  python scripts/simulate_traffic.py --rate 500 --attack-mix 0.1 --duration 120
"""

import json
import random
import time
from typing import Dict, List

import click
import structlog

from src.ingestion.kafka_producer import NIDSProducer
from src.config import settings

logger = structlog.get_logger(__name__)

# Each profile defines (min, max) ranges so build_flow samples fresh values per call.
ATTACK_PROFILES: Dict[str, Dict] = {
    "syn_flood": {
        "protocol": "TCP",
        "syn_ratio": 0.95,
        "packet_count_range": (500, 2000),
        "packets_per_second_range": (800.0, 3000.0),
        "bytes_per_second_range": (50_000.0, 200_000.0),
        "dst_port": 80,
        "tcp_flags": "S",
        "ttl": 64,
    },
    "port_scan": {
        "protocol": "TCP",
        "unique_dst_ports_1m_range": (60, 200),
        "packet_count_range": (100, 500),
        "packets_per_second_range": (10.0, 50.0),
        "syn_ratio": 0.8,
        "rst_ratio": 0.6,
        "tcp_flags": "SR",
        "flow_duration_range": (5.0, 30.0),
        "ttl": 128,
    },
    "ssh_brute_force": {
        "protocol": "TCP",
        "dst_port": 22,
        "failed_auth_count_range": (15, 50),
        "conn_rate_1m_range": (8.0, 20.0),
        "packet_count_range": (50, 200),
        "packets_per_second_range": (5.0, 15.0),
        "ttl": 64,
    },
    "udp_flood": {
        "protocol": "UDP",
        "packets_per_second_range": (6000.0, 20000.0),
        "bytes_per_second_range": (500_000.0, 2_000_000.0),
        "packet_count_range": (10000, 50000),
        "flow_duration_range": (1.0, 10.0),
        "dst_port": 53,
        "ttl": 64,
    },
    "slowloris": {
        "protocol": "TCP",
        "dst_port": 80,
        "flow_duration_range": (180.0, 600.0),
        "packets_per_second_range": (0.05, 0.3),
        "packet_count_range": (5, 30),
        "bytes_per_second_range": (10.0, 50.0),
        "ttl": 64,
    },
    "dns_amplification": {
        "protocol": "UDP",
        "dst_port": 53,
        "fwd_bytes": 50,
        "bwd_bytes_range": (2000, 10000),
        "packet_count_range": (100, 500),
        "packets_per_second_range": (50.0, 200.0),
        "ttl": 128,
    },
}

BENIGN_PROFILES: List[Dict] = [
    {"protocol": "TCP",  "dst_port": 443,  "pps_range": (1.0, 20.0),   "bps_range": (500.0, 50_000.0),    "dur_range": (0.5, 60.0),   "ttl": 64},
    {"protocol": "TCP",  "dst_port": 80,   "pps_range": (2.0, 30.0),   "bps_range": (1000.0, 100_000.0),  "dur_range": (0.1, 30.0),   "ttl": 64},
    {"protocol": "UDP",  "dst_port": 53,   "pps_range": (0.1, 5.0),    "bps_range": (50.0, 2_000.0),      "dur_range": (0.01, 1.0),   "ttl": 128},
    {"protocol": "TCP",  "dst_port": 22,   "pps_range": (0.5, 5.0),    "bps_range": (200.0, 5_000.0),     "dur_range": (10.0, 300.0), "ttl": 64},
    {"protocol": "ICMP", "dst_port": None, "pps_range": (0.1, 2.0),    "bps_range": (50.0, 500.0),        "dur_range": (0.1, 5.0),    "ttl": 64},
]


def _sample(spec: Dict, key: str, is_int: bool = False):
    """Sample a value from a (min, max) range stored under key+'_range', or return a fixed value."""
    range_key = key + "_range"
    if range_key in spec:
        lo, hi = spec[range_key]
        return random.randint(int(lo), int(hi)) if is_int else random.uniform(lo, hi)
    return spec.get(key)


def random_ip(private: bool = False) -> str:
    if private:
        return f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
    prefixes = ["1.1", "8.8", "45.33", "203.0", "151.101", "104.16"]
    return f"{random.choice(prefixes)}.{random.randint(0,255)}.{random.randint(1,254)}"


def build_flow(attack_type: str = "") -> Dict:
    if attack_type and attack_type in ATTACK_PROFILES:
        spec = ATTACK_PROFILES[attack_type]
        label = attack_type
        dst_port = spec.get("dst_port", random.randint(1, 1024))
        protocol = spec.get("protocol", "TCP")
        packet_count = _sample(spec, "packet_count", is_int=True) or random.randint(5, 100)
        flow_duration = _sample(spec, "flow_duration") or random.uniform(0.01, 60)
        pps = _sample(spec, "packets_per_second") or random.uniform(1, 50)
        bps = _sample(spec, "bytes_per_second") or random.uniform(100, 10_000)
        syn_ratio = spec.get("syn_ratio", random.uniform(0, 0.3))
        rst_ratio = spec.get("rst_ratio", random.uniform(0, 0.1))
        unique_ports = _sample(spec, "unique_dst_ports_1m", is_int=True) or random.randint(1, 5)
        failed_auth = _sample(spec, "failed_auth_count", is_int=True) or 0
        conn_rate = _sample(spec, "conn_rate_1m") or random.uniform(0.1, 3)
        fwd_bytes = spec.get("fwd_bytes", random.randint(100, 50_000))
        bwd_bytes = _sample(spec, "bwd_bytes", is_int=True) or random.randint(100, 50_000)
        tcp_flags = spec.get("tcp_flags", "SA")
        ttl = spec.get("ttl", 64)
    else:
        p = random.choice(BENIGN_PROFILES)
        label = "benign"
        dst_port = p["dst_port"] or random.randint(1, 1024)
        protocol = p["protocol"]
        packet_count = random.randint(5, 100)
        flow_duration = random.uniform(*p["dur_range"])
        pps = random.uniform(*p["pps_range"])
        bps = random.uniform(*p["bps_range"])
        syn_ratio = random.uniform(0, 0.2)
        rst_ratio = random.uniform(0, 0.05)
        unique_ports = random.randint(1, 3)
        failed_auth = 0
        conn_rate = random.uniform(0.1, 2)
        fwd_bytes = random.randint(100, 20_000)
        bwd_bytes = random.randint(100, 20_000)
        tcp_flags = "SA"
        ttl = p["ttl"]

    return {
        "timestamp": time.time(),
        "src_ip": random_ip(private=True),
        "dst_ip": random_ip(private=False),
        "src_port": random.randint(1024, 65535),
        "dst_port": dst_port,
        "protocol": protocol,
        "packet_count": packet_count,
        "total_bytes": int(bps * max(flow_duration, 0.01)),
        "flow_duration": flow_duration,
        "packets_per_second": pps,
        "bytes_per_second": bps,
        "mean_packet_size": random.uniform(64, 1500),
        "std_packet_size": random.uniform(0, 200),
        "syn_ratio": syn_ratio,
        "rst_ratio": rst_ratio,
        "fin_ratio": random.uniform(0, 0.1),
        "unique_dst_ports_1m": unique_ports,
        "failed_auth_count": failed_auth,
        "conn_rate_1m": conn_rate,
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        "tcp_flags": tcp_flags,
        "ttl": ttl,
        "_label": label,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@click.command()
@click.option("--rate", default=100, help="Flows per second to generate", type=int)
@click.option("--attack-mix", default=0.05, help="Fraction of flows that are attacks", type=float)
@click.option("--duration", default=60, help="Duration in seconds (0 = infinite)", type=int)
@click.option("--topic", default=settings.KAFKA_TOPIC_PROCESSED_FEATURES, help="Kafka topic")
@click.option("--dry-run", is_flag=True, help="Print instead of publishing to Kafka")
def main(rate: int, attack_mix: float, duration: int, topic: str, dry_run: bool) -> None:
    producer = NIDSProducer()
    if not dry_run:
        producer.start()

    attack_types = list(ATTACK_PROFILES.keys())
    interval = 1.0 / max(rate, 1)
    deadline = time.monotonic() + duration if duration > 0 else float("inf")
    sent = 0
    attacks = 0

    logger.info(
        "simulation_start",
        rate=rate,
        attack_mix=attack_mix,
        duration=duration,
        dry_run=dry_run,
    )

    try:
        while time.monotonic() < deadline:
            t0 = time.monotonic()

            is_attack = random.random() < attack_mix
            attack_type = random.choice(attack_types) if is_attack else ""
            flow = build_flow(attack_type=attack_type)

            key = f"{flow['src_ip']}:{flow['src_port']}->{flow['dst_ip']}:{flow['dst_port']}"

            if dry_run:
                print(json.dumps(flow, indent=2))
            else:
                producer.produce(topic=topic, key=key, value=flow)

            sent += 1
            if is_attack:
                attacks += 1

            elapsed = time.monotonic() - t0
            sleep = max(0, interval - elapsed)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass
    finally:
        if not dry_run:
            producer.flush()
            producer.close()
        logger.info(
            "simulation_done",
            sent=sent,
            attacks=attacks,
            attack_rate=round(attacks / max(sent, 1), 4),
        )


if __name__ == "__main__":
    main()
