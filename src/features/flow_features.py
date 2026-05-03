"""
Flow-based feature extraction.
Operates on a list of raw packet dicts that share the same 5-tuple and
produces a single feature vector representing the flow.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


FEATURE_NAMES = [
    "packet_count",
    "total_bytes",
    "total_payload_bytes",
    "flow_duration",
    "bytes_per_second",
    "packets_per_second",
    "mean_packet_size",
    "std_packet_size",
    "min_packet_size",
    "max_packet_size",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "syn_ratio",
    "rst_ratio",
    "fin_ratio",
    "mean_ttl",
    "unique_dst_ports",
    "fwd_bytes",
    "bwd_bytes",
    "fwd_pkt_count",
    "bwd_pkt_count",
    "fwd_bwd_ratio",
    "header_to_payload_ratio",
    "ip_flags_df_count",
    "ip_flags_mf_count",
]


class FlowFeatureExtractor:
    """
    Extracts flow-level features from a collection of packets belonging
    to the same network flow (identified by 5-tuple).
    """

    def extract(self, packets: List[Dict], src_ip: str) -> Dict[str, float]:
        if not packets:
            return {name: 0.0 for name in FEATURE_NAMES}

        timestamps = [p.get("timestamp", 0.0) for p in packets]
        ip_lens = [p.get("ip_len", 0) or 0 for p in packets]
        payload_lens = [p.get("payload_len", 0) or 0 for p in packets]
        ttls = [p.get("ttl", 64) or 64 for p in packets]
        ip_flags = [p.get("ip_flags", 0) or 0 for p in packets]
        tcp_flags_list = [p.get("tcp_flags", "") or "" for p in packets]

        flow_duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
        safe_duration = max(flow_duration, 1e-6)

        packet_count = len(packets)
        total_bytes = sum(ip_lens)
        total_payload = sum(payload_lens)
        total_headers = total_bytes - total_payload

        syn = sum(1 for f in tcp_flags_list if "S" in f)
        ack = sum(1 for f in tcp_flags_list if "A" in f)
        fin = sum(1 for f in tcp_flags_list if "F" in f)
        rst = sum(1 for f in tcp_flags_list if "R" in f)
        psh = sum(1 for f in tcp_flags_list if "P" in f)
        urg = sum(1 for f in tcp_flags_list if "U" in f)

        fwd_pkts = [p for p in packets if p.get("src_ip") == src_ip]
        bwd_pkts = [p for p in packets if p.get("src_ip") != src_ip]
        fwd_bytes = sum(p.get("ip_len", 0) or 0 for p in fwd_pkts)
        bwd_bytes = sum(p.get("ip_len", 0) or 0 for p in bwd_pkts)

        df_count = sum(1 for f in ip_flags if f & 0x02)  # Don't Fragment
        mf_count = sum(1 for f in ip_flags if f & 0x01)  # More Fragments

        dst_ports = set(p.get("dst_port") for p in packets if p.get("dst_port"))

        mean_size = float(np.mean(ip_lens)) if ip_lens else 0.0
        std_size = float(np.std(ip_lens)) if len(ip_lens) > 1 else 0.0

        return {
            "packet_count": float(packet_count),
            "total_bytes": float(total_bytes),
            "total_payload_bytes": float(total_payload),
            "flow_duration": flow_duration,
            "bytes_per_second": total_bytes / safe_duration,
            "packets_per_second": packet_count / safe_duration,
            "mean_packet_size": mean_size,
            "std_packet_size": std_size,
            "min_packet_size": float(min(ip_lens, default=0)),
            "max_packet_size": float(max(ip_lens, default=0)),
            "syn_count": float(syn),
            "ack_count": float(ack),
            "fin_count": float(fin),
            "rst_count": float(rst),
            "psh_count": float(psh),
            "urg_count": float(urg),
            "syn_ratio": syn / max(packet_count, 1),
            "rst_ratio": rst / max(packet_count, 1),
            "fin_ratio": fin / max(packet_count, 1),
            "mean_ttl": float(np.mean(ttls)) if ttls else 0.0,
            "unique_dst_ports": float(len(dst_ports)),
            "fwd_bytes": float(fwd_bytes),
            "bwd_bytes": float(bwd_bytes),
            "fwd_pkt_count": float(len(fwd_pkts)),
            "bwd_pkt_count": float(len(bwd_pkts)),
            "fwd_bwd_ratio": len(fwd_pkts) / max(len(bwd_pkts), 1),
            "header_to_payload_ratio": total_headers / max(total_payload, 1),
            "ip_flags_df_count": float(df_count),
            "ip_flags_mf_count": float(mf_count),
        }

    def to_vector(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
