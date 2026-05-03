"""
In-process sliding and tumbling window implementations used
by the feature extractor when Spark is not available (e.g., unit tests
or lightweight deployment scenarios).
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class WindowedRecord(Generic[T]):
    timestamp: float
    key: str
    value: T


class SlidingWindow(Generic[T]):
    """
    Time-based sliding window keyed by an arbitrary string.
    Records older than `window_size` seconds are evicted on every `add()`.
    """

    def __init__(self, window_size: float, slide_size: float) -> None:
        if slide_size > window_size:
            raise ValueError("slide_size must be <= window_size")
        self.window_size = window_size
        self.slide_size = slide_size
        self._buckets: Dict[str, Deque[WindowedRecord]] = defaultdict(deque)
        self._last_slide: Dict[str, float] = {}

    def add(self, key: str, value: T, ts: Optional[float] = None) -> None:
        now = ts or time.monotonic()
        self._buckets[key].append(WindowedRecord(timestamp=now, key=key, value=value))
        self._evict(key, now)

    def _evict(self, key: str, now: float) -> None:
        cutoff = now - self.window_size
        bucket = self._buckets[key]
        while bucket and bucket[0].timestamp < cutoff:
            bucket.popleft()

    def get_window(self, key: str, now: Optional[float] = None) -> List[T]:
        ts = now or time.monotonic()
        self._evict(key, ts)
        return [r.value for r in self._buckets[key]]

    def is_slide_due(self, key: str, now: Optional[float] = None) -> bool:
        ts = now or time.monotonic()
        last = self._last_slide.get(key, 0.0)
        if ts - last >= self.slide_size:
            self._last_slide[key] = ts
            return True
        return False

    def all_keys(self) -> List[str]:
        return [k for k, v in self._buckets.items() if v]


class TumblingWindow(Generic[T]):
    """
    Time-based tumbling (non-overlapping) window keyed by an arbitrary string.
    Fires when `window_size` seconds have elapsed since the first record in the window.
    """

    def __init__(self, window_size: float, on_fire: Callable[[str, List[T]], None]) -> None:
        self.window_size = window_size
        self.on_fire = on_fire
        self._buckets: Dict[str, List[T]] = defaultdict(list)
        self._window_start: Dict[str, float] = {}

    def add(self, key: str, value: T, ts: Optional[float] = None) -> None:
        now = ts or time.monotonic()

        if key not in self._window_start:
            self._window_start[key] = now

        self._buckets[key].append(value)

        if now - self._window_start[key] >= self.window_size:
            self._fire(key)

    def _fire(self, key: str) -> None:
        records = self._buckets.pop(key, [])
        self._window_start.pop(key, None)
        if records:
            self.on_fire(key, records)

    def flush_all(self) -> None:
        for key in list(self._buckets.keys()):
            self._fire(key)


@dataclass
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def __str__(self) -> str:
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}/{self.protocol}"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class FlowTracker:
    """
    Tracks per-flow state using a sliding window.
    Used by the behavioral feature extractor to maintain connection state.
    """

    def __init__(self, flow_timeout: float = 120.0) -> None:
        self.flow_timeout = flow_timeout
        self._flows: Dict[str, Dict] = {}
        self._last_seen: Dict[str, float] = {}

    def update(self, flow_key: FlowKey, packet: Dict) -> Dict:
        key = str(flow_key)
        now = packet.get("timestamp", time.time())

        if key not in self._flows:
            self._flows[key] = {
                "flow_key": str(flow_key),
                "src_ip": flow_key.src_ip,
                "dst_ip": flow_key.dst_ip,
                "src_port": flow_key.src_port,
                "dst_port": flow_key.dst_port,
                "protocol": flow_key.protocol,
                "packet_count": 0,
                "total_bytes": 0,
                "first_seen": now,
                "last_seen": now,
                "syn_count": 0,
                "rst_count": 0,
                "fin_count": 0,
                "timestamps": [],
            }

        flow = self._flows[key]
        flow["packet_count"] += 1
        flow["total_bytes"] += packet.get("ip_len", 0)
        flow["last_seen"] = now
        flow["timestamps"].append(now)

        flags = packet.get("tcp_flags", "") or ""
        if "S" in flags:
            flow["syn_count"] += 1
        if "R" in flags:
            flow["rst_count"] += 1
        if "F" in flags:
            flow["fin_count"] += 1

        self._last_seen[key] = now
        return flow

    def evict_expired(self, now: Optional[float] = None) -> List[Dict]:
        ts = now or time.time()
        expired_keys = [
            k for k, last in self._last_seen.items()
            if ts - last > self.flow_timeout
        ]
        evicted = []
        for k in expired_keys:
            evicted.append(self._flows.pop(k))
            self._last_seen.pop(k)
        return evicted

    def active_flow_count(self) -> int:
        return len(self._flows)
