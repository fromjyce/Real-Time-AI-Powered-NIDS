"""
Redis client wrapper providing simple get/set/set-member operations
with automatic JSON serialisation and TTL management.
"""

import json
from typing import Any, Optional, Set

import structlog
import redis as redis_lib
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = structlog.get_logger(__name__)


class RedisClient:
    def __init__(self, url: Optional[str] = None) -> None:
        self._pool = redis_lib.ConnectionPool.from_url(
            url or settings.REDIS_URL,
            max_connections=20,
            decode_responses=True,
        )
        self._redis = redis_lib.Redis(connection_pool=self._pool)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=5))
    def get(self, key: str) -> Optional[str]:
        try:
            return self._redis.get(key)
        except redis_lib.RedisError as e:
            logger.error("redis_get_failed", key=key, error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=5))
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not isinstance(value, str):
            value = json.dumps(value)
        try:
            self._redis.set(key, value, ex=ttl)
        except redis_lib.RedisError as e:
            logger.error("redis_set_failed", key=key, error=str(e))
            raise

    def delete(self, key: str) -> None:
        self._redis.delete(key)

    def exists(self, key: str) -> bool:
        return bool(self._redis.exists(key))

    def incr(self, key: str, amount: int = 1) -> int:
        return int(self._redis.incrby(key, amount))

    def expire(self, key: str, ttl: int) -> None:
        self._redis.expire(key, ttl)

    # ── Set operations ────────────────────────────────────────────────────────

    def sadd(self, key: str, *members: str) -> int:
        return int(self._redis.sadd(key, *members))

    def srem(self, key: str, *members: str) -> int:
        return int(self._redis.srem(key, *members))

    def smembers(self, key: str) -> Set[str]:
        return self._redis.smembers(key)

    def is_member(self, key: str, member: str) -> bool:
        return bool(self._redis.sismember(key, member))

    # ── Hash operations ───────────────────────────────────────────────────────

    def hset(self, key: str, mapping: dict) -> None:
        self._redis.hset(key, mapping=mapping)

    def hget(self, key: str, field: str) -> Optional[str]:
        return self._redis.hget(key, field)

    def hgetall(self, key: str) -> dict:
        return self._redis.hgetall(key)

    # ── Sorted Set operations ─────────────────────────────────────────────────

    def zadd(self, key: str, mapping: dict) -> None:
        self._redis.zadd(key, mapping)

    def zrangebyscore(self, key: str, min_score: float, max_score: float) -> list:
        return self._redis.zrangebyscore(key, min_score, max_score)

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        return int(self._redis.zremrangebyscore(key, min_score, max_score))

    def ping(self) -> bool:
        try:
            return self._redis.ping()
        except redis_lib.RedisError:
            return False
