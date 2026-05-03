"""
Threat intelligence integration.
Checks source IPs against:
  - AbuseIPDB (reputation score)
  - VirusTotal (IP reports)
  - A locally maintained Redis-backed blocklist
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import structlog

from src.config import settings
from src.storage.redis_client import RedisClient

logger = structlog.get_logger(__name__)

BLOCKLIST_KEY = "nids:threat:blocklist"
CACHE_KEY_PREFIX = "nids:threat:ip:"
CACHE_TTL = 3600  # 1 hour per IP lookup


class ThreatIntelligenceClient:
    """
    Async client that enriches IP addresses with reputation data.
    Results are cached in Redis to avoid hammering external APIs.
    """

    ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
    VIRUSTOTAL_URL = "https://www.virustotal.com/api/v3/ip_addresses/{ip}"

    def __init__(self, redis: Optional[RedisClient] = None) -> None:
        self._redis = redis or RedisClient()
        self._local_blocklist: Set[str] = set()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _cache_key(self, ip: str) -> str:
        return f"{CACHE_KEY_PREFIX}{ip}"

    def is_blocklisted(self, ip: str) -> bool:
        if ip in self._local_blocklist:
            return True
        return bool(self._redis.is_member(BLOCKLIST_KEY, ip))

    def add_to_blocklist(self, ip: str) -> None:
        self._local_blocklist.add(ip)
        self._redis.sadd(BLOCKLIST_KEY, ip)
        logger.info("ip_blocklisted", ip=ip)

    def remove_from_blocklist(self, ip: str) -> None:
        self._local_blocklist.discard(ip)
        self._redis.srem(BLOCKLIST_KEY, ip)

    async def _check_cache(self, ip: str) -> Optional[Dict]:
        raw = self._redis.get(self._cache_key(ip))
        if raw:
            return json.loads(raw)
        return None

    def _write_cache(self, ip: str, data: Dict) -> None:
        self._redis.set(self._cache_key(ip), json.dumps(data), ttl=CACHE_TTL)

    async def check_abuseipdb(self, ip: str) -> Optional[Dict]:
        if not settings.ABUSEIPDB_API_KEY:
            return None

        session = await self._get_session()
        headers = {
            "Key": settings.ABUSEIPDB_API_KEY,
            "Accept": "application/json",
        }
        params = {"ipAddress": ip, "maxAgeInDays": 90, "verbose": ""}

        try:
            async with session.get(self.ABUSEIPDB_URL, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", {})
        except aiohttp.ClientError as e:
            logger.warning("abuseipdb_request_failed", ip=ip, error=str(e))
        return None

    async def check_virustotal(self, ip: str) -> Optional[Dict]:
        if not settings.VIRUSTOTAL_API_KEY:
            return None

        session = await self._get_session()
        url = self.VIRUSTOTAL_URL.format(ip=ip)
        headers = {"x-apikey": settings.VIRUSTOTAL_API_KEY}

        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    attrs = data.get("data", {}).get("attributes", {})
                    stats = attrs.get("last_analysis_stats", {})
                    return {
                        "malicious": stats.get("malicious", 0),
                        "suspicious": stats.get("suspicious", 0),
                        "reputation": attrs.get("reputation", 0),
                        "country": attrs.get("country", ""),
                        "asn": attrs.get("asn", ""),
                    }
        except aiohttp.ClientError as e:
            logger.warning("virustotal_request_failed", ip=ip, error=str(e))
        return None

    async def enrich_ip(self, ip: str) -> Dict:
        """
        Returns enriched reputation data for an IP address.
        Falls back to cached results when available.
        """
        cached = await self._check_cache(ip)
        if cached:
            return cached

        abuse_data, vt_data = await asyncio.gather(
            self.check_abuseipdb(ip),
            self.check_virustotal(ip),
            return_exceptions=True,
        )

        result = {
            "ip": ip,
            "checked_at": time.time(),
            "is_blocklisted": self.is_blocklisted(ip),
            "abuseipdb": abuse_data if isinstance(abuse_data, dict) else None,
            "virustotal": vt_data if isinstance(vt_data, dict) else None,
            "threat_score": 0.0,
        }

        abuse_score = 0.0
        vt_score = 0.0

        if isinstance(abuse_data, dict):
            abuse_score = min(abuse_data.get("abuseConfidenceScore", 0) / 100.0, 1.0)
            if abuse_score > 0.7:
                self.add_to_blocklist(ip)

        if isinstance(vt_data, dict):
            malicious = vt_data.get("malicious", 0)
            suspicious = vt_data.get("suspicious", 0)
            vt_score = min((malicious + 0.5 * suspicious) / 10.0, 1.0)

        result["threat_score"] = 0.6 * abuse_score + 0.4 * vt_score

        self._write_cache(ip, result)
        return result

    async def bulk_enrich(self, ips: List[str]) -> Dict[str, Dict]:
        tasks = [self.enrich_ip(ip) for ip in set(ips)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            ip: r if isinstance(r, dict) else {}
            for ip, r in zip(ips, results)
        }

    def sync_enrich(self, ip: str) -> Dict:
        """Synchronous wrapper for use in non-async contexts."""
        return asyncio.run(self.enrich_ip(ip))
