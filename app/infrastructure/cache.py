from __future__ import annotations

import json
import time
from typing import Any


class MemoryTTLCache:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self.ttl_seconds = ttl_seconds
        self._values: dict[str, tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Any | None:
        item = self._values.get(key)
        if not item:
            self.misses += 1
            return None
        expires_at, value = item
        if expires_at <= time.monotonic():
            self._values.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return value

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds or self.ttl_seconds
        self._values[key] = (time.monotonic() + ttl, value)

    async def ping(self) -> bool:
        return True


class RedisCache:
    def __init__(self, url: str, ttl_seconds: int = 600) -> None:
        try:
            from redis.asyncio import from_url
        except ImportError as exc:
            raise RuntimeError("Install FiberMind with the 'infra' extra to use Redis") from exc
        self.client = from_url(url, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Any | None:
        raw = await self.client.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        await self.client.set(
            key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds or self.ttl_seconds
        )

    async def ping(self) -> bool:
        return bool(await self.client.ping())
