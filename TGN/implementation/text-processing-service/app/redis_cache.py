import hashlib
import json
import os
import re
from typing import Any


def normalize_prediction_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def build_prediction_cache_key(artifact_uri: str, normalized_text: str) -> str:
    digest = hashlib.sha256(f"{artifact_uri}\n{normalized_text}".encode("utf-8")).hexdigest()
    return f"pred:text:v1:{digest}"


def build_prediction_lock_key(cache_key: str) -> str:
    return f"lock:{cache_key}"


def build_prediction_semantic_index_key(artifact_uri: str) -> str:
    digest = hashlib.sha256(artifact_uri.encode("utf-8")).hexdigest()
    return f"pred:text:v1:index:{digest}"


class RedisTextCache:
    def __init__(self) -> None:
        self.redis_url = os.environ.get("REDIS_URL", "").strip()
        self.default_ttl_sec = int(os.environ.get("TEXT_PREDICTION_CACHE_TTL_SEC", "604800"))
        self.lock_ttl_sec = int(os.environ.get("TEXT_PREDICTION_CACHE_LOCK_TTL_SEC", "30"))
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.redis_url)

    def _get_client(self):
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client

        try:
            import redis
        except ImportError:
            return None

        self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def get_json(self, key: str) -> dict[str, Any] | None:
        client = self._get_client()
        if client is None:
            return None

        try:
            raw = client.get(key)
        except Exception:
            return None
        if raw is None:
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        client = self._get_client()
        if client is None:
            return

        try:
            client.set(key, json.dumps(value), ex=self.default_ttl_sec)
        except Exception:
            return

    def acquire_lock(self, key: str) -> bool:
        client = self._get_client()
        if client is None:
            return False

        try:
            return bool(client.set(key, "1", nx=True, ex=self.lock_ttl_sec))
        except Exception:
            return False

    def release_lock(self, key: str) -> None:
        client = self._get_client()
        if client is None:
            return

        try:
            client.delete(key)
        except Exception:
            return

    def add_index_member(self, key: str, member: str) -> None:
        client = self._get_client()
        if client is None:
            return

        try:
            client.sadd(key, member)
            client.expire(key, self.default_ttl_sec)
        except Exception:
            return

    def get_index_members(self, key: str, limit: int | None = None) -> list[str]:
        client = self._get_client()
        if client is None:
            return []

        try:
            members = client.smembers(key)
        except Exception:
            return []

        if not members:
            return []

        ordered = sorted(str(member) for member in members)
        if limit is not None and limit >= 0:
            return ordered[:limit]
        return ordered
