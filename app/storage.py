"""
Pluggable key/value storage for interview session state.

Locally (uvicorn, long-running process) plain in-memory storage is fine.
On Vercel, each request may hit a fresh serverless instance, so session
state must live somewhere external. This module transparently switches to
Upstash Redis (REST API, works great from serverless functions -- no
persistent TCP connection needed) whenever these env vars are set:

    UPSTASH_REDIS_REST_URL
    UPSTASH_REDIS_REST_TOKEN

Free tier: https://upstash.com -> create a Redis database -> copy the
"REST API" URL + token into your Vercel project's environment variables.

If those env vars are absent, falls back to an in-memory dict -- correct
for local `uvicorn` runs, but NOT safe on Vercel (state won't survive
between cold starts).
"""

from __future__ import annotations

import os
import threading
from typing import Optional

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

_UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
_UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
USE_UPSTASH = bool(_UPSTASH_URL and _UPSTASH_TOKEN and requests is not None)

_DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours -- plenty for one interview session

_lock = threading.Lock()
_memory_store: dict = {}


def _headers():
    return {"Authorization": f"Bearer {_UPSTASH_TOKEN}"}


def get(key: str) -> Optional[str]:
    if USE_UPSTASH:
        resp = requests.get(f"{_UPSTASH_URL}/get/{key}", headers=_headers(), timeout=5)
        resp.raise_for_status()
        return resp.json().get("result")
    with _lock:
        return _memory_store.get(key)


def set(key: str, value: str, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
    if USE_UPSTASH:
        resp = requests.post(
            f"{_UPSTASH_URL}/set/{key}",
            headers=_headers(),
            data=value.encode("utf-8"),
            params={"EX": ttl_seconds},
            timeout=5,
        )
        resp.raise_for_status()
        return
    with _lock:
        _memory_store[key] = value


def exists(key: str) -> bool:
    return get(key) is not None
