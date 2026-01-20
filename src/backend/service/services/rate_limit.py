from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

from fastapi import HTTPException, Request


@dataclass
class _Bucket:
    timestamps: Deque[float]


_BUCKETS: dict[str, _Bucket] = {}


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(int(raw), 0)
    except Exception:
        return default


def _client_key(request: Request, user_id: str | None) -> str:
    host = getattr(getattr(request, "client", None), "host", None) or "unknown"
    if user_id:
        return f"user:{user_id}"
    return f"ip:{host}"


def check_rate_limit(
    *,
    request: Request,
    scope: str,
    user_id: str | None,
    limit: int,
    window_s: float,
) -> None:
    """Very small in-memory rate limiter (best-effort).

    Notes:
    - Per-process only (not shared across multiple workers/replicas).
    - Good enough to dampen obvious abuse / accidental spamming.
    """

    if limit <= 0:
        return

    now = time.monotonic()
    key = f"{scope}:{_client_key(request, user_id)}"
    bucket = _BUCKETS.get(key)
    if bucket is None:
        bucket = _Bucket(timestamps=deque())
        _BUCKETS[key] = bucket

    cutoff = now - window_s
    while bucket.timestamps and bucket.timestamps[0] < cutoff:
        bucket.timestamps.popleft()

    if len(bucket.timestamps) >= limit:
        retry_after = int(max(bucket.timestamps[0] + window_s - now, 0))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {scope}. Try again in ~{retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    bucket.timestamps.append(now)


def limit_jobs_submit(request: Request, user_id: str | None) -> None:
    per_min = _get_env_int("RATE_LIMIT_JOBS_SUBMIT_PER_MIN", 30)
    check_rate_limit(
        request=request,
        scope="jobs_submit",
        user_id=user_id,
        limit=per_min,
        window_s=60.0,
    )


def limit_upload(request: Request, user_id: str | None) -> None:
    per_min = _get_env_int("RATE_LIMIT_UPLOAD_PER_MIN", 10)
    check_rate_limit(
        request=request,
        scope="upload",
        user_id=user_id,
        limit=per_min,
        window_s=60.0,
    )
