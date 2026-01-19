"""
Redis Queue (RQ) integration for background job processing.

This module provides the queue infrastructure for processing long-running
training jobs asynchronously. Jobs are queued and processed by RQ workers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from redis import Redis
    from rq import Queue


def normalize_redis_url(value: str) -> str:
    """
    Normalize common REDIS_URL formats to a URL that `redis.Redis.from_url` accepts.

    Handles:
    - Missing scheme (e.g. `host:6379` or `:password@host:6379`)
    - Netloc-only URLs (e.g. `//:password@host:6379`)
    - Aliases like `tcp://` and `tls://`
    - Accidental `REDIS_URL=...` prefix (common when copying `.env` lines)
    - Extra/mismatched quotes
    """
    value = (value or "").strip()
    if value.lower().startswith("redis_url="):
        value = value.split("=", 1)[1].strip()

    value = value.strip()
    if value[:1] in {'"', "'"}:
        value = value[1:]
    if value[-1:] in {'"', "'"}:
        value = value[:-1]
    value = value.strip()

    if not value:
        return ""

    # Normalize scheme casing (URL schemes are case-insensitive, but string checks aren't).
    if "://" in value:
        scheme, rest = value.split("://", 1)
        value = f"{scheme.lower()}://{rest}"

    if value.startswith(("redis://", "rediss://", "unix://")):
        return value

    if value.startswith("//"):
        return "redis:" + value

    if value.startswith(("tcp://", "redis+tcp://")):
        return "redis://" + value.split("://", 1)[1]
    if value.startswith(("ssl://", "tls://", "redis+ssl://")):
        return "rediss://" + value.split("://", 1)[1]

    if "://" not in value:
        return f"redis://{value}"

    return value


@dataclass
class RQIntegration:
    """Holds RQ connection and queue references."""

    available: bool
    redis: "Redis | None"
    queue: "Queue | None"
    queue_name: str = "training"
    error: str | None = None


def create_rq_integration() -> RQIntegration:
    """
    Create RQ integration from environment variables.

    Environment Variables:
        REDIS_URL: Redis connection URL (default: redis://localhost:6379)

    Returns:
        RQIntegration instance with queue ready for use.
    """
    redis_url = normalize_redis_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    if not redis_url:
        redis_url = "redis://localhost:6379"
    parsed = urlparse(redis_url)
    redis_host = parsed.hostname or "localhost"
    redis_port = parsed.port or 6379

    try:
        from redis import Redis
        from rq import Queue

        redis_conn = Redis.from_url(redis_url)
        # Test connection
        redis_conn.ping()

        queue = Queue("training", connection=redis_conn)

        print(f"✓ RQ connected to Redis at {redis_host}:{redis_port}")
        return RQIntegration(
            available=True,
            redis=redis_conn,
            queue=queue,
            queue_name="training",
            error=None,
        )
    except ImportError as exc:
        msg = f"Missing dependency for queue: {exc}"
        print(f"Warning: RQ not available ({msg})")
        return RQIntegration(available=False, redis=None, queue=None, error=msg)
    except Exception as exc:
        msg = f"Could not connect to Redis at {redis_host}:{redis_port}: {exc}"
        print(f"Warning: {msg}")
        print("  Jobs will run synchronously (no queuing).")
        return RQIntegration(available=False, redis=None, queue=None, error=msg)


def get_job_status(rq: RQIntegration, job_id: str) -> dict:
    """
    Get the status of a queued job.

    Args:
        rq: RQ integration instance
        job_id: The job ID to check

    Returns:
        Dict with job status information
    """
    if not rq.available or not rq.redis:
        return {"error": "Queue not available"}

    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=rq.redis)

        status = {
            "job_id": job_id,
            "status": job.get_status(),
            "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            "exc_info": job.exc_info if job.is_failed else None,
        }

        # Get position in queue if still queued
        if job.get_status() == "queued":
            try:
                queue_jobs = rq.queue.job_ids if rq.queue else []
                if job_id in queue_jobs:
                    status["position"] = queue_jobs.index(job_id) + 1
                    status["queue_length"] = len(queue_jobs)
            except Exception:
                pass

        # Get result if finished
        if job.get_status() == "finished":
            result = job.latest_result()
            if result and result.type.name == "SUCCESSFUL":
                status["result"] = result.return_value

        return status

    except Exception as exc:
        return {"error": str(exc), "job_id": job_id}


def get_queue_info(rq: RQIntegration) -> dict:
    """Get information about the queue status."""
    if not rq.available or not rq.queue:
        info: dict[str, object] = {"available": False}
        if rq.error:
            info["error"] = rq.error
        return info

    try:
        from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry

        # Get jobs in different states
        queued_ids = rq.queue.job_ids[:20]
        
        registry = StartedJobRegistry(queue=rq.queue)
        started_ids = registry.get_job_ids()
        
        finished_registry = FinishedJobRegistry(queue=rq.queue)
        finished_ids = finished_registry.get_job_ids()[:10]  # Last 10 finished
        
        failed_registry = FailedJobRegistry(queue=rq.queue)
        failed_ids = failed_registry.get_job_ids()[:10]

        # Combine all interesting job IDs (running first, then queued)
        job_ids = started_ids + queued_ids + finished_ids + failed_ids

        return {
            "available": True,
            "queue_name": rq.queue_name,
            "queued_jobs": len(rq.queue),
            "started_jobs": len(started_ids),
            "job_ids": job_ids[:50],  # Cap at 50
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
