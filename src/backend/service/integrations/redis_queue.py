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


@dataclass
class RQIntegration:
    """Holds RQ connection and queue references."""

    available: bool
    redis: "Redis | None"
    queue: "Queue | None"
    queue_name: str = "training"


def create_rq_integration() -> RQIntegration:
    """
    Create RQ integration from environment variables.

    Environment Variables:
        REDIS_URL: Redis connection URL (default: redis://localhost:6379)

    Returns:
        RQIntegration instance with queue ready for use.
    """
    redis_url = (os.getenv("REDIS_URL", "redis://localhost:6379") or "").strip()
    if (redis_url.startswith('"') and redis_url.endswith('"')) or (redis_url.startswith("'") and redis_url.endswith("'")):
        redis_url = redis_url[1:-1].strip()
    if redis_url and not redis_url.startswith(("redis://", "rediss://", "unix://")):
        if redis_url.startswith(("tcp://", "redis+tcp://")):
            redis_url = "redis://" + redis_url.split("://", 1)[1]
        elif redis_url.startswith(("ssl://", "tls://", "redis+ssl://")):
            redis_url = "rediss://" + redis_url.split("://", 1)[1]
        elif "://" not in redis_url:
            redis_url = f"redis://{redis_url}"
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
        )
    except ImportError as exc:
        print(f"Warning: RQ not available (missing dependency): {exc}")
        return RQIntegration(available=False, redis=None, queue=None)
    except Exception as exc:
        print(f"Warning: Could not connect to Redis: {exc}")
        print("  Jobs will run synchronously (no queuing).")
        return RQIntegration(available=False, redis=None, queue=None)


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
        return {"available": False}

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
