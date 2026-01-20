"""
Stale job cleanup service.

Detects and cleans up "zombie" jobs - jobs that are stuck in "started" state
because their worker died (Modal container killed, OOM, heartbeat timeout, etc.)
without proper cleanup.

Detection is based on:
1. job.meta.updated_at - workers update this every ~1 second. If stale, worker is dead.
2. job.timeout - if job has been running longer than its timeout, it's stuck.

If a job is "started" but hasn't been updated in STALE_JOB_THRESHOLD_S, or has
exceeded its timeout, we consider it abandoned and clean it up.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis

from ..config import STALE_JOB_THRESHOLD_S

logger = logging.getLogger(__name__)


def _parse_timeout_to_seconds(timeout) -> int | None:
    """Parse RQ timeout value to seconds."""
    if timeout is None:
        return None
    if isinstance(timeout, (int, float)):
        return int(timeout)
    if isinstance(timeout, str):
        timeout = timeout.strip().lower()
        if timeout.endswith("h"):
            return int(float(timeout[:-1]) * 3600)
        if timeout.endswith("m"):
            return int(float(timeout[:-1]) * 60)
        if timeout.endswith("s"):
            return int(float(timeout[:-1]))
        try:
            return int(timeout)
        except ValueError:
            return None
    return None


def cleanup_stale_jobs(
    redis_conn: "Redis",
    queue_name: str = "training",
    threshold_seconds: int | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Scan for and clean up stale/zombie jobs.

    A job is considered stale if:
    1. It's in the started registry (rq:wip:<queue> or rq:started:<queue>)
    2. Its meta.updated_at is older than threshold_seconds (default: STALE_JOB_THRESHOLD_S)
    3. OR it has no updated_at and its enqueued_at/started_at is older than threshold
    4. OR it has exceeded its job_timeout (even if still updating meta)

    Args:
        redis_conn: Redis connection
        queue_name: RQ queue name (default: "training")
        threshold_seconds: Age threshold in seconds (default: STALE_JOB_THRESHOLD_S)
        dry_run: If True, only detect but don't clean up

    Returns:
        Dict with cleanup results:
        - scanned: number of started jobs checked
        - stale: list of job IDs detected as stale
        - cleaned: list of job IDs that were cleaned up
        - errors: list of errors encountered
    """
    if threshold_seconds is None:
        threshold_seconds = STALE_JOB_THRESHOLD_S

    from rq.job import Job

    result = {
        "scanned": 0,
        "stale": [],
        "cleaned": [],
        "errors": [],
        "threshold_seconds": threshold_seconds,
        "dry_run": dry_run,
    }

    now = datetime.now(timezone.utc)

    # Collect job IDs from started registries
    started_job_ids: set[str] = set()
    for key in (f"rq:wip:{queue_name}", f"rq:started:{queue_name}"):
        try:
            # zset members can be "job_id" or "job_id:execution_id"
            members = redis_conn.zrange(key, 0, -1)
            for member in members:
                value = (
                    member.decode("utf-8") if isinstance(member, bytes) else str(member)
                )
                # Extract job_id (strip execution_id suffix if present)
                job_id = value.split(":")[0] if ":" in value else value
                started_job_ids.add(job_id)
        except Exception as e:
            result["errors"].append(f"Failed to scan {key}: {e}")

    result["scanned"] = len(started_job_ids)

    for job_id in started_job_ids:
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            meta = job.meta or {}

            # Determine the "last seen" timestamp
            last_seen: datetime | None = None

            # Primary: meta.updated_at (workers update this every ~1s)
            updated_at = meta.get("updated_at")
            if updated_at:
                if isinstance(updated_at, str):
                    try:
                        last_seen = datetime.fromisoformat(
                            updated_at.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
                elif isinstance(updated_at, datetime):
                    last_seen = updated_at

            # Fallback: meta.started_at
            if last_seen is None:
                started_at = meta.get("started_at")
                if started_at:
                    if isinstance(started_at, str):
                        try:
                            last_seen = datetime.fromisoformat(
                                started_at.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass
                    elif isinstance(started_at, datetime):
                        last_seen = started_at

            # Fallback: job.started_at (RQ attribute)
            if last_seen is None and job.started_at:
                last_seen = job.started_at
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)

            # Fallback: job.enqueued_at
            if last_seen is None and job.enqueued_at:
                last_seen = job.enqueued_at
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)

            if last_seen is None:
                # Can't determine age, skip (shouldn't happen in practice)
                continue

            age_seconds = (now - last_seen).total_seconds()
            is_stale = age_seconds > threshold_seconds
            stale_reason = "no_heartbeat" if is_stale else None

            # Also check if job has exceeded its timeout (even if still updating)
            # This catches jobs stuck in infinite loops
            if not is_stale and job.started_at:
                job_started = job.started_at
                if job_started.tzinfo is None:
                    job_started = job_started.replace(tzinfo=timezone.utc)

                running_seconds = (now - job_started).total_seconds()
                job_timeout = _parse_timeout_to_seconds(job.timeout)

                # Add 5 minute grace period beyond timeout
                if job_timeout and running_seconds > (job_timeout + 300):
                    is_stale = True
                    stale_reason = "timeout_exceeded"
                    age_seconds = running_seconds

            if is_stale:
                result["stale"].append(
                    {
                        "job_id": job_id,
                        "age_seconds": int(age_seconds),
                        "last_seen": last_seen.isoformat(),
                        "phase": meta.get("status"),
                        "user_id": meta.get("user_id"),
                        "name": meta.get("name"),
                        "reason": stale_reason,
                    }
                )

                if not dry_run:
                    cleanup_result = _purge_zombie_job(
                        redis_conn=redis_conn,
                        job_id=job_id,
                        queue_name=queue_name,
                        reason=stale_reason,
                    )
                    if cleanup_result.get("success"):
                        result["cleaned"].append(job_id)
                    else:
                        result["errors"].append(
                            f"Failed to clean {job_id}: {cleanup_result.get('error')}"
                        )

        except Exception as e:
            # Job might have been deleted by another process
            if "NoSuchJobError" not in str(type(e).__name__):
                result["errors"].append(f"Error checking job {job_id}: {e}")

    if result["stale"]:
        action = "Detected" if dry_run else "Cleaned up"
        logger.info(
            f"{action} {len(result['stale'])} stale job(s): "
            f"{[j['job_id'] for j in result['stale']]}"
        )

    return result


def _purge_zombie_job(
    redis_conn: "Redis",
    job_id: str,
    queue_name: str = "training",
    reason: str | None = None,
) -> dict:
    """
    Purge a zombie job from Redis.

    This removes the job from all registries and marks it as failed/stopped
    so it doesn't show as "running" in the UI.

    Args:
        redis_conn: Redis connection
        job_id: Job ID to purge
        queue_name: RQ queue name
        reason: Why the job was purged (no_heartbeat, timeout_exceeded, etc.)

    Returns:
        Dict with success status and details
    """
    from rq.job import Job

    result = {
        "success": False,
        "job_id": job_id,
        "removed_from": [],
    }

    error_messages = {
        "no_heartbeat": "Worker died (no heartbeat - cleaned up by stale job detector)",
        "timeout_exceeded": "Job exceeded timeout (cleaned up by stale job detector)",
    }

    try:
        # Update job meta to reflect that it was abandoned/cleaned up
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            meta = job.meta or {}
            meta["status"] = "failed"
            meta["error"] = error_messages.get(
                reason, f"Cleaned up by stale job detector ({reason})"
            )
            meta["stopped_phase"] = meta.get("status") or "unknown"
            meta["completed_at"] = datetime.now(timezone.utc).isoformat()
            meta["cleanup_reason"] = reason or "stale_job_cleanup"
            job.meta = meta
            job.save_meta()
        except Exception:
            pass  # Job might be corrupted, continue with cleanup

        # Remove from queue list
        try:
            removed = redis_conn.lrem(f"rq:queue:{queue_name}", 0, job_id)
            if removed:
                result["removed_from"].append(f"rq:queue:{queue_name}")
        except Exception:
            pass

        # Remove from started/wip registries (zsets)
        for key in (f"rq:wip:{queue_name}", f"rq:started:{queue_name}"):
            try:
                # Scan for job_id or job_id:* entries
                for member, _score in redis_conn.zscan_iter(key, match=f"{job_id}*"):
                    value = (
                        member.decode("utf-8")
                        if isinstance(member, bytes)
                        else str(member)
                    )
                    if value == job_id or value.startswith(job_id + ":"):
                        redis_conn.zrem(key, member)
                        result["removed_from"].append(f"{key}:{value}")
            except Exception:
                pass

        # Move to failed registry so it shows up correctly in RQ dashboard
        try:
            from rq.registry import FailedJobRegistry, StartedJobRegistry
            from rq import Queue

            queue = Queue(queue_name, connection=redis_conn)
            started_registry = StartedJobRegistry(queue=queue)
            failed_registry = FailedJobRegistry(queue=queue)

            # Remove from started registry (if still there)
            try:
                started_registry.remove(job_id, delete_job=False)
            except Exception:
                pass

            # Add to failed registry with an expiry
            try:
                failed_registry.add(job_id, ttl=3600)  # Keep in failed for 1 hour
                result["removed_from"].append("moved_to_failed_registry")
            except Exception:
                pass

        except Exception:
            pass

        result["success"] = True
        logger.info(f"Purged zombie job {job_id}: {result['removed_from']}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Failed to purge zombie job {job_id}: {e}")

    return result


def get_stale_job_summary(
    redis_conn: "Redis",
    queue_name: str = "training",
    threshold_seconds: int | None = None,
) -> dict:
    """
    Get a summary of potentially stale jobs without cleaning them up.

    Useful for the /api/queue endpoint to warn about zombie jobs.
    """
    return cleanup_stale_jobs(
        redis_conn=redis_conn,
        queue_name=queue_name,
        threshold_seconds=threshold_seconds,
        dry_run=True,
    )
