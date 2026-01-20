#!/usr/bin/env python3
"""Test script for zombie job detection and cleanup.

This script simulates various zombie job scenarios and verifies that the
stale job cleanup system handles them correctly.

Usage:
    # Run all tests (requires Redis)
    python scripts/test_zombie_cleanup.py

    # Run with custom Redis URL
    REDIS_URL=redis://localhost:6379 python scripts/test_zombie_cleanup.py

    # Dry-run mode (don't actually create/clean jobs)
    python scripts/test_zombie_cleanup.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add the backend service to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "backend"))


def _normalize_redis_url(value: str) -> str:
    """Normalize Redis URL to standard format."""
    value = (value or "").strip()
    if not value:
        return "redis://localhost:6379"
    if value.startswith(("redis://", "rediss://")):
        return value
    if "://" not in value:
        return f"redis://{value}"
    return value


def test_stale_job_detection(redis_conn, queue_name: str = "test_queue"):
    """Test that stale jobs are correctly detected."""
    from rq import Queue
    from rq.job import Job
    from rq.registry import StartedJobRegistry

    from service.services.stale_job_cleanup import cleanup_stale_jobs

    print("\n=== Test: Stale Job Detection ===")

    # Create a fake "started" job with old updated_at
    queue = Queue(queue_name, connection=redis_conn)

    # Create a dummy job
    def dummy_task():
        pass

    job = queue.enqueue(dummy_task, job_timeout=3600)
    job_id = job.id
    print(f"Created test job: {job_id}")

    # Manually move to started registry and set old updated_at
    started_registry = StartedJobRegistry(queue=queue)

    # Simulate job being "started" with old timestamp
    old_time = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    job.meta["status"] = "training"
    job.meta["updated_at"] = old_time
    job.meta["started_at"] = old_time
    job.save_meta()

    # Add to started registry
    started_registry.add(job, ttl=-1)

    # Remove from queue (simulate worker picked it up)
    queue.remove(job_id)

    print(f"Simulated started job with updated_at = {old_time}")

    # Run stale job detection (dry run)
    result = cleanup_stale_jobs(
        redis_conn=redis_conn,
        queue_name=queue_name,
        threshold_seconds=600,  # 10 minutes
        dry_run=True,
    )

    print(f"Detection result: {json.dumps(result, indent=2, default=str)}")

    # Verify the job was detected as stale
    stale_ids = [s["job_id"] for s in result.get("stale", [])]
    if job_id in stale_ids:
        print(f"[PASS] Job {job_id} correctly detected as stale")
    else:
        print(f"[FAIL] Job {job_id} NOT detected as stale")
        return False

    # Now actually clean it up
    result = cleanup_stale_jobs(
        redis_conn=redis_conn,
        queue_name=queue_name,
        threshold_seconds=600,
        dry_run=False,
    )

    if job_id in result.get("cleaned", []):
        print(f"[PASS] Job {job_id} successfully cleaned up")
    else:
        print(f"[FAIL] Job {job_id} NOT cleaned up")
        return False

    # Verify it's no longer in started registry
    started_ids = started_registry.get_job_ids()
    if job_id not in started_ids:
        print(f"[PASS] Job {job_id} removed from started registry")
    else:
        print(f"[FAIL] Job {job_id} still in started registry")
        return False

    # Cleanup
    try:
        job.delete()
    except Exception:
        pass

    return True


def test_fresh_job_not_cleaned(redis_conn, queue_name: str = "test_queue"):
    """Test that fresh jobs are NOT cleaned up."""
    from rq import Queue
    from rq.job import Job
    from rq.registry import StartedJobRegistry

    from service.services.stale_job_cleanup import cleanup_stale_jobs

    print("\n=== Test: Fresh Job Not Cleaned ===")

    queue = Queue(queue_name, connection=redis_conn)

    def dummy_task():
        pass

    job = queue.enqueue(dummy_task, job_timeout=3600)
    job_id = job.id
    print(f"Created test job: {job_id}")

    # Simulate job being "started" with FRESH timestamp
    fresh_time = datetime.now(timezone.utc).isoformat()
    job.meta["status"] = "training"
    job.meta["updated_at"] = fresh_time
    job.meta["started_at"] = fresh_time
    job.save_meta()

    started_registry = StartedJobRegistry(queue=queue)
    started_registry.add(job, ttl=-1)
    queue.remove(job_id)

    print(f"Simulated started job with updated_at = {fresh_time}")

    # Run stale job detection
    result = cleanup_stale_jobs(
        redis_conn=redis_conn,
        queue_name=queue_name,
        threshold_seconds=600,
        dry_run=False,
    )

    print(f"Cleanup result: {json.dumps(result, indent=2, default=str)}")

    # Verify the job was NOT cleaned
    stale_ids = [s["job_id"] for s in result.get("stale", [])]
    if job_id not in stale_ids:
        print(f"[PASS] Fresh job {job_id} correctly NOT detected as stale")
    else:
        print(f"[FAIL] Fresh job {job_id} incorrectly detected as stale")
        return False

    # Cleanup
    try:
        started_registry.remove(job_id)
        job.delete()
    except Exception:
        pass

    return True


def test_timeout_exceeded_detection(redis_conn, queue_name: str = "test_queue"):
    """Test that jobs exceeding timeout are detected even if updating."""
    from rq import Queue
    from rq.job import Job
    from rq.registry import StartedJobRegistry

    from service.services.stale_job_cleanup import cleanup_stale_jobs

    print("\n=== Test: Timeout Exceeded Detection ===")

    queue = Queue(queue_name, connection=redis_conn)

    def dummy_task():
        pass

    # Create job with short timeout
    job = queue.enqueue(dummy_task, job_timeout=60)  # 1 minute timeout
    job_id = job.id
    print(f"Created test job with 60s timeout: {job_id}")

    # Simulate job "started" 10 minutes ago but still updating (stuck in loop)
    old_started = datetime.now(timezone.utc) - timedelta(minutes=10)
    fresh_updated = datetime.now(timezone.utc).isoformat()

    job.meta["status"] = "training"
    job.meta["updated_at"] = fresh_updated  # Recent heartbeat
    job.meta["started_at"] = old_started.isoformat()
    job.save_meta()

    # Set RQ's started_at (used for timeout check)
    job.started_at = old_started
    job.save()

    started_registry = StartedJobRegistry(queue=queue)
    started_registry.add(job, ttl=-1)
    queue.remove(job_id)

    print(
        f"Simulated: started_at = {old_started.isoformat()}, updated_at = {fresh_updated}"
    )
    print(f"Job timeout = 60s, running for ~600s (should exceed timeout)")

    # Run cleanup (use long staleness threshold so only timeout kicks in)
    result = cleanup_stale_jobs(
        redis_conn=redis_conn,
        queue_name=queue_name,
        threshold_seconds=3600,  # 1 hour (won't trigger on staleness)
        dry_run=True,
    )

    print(f"Detection result: {json.dumps(result, indent=2, default=str)}")

    # Check if detected
    stale_jobs = result.get("stale", [])
    job_entry = next((s for s in stale_jobs if s["job_id"] == job_id), None)

    if job_entry and job_entry.get("reason") == "timeout_exceeded":
        print(f"[PASS] Job {job_id} correctly detected as timeout_exceeded")
    else:
        print(f"[FAIL] Job {job_id} NOT detected as timeout_exceeded")
        # Cleanup and return
        try:
            started_registry.remove(job_id)
            job.delete()
        except Exception:
            pass
        return False

    # Cleanup
    try:
        started_registry.remove(job_id)
        job.delete()
    except Exception:
        pass

    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Test zombie job cleanup")
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379"),
        help="Redis URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually create/clean jobs (just test imports)",
    )
    parser.add_argument(
        "--queue",
        default="test_zombie_cleanup",
        help="Queue name to use for tests (default: test_zombie_cleanup)",
    )

    args = parser.parse_args(argv)

    redis_url = _normalize_redis_url(args.redis_url)
    print(f"Redis URL: {redis_url.split('@')[-1]}")  # Hide password

    if args.dry_run:
        print("\n[DRY-RUN] Only testing imports...")
        try:
            from service.services.stale_job_cleanup import (
                cleanup_stale_jobs,
                get_stale_job_summary,
                _purge_zombie_job,
            )

            print("[PASS] All imports successful")
            return 0
        except Exception as e:
            print(f"[FAIL] Import error: {e}")
            return 1

    # Connect to Redis
    try:
        from redis import Redis

        redis_conn = Redis.from_url(redis_url)
        redis_conn.ping()
        print("[OK] Connected to Redis")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Redis: {e}")
        return 2

    # Run tests
    results = []

    try:
        results.append(
            ("Stale Job Detection", test_stale_job_detection(redis_conn, args.queue))
        )
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        results.append(("Stale Job Detection", False))

    try:
        results.append(
            (
                "Fresh Job Not Cleaned",
                test_fresh_job_not_cleaned(redis_conn, args.queue),
            )
        )
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        results.append(("Fresh Job Not Cleaned", False))

    try:
        results.append(
            (
                "Timeout Exceeded Detection",
                test_timeout_exceeded_detection(redis_conn, args.queue),
            )
        )
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        results.append(("Timeout Exceeded Detection", False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
