"""
Job queue API endpoints.

Provides endpoints for submitting training jobs to the queue,
checking job status, and managing the queue.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
import time
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from ..config import (
    MODAL_APP_NAME,
    MODAL_FUNCTION_NAME,
    MODAL_INSTANT_SPAWN,
    MODAL_POLL_URL,
    MODAL_SPAWN_LOCK_TTL_S,
    MODAL_TRIGGER_TOKEN,
    START_LOCAL_RQ_WORKER,
)
from ..integrations.redis_queue import (
    create_rq_integration,
    get_job_status,
    get_queue_info,
    normalize_redis_url,
)
from ..schemas import GenerateRequest, validate_limits

router = APIRouter()

class RenameJobRequest(BaseModel):
    name: str | None = None


def _get_rq_or_reconnect(http_request: Request):
    rq = getattr(http_request.app.state, "rq", None)
    if rq and getattr(rq, "available", False):
        return rq

    # Redis might come up after the API starts; retry occasionally so a restart isn't required.
    now = time.monotonic()
    last = float(getattr(http_request.app.state, "rq_reconnect_last", 0.0) or 0.0)
    if (now - last) < 5.0:
        return rq
    http_request.app.state.rq_reconnect_last = now

    try:
        new_rq = create_rq_integration()
        http_request.app.state.rq = new_rq
        return new_rq
    except Exception:
        return rq


def _maybe_trigger_modal_gpu_worker(rq) -> dict[str, object]:
    """
    Best-effort: spawn a Modal GPU worker right after a job is enqueued.

    This is on-demand (no schedule required).
    """
    if START_LOCAL_RQ_WORKER or not MODAL_INSTANT_SPAWN:
        return {"triggered": False, "reason": "disabled"}
    if not rq or not getattr(rq, "available", False) or not getattr(rq, "redis", None):
        return {"triggered": False, "reason": "rq_unavailable"}

    redis_conn = rq.redis

    # If we're calling the poller, do NOT acquire the spawn lock here.
    # The poller itself owns the lock and will spawn the GPU burst worker if needed.
    calls_poller = MODAL_FUNCTION_NAME in {"poll_queue", "poll_queue_http"}
    lock_key = "pyreflect:modal_worker_lock"
    lock_value = f"{uuid.uuid4()}:{int(time.time())}" if not calls_poller else None

    # Prefer HTTP trigger when configured: doesn't require Modal auth on the backend.
    if MODAL_POLL_URL:
        try:
            import requests

            params = {"token": MODAL_TRIGGER_TOKEN} if MODAL_TRIGGER_TOKEN else None
            resp = requests.post(MODAL_POLL_URL, params=params, timeout=10)
            if resp.ok:
                return {"triggered": True, "via": "http", "target": "poll_queue_http"}
            return {
                "triggered": False,
                "reason": "modal_http_failed",
                "error": f"status={resp.status_code} body={resp.text[:200]}",
            }
        except Exception as http_exc:
            return {
                "triggered": False,
                "reason": "modal_http_failed",
                "error": str(http_exc)[:400],
            }

    try:
        import modal

        fn = modal.Function.lookup(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
        if calls_poller:
            fn.spawn()
            return {"triggered": True, "via": "modal_client", "target": MODAL_FUNCTION_NAME}

        # Direct burst worker spawn: acquire lock and pass lock_value so the worker can release it.
        if lock_value is None:
            return {"triggered": False, "reason": "invalid_config"}
        try:
            acquired = redis_conn.set(
                lock_key, lock_value, nx=True, ex=max(int(MODAL_SPAWN_LOCK_TTL_S), 60)
            )
        except Exception:
            return {"triggered": False, "reason": "redis_lock_error"}
        if not acquired:
            return {"triggered": False, "reason": "spawn_lock_held"}

        fn.spawn(lock_value)
        return {"triggered": True, "via": "modal_client", "target": MODAL_FUNCTION_NAME}
    except Exception as exc:
        print(f"Warning: Failed to trigger Modal worker spawn via Modal client: {exc}")
        if lock_value is not None:
            # Release lock if we still own it so an alternate trigger can recover quickly.
            try:
                current = redis_conn.get(lock_key)
                if current is not None and current.decode("utf-8") == lock_value:
                    redis_conn.delete(lock_key)
            except Exception:
                pass

        return {"triggered": False, "reason": "modal_spawn_failed", "error": str(exc)[:400]}


@router.post("/jobs/submit")
async def submit_job(
    request: GenerateRequest,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    """
    Submit a training job to the queue.

    Returns immediately with a job_id that can be used to poll for status.
    If the queue is not available, falls back to synchronous execution.
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available:
        reason = getattr(rq, "error", None) if rq else None
        raise HTTPException(
            status_code=503,
            detail=(
                "Job queue not available."
                + (f" ({reason})" if reason else "")
                + " Check REDIS_URL / Redis connectivity, or use /api/generate/stream for synchronous execution."
            ),
        )

    # Guardrail: if local workers are disabled, don't accept jobs that can never
    # be consumed by remote workers (e.g. Modal cannot reach localhost Redis).
    if not START_LOCAL_RQ_WORKER:
        try:
            redis_url = normalize_redis_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
            parsed = urlparse(redis_url)
            redis_host = parsed.hostname or "localhost"
            if redis_host in {"localhost", "127.0.0.1", "::1"}:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "START_LOCAL_RQ_WORKER=false but REDIS_URL points to localhost. "
                        "Remote workers (Modal) cannot reach this queue. "
                        "Point REDIS_URL at a public/managed Redis or set START_LOCAL_RQ_WORKER=true."
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            # Best-effort only; enqueue may still fail later with a clear error.
            pass

    # Validate limits
    validate_limits(request.generator, request.training)

    # Build job parameters
    job_params = {
        "layers": [layer.model_dump() for layer in request.layers],
        "generator": request.generator.model_dump(),
        "training": request.training.model_dump(),
    }

    from ..config import RQ_JOB_TIMEOUT

    try:
        from ..jobs import run_training_job

        job = rq.queue.enqueue(
            run_training_job,
            job_params,
            user_id=x_user_id,
            name=request.name,
            job_timeout=RQ_JOB_TIMEOUT,
            result_ttl=3600,  # Keep results for 1 hour
        )

        # Persist enough info for UI actions like "retry"
        try:
            job.meta["job_params"] = job_params
            job.meta["user_id"] = x_user_id
            job.meta["name"] = request.name
            job.save_meta()
        except Exception:
            pass

        import asyncio

        remote_worker = await asyncio.to_thread(_maybe_trigger_modal_gpu_worker, rq)
        return {
            "job_id": job.id,
            "status": "queued",
            "message": "Job submitted successfully. Poll /api/jobs/{job_id} for status.",
            "queue_position": len(rq.queue),
            "remote_worker": remote_worker,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {exc}") from exc


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, http_request: Request):
    """
    Get the status of a submitted job.

    Returns job status, progress, logs, and result (if completed).
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available:
        raise HTTPException(status_code=503, detail="Job queue not available.")

    status = get_job_status(rq, job_id)

    if "error" in status and "not found" in status["error"].lower():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Also try to get job meta for progress/logs
    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=rq.redis)
        meta = job.meta or {}
        status["meta"] = {
            "status": meta.get("status"),
            "progress": meta.get("progress"),
            "logs": meta.get("logs", [])[-20:],  # Last 20 log entries
            "started_at": meta.get("started_at"),
            "completed_at": meta.get("completed_at"),
            "updated_at": meta.get("updated_at"),
            "user_id": meta.get("user_id"),
            "name": meta.get("name"),
            "retried_from": meta.get("retried_from"),
        }
    except Exception:
        pass

    return status


@router.patch("/jobs/{job_id}/name")
async def rename_job(
    job_id: str,
    request: RenameJobRequest,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    """
    Rename a job (stored in Redis meta).

    Also claims the job for the user if it isn't claimed yet.
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available or not rq.redis:
        raise HTTPException(status_code=503, detail="Job queue not available.")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    name = request.name
    if name is not None:
        name = name.strip()
        if name == "":
            name = None
        if name is not None and len(name) > 80:
            raise HTTPException(status_code=400, detail="Name too long (max 80 chars)")

    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=rq.redis)
        meta = job.meta or {}

        existing_user = meta.get("user_id")
        if existing_user and existing_user != x_user_id:
            raise HTTPException(status_code=403, detail="Job already claimed by another user")

        meta["name"] = name
        meta["user_id"] = existing_user or x_user_id
        job.meta = meta
        job.save_meta()
        return {"job_id": job_id, "name": name}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to rename job: {exc}") from exc


@router.post("/jobs/{job_id}/claim")
async def claim_job(job_id: str, http_request: Request, x_user_id: str | None = Header(default=None)):
    """
    Attach a job to a user after it was created (e.g. user logs in mid-run).

    This sets job.meta.user_id so the worker can save results to history when complete.
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available or not rq.redis:
        raise HTTPException(status_code=503, detail="Job queue not available.")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=rq.redis)
        meta = job.meta or {}

        existing = meta.get("user_id")
        if existing and existing != x_user_id:
            raise HTTPException(status_code=403, detail="Job already claimed by another user")

        meta["user_id"] = x_user_id
        job.meta = meta
        job.save_meta()

        # If the job already finished, attempt to persist it to history immediately.
        mongo = getattr(http_request.app.state, "mongo", None)
        generations = getattr(mongo, "generations", None) if mongo else None
        if generations is not None and job.get_status() == "finished":
            try:
                result = None
                try:
                    latest = job.latest_result()
                    if latest and getattr(latest, "type", None) and latest.type.name == "SUCCESSFUL":
                        result = latest.return_value
                except Exception:
                    result = None
                if result is None:
                    try:
                        result = job.result
                    except Exception:
                        result = None

                if isinstance(result, dict):
                    model_id = (result.get("model_id") if isinstance(result.get("model_id"), str) else None)
                    if model_id:
                        existing = generations.find_one(
                            {"user_id": x_user_id, "result.model_id": model_id},
                            {"_id": 1},
                        )
                        if not existing:
                            job_params = meta.get("job_params")
                            if not isinstance(job_params, dict):
                                try:
                                    if job.args and isinstance(job.args[0], dict):
                                        job_params = job.args[0]
                                except Exception:
                                    job_params = None

                            if isinstance(job_params, dict):
                                created_at = job.ended_at or datetime.now(timezone.utc)
                                doc = {
                                    "user_id": x_user_id,
                                    "created_at": created_at,
                                    "name": meta.get("name"),
                                    "params": job_params,
                                    "result": result,
                                }
                                generations.insert_one(doc)
            except Exception:
                # Don't block claim if persistence fails
                pass

        return {"job_id": job_id, "user_id": x_user_id, "status": "claimed"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to claim job: {exc}") from exc


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, http_request: Request):
    """
    Retry a failed/finished job by re-enqueueing it with the same parameters.

    Requires the original job to have stored job_params in Redis meta.
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available or not rq.redis or not rq.queue:
        raise HTTPException(status_code=503, detail="Job queue not available.")

    try:
        from rq.job import Job

        old_job = Job.fetch(job_id, connection=rq.redis)
        old_status = old_job.get_status()
        if old_status not in ("failed", "finished", "canceled", "stopped"):
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is {old_status}; only failed/finished jobs can be retried.",
            )

        meta = old_job.meta or {}

        # Prefer explicitly stored params; fall back to the original job args for
        # backwards compatibility (jobs created before retry support).
        job_params = meta.get("job_params")
        if not isinstance(job_params, dict):
            try:
                if old_job.args and isinstance(old_job.args[0], dict):
                    job_params = old_job.args[0]
            except Exception:
                job_params = None

        if not isinstance(job_params, dict):
            raise HTTPException(
                status_code=400,
                detail="Job parameters not available for retry.",
            )

        from ..jobs import run_training_job
        from ..config import RQ_JOB_TIMEOUT

        # Preserve old user/name when possible, but don't require meta to exist.
        old_user_id = (
            meta.get("user_id") if meta.get("user_id") else (old_job.kwargs or {}).get("user_id")
        )
        old_name = meta.get("name") if meta.get("name") else (old_job.kwargs or {}).get("name")

        new_job = rq.queue.enqueue(
            run_training_job,
            job_params,
            user_id=old_user_id,
            name=old_name,
            job_timeout=RQ_JOB_TIMEOUT,
            result_ttl=3600,
        )
        try:
            new_job.meta["job_params"] = job_params
            new_job.meta["user_id"] = old_user_id
            new_job.meta["name"] = old_name
            new_job.meta["retried_from"] = job_id
            new_job.save_meta()
        except Exception:
            pass

        return {
            "job_id": new_job.id,
            "status": "queued",
            "message": f"Job retried from {job_id}. Poll /api/jobs/{new_job.id} for status.",
            "queue_position": len(rq.queue),
            "remote_worker": _maybe_trigger_modal_gpu_worker(rq),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retry job: {exc}") from exc


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str, http_request: Request):
    """
    Request a running job to stop after the current epoch.

    Sets a flag in job meta that the worker checks between epochs.
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available or not rq.redis:
        raise HTTPException(status_code=503, detail="Job queue not available.")

    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=rq.redis)
        status = job.get_status()

        if status != "started":
            raise HTTPException(status_code=400, detail=f"Job is {status}, not running")

        # Set stop flag in meta
        meta = job.meta or {}
        meta["stop_requested"] = True
        job.meta = meta
        job.save_meta()

        return {"job_id": job_id, "status": "stop_requested", "message": "Stop requested. Job will stop after current epoch."}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop job: {exc}") from exc


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, http_request: Request):
    """
    Cancel a queued job.

    Only works for jobs that haven't started yet.
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available:
        raise HTTPException(status_code=503, detail="Job queue not available.")

    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=rq.redis)
        status = job.get_status()

        if status == "started":
            raise HTTPException(status_code=400, detail="Cannot cancel a running job. Use /stop instead.")

        if status == "finished":
            raise HTTPException(status_code=400, detail="Job already completed")

        job.cancel()
        return {"job_id": job_id, "status": "cancelled", "message": "Job cancelled successfully"}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {exc}") from exc


@router.delete("/jobs/{job_id}/delete")
async def delete_job(job_id: str, http_request: Request):
    """
    Delete a job record from Redis (useful to clear failed jobs from the UI).

    Refuses to delete running jobs.
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available or not rq.redis or not rq.queue:
        raise HTTPException(status_code=503, detail="Job queue not available.")

    try:
        from rq.job import Job
        from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry

        job = Job.fetch(job_id, connection=rq.redis)
        status = job.get_status()

        if status == "started":
            raise HTTPException(status_code=400, detail="Cannot delete a running job")

        # Best-effort removal from queue/registries
        try:
            rq.queue.remove(job_id)
        except Exception:
            pass
        try:
            StartedJobRegistry(queue=rq.queue).remove(job_id, delete_job=False)
        except Exception:
            pass
        try:
            FinishedJobRegistry(queue=rq.queue).remove(job_id, delete_job=False)
        except Exception:
            pass
        try:
            FailedJobRegistry(queue=rq.queue).remove(job_id, delete_job=False)
        except Exception:
            pass

        try:
            job.delete()
        except Exception:
            # Fall back to manual delete if older rq API
            try:
                rq.redis.delete(job.key)
            except Exception:
                raise

        return {"job_id": job_id, "status": "deleted"}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {exc}") from exc


@router.delete("/jobs/purge")
async def purge_user_jobs(
    http_request: Request,
    x_user_id: str | None = Header(default=None),
    include_queued: bool = False,
):
    """
    Delete non-running job records associated with the current user from Redis/RQ.

    - Default: deletes finished/failed/etc jobs, but keeps queued jobs.
    - Set include_queued=true to also delete queued jobs.
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq or not rq.available or not rq.redis or not rq.queue:
        raise HTTPException(status_code=503, detail="Job queue not available.")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        from rq.job import Job
        from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry

        started_registry = StartedJobRegistry(queue=rq.queue)
        finished_registry = FinishedJobRegistry(queue=rq.queue)
        failed_registry = FailedJobRegistry(queue=rq.queue)

        candidate_ids = set(rq.queue.job_ids)
        candidate_ids.update(started_registry.get_job_ids())
        candidate_ids.update(finished_registry.get_job_ids())
        candidate_ids.update(failed_registry.get_job_ids())

        deleted = 0
        skipped_running = 0
        skipped_unowned = 0
        skipped_queued = 0
        missing = 0

        for job_id in candidate_ids:
            try:
                job = Job.fetch(job_id, connection=rq.redis)
            except Exception:
                missing += 1
                continue

            meta = job.meta or {}
            if meta.get("user_id") != x_user_id:
                skipped_unowned += 1
                continue

            status = job.get_status()
            if status == "started":
                skipped_running += 1
                continue
            if status == "queued" and not include_queued:
                skipped_queued += 1
                continue

            # Best-effort removal from queue/registries, then delete job key.
            try:
                rq.queue.remove(job_id)
            except Exception:
                pass
            for registry in (started_registry, finished_registry, failed_registry):
                try:
                    registry.remove(job_id, delete_job=False)
                except Exception:
                    pass
            try:
                job.delete()
            except Exception:
                try:
                    rq.redis.delete(job.key)
                except Exception:
                    pass

            deleted += 1

        return {
            "user_id": x_user_id,
            "deleted": deleted,
            "skipped_running": skipped_running,
            "skipped_queued": skipped_queued,
            "skipped_unowned": skipped_unowned,
            "missing": missing,
            "include_queued": include_queued,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to purge jobs: {exc}") from exc


@router.get("/queue")
async def queue_status(
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    """
    Get the current queue status.

    Returns queue availability, length, and job IDs.
    Jobs are filtered by user - only returns jobs belonging to the authenticated user,
    or jobs with no user_id set (unclaimed jobs that can be claimed on login).
    """
    rq = _get_rq_or_reconnect(http_request)

    if not rq:
        return {"available": False, "message": "Queue integration not configured"}

    info = get_queue_info(rq)
    if not info.get("available"):
        # Bubble up the last-known queue init error (safe: does not include REDIS_URL creds).
        reason = getattr(rq, "error", None)
        if reason and not info.get("error"):
            info["error"] = reason

    try:
        redis_url = normalize_redis_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        parsed = urlparse(redis_url)
        redis_host = parsed.hostname or "localhost"
        redis_db = 0
        try:
            if parsed.path and parsed.path != "/":
                redis_db = int(parsed.path.lstrip("/"))
        except Exception:
            redis_db = 0
        info["redis"] = {
            "scheme": parsed.scheme,
            "host": redis_host,
            "port": parsed.port,
            "db": redis_db,
        }
        info["remote_workers_compatible"] = redis_host not in {"localhost", "127.0.0.1", "::1"}
    except Exception:
        pass
    info["local_worker_enabled"] = START_LOCAL_RQ_WORKER

    # Filter job_ids by user ownership
    if info.get("available") and info.get("job_ids"):
        from rq.job import Job

        filtered_ids = []
        for job_id in info["job_ids"]:
            try:
                job = Job.fetch(job_id, connection=rq.redis)
                meta = job.meta or {}
                job_user_id = meta.get("user_id")
                # Show jobs that:
                # 1. Belong to the current user (if logged in)
                # 2. Have no user_id (unclaimed - can be claimed by the current user)
                if x_user_id:
                    # Logged in: show user's jobs and unclaimed jobs
                    if job_user_id is None or job_user_id == x_user_id:
                        filtered_ids.append(job_id)
                else:
                    # Not logged in: only show unclaimed jobs (no user_id)
                    if job_user_id is None:
                        filtered_ids.append(job_id)
            except Exception:
                # Job not found or error - skip it
                pass

        info["job_ids"] = filtered_ids

    # Add worker info if available
    if rq.available and rq.redis:
        try:
            from rq import Worker

            workers = Worker.all(connection=rq.redis)
            info["workers"] = [
                {
                    "name": w.name,
                    "state": w.state,
                    "current_job": w.get_current_job_id(),
                }
                for w in workers
            ]
        except Exception:
            info["workers"] = []

    # Opportunistic on-demand spawn: if there are queued jobs and no workers are running,
    # try to trigger a remote worker. This makes the system "self-healing" even if the
    # initial trigger after enqueue failed.
    try:
        if (
            info.get("available")
            and not START_LOCAL_RQ_WORKER
            and info.get("remote_workers_compatible")
            and int(info.get("queued_jobs") or 0) > 0
            and int(info.get("started_jobs") or 0) == 0
            and len(info.get("workers") or []) == 0
        ):
            import asyncio
            import time

            # Throttle spawn attempts so UI polling doesn't spam Modal.
            should_attempt = True
            try:
                if rq.redis:
                    should_attempt = bool(
                        rq.redis.set("pyreflect:modal_spawn_attempt", str(int(time.time())), nx=True, ex=5)
                    )
            except Exception:
                should_attempt = True

            if should_attempt:
                info["remote_worker"] = await asyncio.to_thread(_maybe_trigger_modal_gpu_worker, rq)
    except Exception:
        pass

    return info


@router.post("/queue/spawn")
async def spawn_remote_worker(http_request: Request):
    """
    Debug endpoint: attempt to trigger a remote Modal worker spawn immediately.

    Useful to diagnose why instant spawn isn't happening (e.g. Modal not installed
    on backend, missing Modal auth, Redis lock held).
    """
    rq = _get_rq_or_reconnect(http_request)
    if not rq or not rq.available:
        raise HTTPException(status_code=503, detail="Job queue not available.")
    return {"remote_worker": _maybe_trigger_modal_gpu_worker(rq)}
