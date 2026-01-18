"""
PyReflect Modal GPU Worker

This app runs an RQ worker on a Modal GPU to process queued training jobs.

Design:
- A lightweight cron polls Redis for pending jobs.
- When jobs are present, it spawns a GPU container that runs an `rq` worker in
  burst mode (process jobs, then exit).
- This keeps costs low: no GPU container runs while the queue is empty.

Deploy with: `modal deploy src/backend/modal_worker.py`
"""

from __future__ import annotations

from pathlib import Path

import modal

# Create Modal app
app = modal.App("pyreflect-worker")

_HERE = Path(__file__).resolve().parent
_SERVICE_DIR = _HERE / "service"

# Define the container image with all dependencies.
#
# NOTE: If `torch.cuda.is_available()` is false at runtime, you likely installed a
# CPU-only PyTorch wheel. Use a CUDA-enabled torch build for Modal GPUs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Torch 2.2.x is not compatible with NumPy 2.x at runtime (breaks torch<->numpy
    # interop and emits warnings). Pin NumPy <2 inside Modal to keep training stable.
    .pip_install("numpy<2")
    # IMPORTANT: PyPI `torch` is often CPU-only. Install a CUDA wheel explicitly so
    # the Modal GPU is actually used.
    # NOTE: The CUDA wheels use a local version tag (e.g. `2.2.2+cu121`), so pin it
    # explicitly to avoid accidentally pulling the CPU-only wheel from PyPI.
    .pip_install("torch==2.2.2+cu121", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "redis",
        "rq",
        "scipy",
        "scikit-learn",
        "python-dotenv",
        "requests",
        "pymongo",
        "huggingface_hub",
        # Install the same pyreflect build as the backend (the PyPI `pyreflect`
        # package is a different project and is missing required modules).
        "pyreflect @ https://github.com/williamQyq/pyreflect/archive/refs/heads/main.zip",
    )
    # Bundle only the backend service package so the RQ worker can import
    # `service.jobs.run_training_job` without pulling in unrelated repo files.
    .add_local_dir(_SERVICE_DIR, remote_path="/root/backend/service")
)

def _normalize_redis_url(value: str) -> str:
    """
    Redis client expects a URL with an explicit scheme.

    Users often set `REDIS_URL` to `host:6379` or `:password@host:6379`.
    Accept those by assuming `redis://`.
    """
    value = (value or "").strip()
    if value.lower().startswith("redis_url="):
        value = value.split("=", 1)[1].strip()

    # Strip wrapping quotes, even if mismatched (common when secrets get saved with
    # accidental extra quotes).
    value = value.strip()
    if value[:1] in {'"', "'"}:
        value = value[1:]
    if value[-1:] in {'"', "'"}:
        value = value[:-1]
    value = value.strip()

    # Normalize scheme casing (URL schemes are case-insensitive, but our checks aren't).
    if "://" in value:
        scheme, rest = value.split("://", 1)
        value = f"{scheme.lower()}://{rest}"

    if value.startswith(("redis://", "rediss://", "unix://")):
        return value

    # Support netloc-only URLs like `//:password@host:6379` (scheme omitted).
    if value.startswith("//"):
        return "redis:" + value

    # Accept common aliases.
    if value.startswith(("tcp://", "redis+tcp://")):
        return "redis://" + value.split("://", 1)[1]
    if value.startswith(("ssl://", "tls://", "redis+ssl://")):
        return "rediss://" + value.split("://", 1)[1]

    # Bare host:port or :password@host:port
    if "://" not in value:
        return f"redis://{value}"

    # Unknown scheme (e.g. http://) -> keep as-is, but fail with clearer message upstream.
    return value


@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU (cheapest)
    timeout=4 * 60 * 60,  # Hard cap for the whole burst worker session
    secrets=[modal.Secret.from_name("pyreflect-redis")],  # Redis credentials
)
def run_rq_worker_burst(lock_value: str):
    """
    Run a real RQ worker (SimpleWorker) in burst mode on a GPU container.

    This executes `service.jobs.run_training_job` exactly like a local RQ worker,
    so job status/result fields, logs, HF uploads, and history persistence match.
    """
    import os
    import socket
    import sys
    import uuid
    from urllib.parse import urlparse

    from redis import Redis
    from rq import Queue
    from rq.worker import SimpleWorker

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL not set (configure Modal secret 'pyreflect-redis').")
    redis_url = _normalize_redis_url(redis_url)

    parsed = urlparse(redis_url)
    if parsed.scheme not in {"redis", "rediss", "unix"}:
        raise RuntimeError(
            f"Invalid REDIS_URL scheme '{parsed.scheme}'. Use redis:// or rediss:// (example: redis://:PASSWORD@HOST:6379)."
        )
    redis_host = parsed.hostname or "unknown"
    redis_port = parsed.port or 6379
    print(f"Redis: {parsed.scheme}://{redis_host}:{redis_port}")
    if redis_host in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(
            "REDIS_URL points to localhost; Modal cannot reach Redis on your local machine. "
            "Use a Redis instance reachable from the public internet (or your Modal network)."
        )

    # Ensure backend sources are importable (jobs are stored as `service.*` callables).
    sys.path.insert(0, "/root/backend")

    redis_conn = Redis.from_url(redis_url)
    redis_conn.ping()

    try:
        try:
            import torch

            cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            print(f"torch.cuda.is_available(): {cuda_ok}")
            print(f"torch.version.cuda: {getattr(getattr(torch, 'version', None), 'cuda', None)}")
            if not cuda_ok:
                raise RuntimeError(
                    "CUDA is not available inside this Modal GPU container. "
                    "Install a CUDA-enabled PyTorch build."
                )
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            raise RuntimeError(f"GPU sanity check failed: {exc}") from exc

        queue = Queue("training", connection=redis_conn)
        worker_name = f"gpu-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"

        worker = SimpleWorker([queue], connection=redis_conn, name=worker_name)
        print(f"✅ Starting RQ SimpleWorker '{worker_name}' (burst mode)")
        worker.work(burst=True)
    finally:
        # Release the poller lock if we still own it.
        lock_key = "pyreflect:modal_worker_lock"
        try:
            current = redis_conn.get(lock_key)
            if current is not None and current.decode("utf-8") == lock_value:
                redis_conn.delete(lock_key)
        except Exception:
            pass


@app.function(
    image=image,
    schedule=modal.Cron("* * * * *"),  # Every minute
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def poll_queue():
    """
    Cron job that checks the queue and spawns workers for pending jobs.
    This ensures jobs are processed even when no worker is running.
    """
    import os
    import time
    import uuid
    from urllib.parse import urlparse

    from redis import Redis
    from rq import Queue
    from rq import Worker
    from rq.registry import StartedJobRegistry

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        print("REDIS_URL not set; poll_queue is idle.")
        return
    redis_url = _normalize_redis_url(redis_url)
    parsed = urlparse(redis_url)
    redis_scheme = parsed.scheme or "redis"
    redis_host = parsed.hostname or "unknown"
    redis_port = parsed.port or 6379
    redis_db = 0
    try:
        if parsed.path and parsed.path != "/":
            redis_db = int(parsed.path.lstrip("/"))
    except Exception:
        redis_db = 0
    if redis_host in {"localhost", "127.0.0.1", "::1"}:
        print("WARNING: REDIS_URL points to localhost; Modal cannot reach your local Redis.")
        return

    redis_conn = Redis.from_url(redis_url)
    queue = Queue("training", connection=redis_conn)
    started_ids = StartedJobRegistry(queue=queue).get_job_ids()
    workers = Worker.all(connection=redis_conn)
    gpu_workers = [w for w in workers if (w.name or "").lower().startswith("gpu-")]

    queued = len(queue)
    started = len(started_ids)
    print(
        f"Redis: {redis_scheme}://{redis_host}:{redis_port} db={redis_db} "
        f"queued={queued} started={started} workers={len(workers)} gpu_workers={len(gpu_workers)}"
    )
    if queued > 0 and len(gpu_workers) == 0:
        # Use a Redis lock to avoid spawning overlapping burst workers.
        lock_key = "pyreflect:modal_worker_lock"
        lock_value = f"{uuid.uuid4()}:{int(time.time())}"
        acquired = redis_conn.set(lock_key, lock_value, nx=True, ex=15 * 60)
        if acquired:
            print(f"📋 {queued} jobs queued, spawning GPU worker...")
            run_rq_worker_burst.spawn(lock_value)
        else:
            try:
                ttl = redis_conn.ttl(lock_key)
                existing = redis_conn.get(lock_key)
                existing_str = existing.decode("utf-8") if existing is not None else "?"
                print(f"⏳ Spawn lock held (ttl={ttl}s, value={existing_str}); will retry next tick.")

                # Self-heal: if an older deployment left a very long TTL lock behind,
                # clear it when no workers are running so we can spawn again.
                if started == 0 and len(workers) == 0 and isinstance(ttl, int) and ttl > 20 * 60:
                    print(f"🧹 Clearing stale spawn lock (ttl={ttl}s) and retrying spawn...")
                    try:
                        redis_conn.delete(lock_key)
                    except Exception:
                        pass
                    retry_value = f"{uuid.uuid4()}:{int(time.time())}"
                    retry_acquired = redis_conn.set(lock_key, retry_value, nx=True, ex=15 * 60)
                    if retry_acquired:
                        print(f"📋 {queued} jobs queued, spawning GPU worker...")
                        run_rq_worker_burst.spawn(retry_value)
            except Exception:
                print("⏳ Spawn lock held; will retry next tick.")


# For local testing
if __name__ == "__main__":
    print("Deploy with: modal deploy src/backend/modal_worker.py")
    print("Or run locally: modal run src/backend/modal_worker.py::run_rq_worker_burst")
