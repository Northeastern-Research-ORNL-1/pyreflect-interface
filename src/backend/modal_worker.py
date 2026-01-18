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
    # IMPORTANT: PyPI `torch` is often CPU-only. Install a CUDA wheel explicitly so
    # the Modal GPU is actually used.
    .pip_install("torch==2.2.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "redis",
        "rq",
        "numpy",
        "scipy",
        "scikit-learn",
        "python-dotenv",
        "requests",
        "pymongo",
        "huggingface_hub",
        "pyreflect",
    )
    # Bundle only the backend service package so the RQ worker can import
    # `service.jobs.run_training_job` without pulling in unrelated repo files.
    .add_local_dir(_SERVICE_DIR, remote_path="/root/backend/service")
)


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

    parsed = urlparse(redis_url)
    redis_host = parsed.hostname or "unknown"
    redis_port = parsed.port or 6379
    print(f"Redis: {redis_host}:{redis_port}")
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


@app.function(image=image, schedule=modal.Cron("* * * * *"))  # Every minute
def poll_queue():
    """
    Cron job that checks the queue and spawns workers for pending jobs.
    This ensures jobs are processed even when no worker is running.
    """
    import os
    import uuid
    from urllib.parse import urlparse

    from redis import Redis
    from rq import Queue

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return

    parsed = urlparse(redis_url)
    redis_host = parsed.hostname or "unknown"
    redis_port = parsed.port or 6379
    if redis_host in {"localhost", "127.0.0.1", "::1"}:
        print("WARNING: REDIS_URL points to localhost; Modal cannot reach your local Redis.")
        return

    redis_conn = Redis.from_url(redis_url)
    queue = Queue("training", connection=redis_conn)

    pending = len(queue)
    print(f"Redis: {redis_host}:{redis_port} pending={pending}")
    if pending > 0:
        # Use a Redis lock to avoid spawning overlapping burst workers.
        lock_key = "pyreflect:modal_worker_lock"
        lock_value = str(uuid.uuid4())
        acquired = redis_conn.set(lock_key, lock_value, nx=True, ex=4 * 60 * 60)
        if acquired:
            print(f"📋 {pending} jobs pending, spawning GPU worker...")
            run_rq_worker_burst.spawn(lock_value)


# For local testing
if __name__ == "__main__":
    print("Deploy with: modal deploy src/backend/modal_worker.py")
    print("Or run locally: modal run src/backend/modal_worker.py::run_rq_worker_burst")
