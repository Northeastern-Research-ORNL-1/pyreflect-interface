"""
PyReflect Modal GPU Worker

This app runs an RQ worker on a Modal GPU to process queued training jobs.

Design:
- A lightweight poller checks Redis for pending jobs.
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
poller_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Keep the fallback poller lightweight: it only needs Redis + RQ to check the queue.
    # Modal HTTP endpoints now require FastAPI to be installed explicitly.
    .pip_install("fastapi", "redis", "rq")
)

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


# GPU tiers available for training (Modal pricing as of Jan 2026)
GPU_TIERS = {
    "T4": "T4",             # $0.59/hr, 16GB VRAM
    "L4": "L4",             # $0.80/hr, 24GB VRAM
    "A10G": "A10G",         # $1.10/hr, 24GB VRAM
    "L40S": "L40S",         # $1.95/hr, 48GB VRAM
    "A100": "A100",         # $2.10/hr, 40GB VRAM
    "A100-80GB": "A100-80GB",  # $2.50/hr, 80GB VRAM
    "H100": "H100",         # $3.95/hr, 80GB VRAM
    "H200": "H200",         # $4.54/hr, 141GB VRAM
    "B200": "B200",         # $6.25/hr, 192GB VRAM
}

DEFAULT_GPU = "T4"


def _run_rq_worker_impl(lock_value: str, gpu_name: str):
    """
    Core RQ worker implementation shared by all GPU-specific functions.
    
    Runs a real RQ worker (SimpleWorker) in burst mode on a GPU container.
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

    print(f"🎮 GPU Worker starting on {gpu_name}")

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
        worker_name = f"gpu-{gpu_name.lower()}-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"

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


# --- GPU-specific worker functions ---
# Each function uses a different GPU tier. The poller/backend spawns the appropriate one.

@app.function(
    image=image,
    gpu="T4",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_t4(lock_value: str):
    """RQ worker on T4 GPU (~$0.59/hr, 16GB VRAM)"""
    _run_rq_worker_impl(lock_value, "T4")


@app.function(
    image=image,
    gpu="L4",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_l4(lock_value: str):
    """RQ worker on L4 GPU (~$0.80/hr, 24GB VRAM)"""
    _run_rq_worker_impl(lock_value, "L4")


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_a10g(lock_value: str):
    """RQ worker on A10G GPU (~$1.10/hr, 24GB VRAM)"""
    _run_rq_worker_impl(lock_value, "A10G")


@app.function(
    image=image,
    gpu="L40S",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_l40s(lock_value: str):
    """RQ worker on L40S GPU ($1.95/hr, 48GB VRAM)"""
    _run_rq_worker_impl(lock_value, "L40S")


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_a100(lock_value: str):
    """RQ worker on A100 GPU ($2.10/hr, 40GB VRAM)"""
    _run_rq_worker_impl(lock_value, "A100")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_a100_80gb(lock_value: str):
    """RQ worker on A100-80GB GPU ($2.50/hr, 80GB VRAM)"""
    _run_rq_worker_impl(lock_value, "A100-80GB")


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_h100(lock_value: str):
    """RQ worker on H100 GPU ($3.95/hr, 80GB VRAM)"""
    _run_rq_worker_impl(lock_value, "H100")


@app.function(
    image=image,
    gpu="H200",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_h200(lock_value: str):
    """RQ worker on H200 GPU ($4.54/hr, 141GB VRAM)"""
    _run_rq_worker_impl(lock_value, "H200")


@app.function(
    image=image,
    gpu="B200",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_b200(lock_value: str):
    """RQ worker on B200 GPU ($6.25/hr, 192GB VRAM)"""
    _run_rq_worker_impl(lock_value, "B200")


# Backwards compatibility alias
def run_rq_worker_burst(lock_value: str):
    """Deprecated: Use run_rq_worker_t4 instead."""
    run_rq_worker_t4(lock_value)


def get_gpu_worker_fn(gpu: str):
    """Get the appropriate worker function for the given GPU tier."""
    gpu = (gpu or DEFAULT_GPU).upper()
    gpu_map = {
        "T4": run_rq_worker_t4,
        "L4": run_rq_worker_l4,
        "A10G": run_rq_worker_a10g,
        "L40S": run_rq_worker_l40s,
        "A100": run_rq_worker_a100,
        "A100-80GB": run_rq_worker_a100_80gb,
        "H100": run_rq_worker_h100,
        "H200": run_rq_worker_h200,
        "B200": run_rq_worker_b200,
    }
    if gpu in gpu_map:
        return gpu_map[gpu]
    print(f"Unknown GPU tier '{gpu}', falling back to {DEFAULT_GPU}")
    return run_rq_worker_t4


@app.function(
    image=poller_image,
    cpu=0.125,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def poll_queue():
    """
    On-demand poller that checks the queue and spawns GPU burst workers for pending jobs.

    This is designed to be triggered immediately after enqueue (backend -> Modal),
    without relying on a periodic schedule.
    """
    return _poll_queue_impl()


@app.function(
    image=poller_image,
    cpu=0.125,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
@modal.fastapi_endpoint(method="POST")
def poll_queue_http(token: str | None = None):
    """
    HTTP trigger for the poller (optional).

    If `MODAL_TRIGGER_TOKEN` is set in the Modal secret, the caller must provide
    `?token=...` to authorize. This endpoint is useful when the backend cannot
    authenticate to Modal to call functions directly.
    """
    import os
    from fastapi import HTTPException

    expected = os.environ.get("MODAL_TRIGGER_TOKEN")
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="unauthorized")
    return _poll_queue_impl()


def _get_requested_gpu_from_queue(queue) -> str:
    """Check the first queued job for GPU preference."""
    try:
        job_ids = queue.job_ids
        if job_ids:
            job = queue.fetch_job(job_ids[0])
            if job and job.args:
                job_params = job.args[0] if job.args else {}
                if isinstance(job_params, dict):
                    return job_params.get("gpu", DEFAULT_GPU)
    except Exception:
        pass
    return DEFAULT_GPU


def _poll_queue_impl() -> dict:
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
        return {"ok": False, "error": "REDIS_URL not set"}
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
        return {"ok": False, "error": "REDIS_URL points to localhost"}

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
        lock_key = "pyreflect:modal_worker_lock"
        lock_value = f"{uuid.uuid4()}:{int(time.time())}"
        lock_ttl_s = int(os.environ.get("MODAL_SPAWN_LOCK_TTL_S", "900"))
        lock_ttl_s = max(lock_ttl_s, 60)
        acquired = redis_conn.set(lock_key, lock_value, nx=True, ex=lock_ttl_s)
        if acquired:
            # Get the GPU preference from the first queued job
            requested_gpu = _get_requested_gpu_from_queue(queue)
            worker_fn = get_gpu_worker_fn(requested_gpu)
            print(f"📋 {queued} jobs queued, spawning {requested_gpu} GPU worker...")
            worker_fn.spawn(lock_value)
            return {"ok": True, "spawned": True, "queued": queued, "started": started, "gpu": requested_gpu}

        try:
            ttl = redis_conn.ttl(lock_key)
            existing = redis_conn.get(lock_key)
            existing_str = existing.decode("utf-8") if existing is not None else "?"
            print(f"⏳ Spawn lock held (ttl={ttl}s, value={existing_str}); will retry next tick.")

            if started == 0 and len(workers) == 0:
                existing_ts: int | None = None
                try:
                    existing_ts = int(existing_str.split(":")[-1])
                except Exception:
                    existing_ts = None

                stale_after_s = max(lock_ttl_s, 120)
                is_stale = False
                if existing_ts is not None:
                    is_stale = (time.time() - existing_ts) > stale_after_s
                elif isinstance(ttl, int) and ttl > stale_after_s:
                    is_stale = True

                if is_stale:
                    print(f"🧹 Clearing stale spawn lock (ttl={ttl}s) and retrying spawn...")
                    try:
                        redis_conn.delete(lock_key)
                    except Exception:
                        pass
                    retry_value = f"{uuid.uuid4()}:{int(time.time())}"
                    retry_acquired = redis_conn.set(lock_key, retry_value, nx=True, ex=lock_ttl_s)
                    if retry_acquired:
                        requested_gpu = _get_requested_gpu_from_queue(queue)
                        worker_fn = get_gpu_worker_fn(requested_gpu)
                        print(f"📋 {queued} jobs queued, spawning {requested_gpu} GPU worker...")
                        worker_fn.spawn(retry_value)
                        return {
                            "ok": True,
                            "spawned": True,
                            "queued": queued,
                            "started": started,
                            "gpu": requested_gpu,
                            "stale_lock_cleared": True,
                        }
        except Exception:
            pass

        return {"ok": True, "spawned": False, "queued": queued, "started": started, "reason": "lock_held"}

    return {"ok": True, "spawned": False, "queued": queued, "started": started}


# For local testing
if __name__ == "__main__":
    print("Deploy with: modal deploy src/backend/modal_worker.py")
    print("Or run locally: modal run src/backend/modal_worker.py::run_rq_worker_burst")
