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

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"

# B200 (Blackwell, sm_100) needs a torch wheel built with sm_100 support.
# We run all GPU tiers on CUDA 13.x wheels for consistency.
TORCH_INDEX_URL_B200 = TORCH_INDEX_URL


def _build_worker_image(
    *,
    torch_index_url: str,
    torch_pre: bool = False,
    install_torchvision: bool = False,
) -> modal.Image:
    img = (
        modal.Image.debian_slim(python_version="3.11")
        # Match upstream pyreflect pin.
        .pip_install("numpy==2.1.0")
        .pip_install(
            "redis",
            "rq",
            "python-dotenv",
            "requests",
            "pymongo",
            "huggingface_hub",
            "pydantic",
            "fastapi",
        )
    )

    # NOTE: The upstream `pyreflect` project pins `torch==2.5.1` in its metadata.
    # That build does not include sm_100 kernels, so it cannot run on B200.
    # We install `pyreflect` with `--no-deps` and manage its dependencies explicitly,
    # so we can run a CUDA 13 torch build while still using pyreflect's code.

    torch_packages = "torch torchvision" if install_torchvision else "torch"
    pre_flag = "--pre " if torch_pre else ""
    img = img.run_commands(
        # Start clean (prevents a dependency from "sticking" to cu12 builds).
        "pip uninstall -y torch torchvision torchaudio numpy || true",
        # Keep numpy pinned to what pyreflect expects.
        "pip install --no-cache-dir --upgrade --force-reinstall numpy==2.1.0",
        # Install CUDA 13 torch wheels.
        f"pip install --no-cache-dir --upgrade --force-reinstall {pre_flag}{torch_packages} --index-url {torch_index_url}",
        # Re-pin numpy in case the torch install pulled a newer one.
        "pip install --no-cache-dir --upgrade --force-reinstall numpy==2.1.0",
        # Install pyreflect runtime deps (excluding torch/numpy) and allow prereleases for refl1d.
        "pip install --no-cache-dir --upgrade opencv-python pandas seaborn scikit-learn scipy typer pyyaml llvmlite numba refnx tqdm allpairspy",
        "pip install --no-cache-dir --upgrade --pre refl1d",
        # Install pyreflect code without enforcing its pinned torch version.
        "pip install --no-cache-dir --upgrade --force-reinstall --no-deps "
        "'pyreflect @ https://github.com/williamQyq/pyreflect/archive/refs/heads/main.zip'",
        # Build-time sanity checks.
        'python -c "import sys; import numpy as np; import torch; '
        "print('numpy.__version__:', np.__version__); "
        "print('torch.__version__:', torch.__version__); "
        "v=getattr(getattr(torch,'version',None),'cuda',None); "
        "print('torch.version.cuda:', v); "
        "major=int(str(v).split('.')[0]) if v else 0; "
        "sys.exit(0 if (np.__version__=='2.1.0' and major>=13) else 1)\"",
    )

    # Add local files last (Modal best practice; avoids rebuilds).
    return img.add_local_dir(_SERVICE_DIR, remote_path="/root/backend/service")


image = _build_worker_image(torch_index_url=TORCH_INDEX_URL, install_torchvision=True)

# Keep a separate symbol for clarity; both resolve to CUDA 13 wheels.
image_b200 = _build_worker_image(
    torch_index_url=TORCH_INDEX_URL_B200, install_torchvision=True
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
    "T4": "T4",  # $0.59/hr, 16GB VRAM
    "L4": "L4",  # $0.80/hr, 24GB VRAM
    "A10G": "A10G",  # $1.10/hr, 24GB VRAM
    "L40S": "L40S",  # $1.95/hr, 48GB VRAM
    "A100": "A100",  # $2.10/hr, 40GB VRAM
    "A100-80GB": "A100-80GB",  # $2.50/hr, 80GB VRAM
    "H100": "H100",  # $3.95/hr, 80GB VRAM
    "H200": "H200",  # $4.54/hr, 141GB VRAM
    "B200": "B200",  # $6.25/hr, 192GB VRAM
}

GPU_FALLBACK_ORDER = [
    "B200",
    "H200",
    "H100",
    "A100-80GB",
    "A100",
    "L40S",
    "A10G",
    "L4",
    "T4",
]

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
        raise RuntimeError(
            "REDIS_URL not set (configure Modal secret 'pyreflect-redis')."
        )
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

    handed_off_lock = False
    try:
        try:
            import torch

            cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            print(f"torch.__version__: {getattr(torch, '__version__', None)}")
            print(f"torch.cuda.is_available(): {cuda_ok}")
            print(
                f"torch.version.cuda: {getattr(getattr(torch, 'version', None), 'cuda', None)}"
            )
            if not cuda_ok:
                print(
                    "⚠️ CUDA is not available inside this Modal GPU container. "
                    "Falling back to CPU execution."
                )
            else:
                try:
                    arch_list = (
                        torch.cuda.get_arch_list()
                        if getattr(torch.cuda, "get_arch_list", None)
                        else None
                    )
                    print(f"torch.cuda.get_arch_list(): {arch_list}")
                except Exception as arch_exc:
                    print(f"torch.cuda.get_arch_list() failed: {arch_exc}")
                supported, reason = _cuda_arch_supported(torch)
                if not supported:
                    fallback_gpu = _next_gpu_tier(gpu_name)
                    print(
                        f"⚠️ GPU {gpu_name} not supported by this torch build: {reason}"
                    )
                    if fallback_gpu:
                        print(f"↪️ Falling back to {fallback_gpu} GPU worker")
                        try:
                            get_gpu_worker_fn(fallback_gpu).spawn(lock_value)
                            handed_off_lock = True
                            return
                        except Exception as spawn_exc:
                            print(
                                f"Warning: Failed to spawn {fallback_gpu} worker: {spawn_exc}"
                            )
                    print(
                        "↪️ No lower GPU tier available; falling back to CPU execution"
                    )
                else:
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            raise RuntimeError(f"GPU sanity check failed: {exc}") from exc

        queue = Queue("training", connection=redis_conn)
        worker_name = (
            f"gpu-{gpu_name.lower()}-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        )

        worker = SimpleWorker([queue], connection=redis_conn, name=worker_name)
        print(f"✅ Starting RQ SimpleWorker '{worker_name}' (burst mode)")
        worker.work(burst=True)
    finally:
        # Release the poller lock if we still own it.
        lock_key = "pyreflect:modal_worker_lock"
        try:
            # If we spawned a fallback worker, let the fallback worker release the lock.
            if not handed_off_lock:
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
    image=image_b200,
    gpu="B200",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("pyreflect-redis")],
)
def run_rq_worker_b200(lock_value: str):
    """RQ worker on B200 GPU ($6.25/hr, 192GB VRAM)"""
    _run_rq_worker_impl(lock_value, "B200")


@app.function(
    image=image_b200,
    gpu="B200",
    timeout=10 * 60,
)
def torch_diagnostics_b200():
    """Sanity-check torch build/runtime on a B200 container."""
    import subprocess

    import torch

    out: dict[str, object] = {
        "torch.__version__": getattr(torch, "__version__", None),
        "torch.version.cuda": getattr(getattr(torch, "version", None), "cuda", None),
    }
    try:
        out["torch.cuda.get_arch_list"] = (
            list(torch.cuda.get_arch_list())
            if getattr(torch.cuda, "get_arch_list", None)
            else None
        )
    except Exception as exc:
        out["torch.cuda.get_arch_list"] = f"error: {exc}"

    try:
        out["torch.cuda.is_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        out["torch.cuda.is_available"] = f"error: {exc}"

    try:
        out["torch.cuda.get_device_capability"] = tuple(
            torch.cuda.get_device_capability()
        )
    except Exception as exc:
        out["torch.cuda.get_device_capability"] = f"error: {exc}"

    try:
        out["torch.cuda.get_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        out["torch.cuda.get_device_name"] = f"error: {exc}"

    try:
        out["nvidia-smi"] = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.STDOUT
        )
    except Exception as exc:
        out["nvidia-smi"] = f"error: {exc}"

    print(out)
    return out


@app.function(
    image=image_b200,
    gpu="B200",
    timeout=10 * 60,
)
def pyreflect_diagnostics_b200():
    """Sanity-check pyreflect imports against the installed torch build."""
    import importlib
    import traceback

    try:
        from importlib import metadata as importlib_metadata
    except Exception:  # pragma: no cover
        import importlib_metadata  # type: ignore

    import torch

    out: dict[str, object] = {
        "torch.__version__": getattr(torch, "__version__", None),
        "torch.version.cuda": getattr(getattr(torch, "version", None), "cuda", None),
    }

    # Basic CUDA execution test (catches arch mismatches early).
    try:
        if torch.cuda.is_available():
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            out["cuda_tensor_ok"] = bool((x * 2).sum().item() == 12.0)
        else:
            out["cuda_tensor_ok"] = False
    except Exception as exc:
        out["cuda_tensor_ok"] = f"error: {exc}"

    try:
        out["pyreflect.version"] = importlib_metadata.version("pyreflect")
    except Exception as exc:
        out["pyreflect.version"] = f"error: {exc}"

    checks: list[tuple[str, str, str | None]] = [
        ("pyreflect", "pyreflect", None),
        (
            "pyreflect.input.reflectivity_data_generator",
            "pyreflect.input.reflectivity_data_generator",
            "ReflectivityDataGenerator",
        ),
        (
            "pyreflect.input.data_processor",
            "pyreflect.input.data_processor",
            "DataProcessor",
        ),
        ("pyreflect.models.cnn", "pyreflect.models.cnn", "CNN"),
        ("pyreflect.config.runtime", "pyreflect.config.runtime", "DEVICE"),
        (
            "pyreflect.pipelines.reflectivity_pipeline",
            "pyreflect.pipelines",
            "reflectivity_pipeline",
        ),
        (
            "pyreflect.pipelines.train_autoencoder_mlp_chi_pred",
            "pyreflect.pipelines",
            "train_autoencoder_mlp_chi_pred",
        ),
        (
            "pyreflect.pipelines.sld_profile_pred_chi",
            "pyreflect.pipelines",
            "sld_profile_pred_chi",
        ),
    ]

    for label, module_name, attr in checks:
        try:
            mod = importlib.import_module(module_name)
            if attr:
                getattr(mod, attr)
            out[f"import:{label}"] = True
        except Exception as exc:
            out[f"import:{label}"] = f"error: {exc}\n{traceback.format_exc()}"

    print(out)
    return out


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
            return {
                "ok": True,
                "spawned": True,
                "queued": queued,
                "started": started,
                "gpu": requested_gpu,
            }

        try:
            ttl = redis_conn.ttl(lock_key)
            existing = redis_conn.get(lock_key)
            existing_str = existing.decode("utf-8") if existing is not None else "?"
            print(
                f"⏳ Spawn lock held (ttl={ttl}s, value={existing_str}); will retry next tick."
            )

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
                    print(
                        f"🧹 Clearing stale spawn lock (ttl={ttl}s) and retrying spawn..."
                    )
                    try:
                        redis_conn.delete(lock_key)
                    except Exception:
                        pass
                    retry_value = f"{uuid.uuid4()}:{int(time.time())}"
                    retry_acquired = redis_conn.set(
                        lock_key, retry_value, nx=True, ex=lock_ttl_s
                    )
                    if retry_acquired:
                        requested_gpu = _get_requested_gpu_from_queue(queue)
                        worker_fn = get_gpu_worker_fn(requested_gpu)
                        print(
                            f"📋 {queued} jobs queued, spawning {requested_gpu} GPU worker..."
                        )
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

        return {
            "ok": True,
            "spawned": False,
            "queued": queued,
            "started": started,
            "reason": "lock_held",
        }

    return {"ok": True, "spawned": False, "queued": queued, "started": started}


# For local testing
if __name__ == "__main__":
    print("Deploy with: modal deploy src/backend/modal_worker.py")
    print("Or run locally: modal run src/backend/modal_worker.py::run_rq_worker_burst")


def _normalize_cuda_arch(arch: str) -> int | None:
    digits = "".join(ch for ch in arch if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _cuda_arch_supported(torch) -> tuple[bool, str | None]:
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return False, "CUDA is not available in this container"

    try:
        capability = torch.cuda.get_device_capability()
        device_arch = capability[0] * 10 + capability[1]
    except Exception as exc:
        return False, f"Failed to read CUDA device capability: {exc}"

    arch_list: list[str] = []
    if getattr(torch.cuda, "get_arch_list", None):
        try:
            arch_list = list(torch.cuda.get_arch_list())
        except Exception:
            arch_list = []

    if not arch_list:
        return True, None

    arch_numbers = [
        arch_number
        for arch in arch_list
        if (arch_number := _normalize_cuda_arch(arch)) is not None
    ]
    if not arch_numbers:
        return True, None

    if device_arch in arch_numbers:
        return True, None

    max_arch = max(arch_numbers)
    min_arch = min(arch_numbers)
    if device_arch > max_arch:
        return (
            False,
            f"CUDA capability sm_{capability[0]}{capability[1]} exceeds torch build arch list {arch_list}",
        )
    if device_arch < min_arch:
        return (
            False,
            f"CUDA capability sm_{capability[0]}{capability[1]} older than torch build arch list {arch_list}",
        )
    return (
        False,
        f"CUDA capability sm_{capability[0]}{capability[1]} not explicitly supported by torch build arch list {arch_list}",
    )


def _next_gpu_tier(current: str) -> str | None:
    current = (current or "").upper()
    if current not in GPU_FALLBACK_ORDER:
        return None
    idx = GPU_FALLBACK_ORDER.index(current)
    if idx + 1 >= len(GPU_FALLBACK_ORDER):
        return None
    return GPU_FALLBACK_ORDER[idx + 1]
