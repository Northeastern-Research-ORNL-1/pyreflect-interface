from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import BACKEND_ROOT, CORS_ORIGINS, HF_REPO_ID, HF_TOKEN, MONGODB_URI
from .integrations.huggingface import init_huggingface
from .integrations.mongo import init_mongo, mongo_keepalive
from .integrations.redis_queue import create_rq_integration
from .routers.generate import router as generate_router
from .routers.health import router as health_router
from .routers.history import router as history_router
from .routers.jobs import router as jobs_router
from .routers.models import router as models_router
from .routers.status import router as status_router
from .routers.upload import router as upload_router
from .services.pyreflect_runtime import PYREFLECT
from .settings_store import ensure_backend_layout


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import subprocess
        import sys

        mongo = init_mongo(MONGODB_URI)
        hf = init_huggingface(HF_TOKEN, HF_REPO_ID)
        rq = create_rq_integration()
        app.state.mongo = mongo
        app.state.hf = hf
        app.state.rq = rq

        print("PyReflect Interface Backend starting...")
        print(f"   pyreflect available: {PYREFLECT.available}")
        print(f"   MongoDB available: {mongo.available}")
        print(f"   Redis Queue available: {rq.available}")
        ensure_backend_layout()

        # Start RQ worker subprocess if queue is available
        worker_process = None
        if rq.available:
            try:
                # Use SimpleWorker to avoid macOS fork() crash with PyTorch
                # SimpleWorker runs jobs in the main process (no forking)
                worker_process = subprocess.Popen(
                    [
                        sys.executable, "-m", "rq.cli", "worker", "training",
                        "--path", ".",
                        "--worker-class", "rq.worker.SimpleWorker",
                    ],
                    # Don't pipe stdout/stderr unless you're actively draining them;
                    # otherwise the worker can deadlock once the OS pipe buffer fills.
                    cwd=str(BACKEND_ROOT),
                )
                print(f"   RQ worker started (PID: {worker_process.pid})")
            except Exception as exc:
                print(f"   Warning: Failed to start RQ worker: {exc}")
                worker_process = None

        keepalive_task = None
        if mongo.available and mongo.client is not None:
            keepalive_task = asyncio.create_task(mongo_keepalive(mongo.client))

        yield

        # Cleanup
        if worker_process:
            print("Stopping RQ worker...")
            worker_process.terminate()
            try:
                worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker_process.kill()

        if keepalive_task:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
        print("PyReflect Interface Backend shutting down...")

    app = FastAPI(
        title="PyReflect Interface API",
        description="REST API for neutron reflectivity analysis using pyreflect",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.mongo = None
    app.state.hf = None
    app.state.rq = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api")
    app.include_router(generate_router, prefix="/api")
    app.include_router(history_router, prefix="/api")
    app.include_router(upload_router, prefix="/api")
    app.include_router(status_router, prefix="/api")
    app.include_router(models_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")

    return app
