"""
Background job functions for RQ workers.

These functions are designed to be run in a separate worker process.
They should NOT depend on any web request context.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_MONGO_CLIENT = None
_MONGO_URI: str | None = None


def run_training_job(
    job_params: dict[str, Any],
    *,
    user_id: str | None = None,
    name: str | None = None,
    hf_config: dict | None = None,
    mongo_uri: str | None = None,
    reuse_model_id: str | None = None,
) -> dict[str, Any]:
    """
    Run a training job in the background.

    This function is called by the RQ worker and runs the full training pipeline.
    Progress is stored in Redis so the API can poll for updates.

    Args:
        job_params: Training parameters (layers, generator, training config)
        user_id: Optional user ID for tracking
        name: Optional job name
        hf_config: Optional HuggingFace config (repo_id). Prefer env vars (`HF_TOKEN`, `HF_REPO_ID`).
        mongo_uri: Optional MongoDB URI for saving results (falls back to `MONGODB_URI` env var)
        reuse_model_id: Optional model ID to reuse training data from HuggingFace (for retry)

    Returns:
        Dict containing the training results
    """
    from rq import get_current_job

    from ..config import (
        CHECKPOINT_EVERY_N_EPOCHS,
        LEARNING_RATE,
        LOCAL_MODEL_WAIT_POLL_S,
        LOCAL_MODEL_WAIT_TIMEOUT_S,
        MAX_LOCAL_MODELS,
        MODELS_DIR,
        SPLIT_RATIO,
        WEIGHT_DECAY,
    )
    from ..integrations.huggingface import (
        HuggingFaceIntegration,
        upload_model,
        upload_training_data,
        download_training_data,
    )
    from ..services.checkpointing import (
        Checkpoint,
        delete_checkpoint,
        load_checkpoint,
        save_checkpoint,
    )
    from ..services.local_model_limit import save_torch_state_dict_with_local_limit
    from ..services.pyreflect_runtime import PYREFLECT, resolve_torch_device

    global _MONGO_CLIENT, _MONGO_URI

    job = get_current_job()
    logs: list[str] = []

    import os

    meta_flush_interval_s = float(os.getenv("RQ_META_FLUSH_INTERVAL_S", "1.0"))
    max_log_lines = int(os.getenv("RQ_MAX_LOG_LINES", "250"))
    stop_poll_interval_s = float(os.getenv("RQ_STOP_POLL_INTERVAL_S", "1.0"))
    last_meta_flush = 0.0
    pending_meta: dict[str, Any] = {}
    last_stop_poll = 0.0
    cached_stop_requested = False

    class StopRequested(RuntimeError):
        def __init__(self, phase: str):
            super().__init__(f"Stop requested during {phase}")
            self.phase = phase

    class PauseRequested(RuntimeError):
        def __init__(self, phase: str, epoch: int):
            super().__init__(f"Pause requested during {phase} at epoch {epoch}")
            self.phase = phase
            self.epoch = epoch

    def _flush_meta(*, force: bool = False) -> None:
        nonlocal last_meta_flush, pending_meta
        if not job:
            return
        now = time.monotonic()
        if not force and (now - last_meta_flush) < meta_flush_interval_s:
            return

        try:
            meta = job.get_meta(refresh=True) or {}
        except Exception:
            meta = job.meta or {}
        meta.update(pending_meta)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        job.meta = meta
        job.save_meta()

        pending_meta = {}
        last_meta_flush = now

    def _get_stop_requested() -> bool:
        """Refresh job meta periodically and read stop_requested flag."""
        nonlocal last_stop_poll, cached_stop_requested
        if not job:
            return False

        now = time.monotonic()
        if (now - last_stop_poll) < stop_poll_interval_s:
            return cached_stop_requested

        try:
            meta = job.get_meta(refresh=True) or {}
        except Exception:
            meta = job.meta or {}
        cached_stop_requested = bool(meta.get("stop_requested"))
        last_stop_poll = now
        return cached_stop_requested

    def _get_pause_requested() -> bool:
        """Check if pause was requested (save checkpoint and stop)."""
        if not job:
            return False
        try:
            meta = job.get_meta(refresh=True) or {}
        except Exception:
            meta = job.meta or {}
        return bool(meta.get("pause_requested"))

    def _raise_if_stopped(phase: str) -> None:
        if _get_stop_requested():
            raise StopRequested(phase)

    def set_meta(fields: dict[str, Any], *, force: bool = False) -> None:
        """
        Safely update job meta without clobbering fields set by the API.

        RQ stores meta as a single serialized blob; `job.save_meta()` overwrites
        the whole thing. Since the API can set flags like `stop_requested` while
        the worker is running, we always refresh meta from Redis, then merge.
        """
        pending_meta.update(fields)
        _flush_meta(force=force)

    def log(message: str) -> None:
        """Add a log message and update job meta."""
        logs.append(message)
        if max_log_lines > 0 and len(logs) > max_log_lines:
            del logs[: len(logs) - max_log_lines]
        # Also print to stdout so Modal captures logs in real-time
        print(message, flush=True)
        set_meta({"logs": logs})

    def update_progress(
        epoch: int, total: int, train_loss: float, val_loss: float
    ) -> None:
        """Update training progress in job meta."""
        set_meta(
            {
                "progress": {
                    "epoch": epoch,
                    "total": total,
                    "trainLoss": train_loss,
                    "valLoss": val_loss,
                }
            }
        )

    # Initialize job meta
    init_meta: dict[str, Any] = {
        "status": "initializing",
        "logs": logs,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if user_id:
        init_meta["user_id"] = user_id
    if name:
        init_meta["name"] = name
    set_meta(init_meta, force=True)

    if not PYREFLECT.available:
        raise RuntimeError(
            "pyreflect not available. Please install pyreflect dependencies."
        )

    ReflectivityDataGenerator = PYREFLECT.ReflectivityDataGenerator
    DataProcessor = PYREFLECT.DataProcessor
    CNN = PYREFLECT.CNN
    runtime_device = PYREFLECT.DEVICE
    torch = PYREFLECT.torch
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    # Prefer CUDA when available (e.g. Modal GPU workers), regardless of what the
    # upstream `pyreflect.config.runtime.DEVICE` defaulted to.
    device, device_reason = resolve_torch_device(
        torch, runtime_device=runtime_device, prefer_cuda=True
    )
    if device_reason:
        log(f"Warning: {device_reason}")

    log(f"Device selected: {device!s}")
    try:
        if torch is not None and getattr(torch, "cuda", None):
            log(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    except Exception:
        pass

    # =====================
    # Remote Integration Preflight
    # =====================
    def _get_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    hf_repo_id = None
    if isinstance(hf_config, dict):
        hf_repo_id = hf_config.get("repo_id")
    hf_repo_id = hf_repo_id or os.getenv("HF_REPO_ID")
    hf_token = os.getenv("HF_TOKEN")

    storage_mode = (os.getenv("MODEL_STORAGE") or "").strip().lower()
    if not storage_mode:
        storage_mode = "hf" if (hf_repo_id and hf_token) else "local"
    if storage_mode in {"huggingface", "huggingface_only", "hf_only"}:
        storage_mode = "hf"

    log(f"Model storage: {storage_mode}")

    hf: HuggingFaceIntegration | None = None
    if hf_repo_id and hf_token:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            # Pre-create repo so upload is only the file transfer.
            try:
                api.create_repo(
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                    private=False,
                )
            except Exception:
                pass
            hf = HuggingFaceIntegration(available=True, repo_id=hf_repo_id, api=api)
            log(f"Hugging Face ready: {hf_repo_id}")
        except Exception as exc:
            log(f"Warning: Hugging Face preflight failed: {exc}")
            hf = None

    if storage_mode == "hf" and hf is None:
        raise RuntimeError(
            "MODEL_STORAGE=hf requires HF_REPO_ID + HF_TOKEN (and a valid Hugging Face auth)."
        )

    mongo_uri = (mongo_uri or os.getenv("MONGODB_URI") or "").strip() or None
    require_mongo = _get_bool("REQUIRE_MONGO_SAVE", default=False)
    if mongo_uri and (require_mongo or user_id):
        try:
            from pymongo.mongo_client import MongoClient
            from pymongo.server_api import ServerApi

            if _MONGO_CLIENT is None or _MONGO_URI != mongo_uri:
                _MONGO_URI = mongo_uri
                _MONGO_CLIENT = MongoClient(
                    mongo_uri,
                    server_api=ServerApi("1"),
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000,
                )
            _MONGO_CLIENT.admin.command("ping")
            log("MongoDB reachable")
        except Exception as exc:
            log(f"Warning: MongoDB preflight failed: {exc}")
            if require_mongo:
                raise RuntimeError(f"MongoDB not reachable: {exc}") from exc
            mongo_uri = None

    # Extract parameters
    from ..schemas import FilmLayer, GeneratorParams, validate_layer_bounds

    gen_params = job_params.get("generator", {})
    train_params = job_params.get("training", {})
    layer_desc_payload = job_params.get("layers", [])

    # Coerce generator params so layerBound (if present) is validated/normalized.
    gen_params_model = GeneratorParams(**gen_params)
    layer_models: list[FilmLayer] = []
    try:
        if isinstance(layer_desc_payload, list):
            layer_models = [FilmLayer(**layer) for layer in layer_desc_payload]
    except Exception as exc:
        raise RuntimeError(f"Invalid layers payload: {exc}") from exc

    try:
        validate_layer_bounds(layer_models, gen_params_model)
    except Exception as exc:
        # validate_layer_bounds raises HTTPException; normalize to RuntimeError for workers.
        detail = getattr(exc, "detail", None)
        raise RuntimeError(str(detail or exc)) from exc

    num_curves = gen_params_model.numCurves
    num_film_layers = gen_params_model.numFilmLayers
    epochs = train_params.get("epochs", 50)
    batch_size = train_params.get("batchSize", 32)
    layers = train_params.get("layers", [512, 256, 128])
    dropout = train_params.get("dropout", 0.2)

    total_start = time.perf_counter()

    # Generate model_id upfront so it can be used for both .npy and .pth files
    model_id = str(uuid.uuid4())
    log(f"Model bundle ID: {model_id}")
    set_meta({"model_id": model_id}) # Persist for retry reuse


    try:
        # =====================
        # Try to Reuse Training Data (for retry jobs)
        # =====================
        nr_curves = None
        sld_curves = None
        data_reused = False
        gen_time = 0.0
        
        if reuse_model_id and hf:
            log(f"Attempting to reuse training data from models/{reuse_model_id}/...")
            set_meta({"status": "downloading_training_data"})
            try:
                result = download_training_data(hf, reuse_model_id)
                if result is not None:
                    nr_curves, sld_curves = result
                    # Verify the downloaded data matches expected curve count
                    if nr_curves.shape[0] == num_curves and sld_curves.shape[0] == num_curves:
                        data_reused = True
                        log(f"Reusing training data from {reuse_model_id}: NR shape={nr_curves.shape}, SLD shape={sld_curves.shape}")
                    else:
                        log(f"Warning: Downloaded data has {nr_curves.shape[0]} curves but expected {num_curves}, regenerating...")
                        nr_curves = None
                        sld_curves = None
            except Exception as exc:
                log(f"Warning: Could not download training data: {exc}, will generate fresh data")
        elif reuse_model_id and not hf:
            log("Warning: reuse_model_id provided but HuggingFace not configured, will generate fresh data")
        
        # =====================
        # Data Generation (if not reused)
        # =====================
        if not data_reused:
            log(
                f"Generating {num_curves} synthetic curves with {num_film_layers} film layers..."
            )
            set_meta({"status": "generating"})

            gen_start = time.perf_counter()
            layer_desc = None
            layer_bound = None
            if gen_params_model.layerBound:
                layer_desc = layer_desc_payload
                layer_bound = [b.model_dump() for b in gen_params_model.layerBound]

            data_generator = ReflectivityDataGenerator(
                num_layers=num_film_layers,
                layer_desc=layer_desc,
                layer_bound=layer_bound,
            )

            # Generation can be extremely slow (refl1d loops). Generate in chunks so we
            # can honor stop requests and provide basic progress.
            generation_chunk_size = int(os.getenv("RQ_GENERATION_CHUNK_SIZE", "50"))
            if generation_chunk_size <= 0:
                generation_chunk_size = num_curves

            _raise_if_stopped("generation")
            first_n = min(num_curves, max(1, generation_chunk_size))
            nr0, sld0 = data_generator.generate(first_n)

            # pyreflect may return more curves than requested; only take what we need
            actual_first = min(nr0.shape[0], num_curves)
            nr0 = nr0[:actual_first]
            sld0 = sld0[:actual_first]

            # Pre-allocate full arrays to avoid holding many chunk objects.
            nr_curves = np.empty((num_curves,) + tuple(nr0.shape[1:]), dtype=nr0.dtype)
            sld_curves = np.empty((num_curves,) + tuple(sld0.shape[1:]), dtype=sld0.dtype)
            nr_curves[:actual_first] = nr0
            sld_curves[:actual_first] = sld0
            generated = actual_first

            set_meta({"generation": {"done": generated, "total": num_curves}})
            if generated < num_curves:
                log(f"   Generated {generated}/{num_curves} curves...")

            while generated < num_curves:
                _raise_if_stopped("generation")
                n = min(generation_chunk_size, num_curves - generated)
                nr_chunk, sld_chunk = data_generator.generate(n)
                # pyreflect may return more curves than requested; only take what we need
                actual_n = min(nr_chunk.shape[0], num_curves - generated)
                nr_curves[generated : generated + actual_n] = nr_chunk[:actual_n]
                sld_curves[generated : generated + actual_n] = sld_chunk[:actual_n]
                generated += actual_n
                set_meta({"generation": {"done": generated, "total": num_curves}})
                log(f"   Generated {generated}/{num_curves} curves...")

            gen_time = time.perf_counter() - gen_start

            log(f"   Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}")
            log(f"Generation took {gen_time:.2f}s")

        _raise_if_stopped("post_generation")

        # =====================
        # Upload Training Data (if HF configured)
        # =====================
        if hf and storage_mode == "hf":
            log("Uploading training data to Hugging Face...")
            set_meta({"status": "uploading_training_data"})
            try:
                if upload_training_data(hf, nr_curves, sld_curves, model_id):
                    log(f"Training data uploaded to models/{model_id}/")
                else:
                    log("Warning: Training data upload failed")
            except Exception as exc:
                log(f"Warning: Could not upload training data: {exc}")

        # =====================
        # Preprocessing
        # =====================
        log("Preprocessing data...")
        set_meta({"status": "preprocessing"})

        _raise_if_stopped("preprocessing")
        nr_log = np.array(nr_curves, copy=True)
        nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
        nr_stats = _compute_norm_stats(nr_log)
        normalized_nr = DataProcessor.normalize_xy_curves(
            nr_curves,
            apply_log=True,
            min_max_stats=nr_stats,
        )

        _raise_if_stopped("preprocessing")
        sld_stats = _compute_norm_stats(sld_curves)
        normalized_sld = DataProcessor.normalize_xy_curves(
            sld_curves,
            apply_log=False,
            min_max_stats=sld_stats,
        )

        reshaped_nr = normalized_nr[:, 1:2, :]

        _raise_if_stopped("post_preprocessing")

        # =====================
        # Training
        # =====================
        log(f"Training CNN model ({epochs} epochs, batch size {batch_size})...")
        set_meta({"status": "training"})

        model = CNN(layers=layers, dropout_prob=dropout).to(device)
        model.train()

        list_arrays = DataProcessor.split_arrays(
            reshaped_nr, normalized_sld, size_split=SPLIT_RATIO
        )
        tensor_arrays = DataProcessor.convert_tensors(list_arrays)
        _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
            *tensor_arrays,
            batch_size=batch_size,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        loss_fn = torch.nn.MSELoss()

        epoch_list = []
        train_losses = []
        val_losses = []
        start_epoch = 0
        best_val_loss = float("inf")

        # Try to load checkpoint for resume
        resumed_from_checkpoint = False
        if job and hf and hf.api:
            checkpoint = load_checkpoint(hf.api, torch, job.id, device, log_fn=log)
            if checkpoint:
                try:
                    model.load_state_dict(checkpoint.model_state_dict)
                    optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                    epoch_list = checkpoint.epoch_list
                    train_losses = checkpoint.train_losses
                    val_losses = checkpoint.val_losses
                    start_epoch = checkpoint.epoch
                    best_val_loss = checkpoint.best_val_loss
                    resumed_from_checkpoint = True
                    log(f"Resumed from checkpoint at epoch {start_epoch}")
                    set_meta(
                        {
                            "resumed_from_checkpoint": True,
                            "resumed_at_epoch": start_epoch,
                        }
                    )
                except Exception as exc:
                    log(f"Warning: Could not restore checkpoint state: {exc}")
                    # Reset and start fresh
                    epoch_list = []
                    train_losses = []
                    val_losses = []
                    start_epoch = 0

        stopped_early = False
        training_start = time.perf_counter()

        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0

            # Progress bar for each epoch showing batch progress
            batch_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            )
            for X_batch, y_batch in batch_pbar:
                # Check stop frequently so /stop can interrupt mid-epoch.
                if _get_stop_requested():
                    # Check if this is a pause request (save checkpoint before stopping)
                    if _get_pause_requested():
                        raise PauseRequested("training", epoch)
                    raise StopRequested("training")
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # Show running average loss
                batch_pbar.set_postfix(loss=f"{running_loss / (batch_pbar.n + 1):.4f}")

            train_loss = running_loss / len(train_loader)

            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    if _get_stop_requested():
                        if _get_pause_requested():
                            raise PauseRequested("validation", epoch)
                        raise StopRequested("training")
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_running_loss += loss_fn(outputs, y_batch).item()
            val_loss = val_running_loss / len(valid_loader)

            epoch_list.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Log final epoch stats after progress bar completes
            print(f"  -> Train: {train_loss:.6f}, Val: {val_loss:.6f}", flush=True)

            update_progress(epoch + 1, epochs, train_loss, val_loss)

            # Save checkpoint every N epochs (if enabled and HF available)
            if (
                CHECKPOINT_EVERY_N_EPOCHS > 0
                and hf
                and job
                and (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0
                and (epoch + 1) < epochs  # Don't checkpoint on final epoch
            ):
                try:
                    # Move model to CPU for checkpointing
                    cpu_state = {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    }
                    # Deep copy optimizer state and move tensors to CPU
                    # (must copy to avoid mutating optimizer's internal state)
                    opt_state = copy.deepcopy(optimizer.state_dict())
                    for state in opt_state.get("state", {}).values():
                        for k, v in state.items():
                            if hasattr(v, "cpu"):
                                state[k] = v.cpu()

                    ckpt = Checkpoint(
                        job_id=job.id,
                        epoch=epoch + 1,
                        model_state_dict=cpu_state,
                        optimizer_state_dict=opt_state,
                        train_losses=train_losses.copy(),
                        val_losses=val_losses.copy(),
                        epoch_list=epoch_list.copy(),
                        best_val_loss=best_val_loss,
                        saved_at=datetime.now(timezone.utc).isoformat(),
                        nr_stats=nr_stats,
                        sld_stats=sld_stats,
                    )
                    save_checkpoint(hf.api, torch, ckpt, log_fn=log)
                    set_meta({"last_checkpoint_epoch": epoch + 1})
                except Exception as exc:
                    log(f"Warning: Checkpoint save failed: {exc}")

            # Check for stop request after each epoch
            if _get_stop_requested():
                log(f"Stop requested - stopping after epoch {epoch + 1}")
                stopped_early = True
                break

        training_time = time.perf_counter() - training_start
        if stopped_early:
            set_meta({"stopped_early": True})
            log(f"Training stopped early after {len(epoch_list)} epochs")

    except StopRequested as exc:
        log(str(exc))
        set_meta(
            {
                "status": "stopped",
                "stopped_phase": exc.phase,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "logs": logs,
            },
            force=True,
        )
        return {"stopped": True, "phase": exc.phase}

    except PauseRequested as exc:
        log(str(exc))
        log("Saving checkpoint before pausing...")

        # Save checkpoint so training can be resumed
        if hf and hf.api and job:
            try:
                cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                # Deep copy optimizer state and move tensors to CPU
                # (must copy to avoid mutating optimizer's internal state)
                opt_state = copy.deepcopy(optimizer.state_dict())
                for state in opt_state.get("state", {}).values():
                    for k, v in state.items():
                        if hasattr(v, "cpu"):
                            state[k] = v.cpu()

                ckpt = Checkpoint(
                    job_id=job.id,
                    epoch=exc.epoch,  # Save at the epoch we paused at
                    model_state_dict=cpu_state,
                    optimizer_state_dict=opt_state,
                    train_losses=train_losses.copy(),
                    val_losses=val_losses.copy(),
                    epoch_list=epoch_list.copy(),
                    best_val_loss=best_val_loss,
                    saved_at=datetime.now(timezone.utc).isoformat(),
                    nr_stats=nr_stats,
                    sld_stats=sld_stats,
                )
                save_checkpoint(hf.api, torch, ckpt, log_fn=log)
                log(f"Checkpoint saved at epoch {exc.epoch}")
            except Exception as ckpt_exc:
                log(f"Warning: Failed to save checkpoint on pause: {ckpt_exc}")

        set_meta(
            {
                "status": "paused",
                "paused_phase": exc.phase,
                "paused_at_epoch": exc.epoch,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "logs": logs,
            },
            force=True,
        )
        return {"paused": True, "phase": exc.phase, "epoch": exc.epoch}

    # =====================
    # Save Model
    # =====================
    log("Saving model...")
    set_meta({"status": "saving"})

    # Saving tensors on MPS/GPU can be extremely slow or hang; always move to CPU first.
    log("Preparing model for save (moving tensors to CPU)...")
    try:
        raw_state_dict = model.state_dict()
        cpu_state_dict = {}
        for key, value in raw_state_dict.items():
            try:
                cpu_state_dict[key] = value.detach().cpu()  # type: ignore[union-attr]
            except Exception:
                cpu_state_dict[key] = value
    except Exception as exc:
        log(f"Warning: Failed to prepare CPU state_dict: {exc}")
        cpu_state_dict = model.state_dict()

    import io

    model_path: Path | None = None
    model_size_mb: float | None = None

    if storage_mode == "hf":
        set_meta({"status": "uploading"})
        log("Uploading model to Hugging Face (no local storage)...")
        try:
            buffer = io.BytesIO()
            torch.save(cpu_state_dict, buffer)
            size_bytes = buffer.tell()
            model_size_mb = size_bytes / (1024 * 1024)
            buffer.seek(0)

            if hf is None or not upload_model(hf, buffer, model_id):
                raise RuntimeError("Hugging Face upload failed.")
            log(
                f"Model uploaded to Hugging Face Hub: {model_id}.pth ({model_size_mb:.2f} MB)"
            )
        except Exception as exc:
            raise RuntimeError(f"Hugging Face upload failed: {exc}") from exc
    else:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_id}.pth"
        try:
            waiting_marked = False
            for msg in save_torch_state_dict_with_local_limit(
                torch=torch,
                state_dict=cpu_state_dict,
                model_path=model_path,
                models_dir=MODELS_DIR,
                max_models=MAX_LOCAL_MODELS,
                timeout_s=LOCAL_MODEL_WAIT_TIMEOUT_S,
                poll_s=LOCAL_MODEL_WAIT_POLL_S,
                user_id=user_id,
            ):
                if job and not waiting_marked:
                    set_meta({"status": "waiting_for_local_model_slot"})
                    waiting_marked = True
                log(msg)
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from exc

        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        log(f"Model saved locally: {model_id}.pth ({model_size_mb:.2f} MB)")

    # =====================
    # Inference
    # =====================
    log("Running inference on test sample...")
    set_meta({"status": "inference"})

    split_idx = int(len(nr_curves) * SPLIT_RATIO)
    test_idx = split_idx

    gt_nr = nr_curves[test_idx]
    gt_sld = sld_curves[test_idx]

    inference_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        test_nr_normalized = normalized_nr[test_idx : test_idx + 1, 1:2, :]
        test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(device)
        pred_sld_normalized = model(test_input).cpu().numpy()

    pred_sld_denorm = DataProcessor.denormalize_xy_curves(
        pred_sld_normalized, stats=sld_stats, apply_exp=False
    )
    pred_sld_y = pred_sld_denorm[0, 1, :]
    pred_sld_z = pred_sld_denorm[0, 0, :]

    sld_z = np.linspace(0, 450, len(gt_sld[1]))

    # Compute NR from predicted SLD
    computed_nr = gt_nr[1].tolist()
    if (
        PYREFLECT.compute_nr_available
        and compute_nr_from_sld is not None
        and not _get_stop_requested()
    ):
        log("Computing NR from predicted SLD...")
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            _, computed_r = compute_nr_from_sld(
                pred_sld_profile, Q=gt_nr[0], order="substrate_to_air"
            )
            computed_nr = computed_r.tolist()
        except Exception as exc:
            log(f"Warning: Could not compute NR from predicted SLD: {exc}")
    else:
        if _get_stop_requested():
            log("Stop requested - skipping compute_nr_from_sld; using ground truth NR.")
        else:
            log("Warning: compute_nr_from_sld not available; using ground truth NR.")

    # Calculate metrics
    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        {
            "x": int(i),
            "predicted": float(pred_sld_y[idx]),
            "actual": float(gt_sld[1][idx]),
        }
        for i, idx in enumerate(sample_indices)
    ]

    final_mse = val_losses[-1] if val_losses else 0.0
    r2 = 1 - (final_mse / np.var(normalized_sld[:, 1, :]))
    mae = float(np.mean(np.abs(pred_sld_y - gt_sld[1])))
    inference_time = time.perf_counter() - inference_start
    total_time = time.perf_counter() - total_start

    log(
        f"Timing: generation {gen_time:.2f}s, training {training_time:.2f}s, "
        f"inference {inference_time:.2f}s, total {total_time:.2f}s"
    )

    # =====================
    # Build Result
    # =====================
    result = {
        "nr": {
            "q": gt_nr[0].tolist(),
            "groundTruth": gt_nr[1].tolist(),
            "computed": computed_nr,
        },
        "sld": {
            "z": sld_z.tolist(),
            "groundTruth": gt_sld[1].tolist(),
            "predicted": pred_sld_y.tolist(),
        },
        "training": {
            "epochs": epoch_list,
            "trainingLoss": train_losses,
            "validationLoss": val_losses,
        },
        "chi": chi,
        "metrics": {
            "mse": float(final_mse),
            "r2": float(np.clip(r2, 0, 1)),
            "mae": mae,
        },
        "name": name,
        "model_id": model_id,
        "model_size_mb": model_size_mb,
        "timing": {
            "generation": gen_time,
            "training": training_time,
            "inference": inference_time,
            "total": total_time,
        },
    }

    # Save to MongoDB if configured
    runtime_user_id = None
    runtime_name = name
    if job:
        runtime_user_id = (job.meta or {}).get("user_id") or user_id
        runtime_name = (job.meta or {}).get("name") or name
    else:
        runtime_user_id = user_id

    if mongo_uri and runtime_user_id:
        set_meta({"status": "saving_to_history"})
        log("Training complete - moving to history...")
        log("Saving to database...")
        try:
            from pymongo.mongo_client import MongoClient
            from pymongo.server_api import ServerApi

            if _MONGO_CLIENT is None or _MONGO_URI != mongo_uri:
                _MONGO_URI = mongo_uri
                _MONGO_CLIENT = MongoClient(
                    mongo_uri,
                    server_api=ServerApi("1"),
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000,
                )

            client = _MONGO_CLIENT
            client.admin.command("ping")
            db = client["PyReflect"]
            generations = db["generations"]
            doc = {
                "user_id": runtime_user_id,
                "name": runtime_name,
                "created_at": datetime.now(timezone.utc),
                "params": job_params,
                "result": result,
            }
            generations.insert_one(doc)
            log("Results saved to database.")
        except Exception as exc:
            log(f"Warning: Could not save to database: {exc}")

    log("Training complete!")

    # Delete checkpoint on successful completion (no longer needed)
    if hf and hf.api and job:
        delete_checkpoint(hf.api, job.id, log_fn=log)

    # Finalize job meta (force flush so UI sees completion immediately).
    set_meta(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "logs": logs,
        },
        force=True,
    )
    return result


def _compute_norm_stats(curves: np.ndarray) -> dict:
    """Compute normalization statistics for curves."""
    x_points = curves[:, 0, :]
    y_points = curves[:, 1, :]
    return {
        "x": {"min": float(np.min(x_points)), "max": float(np.max(x_points))},
        "y": {"min": float(np.min(y_points)), "max": float(np.max(y_points))},
    }
