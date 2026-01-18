"""
Background job functions for RQ workers.

These functions are designed to be run in a separate worker process.
They should NOT depend on any web request context.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_MONGO_CLIENT = None
_MONGO_URI: str | None = None


def run_training_job(
    job_params: dict[str, Any],
    *,
    user_id: str | None = None,
    name: str | None = None,
    hf_config: dict | None = None,
    mongo_uri: str | None = None,
) -> dict[str, Any]:
    """
    Run a training job in the background.

    This function is called by the RQ worker and runs the full training pipeline.
    Progress is stored in Redis so the API can poll for updates.

    Args:
        job_params: Training parameters (layers, generator, training config)
        user_id: Optional user ID for tracking
        name: Optional job name
        hf_config: Optional HuggingFace config (token, repo_id)
        mongo_uri: Optional MongoDB URI for saving results

    Returns:
        Dict containing the training results
    """
    from rq import get_current_job

    from ..config import (
        LEARNING_RATE,
        LOCAL_MODEL_WAIT_POLL_S,
        LOCAL_MODEL_WAIT_TIMEOUT_S,
        MAX_LOCAL_MODELS,
        MODELS_DIR,
        SPLIT_RATIO,
        WEIGHT_DECAY,
    )
    from ..integrations.huggingface import HuggingFaceIntegration, upload_model
    from ..services.local_model_limit import delete_local_model, save_torch_state_dict_with_local_limit
    from ..services.pyreflect_runtime import PYREFLECT

    job = get_current_job()
    logs: list[str] = []

    def set_meta(fields: dict[str, Any]) -> None:
        """
        Safely update job meta without clobbering fields set by the API.

        RQ stores meta as a single serialized blob; `job.save_meta()` overwrites
        the whole thing. Since the API can set flags like `stop_requested` while
        the worker is running, we always refresh meta from Redis, then merge.
        """
        if not job:
            return
        try:
            meta = job.get_meta(refresh=True) or {}
        except Exception:
            meta = job.meta or {}
        meta.update(fields)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        job.meta = meta
        job.save_meta()

    def log(message: str) -> None:
        """Add a log message and update job meta."""
        logs.append(message)
        set_meta({"logs": logs})

    def update_progress(epoch: int, total: int, train_loss: float, val_loss: float) -> None:
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
    set_meta(init_meta)

    if not PYREFLECT.available:
        raise RuntimeError("pyreflect not available. Please install pyreflect dependencies.")

    ReflectivityDataGenerator = PYREFLECT.ReflectivityDataGenerator
    DataProcessor = PYREFLECT.DataProcessor
    CNN = PYREFLECT.CNN
    runtime_device = PYREFLECT.DEVICE
    torch = PYREFLECT.torch
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    # Prefer CUDA when available (e.g. Modal GPU workers), regardless of what the
    # upstream `pyreflect.config.runtime.DEVICE` defaulted to.
    device = runtime_device
    try:
        if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
            device = torch.device("cuda")
    except Exception:
        device = runtime_device

    log(f"Device selected: {device!s}")
    try:
        if torch is not None and getattr(torch, "cuda", None):
            log(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    except Exception:
        pass

    # Extract parameters
    gen_params = job_params.get("generator", {})
    train_params = job_params.get("training", {})

    num_curves = gen_params.get("numCurves", 1000)
    num_film_layers = gen_params.get("numFilmLayers", 3)
    epochs = train_params.get("epochs", 50)
    batch_size = train_params.get("batchSize", 32)
    layers = train_params.get("layers", [512, 256, 128])
    dropout = train_params.get("dropout", 0.2)

    total_start = time.perf_counter()

    # =====================
    # Data Generation
    # =====================
    log(f"Generating {num_curves} synthetic curves with {num_film_layers} film layers...")
    set_meta({"status": "generating"})

    gen_start = time.perf_counter()
    data_generator = ReflectivityDataGenerator(num_layers=num_film_layers)
    nr_curves, sld_curves = data_generator.generate(num_curves)
    gen_time = time.perf_counter() - gen_start

    log(f"   Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}")
    log(f"Generation took {gen_time:.2f}s")

    # =====================
    # Preprocessing
    # =====================
    log("Preprocessing data...")
    set_meta({"status": "preprocessing"})

    nr_log = np.array(nr_curves, copy=True)
    nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
    nr_stats = _compute_norm_stats(nr_log)
    normalized_nr = DataProcessor.normalize_xy_curves(nr_curves, apply_log=True, min_max_stats=nr_stats)

    sld_stats = _compute_norm_stats(sld_curves)
    normalized_sld = DataProcessor.normalize_xy_curves(sld_curves, apply_log=False, min_max_stats=sld_stats)

    reshaped_nr = normalized_nr[:, 1:2, :]

    # =====================
    # Training
    # =====================
    log(f"Training CNN model ({epochs} epochs, batch size {batch_size})...")
    set_meta({"status": "training"})

    model = CNN(layers=layers, dropout_prob=dropout).to(device)
    model.train()

    list_arrays = DataProcessor.split_arrays(reshaped_nr, normalized_sld, size_split=SPLIT_RATIO)
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(*tensor_arrays, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    epoch_list = []
    train_losses = []
    val_losses = []

    stopped_early = False
    training_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_running_loss += loss_fn(outputs, y_batch).item()
        val_loss = val_running_loss / len(valid_loader)

        epoch_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        update_progress(epoch + 1, epochs, train_loss, val_loss)
        log(f"   Epoch {epoch + 1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        # Check for stop request after each epoch
        if job:
            try:
                meta = job.get_meta(refresh=True) or {}
            except Exception:
                meta = job.meta or {}
            if meta.get("stop_requested"):
                log(f"⚠️ Stop requested - stopping after epoch {epoch + 1}")
                stopped_early = True
                break

    training_time = time.perf_counter() - training_start
    if stopped_early:
        log(f"Training stopped early after {len(epoch_list)} epochs")

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

    model_id = str(uuid.uuid4())
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

    # Optional: upload the model artifact back to the backend so Modal workers
    # don't retain large files on ephemeral disk.
    import os

    upload_url = os.getenv("MODEL_UPLOAD_URL")
    upload_token = os.getenv("MODEL_UPLOAD_TOKEN")
    want_backend_upload = bool(upload_url and upload_token)

    # Upload to HuggingFace if configured
    if hf_config and hf_config.get("token") and hf_config.get("repo_id"):
        set_meta({"status": "uploading"})
        log("Uploading to Hugging Face...")
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_config["token"])
            hf = HuggingFaceIntegration(
                available=True,
                api=api,
                repo_id=hf_config["repo_id"],
            )
            if upload_model(hf, model_path, model_id):
                log("Model uploaded to Hugging Face Hub")
                # Verify and cleanup
                try:
                    if api.file_exists(repo_id=hf.repo_id, filename=f"{model_id}.pth", repo_type="dataset"):
                        if want_backend_upload:
                            log("Verified on HF. Keeping local model temporarily for backend upload.")
                        else:
                            delete_local_model(models_dir=MODELS_DIR, model_id=model_id)
                            log("Verified on HF. Local model file deleted (cleanup)")
                except Exception:
                    pass  # Keep local file if verification fails
            else:
                log("Warning: Model NOT uploaded to HF (Error occurred)")
        except Exception as exc:
            log(f"Warning: HuggingFace upload failed: {exc}")

    if want_backend_upload and model_path.exists():
        set_meta({"status": "uploading_to_backend"})
        log("Uploading model artifact to backend...")
        try:
            import requests

            with model_path.open("rb") as f:
                resp = requests.post(
                    upload_url,  # type: ignore[arg-type]
                    headers={"X-Model-Upload-Token": upload_token},  # type: ignore[arg-type]
                    data={"model_id": model_id, "user_id": user_id or ""},
                    files={"file": (f"{model_id}.pth", f, "application/octet-stream")},
                    timeout=60 * 5,
                )
            if resp.ok:
                try:
                    payload = resp.json()
                except Exception:
                    payload = None
                log(f"Model uploaded to backend (status={resp.status_code}).")
                if payload and payload.get("evicted"):
                    log(f"Backend evicted old models: {payload['evicted']}")
                delete_local_model(models_dir=MODELS_DIR, model_id=model_id)
                log("Deleted local model copy after backend upload (cleanup).")
            else:
                log(f"Warning: Backend model upload failed (status={resp.status_code}): {resp.text[:500]}")
        except Exception as exc:
            log(f"Warning: Backend model upload failed: {exc}")

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

    pred_sld_denorm = DataProcessor.denormalize_xy_curves(pred_sld_normalized, stats=sld_stats, apply_exp=False)
    pred_sld_y = pred_sld_denorm[0, 1, :]
    pred_sld_z = pred_sld_denorm[0, 0, :]

    sld_z = np.linspace(0, 450, len(gt_sld[1]))

    # Compute NR from predicted SLD
    computed_nr = gt_nr[1].tolist()
    if PYREFLECT.compute_nr_available and compute_nr_from_sld is not None:
        log("Computing NR from predicted SLD...")
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            _, computed_r = compute_nr_from_sld(pred_sld_profile, Q=gt_nr[0], order="substrate_to_air")
            computed_nr = computed_r.tolist()
        except Exception as exc:
            log(f"Warning: Could not compute NR from predicted SLD: {exc}")
    else:
        log("Warning: compute_nr_from_sld not available; using ground truth NR.")

    # Calculate metrics
    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        {"x": int(i), "predicted": float(pred_sld_y[idx]), "actual": float(gt_sld[1][idx])}
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
        "nr": {"q": gt_nr[0].tolist(), "groundTruth": gt_nr[1].tolist(), "computed": computed_nr},
        "sld": {"z": sld_z.tolist(), "groundTruth": gt_sld[1].tolist(), "predicted": pred_sld_y.tolist()},
        "training": {"epochs": epoch_list, "trainingLoss": train_losses, "validationLoss": val_losses},
        "chi": chi,
        "metrics": {"mse": float(final_mse), "r2": float(np.clip(r2, 0, 1)), "mae": mae},
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
            from pymongo import MongoClient

            global _MONGO_CLIENT, _MONGO_URI
            if _MONGO_CLIENT is None or _MONGO_URI != mongo_uri:
                _MONGO_URI = mongo_uri
                _MONGO_CLIENT = MongoClient(mongo_uri)

            client = _MONGO_CLIENT
            db = client.get_default_database()
            generations = db.generations
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

    # Finalize job meta
    set_meta(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "logs": logs,
        }
    )

    log("Training complete!")
    return result


def _compute_norm_stats(curves: np.ndarray) -> dict:
    """Compute normalization statistics for curves."""
    x_points = curves[:, 0, :]
    y_points = curves[:, 1, :]
    return {
        "x": {"min": float(np.min(x_points)), "max": float(np.max(x_points))},
        "y": {"min": float(np.min(y_points)), "max": float(np.max(y_points))},
    }
