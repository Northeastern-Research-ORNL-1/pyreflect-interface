from __future__ import annotations

import io
import json
import queue
import threading
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Generator, TextIO, cast

import numpy as np

from ..config import (
    BACKEND_ROOT,
    LEARNING_RATE,
    LOCAL_MODEL_WAIT_POLL_S,
    LOCAL_MODEL_WAIT_TIMEOUT_S,
    MAX_LOCAL_MODELS,
    MODELS_DIR,
    SPLIT_RATIO,
    WEIGHT_DECAY,
)
from ..integrations.huggingface import HuggingFaceIntegration, upload_model
from ..schemas import (
    ChiDataPoint,
    GenerateResponse,
    GeneratorParams,
    Metrics,
    NRData,
    SLDData,
    TrainingData,
    TrainingParams,
)
from .pyreflect_runtime import PYREFLECT, resolve_torch_device
from .local_model_limit import (
    save_torch_state_dict_with_local_limit,
    wait_for_local_model_slot,
)


def compute_norm_stats(curves: np.ndarray) -> dict:
    x_points = curves[:, 0, :]
    y_points = curves[:, 1, :]
    return {
        "x": {"min": float(np.min(x_points)), "max": float(np.max(x_points))},
        "y": {"min": float(np.min(y_points)), "max": float(np.max(y_points))},
    }


def _resolve_model_path(model_id: str | None) -> Path | None:
    """Resolve a model_id to a local .pth path. Returns None if not found."""
    if not model_id:
        return None
    # Reject model IDs containing path separators rather than mutating them.
    if "/" in model_id or "\\" in model_id:
        return None
    candidate = MODELS_DIR / f"{model_id}.pth"
    return candidate if candidate.exists() else None


def _load_normalization_stats_file(stats_path: Path) -> dict:
    """Load normalization stats from a .npy file.

    Supports two formats:
    1. Object-array wrapping a dict with 'nr'/'sld' or 'x'/'y' keys
    2. Structured array with named fields
    """
    data = np.load(stats_path, allow_pickle=True)
    if data.dtype == object:
        item = data.item()
        if isinstance(item, dict):
            # Return the full stats dict so callers can access both 'nr' and
            # 'sld' sub-dicts. Callers that only need the NR sub-dict should
            # index with item["nr"] themselves.
            return item
    raise ValueError(
        f"Could not parse normalization stats from {stats_path}. "
        "Expected a .npy file containing a dict with 'x'/'y' min/max keys."
    )


def generate_with_pyreflect_infer_streaming(
    *,
    layers,
    gen_params: GeneratorParams,
    train_params: TrainingParams,
    model_id: str | None,
    normalization_stats_path: str | None,
    user_id: str | None,
    name: str | None,
    mongo_generations,
    hf: HuggingFaceIntegration,
) -> Generator[str, None, None]:
    """Inference-only streaming path for the synthetic pipeline.

    Skips training entirely. Loads a pre-trained model (.pth) and normalization
    stats (.npy), generates a small set of synthetic test curves, and runs
    inference to predict SLD profiles.
    """

    def emit(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    if not PYREFLECT.available:
        yield emit("error", "pyreflect not available. Please install pyreflect dependencies.")
        return

    ReflectivityDataGenerator = PYREFLECT.ReflectivityDataGenerator
    DataProcessor = PYREFLECT.DataProcessor
    CNN = PYREFLECT.CNN
    runtime_device = PYREFLECT.DEVICE
    torch = PYREFLECT.torch
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    device, device_reason = resolve_torch_device(
        torch, runtime_device=runtime_device, prefer_cuda=True
    )
    if device_reason:
        yield emit("log", f"Warning: {device_reason}")
    yield emit("log", f"[Inference Mode] Device: {device!s}")

    # --- Resolve model path ---
    model_path = _resolve_model_path(model_id)
    if model_path is None:
        from ..settings_store import load_settings, resolve_setting_path
        settings = load_settings()
        nr_sld = settings.get("nr_predict_sld", {})
        model_rel = nr_sld.get("models", {}).get("model")
        if model_rel:
            candidate = resolve_setting_path(model_rel)
            if candidate and candidate.exists():
                model_path = candidate
    if model_path is None:
        pretrained = BACKEND_ROOT / "data" / "pretrained" / "trained_model.pth"
        if pretrained.exists():
            model_path = pretrained
        else:
            yield emit("error", "Pre-trained model not found. Upload a .pth model file first (role: nr_sld_model) or provide a valid model_id.")
            return
    # --- Resolve normalization stats path ---
    norm_path: Path | None = None
    if normalization_stats_path:
        # Reject paths that escape the backend root (path traversal guard)
        try:
            p = (BACKEND_ROOT / normalization_stats_path).resolve()
            p.relative_to(BACKEND_ROOT.resolve())  # raises ValueError if outside
            if p.exists():
                norm_path = p
        except (ValueError, OSError):
            yield emit("error", "Invalid normalization_stats_path: path must remain within the backend data directory.")
            return
    if norm_path is None:
        from ..settings_store import load_settings, resolve_setting_path
        settings = load_settings()
        nr_sld = settings.get("nr_predict_sld", {})
        stats_rel = nr_sld.get("models", {}).get("normalization_stats")
        if stats_rel:
            candidate = resolve_setting_path(stats_rel)
            if candidate and candidate.exists():
                norm_path = candidate
    if norm_path is None:
        pretrained_norm = BACKEND_ROOT / "data" / "pretrained" / "normalization_stat.npy"
        if pretrained_norm.exists():
            norm_path = pretrained_norm
        else:
            yield emit("error", "Normalization stats not found. Upload a normalization_stat.npy file first (role: normalization_stats) or provide a valid path.")
            return

    yield emit("log", f"[Inference Mode] Loading model from: {model_path.name}")
    yield emit("log", f"[Inference Mode] Loading normalization stats from: {norm_path.name}")

    # --- Load normalization stats ---
    try:
        full_stats = _load_normalization_stats_file(norm_path)
        # The .npy file may contain {'nr': {...}, 'sld': {...}} or a flat
        # {'x': ..., 'y': ...} dict.  Pull out sub-dicts when present so that
        # both NR normalisation and SLD *de*normalisation use the correct
        # training-time statistics — never freshly-computed test-curve stats.
        nr_stats = full_stats.get("nr", full_stats)
        sld_stats_from_file: dict | None = full_stats.get("sld")
    except Exception as exc:
        yield emit("error", f"Failed to load normalization stats: {exc}")
        return

    # --- Load model ---
    # Architecture params must come from app settings (matching training config),
    # not from the request's train_params, which reflect user UI state and may differ.
    try:
        from ..settings_store import load_settings as _load_settings
        _settings = _load_settings()
        _model_cfg = _settings.get("nr_predict_sld", {}).get("models", {})
        _cnn_layers = int(_model_cfg.get("layers", 6))
        _cnn_dropout = float(_model_cfg.get("dropout", 0.0873))
    except Exception:
        _cnn_layers, _cnn_dropout = 6, 0.0873  # Optuna best-found defaults

    try:
        model = CNN(layers=_cnn_layers, dropout_prob=_cnn_dropout).to(device)
        state_dict = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        yield emit("log", f"[Inference Mode] Model loaded - CNN(layers={_cnn_layers}, dropout={_cnn_dropout})")
    except Exception as exc:
        yield emit("error", f"Failed to load model: {exc}")
        return

    # --- Generate synthetic curves for inference ---
    num_test_curves = max(1, min(gen_params.numCurves, 100))
    yield emit("log", f"[Inference Mode] Generating {num_test_curves} synthetic test curve(s) for inference...")

    layer_desc = None
    layer_bound = None
    if gen_params.layerBound:
        layer_desc = [
            layer.model_dump() if hasattr(layer, "model_dump") else layer
            for layer in layers
        ]
        layer_bound = [
            b.model_dump() if hasattr(b, "model_dump") else b
            for b in gen_params.layerBound
        ]

    try:
        data_generator = ReflectivityDataGenerator(
            num_layers=gen_params.numFilmLayers,
            layer_desc=layer_desc,
            layer_bound=layer_bound,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            nr_curves, sld_curves = data_generator.generate(num_test_curves)
    except Exception as exc:
        yield emit("error", f"Failed to generate test curves: {exc}")
        return

    yield emit("log", f"[Inference Mode] Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}")

    # --- Normalize using loaded stats ---
    try:
        normalized_nr = DataProcessor.normalize_xy_curves(
            nr_curves, apply_log=True, min_max_stats=nr_stats
        )
        # Use training-time SLD stats for denormalisation.  Fall back to
        # freshly-computed stats only when the .npy file predates the
        # nr/sld split and contains no 'sld' key.
        sld_stats: dict = sld_stats_from_file if sld_stats_from_file is not None else compute_norm_stats(sld_curves)
        normalized_sld = DataProcessor.normalize_xy_curves(
            sld_curves, apply_log=False, min_max_stats=sld_stats
        )
    except Exception as exc:
        yield emit("error", f"Failed to normalize curves: {exc}")
        return

    # --- Run inference on first test sample ---
    test_idx = 0
    gt_nr = nr_curves[test_idx]
    gt_sld = sld_curves[test_idx]

    yield emit("log", "[Inference Mode] Running inference...")

    try:
        with torch.no_grad():
            test_nr_normalized = normalized_nr[test_idx: test_idx + 1, 1:2, :]
            test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(device)
            pred_sld_normalized = model(test_input).cpu().numpy()

        pred_sld_denorm = DataProcessor.denormalize_xy_curves(
            pred_sld_normalized,
            stats=sld_stats,
            apply_exp=False,
        )
        pred_sld_y = pred_sld_denorm[0, 1, :]
        pred_sld_z = pred_sld_denorm[0, 0, :]
    except Exception as exc:
        yield emit("error", f"Inference failed: {exc}")
        return

    # Use the z-grid from the denormalized predictions so the z axis is
    # consistent with the predicted SLD values (same grid used during training).
    sld_z = pred_sld_z

    # --- Compute NR from predicted SLD ---
    if PYREFLECT.compute_nr_available and compute_nr_from_sld is not None:
        yield emit("log", "[Inference Mode] Computing NR from predicted SLD...")
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                _, computed_r = compute_nr_from_sld(
                    pred_sld_profile,
                    Q=gt_nr[0],
                    order="substrate_to_air",
                )
            computed_nr = computed_r.tolist()
        except Exception as exc:
            yield emit("log", f"Warning: Could not compute NR from predicted SLD: {exc}")
            computed_nr = gt_nr[1].tolist()
    else:
        yield emit("log", "Warning: compute_nr_from_sld not available; using ground truth NR.")
        computed_nr = gt_nr[1].tolist()

    # --- Build result ---
    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        {
            "x": int(i),
            "predicted": float(pred_sld_y[idx]),
            "actual": float(gt_sld[1][idx]),
        }
        for i, idx in enumerate(sample_indices)
    ]

    # R² on NR in log-space (standard reflectometry comparison)
    gt_nr_y = gt_nr[1]
    if isinstance(computed_nr, list):
        computed_nr_arr = np.array(computed_nr)
    else:
        computed_nr_arr = computed_nr
    # Clamp to avoid log(0), then compare in log10 space
    gt_log = np.log10(np.clip(gt_nr_y, 1e-12, None))
    comp_log = np.log10(np.clip(computed_nr_arr, 1e-12, None))
    nr_ss_res = np.sum((gt_log - comp_log) ** 2)
    nr_ss_tot = np.sum((gt_log - np.mean(gt_log)) ** 2)
    r2 = float(np.clip(1 - nr_ss_res / nr_ss_tot if nr_ss_tot > 0 else 0.0, 0, 1))
    mae = float(np.mean(np.abs(gt_log - comp_log)))
    mse = float(np.mean((gt_log - comp_log) ** 2))
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
            "epochs": [],
            "trainingLoss": [],
            "validationLoss": [],
        },
        "chi": chi,
        "metrics": {
            "mse": mse,
            "r2": r2,
            "mae": mae,
        },
        "name": name,
        "model_id": model_id,
    }

    yield emit("log", f"[Inference Mode] Done — MAE: {mae:.6f}, R²: {r2:.4f}")
    yield emit("result", result)

    if mongo_generations is not None and user_id:
        from datetime import datetime, timezone
        try:
            doc = {
                "user_id": user_id,
                "name": name,
                "created_at": datetime.now(timezone.utc),
                "mode": "infer",
                "params": {
                    "layers": [layer.model_dump() for layer in layers],
                    "generator": gen_params.model_dump(),
                    "training": train_params.model_dump(),
                },
                "result": result,
            }
            mongo_generations.insert_one(doc)
            yield emit("log", "Results saved to database.")
        except Exception as exc:
            yield emit("log", f"Warning: Could not save to database: {exc}")


def generate_with_pyreflect_streaming(
    *,
    layers,
    gen_params: GeneratorParams,
    train_params: TrainingParams,
    user_id: str | None,
    name: str | None,
    mongo_generations,
    hf: HuggingFaceIntegration,
    mode: str = "train",
    model_id: str | None = None,
    normalization_stats_path: str | None = None,
) -> Generator[str, None, None]:
    # Route to inference-only path when mode == "infer"
    if mode == "infer":
        yield from generate_with_pyreflect_infer_streaming(
            layers=layers,
            gen_params=gen_params,
            train_params=train_params,
            model_id=model_id,
            normalization_stats_path=normalization_stats_path,
            user_id=user_id,
            name=name,
            mongo_generations=mongo_generations,
            hf=hf,
        )
        return

    def emit(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    if not PYREFLECT.available or PYREFLECT.reflectivity_pipeline is None:
        yield emit(
            "error", "pyreflect not available. Please install pyreflect dependencies."
        )
        return

    ReflectivityDataGenerator = PYREFLECT.ReflectivityDataGenerator
    DataProcessor = PYREFLECT.DataProcessor
    CNN = PYREFLECT.CNN
    runtime_device = PYREFLECT.DEVICE
    torch = PYREFLECT.torch
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    device, device_reason = resolve_torch_device(
        torch, runtime_device=runtime_device, prefer_cuda=True
    )
    if device_reason:
        yield emit("log", f"Warning: {device_reason}")
    yield emit("log", f"Device selected: {device!s}")

    total_start = time.perf_counter()

    def emit_warnings(
        context: str, warning_list: list[warnings.WarningMessage]
    ) -> Generator[str, None, None]:
        if not warning_list:
            return
        max_warnings = 10
        for w in warning_list[:max_warnings]:
            yield emit("log", f"Warning ({context}): {w.message}")
        if len(warning_list) > max_warnings:
            yield emit(
                "log",
                f"Warning ({context}): {len(warning_list) - max_warnings} more warnings...",
            )

    HEARTBEAT_INTERVAL = 15.0
    last_heartbeat = [time.perf_counter()]

    def maybe_heartbeat() -> str | None:
        now = time.perf_counter()
        if now - last_heartbeat[0] >= HEARTBEAT_INTERVAL:
            last_heartbeat[0] = now
            return ":keepalive\n\n"
        return None

    class QueueWriter(io.TextIOBase):
        def __init__(self, q: "queue.Queue[str]") -> None:
            super().__init__()
            self.q = q
            self._buffer = ""

        def write(self, s: str) -> int:
            if not s:
                return 0
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self.q.put(line)
            return len(s)

        def flush(self) -> None:
            if self._buffer.strip():
                self.q.put(self._buffer.strip())
            self._buffer = ""

    yield emit(
        "log",
        f"Generating {gen_params.numCurves} synthetic curves with {gen_params.numFilmLayers} film layers...",
    )

    try:
        for msg in wait_for_local_model_slot(
            models_dir=MODELS_DIR,
            max_models=MAX_LOCAL_MODELS,
            timeout_s=LOCAL_MODEL_WAIT_TIMEOUT_S,
            poll_s=LOCAL_MODEL_WAIT_POLL_S,
        ):
            yield emit("log", msg)
            heartbeat = maybe_heartbeat()
            if heartbeat:
                yield heartbeat
    except TimeoutError as exc:
        yield emit("log", f"Error: {exc}")
        yield emit(
            "log",
            "Delete old local models or configure Hugging Face to offload them.",
        )
        yield emit("error", str(exc))
        return
    except Exception as exc:
        yield emit("log", f"Warning: Could not check/wait for local model slots: {exc}")

    layer_desc = None
    layer_bound = None
    if gen_params.layerBound:
        layer_desc = [
            layer.model_dump() if hasattr(layer, "model_dump") else layer
            for layer in layers
        ]
        layer_bound = [
            b.model_dump() if hasattr(b, "model_dump") else b
            for b in gen_params.layerBound
        ]

    data_generator = ReflectivityDataGenerator(
        num_layers=gen_params.numFilmLayers,
        layer_desc=layer_desc,
        layer_bound=layer_bound,
    )
    gen_start = time.perf_counter()
    log_queue: "queue.Queue[str]" = queue.Queue()
    gen_warnings: list[warnings.WarningMessage] = []
    gen_result: dict[str, Any] = {}
    gen_error: list[BaseException] = []

    def run_generate() -> None:
        writer = QueueWriter(log_queue)
        try:
            with warnings.catch_warnings(record=True) as warn_list:
                warnings.simplefilter("always")
                warnings.filterwarnings(
                    "ignore", message=".*data argument is deprecated.*"
                )
                with (
                    redirect_stdout(cast(TextIO, writer)),
                    redirect_stderr(cast(TextIO, writer)),
                ):
                    result = data_generator.generate(gen_params.numCurves)
                gen_warnings.extend(warn_list)
                gen_result["data"] = result
        except Exception as exc:
            gen_error.append(exc)
        finally:
            writer.flush()

    gen_thread = threading.Thread(target=run_generate, daemon=True)
    gen_thread.start()

    while gen_thread.is_alive() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.2)
            if line.strip():
                yield emit("log", line.rstrip())
        except queue.Empty:
            pass

    gen_thread.join()
    if gen_error:
        raise gen_error[0]
    nr_curves, sld_curves = gen_result["data"]
    gen_time = time.perf_counter() - gen_start
    yield emit(
        "log",
        f" Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}",
    )
    yield emit("log", f"Generation took {gen_time:.2f}s")
    for warning_msg in emit_warnings("generation", gen_warnings):
        yield warning_msg

    yield emit("log", "Preprocessing data...")
    nr_log = np.array(nr_curves, copy=True)
    nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
    nr_stats = compute_norm_stats(nr_log)
    normalized_nr = DataProcessor.normalize_xy_curves(
        nr_curves, apply_log=True, min_max_stats=nr_stats
    )

    sld_stats = compute_norm_stats(sld_curves)
    normalized_sld = DataProcessor.normalize_xy_curves(
        sld_curves, apply_log=False, min_max_stats=sld_stats
    )

    reshaped_nr = normalized_nr[:, 1:2, :]

    yield emit(
        "log",
        f"Training CNN model ({train_params.epochs} epochs, batch size {train_params.batchSize})...",
    )
    model = CNN(layers=train_params.layers, dropout_prob=train_params.dropout).to(
        device
    )
    model.train()

    list_arrays = DataProcessor.split_arrays(
        reshaped_nr, normalized_sld, size_split=SPLIT_RATIO
    )
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
        *tensor_arrays, batch_size=train_params.batchSize
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.MSELoss()

    epoch_list = []
    train_losses = []
    val_losses = []

    training_start = time.perf_counter()
    for epoch in range(train_params.epochs):
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

        yield emit(
            "progress",
            {
                "epoch": epoch + 1,
                "total": train_params.epochs,
                "trainLoss": train_loss,
                "valLoss": val_loss,
            },
        )

        heartbeat = maybe_heartbeat()
        if heartbeat:
            yield heartbeat

        yield emit(
            "log",
            f" Epoch {epoch + 1}/{train_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}",
        )

    training_time = time.perf_counter() - training_start

    import uuid

    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.pth"
    yield emit("log", "Preparing model for save (moving tensors to CPU)...")
    try:
        raw_state_dict = model.state_dict()
        cpu_state_dict = {}
        for key, value in raw_state_dict.items():
            try:
                cpu_state_dict[key] = value.detach().cpu()  # type: ignore[union-attr]
            except Exception:
                cpu_state_dict[key] = value
    except Exception as exc:
        yield emit("log", f"Warning: Failed to prepare CPU state_dict: {exc}")
        cpu_state_dict = model.state_dict()
    try:
        for msg in save_torch_state_dict_with_local_limit(
            torch=torch,
            state_dict=cpu_state_dict,
            model_path=model_path,
            models_dir=MODELS_DIR,
            max_models=MAX_LOCAL_MODELS,
            timeout_s=LOCAL_MODEL_WAIT_TIMEOUT_S,
            poll_s=LOCAL_MODEL_WAIT_POLL_S,
            user_id=None,
        ):
            yield emit("log", msg)
            heartbeat = maybe_heartbeat()
            if heartbeat:
                yield heartbeat
    except TimeoutError as exc:
        yield emit("log", f"Error: {exc}")
        yield emit("error", str(exc))
        return
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    yield emit("log", f"Model saved locally: {model_id}.pth ({model_size_mb:.2f} MB)")

    if hf.available and hf.api and hf.repo_id:
        yield emit("log", "Uploading to Hugging Face...")
        if upload_model(hf, model_path, model_id):
            yield emit("log", "Model uploaded to Hugging Face Hub")
            yield emit("log", "Verifying upload...")
            try:
                if hf.api.file_exists(
                    repo_id=hf.repo_id,
                    filename=f"{model_id}.pth",
                    repo_type="dataset",
                ):
                    model_path.unlink()
                    yield emit(
                        "log", "Verified on HF. Local model file deleted (cleanup)"
                    )
                else:
                    yield emit(
                        "log",
                        "Warning: file_exists returned False after upload. keeping local file.",
                    )
            except Exception as exc:
                yield emit("log", f"Warning: Failed to verify/delete: {exc}")
        else:
            yield emit("log", "Warning: Model NOT uploaded to HF (Error occurred)")
    else:
        yield emit("log", "Hugging Face not configured")

    yield emit("log", "Training complete!")
    yield emit("log", f"Training took {training_time:.2f}s")
    yield emit("log", "Running inference on test sample...")

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
        pred_sld_normalized,
        stats=sld_stats,
        apply_exp=False,
    )
    pred_sld_y = pred_sld_denorm[0, 1, :]
    pred_sld_z = pred_sld_denorm[0, 0, :]

    sld_z = np.linspace(0, 450, len(gt_sld[1]))

    if PYREFLECT.compute_nr_available and compute_nr_from_sld is not None:
        yield emit("log", "Computing NR from predicted SLD...")
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            with warnings.catch_warnings(record=True) as nr_warnings:
                warnings.simplefilter("always")
                _, computed_r = compute_nr_from_sld(
                    pred_sld_profile,
                    Q=gt_nr[0],
                    order="substrate_to_air",
                )
            for warning_msg in emit_warnings("computed NR", nr_warnings):
                yield warning_msg
            computed_nr = computed_r.tolist()
        except Exception as exc:
            yield emit(
                "log", f"Warning: Could not compute NR from predicted SLD: {exc}"
            )
            computed_nr = gt_nr[1].tolist()
    else:
        yield emit(
            "log", "Warning: compute_nr_from_sld not available; using ground truth NR."
        )
        computed_nr = gt_nr[1].tolist()

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

    yield emit(
        "log",
        f"Timing: generation {gen_time:.2f}s, training {training_time:.2f}s, inference {inference_time:.2f}s, total {total_time:.2f}s",
    )

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
    }
    yield emit("result", result)

    if mongo_generations is not None and user_id:
        from datetime import datetime, timezone

        try:
            doc = {
                "user_id": user_id,
                "name": name,
                "created_at": datetime.now(timezone.utc),
                "params": {
                    "layers": [layer.model_dump() for layer in layers],
                    "generator": gen_params.model_dump(),
                    "training": train_params.model_dump(),
                },
                "result": result,
            }
            mongo_generations.insert_one(doc)
            yield emit("log", "Results saved to database.")
        except Exception as exc:
            yield emit("log", f"Warning: Could not save to database: {exc}")


def generate_with_pyreflect(
    layers,
    gen_params: GeneratorParams,
    train_params: TrainingParams,
) -> GenerateResponse:
    if not PYREFLECT.available:
        raise RuntimeError("pyreflect not available")

    ReflectivityDataGenerator = PYREFLECT.ReflectivityDataGenerator
    DataProcessor = PYREFLECT.DataProcessor
    CNN = PYREFLECT.CNN
    runtime_device = PYREFLECT.DEVICE
    torch = PYREFLECT.torch
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    device, device_reason = resolve_torch_device(
        torch, runtime_device=runtime_device, prefer_cuda=True
    )
    if device_reason:
        print(f"Warning: {device_reason}")
    print(f"Device selected: {device!s}")

    print(
        f"Generating {gen_params.numCurves} synthetic curves with {gen_params.numFilmLayers} film layers..."
    )

    layer_desc = None
    layer_bound = None
    if gen_params.layerBound:
        layer_desc = [
            layer.model_dump() if hasattr(layer, "model_dump") else layer
            for layer in layers
        ]
        layer_bound = [
            b.model_dump() if hasattr(b, "model_dump") else b
            for b in gen_params.layerBound
        ]

    data_generator = ReflectivityDataGenerator(
        num_layers=gen_params.numFilmLayers,
        layer_desc=layer_desc,
        layer_bound=layer_bound,
    )
    nr_curves, sld_curves = data_generator.generate(gen_params.numCurves)
    print(f" Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}")

    print("Preprocessing data...")
    nr_log = np.array(nr_curves, copy=True)
    nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
    nr_stats = compute_norm_stats(nr_log)
    normalized_nr = DataProcessor.normalize_xy_curves(
        nr_curves, apply_log=True, min_max_stats=nr_stats
    )

    sld_stats = compute_norm_stats(sld_curves)
    normalized_sld = DataProcessor.normalize_xy_curves(
        sld_curves, apply_log=False, min_max_stats=sld_stats
    )

    reshaped_nr = normalized_nr[:, 1:2, :]

    print(
        f"Training CNN model ({train_params.epochs} epochs, batch size {train_params.batchSize})..."
    )
    model = CNN(layers=train_params.layers, dropout_prob=train_params.dropout).to(
        device
    )
    model.train()

    list_arrays = DataProcessor.split_arrays(
        reshaped_nr, normalized_sld, size_split=SPLIT_RATIO
    )
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
        *tensor_arrays, batch_size=train_params.batchSize
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.MSELoss()

    epoch_list: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(train_params.epochs):
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f" Epoch {epoch + 1}/{train_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}"
            )

    import uuid

    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.pth"
    print("Preparing model for save (moving tensors to CPU)...")
    try:
        raw_state_dict = model.state_dict()
        cpu_state_dict = {}
        for key, value in raw_state_dict.items():
            try:
                cpu_state_dict[key] = value.detach().cpu()  # type: ignore[union-attr]
            except Exception:
                cpu_state_dict[key] = value
    except Exception as exc:
        print(f"Warning: Failed to prepare CPU state_dict: {exc}")
        cpu_state_dict = model.state_dict()
    for msg in save_torch_state_dict_with_local_limit(
        torch=torch,
        state_dict=cpu_state_dict,
        model_path=model_path,
        models_dir=MODELS_DIR,
        max_models=MAX_LOCAL_MODELS,
        timeout_s=LOCAL_MODEL_WAIT_TIMEOUT_S,
        poll_s=LOCAL_MODEL_WAIT_POLL_S,
        user_id=None,
    ):
        print(msg)
    print(f"Model saved: {model_id}.pth")

    print("Training complete!")
    print("Running inference on test sample...")

    split_idx = int(len(nr_curves) * SPLIT_RATIO)
    test_idx = split_idx

    gt_nr = nr_curves[test_idx]
    gt_sld = sld_curves[test_idx]

    model.eval()
    with torch.no_grad():
        test_nr_normalized = normalized_nr[test_idx : test_idx + 1, 1:2, :]
        test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(device)
        pred_sld_normalized = model(test_input).cpu().numpy()

    pred_sld_denorm = DataProcessor.denormalize_xy_curves(
        pred_sld_normalized,
        stats=sld_stats,
        apply_exp=False,
    )
    pred_sld_y = pred_sld_denorm[0, 1, :]
    pred_sld_z = pred_sld_denorm[0, 0, :]

    sld_z = np.linspace(0, 450, len(gt_sld[1]))

    if PYREFLECT.compute_nr_available and compute_nr_from_sld is not None:
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            _, computed_r = compute_nr_from_sld(
                pred_sld_profile,
                Q=gt_nr[0],
                order="substrate_to_air",
            )
            computed_nr = computed_r.tolist()
        except Exception as exc:
            print(f"Warning: Could not compute NR from predicted SLD: {exc}")
            computed_nr = gt_nr[1].tolist()
    else:
        print("Warning: compute_nr_from_sld not available; using ground truth NR.")
        computed_nr = gt_nr[1].tolist()

    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        ChiDataPoint(
            x=int(i), predicted=float(pred_sld_y[idx]), actual=float(gt_sld[1][idx])
        )
        for i, idx in enumerate(sample_indices)
    ]

    final_mse = val_losses[-1] if val_losses else 0.0
    r2 = 1 - (final_mse / np.var(normalized_sld[:, 1, :]))
    mae = float(np.mean(np.abs(pred_sld_y - gt_sld[1])))

    return GenerateResponse(
        nr=NRData(
            q=gt_nr[0].tolist(), groundTruth=gt_nr[1].tolist(), computed=computed_nr
        ),
        sld=SLDData(
            z=sld_z.tolist(),
            groundTruth=gt_sld[1].tolist(),
            predicted=pred_sld_y.tolist(),
        ),
        training=TrainingData(
            epochs=epoch_list, trainingLoss=train_losses, validationLoss=val_losses
        ),
        chi=chi,
        metrics=Metrics(mse=float(final_mse), r2=float(np.clip(r2, 0, 1)), mae=mae),
        model_id=model_id,
    )
