from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Generator

import numpy as np

from ..config import (
    LEARNING_RATE,
    MODELS_DIR,
    SPLIT_RATIO,
    WEIGHT_DECAY,
)
from ..integrations.huggingface import HuggingFaceIntegration, upload_model
from ..schemas import GenerateRequest
from ..settings_store import (
    load_settings,
    resolve_setting_path,
    save_settings,
    validate_npy_payload,
)
from .pyreflect_runtime import PYREFLECT, resolve_torch_device


def generate_with_real_data_streaming(
    request: GenerateRequest,
    *,
    user_id: str | None,
    mongo_generations,
    hf: HuggingFaceIntegration,
) -> Generator[str, None, None]:
    def emit(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    if not PYREFLECT.available or PYREFLECT.reflectivity_pipeline is None:
        yield emit("error", "pyreflect not available. Please install pyreflect dependencies.")
        return

    settings = load_settings()
    if request.workflow == "sld_chi":
        yield from _run_sld_chi_workflow(
            settings, request, emit, user_id=user_id, mongo_generations=mongo_generations
        )
        return

    if request.workflow == "nr_sld_chi":
        yield from _run_real_nr_sld_chi(
            settings, request, emit, user_id=user_id, mongo_generations=mongo_generations, hf=hf
        )
        return

    if request.mode == "infer":
        yield from _run_real_nr_sld_infer(settings, request, emit)
    else:
        yield from _run_real_nr_sld_train(
            settings, request, emit, user_id=user_id, mongo_generations=mongo_generations, hf=hf
        )


def _real_nr_sld_train_core(
    settings: dict,
    request: GenerateRequest,
    emit,
    *,
    hf: HuggingFaceIntegration,
) -> Generator[str, None, dict]:
    NRSLDDataProcessor = PYREFLECT.NRSLDDataProcessor
    reflectivity_pipeline = PYREFLECT.reflectivity_pipeline
    DataProcessor = PYREFLECT.DataProcessor
    CNN = PYREFLECT.CNN
    runtime_device = PYREFLECT.DEVICE
    torch = PYREFLECT.torch
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    device, device_reason = resolve_torch_device(torch, runtime_device=runtime_device, prefer_cuda=True)
    if device_reason:
        yield emit("log", f"Warning: {device_reason}")
    yield emit("log", f"Device selected: {device!s}")

    if NRSLDDataProcessor is None or reflectivity_pipeline is None or DataProcessor is None:
        yield emit("error", "NR->SLD workflow not available. Check pyreflect install.")
        return {}

    nr_file = resolve_setting_path(settings["nr_predict_sld"]["file"].get("nr_train"))
    sld_file = resolve_setting_path(settings["nr_predict_sld"]["file"].get("sld_train"))
    model_rel = settings["nr_predict_sld"]["models"].get("model")
    norm_rel = settings["nr_predict_sld"]["models"].get("normalization_stats")

    if not nr_file or not sld_file:
        yield emit("error", "Missing nr_train or sld_train in settings.yml")
        return {}

    if not nr_file.exists() or not sld_file.exists():
        yield emit("error", "Training data files not found. Check settings.yml paths.")
        return {}

    if not model_rel:
        if not request.autoGenerateModelStats:
            yield emit("error", "Missing model path in settings.yml (auto-generate disabled).")
            return {}
        model_rel = f"data/models/model_{int(time.time())}.pth"
        settings["nr_predict_sld"]["models"]["model"] = model_rel
    if not norm_rel:
        if not request.autoGenerateModelStats:
            yield emit("error", "Missing normalization stats path in settings.yml (auto-generate disabled).")
            return {}
        norm_rel = "data/normalization_stat.npy"
        settings["nr_predict_sld"]["models"]["normalization_stats"] = norm_rel

    model_path = resolve_setting_path(model_rel)
    norm_path = resolve_setting_path(norm_rel)
    if model_path is None or norm_path is None:
        yield emit("error", "Invalid model or normalization stats path in settings.yml")
        return {}

    save_settings(settings)

    yield emit("log", "Loading real NR/SLD training data...")
    dproc = NRSLDDataProcessor(nr_file_path=str(nr_file), sld_file_path=str(sld_file)).load_data()
    nr_curves = getattr(dproc, "_nr_arr", None)
    sld_curves = getattr(dproc, "_sld_arr", None)
    if nr_curves is None or sld_curves is None:
        yield emit("error", "Failed to load NR/SLD arrays from .npy files")
        return {}

    yield emit("log", f"NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}")
    yield emit("log", "Preprocessing data and computing normalization stats...")
    reshaped_nr, normalized_sld = reflectivity_pipeline.preprocess(dproc, norm_path)

    yield emit(
        "log",
        f"Training CNN model ({request.training.epochs} epochs, batch size {request.training.batchSize})...",
    )
    model = CNN(layers=request.training.layers, dropout_prob=request.training.dropout).to(device)
    model.train()

    list_arrays = DataProcessor.split_arrays(reshaped_nr, normalized_sld, size_split=SPLIT_RATIO)
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
        *tensor_arrays, batch_size=request.training.batchSize
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    epoch_list = []
    train_losses = []
    val_losses = []

    for epoch in range(request.training.epochs):
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
            {"epoch": epoch + 1, "total": request.training.epochs, "trainLoss": train_loss, "valLoss": val_loss},
        )

        yield emit(
            "log",
            f"   Epoch {epoch + 1}/{request.training.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}",
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    yield emit("log", f"Model saved locally: {model_path.name} ({model_size_mb:.2f} MB)")

    if hf.available:
        yield emit("log", "Uploading to Hugging Face...")
        if upload_model(hf, model_path, model_path.stem):
            yield emit("log", "Model uploaded to Hugging Face Hub")
        else:
            yield emit("log", "Warning: Model NOT uploaded to HF (Error occurred)")

    split_idx = int(len(nr_curves) * SPLIT_RATIO)
    test_idx = min(split_idx, len(nr_curves) - 1)
    gt_nr = nr_curves[test_idx]
    gt_sld = sld_curves[test_idx]

    norm_stats = reflectivity_pipeline.load_normalization_stat(norm_path)
    model.eval()
    with torch.no_grad():
        test_nr_normalized = reshaped_nr[test_idx : test_idx + 1, :, :]
        test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(device)
        pred_sld_normalized = model(test_input).cpu().numpy()

    pred_sld_denorm = DataProcessor.denormalize_xy_curves(
        pred_sld_normalized,
        stats=norm_stats["sld"],
        apply_exp=False,
    )
    pred_sld_z = pred_sld_denorm[0, 0, :]
    pred_sld_y = pred_sld_denorm[0, 1, :]

    computed_nr = gt_nr[1].tolist()
    if PYREFLECT.compute_nr_available and compute_nr_from_sld is not None:
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            _, computed_r = compute_nr_from_sld(pred_sld_profile, Q=gt_nr[0], order="substrate_to_air")
            computed_nr = computed_r.tolist()
        except Exception as exc:
            yield emit("log", f"Warning: Could not compute NR from predicted SLD: {exc}")

    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        {"x": int(i), "predicted": float(pred_sld_y[idx]), "actual": float(gt_sld[1][idx])}
        for i, idx in enumerate(sample_indices)
    ]

    final_mse = val_losses[-1] if val_losses else 0.0
    r2 = 1 - (final_mse / np.var(normalized_sld[:, 1, :]))
    mae = float(np.mean(np.abs(pred_sld_y - gt_sld[1])))

    result = {
        "nr": {"q": gt_nr[0].tolist(), "groundTruth": gt_nr[1].tolist(), "computed": computed_nr},
        "sld": {
            "z": gt_sld[0].tolist(),
            "groundTruth": gt_sld[1].tolist(),
            "predicted": pred_sld_y.tolist(),
        },
        "training": {"epochs": epoch_list, "trainingLoss": train_losses, "validationLoss": val_losses},
        "chi": chi,
        "metrics": {"mse": float(final_mse), "r2": float(np.clip(r2, 0, 1)), "mae": mae},
        "name": request.name,
        "model_id": model_path.stem,
        "model_size_mb": model_size_mb,
    }

    return {"result": result, "predicted_sld": pred_sld_denorm, "model_id": model_path.stem}


def _run_real_nr_sld_train(
    settings: dict,
    request: GenerateRequest,
    emit,
    *,
    user_id: str | None,
    mongo_generations,
    hf: HuggingFaceIntegration,
) -> Generator[str, None, None]:
    payload = yield from _real_nr_sld_train_core(settings, request, emit, hf=hf)
    result = payload.get("result")
    if not result:
        return
    yield emit("result", result)

    if mongo_generations is not None and user_id:
        from datetime import datetime, timezone

        try:
            doc = {
                "user_id": user_id,
                "name": request.name,
                "created_at": datetime.now(timezone.utc),
                "params": {
                    "layers": [layer.model_dump() for layer in request.layers],
                    "generator": request.generator.model_dump(),
                    "training": request.training.model_dump(),
                },
                "result": result,
            }
            mongo_generations.insert_one(doc)
            yield emit("log", "Results saved to database.")
        except Exception as exc:
            yield emit("log", f"Warning: Could not save to database: {exc}")


def _real_nr_sld_infer_core(
    settings: dict,
    request: GenerateRequest,
    emit,
) -> Generator[str, None, dict]:
    NRSLDDataProcessor = PYREFLECT.NRSLDDataProcessor
    reflectivity_pipeline = PYREFLECT.reflectivity_pipeline
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    if NRSLDDataProcessor is None or reflectivity_pipeline is None:
        yield emit("error", "NR->SLD workflow not available. Check pyreflect install.")
        return {}

    nr_file = resolve_setting_path(settings["nr_predict_sld"]["file"].get("experimental_nr_file"))
    model_rel = settings["nr_predict_sld"]["models"].get("model")
    norm_rel = settings["nr_predict_sld"]["models"].get("normalization_stats")

    if not nr_file or not model_rel or not norm_rel:
        yield emit("error", "Missing experimental NR, model, or normalization stats in settings.yml")
        return {}

    model_path = resolve_setting_path(model_rel)
    norm_path = resolve_setting_path(norm_rel)
    if model_path is None or norm_path is None:
        yield emit("error", "Invalid model or normalization stats path in settings.yml")
        return {}

    if not nr_file.exists() or not model_path.exists() or not norm_path.exists():
        yield emit("error", "Required files not found for inference. Check settings.yml paths.")
        return {}

    nr_curves = np.load(nr_file, allow_pickle=True)
    validate_npy_payload("experimental_nr", nr_curves)

    yield emit("log", "Loading model and normalization stats...")
    layers = settings.get("nr_predict_sld", {}).get("models", {}).get("layers", request.training.layers)
    dropout = settings.get("nr_predict_sld", {}).get("models", {}).get("dropout", request.training.dropout)
    model = reflectivity_pipeline.load_nr_sld_model(model_path, layers=layers, dropout_prob=dropout)
    norm_stats = reflectivity_pipeline.load_normalization_stat(norm_path)

    predicted_sld = reflectivity_pipeline.predict_sld_from_nr(model, nr_file, norm_stats)
    pred_curve = predicted_sld[0] if predicted_sld.ndim == 3 else predicted_sld

    gt_nr = nr_curves[0] if nr_curves.ndim == 3 else nr_curves
    pred_sld_z = pred_curve[0]
    pred_sld_y = pred_curve[1]

    computed_nr = gt_nr[1].tolist()
    if PYREFLECT.compute_nr_available and compute_nr_from_sld is not None:
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            _, computed_r = compute_nr_from_sld(pred_sld_profile, Q=gt_nr[0], order="substrate_to_air")
            computed_nr = computed_r.tolist()
        except Exception as exc:
            yield emit("log", f"Warning: Could not compute NR from predicted SLD: {exc}")

    result = {
        "nr": {"q": gt_nr[0].tolist(), "groundTruth": gt_nr[1].tolist(), "computed": computed_nr},
        "sld": {"z": pred_sld_z.tolist(), "groundTruth": pred_sld_y.tolist(), "predicted": pred_sld_y.tolist()},
        "training": {"epochs": [], "trainingLoss": [], "validationLoss": []},
        "chi": [],
        "metrics": {"mse": 0.0, "r2": 0.0, "mae": 0.0},
        "name": request.name,
        "model_id": Path(model_rel).stem,
    }

    yield emit("log", "Inference complete (no ground truth SLD available).")
    return {"result": result, "predicted_sld": predicted_sld, "model_id": Path(model_rel).stem}


def _run_real_nr_sld_infer(
    settings: dict,
    request: GenerateRequest,
    emit,
) -> Generator[str, None, None]:
    payload = yield from _real_nr_sld_infer_core(settings, request, emit)
    result = payload.get("result")
    if not result:
        return
    yield emit("result", result)


def _run_real_nr_sld_chi(
    settings: dict,
    request: GenerateRequest,
    emit,
    *,
    user_id: str | None,
    mongo_generations,
    hf: HuggingFaceIntegration,
) -> Generator[str, None, None]:
    if request.mode == "infer":
        nr_payload = yield from _real_nr_sld_infer_core(settings, request, emit)
    else:
        nr_payload = yield from _real_nr_sld_train_core(settings, request, emit, hf=hf)

    base_result = nr_payload.get("result")
    predicted_sld = nr_payload.get("predicted_sld")
    if not base_result or predicted_sld is None:
        return

    chi_payload = yield from _sld_chi_core(settings, request, emit, expt_override=predicted_sld)
    chi_data = chi_payload.get("chi")
    sld_curve = chi_payload.get("sld_curve")
    if chi_data is None or sld_curve is None:
        return

    base_result["chi"] = chi_data
    base_result["sld"] = {
        "z": sld_curve[0].tolist(),
        "groundTruth": sld_curve[1].tolist(),
        "predicted": sld_curve[1].tolist(),
    }
    base_result["name"] = request.name

    yield emit("result", base_result)

    if mongo_generations is not None and user_id:
        from datetime import datetime, timezone

        try:
            doc = {
                "user_id": user_id,
                "name": request.name,
                "created_at": datetime.now(timezone.utc),
                "params": {
                    "layers": [layer.model_dump() for layer in request.layers],
                    "generator": request.generator.model_dump(),
                    "training": request.training.model_dump(),
                },
                "result": base_result,
            }
            mongo_generations.insert_one(doc)
            yield emit("log", "Results saved to database.")
        except Exception as exc:
            yield emit("log", f"Warning: Could not save to database: {exc}")


def _sld_chi_core(
    settings: dict,
    request: GenerateRequest,
    emit,
    expt_override: np.ndarray | None = None,
) -> Generator[str, None, dict]:
    SLDChiDataProcessor = PYREFLECT.SLDChiDataProcessor
    chi_pred_trainer = PYREFLECT.chi_pred_trainer
    chi_pred_runner = PYREFLECT.chi_pred_runner

    if SLDChiDataProcessor is None or chi_pred_trainer is None or chi_pred_runner is None:
        yield emit("error", "SLD->Chi workflow not available. Check pyreflect install.")
        return {}

    chi_files = settings.get("sld_predict_chi", {}).get("file", {})
    expt_path = resolve_setting_path(chi_files.get("model_experimental_sld_profile"))
    sld_path = resolve_setting_path(chi_files.get("model_sld_file"))
    params_path = resolve_setting_path(chi_files.get("model_chi_params_file"))

    if not expt_path or not sld_path or not params_path:
        yield emit("error", "Missing SLD/Chi files in settings.yml")
        return {}

    if not expt_path.exists() or not sld_path.exists() or not params_path.exists():
        yield emit("error", "SLD/Chi files not found. Check settings.yml paths.")
        return {}

    yield emit("log", "Loading SLD/Chi datasets...")
    data_processor = SLDChiDataProcessor(str(expt_path), str(sld_path), str(params_path))
    data_processor.load_data()
    sld_arr, params_arr = data_processor.preprocess_data()
    yield emit("log", f"SLD shape: {sld_arr.shape}, Chi params shape: {params_arr.shape}")

    yield emit("log", "Training autoencoder + MLP for chi prediction...")
    mlp, autoencoder = chi_pred_trainer.run_model_training(
        X=sld_arr,
        y=params_arr,
        latent_dim=request.training.latentDim,
        batch_size=request.training.batchSize,
        ae_epochs=request.training.aeEpochs,
        mlp_epochs=request.training.mlpEpochs,
    )

    expt_input = expt_override if expt_override is not None else data_processor.expt_arr
    df_predictions, _ = chi_pred_runner.run_model_prediction(mlp, autoencoder, X=expt_input)
    rows = df_predictions.to_dict(orient="records")
    chi_row = rows[0] if rows else {}
    chi_items = [(k, v) for k, v in chi_row.items() if k.startswith("chi")]

    chi_data = [
        {"x": idx + 1, "predicted": float(val), "actual": float(val)}
        for idx, (_, val) in enumerate(sorted(chi_items))
    ]

    expt_arr = expt_input
    if isinstance(expt_arr, np.ndarray) and expt_arr.ndim == 3:
        expt_curve = expt_arr[0]
    else:
        expt_curve = expt_arr

    yield emit("log", "Chi prediction complete (actual values unavailable for experimental input).")
    return {"chi": chi_data, "sld_curve": expt_curve}


def _run_sld_chi_workflow(
    settings: dict,
    request: GenerateRequest,
    emit,
    *,
    user_id: str | None,
    mongo_generations,
) -> Generator[str, None, None]:
    payload = yield from _sld_chi_core(settings, request, emit)
    chi_data = payload.get("chi")
    expt_curve = payload.get("sld_curve")
    if chi_data is None or expt_curve is None:
        return

    result = {
        "nr": {"q": [], "groundTruth": [], "computed": []},
        "sld": {
            "z": expt_curve[0].tolist(),
            "groundTruth": expt_curve[1].tolist(),
            "predicted": expt_curve[1].tolist(),
        },
        "training": {"epochs": [], "trainingLoss": [], "validationLoss": []},
        "chi": chi_data,
        "metrics": {"mse": 0.0, "r2": 0.0, "mae": 0.0},
        "name": request.name,
    }

    yield emit("result", result)

    if mongo_generations is not None and user_id:
        from datetime import datetime, timezone

        try:
            doc = {
                "user_id": user_id,
                "name": request.name,
                "created_at": datetime.now(timezone.utc),
                "params": {
                    "layers": [layer.model_dump() for layer in request.layers],
                    "generator": request.generator.model_dump(),
                    "training": request.training.model_dump(),
                },
                "result": result,
            }
            mongo_generations.insert_one(doc)
            yield emit("log", "Results saved to database.")
        except Exception as exc:
            yield emit("log", f"Warning: Could not save to database: {exc}")
