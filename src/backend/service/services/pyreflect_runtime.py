from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PyreflectRuntime:
    available: bool
    compute_nr_available: bool
    ReflectivityDataGenerator: Any | None
    calculate_reflectivity: Any | None
    DataProcessor: Any | None
    NRSLDDataProcessor: Any | None
    SLDChiDataProcessor: Any | None
    CNN: Any | None
    DEVICE: Any | None
    reflectivity_pipeline: Any | None
    chi_pred_trainer: Any | None
    chi_pred_runner: Any | None
    compute_nr_from_sld: Callable[..., Any] | None
    torch: Any | None


def _normalize_cuda_arch(arch: str) -> int | None:
    digits = "".join(ch for ch in arch if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _cuda_arch_supported(torch: Any) -> tuple[bool, str | None]:
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


def resolve_torch_device(
    torch: Any | None,
    *,
    runtime_device: Any | None = None,
    prefer_cuda: bool = False,
) -> tuple[Any | None, str | None]:
    device = runtime_device
    if torch is None:
        return device, None

    def cpu_device() -> Any:
        try:
            return torch.device("cpu")
        except Exception:
            return "cpu"

    def cuda_device() -> Any:
        try:
            return torch.device("cuda")
        except Exception:
            return "cuda"

    if prefer_cuda and getattr(torch, "cuda", None):
        try:
            if torch.cuda.is_available():
                supported, reason = _cuda_arch_supported(torch)
                if supported:
                    return cuda_device(), None
                fallback = runtime_device
                if fallback is None or str(fallback).startswith("cuda"):
                    fallback = cpu_device()
                return fallback, reason
        except Exception as exc:
            fallback = runtime_device
            if fallback is None or str(fallback).startswith("cuda"):
                fallback = cpu_device()
            return fallback, f"CUDA selection failed: {exc}"

    if runtime_device is not None and str(runtime_device).startswith("cuda") and getattr(torch, "cuda", None):
        supported, reason = _cuda_arch_supported(torch)
        if not supported:
            return cpu_device(), reason

    return device, None


def load_pyreflect_runtime() -> PyreflectRuntime:
    try:
        from pyreflect.input.reflectivity_data_generator import (
            ReflectivityDataGenerator,
            calculate_reflectivity,
        )
        from pyreflect.input.data_processor import DataProcessor
        from pyreflect.input import NRSLDDataProcessor as _NRSLDDataProcessor
        from pyreflect.input import SLDChiDataProcessor as _SLDChiDataProcessor
        from pyreflect.models.cnn import CNN
        from pyreflect.config.runtime import DEVICE
        from pyreflect.pipelines import reflectivity_pipeline as _reflectivity_pipeline
        from pyreflect.pipelines import train_autoencoder_mlp_chi_pred as _chi_pred_trainer
        from pyreflect.pipelines import sld_profile_pred_chi as _chi_pred_runner
        import torch

        compute_nr_from_sld = None
        compute_nr_available = False
        try:
            from pyreflect.pipelines import helper as pipelines_helper

            compute_nr_from_sld = pipelines_helper.compute_nr_from_sld
            compute_nr_available = True
        except ImportError as exc:
            print(f"Warning: compute_nr_from_sld not available: {exc}")

        resolved_device, _ = resolve_torch_device(torch, runtime_device=DEVICE)
        return PyreflectRuntime(
            available=True,
            compute_nr_available=compute_nr_available,
            ReflectivityDataGenerator=ReflectivityDataGenerator,
            calculate_reflectivity=calculate_reflectivity,
            DataProcessor=DataProcessor,
            NRSLDDataProcessor=_NRSLDDataProcessor,
            SLDChiDataProcessor=_SLDChiDataProcessor,
            CNN=CNN,
            DEVICE=resolved_device,
            reflectivity_pipeline=_reflectivity_pipeline,
            chi_pred_trainer=_chi_pred_trainer,
            chi_pred_runner=_chi_pred_runner,
            compute_nr_from_sld=compute_nr_from_sld,
            torch=torch,
        )
    except ImportError as exc:
        print(f"Warning: pyreflect not fully available: {exc}")
        return PyreflectRuntime(
            available=False,
            compute_nr_available=False,
            ReflectivityDataGenerator=None,
            calculate_reflectivity=None,
            DataProcessor=None,
            NRSLDDataProcessor=None,
            SLDChiDataProcessor=None,
            CNN=None,
            DEVICE=None,
            reflectivity_pipeline=None,
            chi_pred_trainer=None,
            chi_pred_runner=None,
            compute_nr_from_sld=None,
            torch=None,
        )


PYREFLECT = load_pyreflect_runtime()
