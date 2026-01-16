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

        return PyreflectRuntime(
            available=True,
            compute_nr_available=compute_nr_available,
            ReflectivityDataGenerator=ReflectivityDataGenerator,
            calculate_reflectivity=calculate_reflectivity,
            DataProcessor=DataProcessor,
            NRSLDDataProcessor=_NRSLDDataProcessor,
            SLDChiDataProcessor=_SLDChiDataProcessor,
            CNN=CNN,
            DEVICE=DEVICE,
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

