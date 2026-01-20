"""Tests for GPU tier schema validation and job submission."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_gpu_tiers_literal_accepts_valid_values() -> None:
    """All defined GPU tiers should be accepted by GenerateRequest."""
    from service.schemas import GenerateRequest, FilmLayer, GeneratorParams, TrainingParams

    valid_gpus = ["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200", "B200"]

    for gpu in valid_gpus:
        req = GenerateRequest(
            layers=[FilmLayer(name="test", sld=1.0, isld=0.0, thickness=100, roughness=10)],
            generator=GeneratorParams(numCurves=100, numFilmLayers=1),
            training=TrainingParams(
                batchSize=32, epochs=10, layers=12, dropout=0.0, latentDim=16, aeEpochs=50, mlpEpochs=50
            ),
            gpu=gpu,
        )
        assert req.gpu == gpu


def test_gpu_tiers_default_is_t4() -> None:
    """Default GPU should be T4."""
    from service.schemas import GenerateRequest, FilmLayer, GeneratorParams, TrainingParams

    req = GenerateRequest(
        layers=[FilmLayer(name="test", sld=1.0, isld=0.0, thickness=100, roughness=10)],
        generator=GeneratorParams(numCurves=100, numFilmLayers=1),
        training=TrainingParams(
            batchSize=32, epochs=10, layers=12, dropout=0.0, latentDim=16, aeEpochs=50, mlpEpochs=50
        ),
    )
    assert req.gpu == "T4"


def test_gpu_tiers_rejects_invalid_value() -> None:
    """Invalid GPU tier should raise validation error."""
    from service.schemas import GenerateRequest, FilmLayer, GeneratorParams, TrainingParams

    with pytest.raises(ValidationError) as exc:
        GenerateRequest(
            layers=[FilmLayer(name="test", sld=1.0, isld=0.0, thickness=100, roughness=10)],
            generator=GeneratorParams(numCurves=100, numFilmLayers=1),
            training=TrainingParams(
                batchSize=32, epochs=10, layers=12, dropout=0.0, latentDim=16, aeEpochs=50, mlpEpochs=50
            ),
            gpu="INVALID_GPU",
        )
    assert "gpu" in str(exc.value).lower()


def test_generate_request_model_dump_includes_gpu() -> None:
    """model_dump() should include the GPU field."""
    from service.schemas import GenerateRequest, FilmLayer, GeneratorParams, TrainingParams

    req = GenerateRequest(
        layers=[FilmLayer(name="test", sld=1.0, isld=0.0, thickness=100, roughness=10)],
        generator=GeneratorParams(numCurves=100, numFilmLayers=1),
        training=TrainingParams(
            batchSize=32, epochs=10, layers=12, dropout=0.0, latentDim=16, aeEpochs=50, mlpEpochs=50
        ),
        gpu="A100",
    )
    dumped = req.model_dump()
    assert dumped["gpu"] == "A100"
