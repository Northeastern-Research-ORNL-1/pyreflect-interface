"""Tests for job submission with GPU selection."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest


def test_job_submit_rejects_invalid_gpu(
    client, tmp_backend_layout
) -> None:
    """Invalid GPU tier should be rejected by validation."""
    response = client.post(
        "/api/jobs/submit",
        json={
            "layers": [{"name": "test", "sld": 1.0, "isld": 0.0, "thickness": 100, "roughness": 10}],
            "generator": {"numCurves": 100, "numFilmLayers": 1},
            "training": {
                "batchSize": 32,
                "epochs": 10,
                "layers": 12,
                "dropout": 0.0,
                "latentDim": 16,
                "aeEpochs": 50,
                "mlpEpochs": 50,
            },
            "gpu": "INVALID_GPU",
        },
        headers={"X-User-ID": "test-user"},
    )

    # FastAPI/Pydantic returns 422 for validation errors
    assert response.status_code == 422
    detail = response.json().get("detail", [])
    # Check that the error mentions the gpu field
    assert any("gpu" in str(err).lower() for err in detail)


def test_job_submit_accepts_valid_gpu_tiers(
    client, tmp_backend_layout
) -> None:
    """All valid GPU tiers should pass validation (may fail on RQ availability)."""
    valid_gpus = ["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200", "B200"]

    for gpu in valid_gpus:
        response = client.post(
            "/api/jobs/submit",
            json={
                "layers": [{"name": "test", "sld": 1.0, "isld": 0.0, "thickness": 100, "roughness": 10}],
                "generator": {"numCurves": 100, "numFilmLayers": 1},
                "training": {
                    "batchSize": 32,
                    "epochs": 10,
                    "layers": 12,
                    "dropout": 0.0,
                    "latentDim": 16,
                    "aeEpochs": 50,
                    "mlpEpochs": 50,
                },
                "gpu": gpu,
            },
            headers={"X-User-ID": "test-user"},
        )

        # Should not be a validation error (422).
        # May be 503 (RQ not available) or 200 (success), but not 422.
        assert response.status_code != 422, f"GPU {gpu} should be valid but got validation error"
