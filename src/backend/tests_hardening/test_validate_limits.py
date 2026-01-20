from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_validate_limits_accepts_within_limits() -> None:
    from service.schemas import GeneratorParams, TrainingParams, validate_limits

    gen = GeneratorParams(numCurves=1, numFilmLayers=1)
    train = TrainingParams(
        batchSize=1,
        epochs=1,
        layers=1,
        dropout=0.0,
        latentDim=2,
        aeEpochs=1,
        mlpEpochs=1,
    )

    validate_limits(
        gen,
        train,
        limits={
            "max_curves": 1,
            "max_film_layers": 1,
            "max_batch_size": 1,
            "max_epochs": 1,
            "max_cnn_layers": 1,
            "max_dropout": 0.0,
            "max_latent_dim": 2,
            "max_ae_epochs": 1,
            "max_mlp_epochs": 1,
        },
    )


def test_validate_limits_collects_all_errors() -> None:
    from service.schemas import GeneratorParams, TrainingParams, validate_limits

    gen = GeneratorParams(numCurves=2, numFilmLayers=2)
    train = TrainingParams(
        batchSize=2,
        epochs=2,
        layers=2,
        dropout=0.5,
        latentDim=3,
        aeEpochs=2,
        mlpEpochs=2,
    )

    with pytest.raises(HTTPException) as exc:
        validate_limits(
            gen,
            train,
            limits={
                "max_curves": 1,
                "max_film_layers": 1,
                "max_batch_size": 1,
                "max_epochs": 1,
                "max_cnn_layers": 1,
                "max_dropout": 0.0,
                "max_latent_dim": 2,
                "max_ae_epochs": 1,
                "max_mlp_epochs": 1,
            },
        )

    assert exc.value.status_code == 400
    # Spot-check a few messages; full concatenation is covered by earlier branches.
    detail = str(exc.value.detail)
    assert "numCurves" in detail
    assert "epochs" in detail
    assert "dropout" in detail
