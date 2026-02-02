from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_validate_layer_bounds_noop_when_missing() -> None:
    from service.schemas import FilmLayer, GeneratorParams, validate_layer_bounds

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]
    gen = GeneratorParams(numCurves=1, numFilmLayers=1)
    validate_layer_bounds(layers, gen)


def test_validate_layer_bounds_rejects_too_few_layers() -> None:
    from service.schemas import (
        FilmLayer,
        GeneratorParams,
        LayerBound,
        validate_layer_bounds,
    )

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0)
    ]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=0, par="sld", bounds=(0.0, 1.0))],
    )

    with pytest.raises(HTTPException) as exc:
        validate_layer_bounds(layers, gen)
    assert exc.value.status_code == 400
    assert "at least substrate" in str(exc.value.detail)


def test_validate_layer_bounds_requires_num_film_layers_match() -> None:
    from service.schemas import (
        FilmLayer,
        GeneratorParams,
        LayerBound,
        validate_layer_bounds,
    )

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="layer_1", sld=4.0, isld=0.0, thickness=50.0, roughness=3.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]

    # len(layers) - 3 == 1, so numFilmLayers must be 1.
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=2,
        layerBound=[LayerBound(i=2, par="thickness", bounds=(10.0, 100.0))],
    )

    with pytest.raises(HTTPException) as exc:
        validate_layer_bounds(layers, gen)
    assert exc.value.status_code == 400
    assert "numFilmLayers must equal" in str(exc.value.detail)


def test_validate_layer_bounds_rejects_out_of_range_i() -> None:
    from service.schemas import (
        FilmLayer,
        GeneratorParams,
        LayerBound,
        validate_layer_bounds,
    )

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="layer_1", sld=4.0, isld=0.0, thickness=50.0, roughness=3.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=99, par="sld", bounds=(0.0, 1.0))],
    )

    with pytest.raises(HTTPException) as exc:
        validate_layer_bounds(layers, gen)
    assert exc.value.status_code == 400
    assert "out of range" in str(exc.value.detail)


def test_validate_layer_bounds_rejects_reversed_bounds() -> None:
    from service.schemas import (
        FilmLayer,
        GeneratorParams,
        LayerBound,
        validate_layer_bounds,
    )

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="layer_1", sld=4.0, isld=0.0, thickness=50.0, roughness=3.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=2, par="sld", bounds=(2.0, 1.0))],
    )

    with pytest.raises(HTTPException) as exc:
        validate_layer_bounds(layers, gen)
    assert exc.value.status_code == 400
    assert "min<=max" in str(exc.value.detail)


def test_validate_layer_bounds_accepts_valid_payload() -> None:
    from service.schemas import (
        FilmLayer,
        GeneratorParams,
        LayerBound,
        validate_layer_bounds,
    )

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="layer_1", sld=4.0, isld=0.0, thickness=50.0, roughness=3.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[
            LayerBound(i=0, par="sld", bounds=(0.0, 10.0)),
            LayerBound(i=2, par="thickness", bounds=(10.0, 100.0)),
        ],
    )
    validate_layer_bounds(layers, gen)


def test_validate_layer_bounds_rejects_bounds_outside_constraints() -> None:
    from service.schemas import (
        FilmLayer,
        GeneratorParams,
        LayerBound,
        validate_layer_bounds,
    )

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="layer_1", sld=4.0, isld=0.0, thickness=50.0, roughness=3.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]
    
    # Test SLD bounds outside allowed range [0, 10]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=2, par="sld", bounds=(-1.0, 15.0))],
    )
    with pytest.raises(HTTPException) as exc:
        validate_layer_bounds(layers, gen)
    assert exc.value.status_code == 400
    assert "must be within" in str(exc.value.detail)
    assert "sld" in str(exc.value.detail)
    
    # Test thickness bounds outside allowed range [0, 1000]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=2, par="thickness", bounds=(10.0, 1500.0))],
    )
    with pytest.raises(HTTPException) as exc:
        validate_layer_bounds(layers, gen)
    assert exc.value.status_code == 400
    assert "must be within" in str(exc.value.detail)
    assert "thickness" in str(exc.value.detail)
    
    # Test isld bounds outside allowed range [0, 1]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=2, par="isld", bounds=(0.0, 2.0))],
    )
    with pytest.raises(HTTPException) as exc:
        validate_layer_bounds(layers, gen)
    assert exc.value.status_code == 400
    assert "must be within" in str(exc.value.detail)
    assert "isld" in str(exc.value.detail)
    
    # Test roughness bounds outside allowed range [0, 200]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=2, par="roughness", bounds=(0.0, 300.0))],
    )
    with pytest.raises(HTTPException) as exc:
        validate_layer_bounds(layers, gen)
    assert exc.value.status_code == 400
    assert "must be within" in str(exc.value.detail)
    assert "roughness" in str(exc.value.detail)
