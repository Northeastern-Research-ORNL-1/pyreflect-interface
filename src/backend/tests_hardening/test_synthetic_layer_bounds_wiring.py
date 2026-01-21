from __future__ import annotations

import types

import pytest


class _Sentinel(Exception):
    pass


def _fake_pyreflect(*, calls: list[dict]) -> object:
    class FakeRDG:
        def __init__(self, *, num_layers, layer_desc=None, layer_bound=None):
            calls.append(
                {
                    "num_layers": num_layers,
                    "layer_desc": layer_desc,
                    "layer_bound": layer_bound,
                }
            )

        def generate(self, _n):
            raise _Sentinel("stop")

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def device(name: str):
            return name

    return types.SimpleNamespace(
        available=True,
        reflectivity_pipeline=object(),
        ReflectivityDataGenerator=FakeRDG,
        DataProcessor=object(),
        CNN=object(),
        DEVICE="cpu",
        torch=FakeTorch,
        compute_nr_available=False,
        compute_nr_from_sld=None,
    )


def test_generate_with_pyreflect_passes_layer_desc_and_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from service.schemas import FilmLayer, GeneratorParams, LayerBound, TrainingParams
    from service.services import synthetic as syn

    calls: list[dict] = []
    monkeypatch.setattr(syn, "PYREFLECT", _fake_pyreflect(calls=calls))

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="layer_1", sld=4.0, isld=0.0, thickness=50.0, roughness=3.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=2, par="thickness", bounds=(10.0, 100.0))],
    )
    train = TrainingParams(
        batchSize=1,
        epochs=1,
        layers=1,
        dropout=0.0,
        latentDim=2,
        aeEpochs=1,
        mlpEpochs=1,
    )

    with pytest.raises(_Sentinel):
        syn.generate_with_pyreflect(layers, gen, train)

    assert calls
    assert calls[0]["num_layers"] == 1
    assert isinstance(calls[0]["layer_desc"], list)
    assert isinstance(calls[0]["layer_bound"], list)


def test_generate_with_pyreflect_omits_layer_desc_without_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from service.schemas import FilmLayer, GeneratorParams, TrainingParams
    from service.services import synthetic as syn

    calls: list[dict] = []
    monkeypatch.setattr(syn, "PYREFLECT", _fake_pyreflect(calls=calls))

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="layer_1", sld=4.0, isld=0.0, thickness=50.0, roughness=3.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]
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

    with pytest.raises(_Sentinel):
        syn.generate_with_pyreflect(layers, gen, train)

    assert calls
    assert calls[0]["layer_desc"] is None
    assert calls[0]["layer_bound"] is None


def test_generate_with_pyreflect_streaming_passes_layer_desc_and_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from service.schemas import FilmLayer, GeneratorParams, LayerBound, TrainingParams
    from service.services import synthetic as syn

    calls: list[dict] = []
    monkeypatch.setattr(syn, "PYREFLECT", _fake_pyreflect(calls=calls))

    layers = [
        FilmLayer(name="substrate", sld=2.0, isld=0.0, thickness=0.0, roughness=1.0),
        FilmLayer(name="siox", sld=3.0, isld=0.0, thickness=10.0, roughness=2.0),
        FilmLayer(name="layer_1", sld=4.0, isld=0.0, thickness=50.0, roughness=3.0),
        FilmLayer(name="air", sld=0.0, isld=0.0, thickness=0.0, roughness=0.0),
    ]
    gen = GeneratorParams(
        numCurves=1,
        numFilmLayers=1,
        layerBound=[LayerBound(i=2, par="thickness", bounds=(10.0, 100.0))],
    )
    train = TrainingParams(
        batchSize=1,
        epochs=1,
        layers=1,
        dropout=0.0,
        latentDim=2,
        aeEpochs=1,
        mlpEpochs=1,
    )

    gen_iter = syn.generate_with_pyreflect_streaming(
        layers=layers,
        gen_params=gen,
        train_params=train,
        user_id=None,
        name=None,
        mongo_generations=None,
        hf=types.SimpleNamespace(available=False, api=None, repo_id=None),
    )

    with pytest.raises(_Sentinel):
        # consume until the fake generator throws
        for _ in gen_iter:
            pass

    assert calls
    assert isinstance(calls[0]["layer_desc"], list)
    assert isinstance(calls[0]["layer_bound"], list)
