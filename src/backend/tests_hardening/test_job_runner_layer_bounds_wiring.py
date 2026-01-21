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
        ReflectivityDataGenerator=FakeRDG,
        DataProcessor=object(),
        CNN=object(),
        DEVICE="cpu",
        torch=FakeTorch,
        compute_nr_available=False,
        compute_nr_from_sld=None,
    )


def test_worker_passes_layer_desc_and_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    from service import jobs

    calls: list[dict] = []
    monkeypatch.setattr(jobs, "PYREFLECT", _fake_pyreflect(calls=calls))

    # No rq context in unit test.
    import rq

    monkeypatch.setattr(rq, "get_current_job", lambda: None)

    job_params = {
        "layers": [
            {
                "name": "substrate",
                "sld": 2.0,
                "isld": 0.0,
                "thickness": 0.0,
                "roughness": 1.0,
            },
            {
                "name": "siox",
                "sld": 3.0,
                "isld": 0.0,
                "thickness": 10.0,
                "roughness": 2.0,
            },
            {
                "name": "layer_1",
                "sld": 4.0,
                "isld": 0.0,
                "thickness": 50.0,
                "roughness": 3.0,
            },
            {
                "name": "air",
                "sld": 0.0,
                "isld": 0.0,
                "thickness": 0.0,
                "roughness": 0.0,
            },
        ],
        "generator": {
            "numCurves": 1,
            "numFilmLayers": 1,
            "layerBound": [{"i": 2, "par": "thickness", "bounds": [10.0, 100.0]}],
        },
        "training": {
            "batchSize": 1,
            "epochs": 1,
            "layers": 1,
            "dropout": 0.0,
            "latentDim": 2,
            "aeEpochs": 1,
            "mlpEpochs": 1,
        },
        "gpu": "T4",
    }

    with pytest.raises(_Sentinel):
        jobs.run_training_job(job_params)

    assert calls
    assert isinstance(calls[0]["layer_desc"], list)
    assert isinstance(calls[0]["layer_bound"], list)


def test_worker_omits_layer_desc_without_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from service import jobs

    calls: list[dict] = []
    monkeypatch.setattr(jobs, "PYREFLECT", _fake_pyreflect(calls=calls))

    import rq

    monkeypatch.setattr(rq, "get_current_job", lambda: None)

    job_params = {
        "layers": [
            {
                "name": "substrate",
                "sld": 2.0,
                "isld": 0.0,
                "thickness": 0.0,
                "roughness": 1.0,
            },
            {
                "name": "siox",
                "sld": 3.0,
                "isld": 0.0,
                "thickness": 10.0,
                "roughness": 2.0,
            },
            {
                "name": "layer_1",
                "sld": 4.0,
                "isld": 0.0,
                "thickness": 50.0,
                "roughness": 3.0,
            },
            {
                "name": "air",
                "sld": 0.0,
                "isld": 0.0,
                "thickness": 0.0,
                "roughness": 0.0,
            },
        ],
        "generator": {
            "numCurves": 1,
            "numFilmLayers": 1,
        },
        "training": {
            "batchSize": 1,
            "epochs": 1,
            "layers": 1,
            "dropout": 0.0,
            "latentDim": 2,
            "aeEpochs": 1,
            "mlpEpochs": 1,
        },
        "gpu": "T4",
    }

    with pytest.raises(_Sentinel):
        jobs.run_training_job(job_params)

    assert calls
    assert calls[0]["layer_desc"] is None
    assert calls[0]["layer_bound"] is None
