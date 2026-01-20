from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


def test_check_rate_limit_allows_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import rate_limit

    rate_limit._BUCKETS.clear()
    monkeypatch.setenv("RATE_LIMIT_JOBS_SUBMIT_PER_MIN", "0")

    class _Client:
        host = "127.0.0.1"

    class _Req:
        client = _Client()

    # should not raise
    rate_limit.limit_jobs_submit(request=_Req(), user_id=None)  # type: ignore[arg-type]


def test_rate_limit_blocks_after_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import rate_limit

    # Keep this test self-contained.
    rate_limit._BUCKETS.clear()
    monkeypatch.setenv("RATE_LIMIT_JOBS_SUBMIT_PER_MIN", "2")

    # avoid touching external RQ by patching it to look available.
    from service.routers import jobs as jobs_router

    class _FakeQueue:
        def __len__(self):
            return 0

        def enqueue(self, *args, **kwargs):
            raise AssertionError("should not enqueue in this test")

    class _FakeRQ:
        available = True
        queue = _FakeQueue()
        redis = object()

    monkeypatch.setattr(jobs_router, "_get_rq_or_reconnect", lambda http_request: _FakeRQ())

    # Make limits validation a no-op so we don't need a full request payload.
    monkeypatch.setattr(jobs_router, "validate_limits", lambda *a, **k: None)

    # Make effective limits cheap.
    monkeypatch.setattr(jobs_router, "get_effective_limits", lambda **k: ({}, True, "local_dev"))

    payload = {
        "layers": [{"name": "air", "sld": 0, "isld": 0, "thickness": 0, "roughness": 0}],
        "generator": {"numCurves": 1, "numFilmLayers": 1},
        "training": {
            "batchSize": 1,
            "epochs": 1,
            "layers": 1,
            "dropout": 0,
            "latentDim": 2,
            "aeEpochs": 1,
            "mlpEpochs": 1,
        },
    }

    # two requests ok; third is rate limited (429).
    r1 = client.post("/api/jobs/submit", json=payload)
    assert r1.status_code in {200, 500, 503}
    r2 = client.post("/api/jobs/submit", json=payload)
    assert r2.status_code in {200, 500, 503}
    r3 = client.post("/api/jobs/submit", json=payload)
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


def test_get_env_int_default_on_bad_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services.rate_limit import _get_env_int

    monkeypatch.setenv("X", "not-an-int")
    assert _get_env_int("X", 7) == 7


def test_check_rate_limit_uses_user_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import rate_limit

    rate_limit._BUCKETS.clear()

    class _Client:
        host = "1.2.3.4"

    class _Req:
        client = _Client()

    # limit=1 per window; second should raise.
    rate_limit.check_rate_limit(
        request=_Req(), scope="s", user_id="u", limit=1, window_s=60.0
    )
    with pytest.raises(HTTPException) as exc:
        rate_limit.check_rate_limit(
            request=_Req(), scope="s", user_id="u", limit=1, window_s=60.0
        )
    assert exc.value.status_code == 429


def test_check_rate_limit_evicts_old_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import rate_limit

    rate_limit._BUCKETS.clear()

    class _Client:
        host = "1.2.3.4"

    class _Req:
        client = _Client()

    times = iter([1000.0, 1011.0])
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: next(times))

    # First call records timestamp at t=1000.
    rate_limit.check_rate_limit(request=_Req(), scope="s", user_id=None, limit=1, window_s=10.0)

    # Second call is after the window; should evict the old timestamp and allow.
    rate_limit.check_rate_limit(request=_Req(), scope="s", user_id=None, limit=1, window_s=10.0)
