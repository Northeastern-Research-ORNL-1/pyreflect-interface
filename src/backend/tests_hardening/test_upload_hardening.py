from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _multipart(files: list[tuple[str, bytes]], roles: list[str] | None = None):
    # starlette TestClient uses httpx; pass bytes for file content.
    upload_files = [("files", (name, content, "application/octet-stream")) for name, content in files]
    data: dict[str, str | list[str]] = {}
    if roles is not None:
        data["roles"] = roles[0] if len(roles) == 1 else roles
    return upload_files, data


def test_upload_requires_user_id_in_production(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.routers import upload as upload_router

    monkeypatch.setattr(upload_router, "IS_PRODUCTION", True)

    upload_files, data = _multipart([("settings.yml", b"root: .\n")], roles=["auto"])
    resp = client.post("/api/upload", files=upload_files, data=data)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "X-User-ID required"


def test_upload_rejects_missing_role_for_non_settings_file(client: TestClient) -> None:
    # In dev mode, but should still require explicit role.
    arr = np.zeros((1, 2, 3), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    upload_files, data = _multipart([("x.npy", buf.getvalue())], roles=["auto"])
    resp = client.post("/api/upload", files=upload_files, data=data, headers={"X-User-ID": "u"})
    assert resp.status_code == 400
    assert "missing upload role" in resp.json()["detail"]


def _norm_stats_payload() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "nr": {
            "x": {"min": 0.0081, "max": 0.1975},
            "y": {"min": -8.0, "max": 0.0},
        },
        "sld": {
            "x": {"min": 0.0, "max": 900.0},
            "y": {"min": -2.0, "max": 8.0},
        },
    }


def test_upload_normalization_stats_accepts_trusted_npy(
    client: TestClient, tmp_backend_layout: dict[str, Path]
) -> None:
    payload = _norm_stats_payload()
    buf = io.BytesIO()
    np.save(buf, payload)

    upload_files, data = _multipart([("normalization_stat.npy", buf.getvalue())], roles=["normalization_stats"])
    resp = client.post("/api/upload", files=upload_files, data=data, headers={"X-User-ID": "u"})
    assert resp.status_code == 200

    data_dir = tmp_backend_layout["data_dir"]
    npy_path = data_dir / "normalization_stat.npy"
    assert npy_path.exists()
    loaded = np.load(npy_path, allow_pickle=True).item()
    assert set(loaded.keys()) == {"nr", "sld"}
    assert loaded["nr"]["x"]["min"] == payload["nr"]["x"]["min"]


def test_upload_normalization_stats_accepts_json_and_converts_to_npy(
    client: TestClient, tmp_backend_layout: dict[str, Path]
) -> None:
    # json payload -> backend writes .npy and unlinks json.
    raw = _norm_stats_payload()
    upload_files, data = _multipart(
        [("normalization_stat.json", json.dumps(raw).encode("utf-8"))],
        roles=["normalization_stats"],
    )
    resp = client.post("/api/upload", files=upload_files, data=data, headers={"X-User-ID": "u"})
    assert resp.status_code == 200

    data_dir = tmp_backend_layout["data_dir"]
    npy_path = data_dir / "normalization_stat.npy"
    assert npy_path.exists()
    loaded = np.load(npy_path, allow_pickle=True).item()
    assert set(loaded.keys()) == {"nr", "sld"}


def test_upload_normalization_stats_accepts_npz_and_converts_to_npy(
    client: TestClient, tmp_backend_layout: dict[str, Path]
) -> None:
    buf = io.BytesIO()
    np.savez(buf, x=np.array([0.0081, 0.1975]), y=np.array([-8.0, 0.0]))
    upload_files, data = _multipart([("normalization_stat.npz", buf.getvalue())], roles=["normalization_stats"])
    resp = client.post("/api/upload", files=upload_files, data=data, headers={"X-User-ID": "u"})
    assert resp.status_code == 200

    data_dir = tmp_backend_layout["data_dir"]
    npy_path = data_dir / "normalization_stat.npy"
    assert npy_path.exists()
    loaded = np.load(npy_path, allow_pickle=True).item()
    assert set(loaded.keys()) == {"nr", "sld"}


def test_upload_npy_disallows_pickles_via_object_array(client: TestClient, tmp_backend_layout: dict[str, Path]) -> None:
    # Make an object array .npy; np.load(..., allow_pickle=False) should fail.
    obj = np.array([{"a": 1}], dtype=object)
    buf = io.BytesIO()
    np.save(buf, obj)
    upload_files, data = _multipart([("nr_train.npy", buf.getvalue())], roles=["nr_train"])
    resp = client.post("/api/upload", files=upload_files, data=data, headers={"X-User-ID": "u"})
    assert resp.status_code == 400

    curves_dir = tmp_backend_layout["curves_dir"]
    assert list(curves_dir.glob("*.npy")) == []


def test_upload_npy_allows_numeric_array(client: TestClient, tmp_backend_layout: dict[str, Path]) -> None:
    q = np.logspace(np.log10(0.0081), np.log10(0.1975), num=308)
    r = np.linspace(1.0, 1e-6, num=308)
    arr = np.stack([q, r], axis=0)[np.newaxis, :, :]
    buf = io.BytesIO()
    np.save(buf, arr)
    upload_files, data = _multipart([("nr_train.npy", buf.getvalue())], roles=["nr_train"])
    resp = client.post("/api/upload", files=upload_files, data=data, headers={"X-User-ID": "u"})
    assert resp.status_code == 200
    curves_dir = tmp_backend_layout["curves_dir"]
    assert (curves_dir / "nr_train.npy").exists()


def test_upload_experimental_nr_converts_l3_to_canonical(
    client: TestClient, tmp_backend_layout: dict[str, Path]
) -> None:
    q = np.linspace(0.0081, 0.1975, num=431)
    r = np.linspace(1.1, 1e-6, num=431)
    dr = np.full_like(q, 0.01)
    arr = np.stack([q, r, dr], axis=1)

    buf = io.BytesIO()
    np.save(buf, arr)
    upload_files, data = _multipart([("NR_EXP.npy", buf.getvalue())], roles=["experimental_nr"])
    resp = client.post("/api/upload", files=upload_files, data=data, headers={"X-User-ID": "u"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["metadata"][0]["canonicalized"] is True
    assert body["metadata"][0]["canonical_shape"] == [1, 2, 308]
    assert "report_path_local" in body["metadata"][0]

    expt_path = tmp_backend_layout["expt_dir"] / "NR_EXP.npy"
    canonical = np.load(expt_path, allow_pickle=False)
    assert canonical.shape == (1, 2, 308)


def test_upload_experimental_nr_rejects_out_of_range_q(
    client: TestClient,
) -> None:
    q = np.linspace(0.0081, 0.2771, num=431)
    r = np.linspace(1.0, 1e-6, num=431)
    dr = np.full_like(q, 0.01)
    arr = np.stack([q, r, dr], axis=1)

    buf = io.BytesIO()
    np.save(buf, arr)
    upload_files, data = _multipart([("NR_EXP.npy", buf.getvalue())], roles=["experimental_nr"])
    resp = client.post("/api/upload", files=upload_files, data=data, headers={"X-User-ID": "u"})
    assert resp.status_code == 400
    assert "q-range out of bounds" in resp.json()["detail"]
