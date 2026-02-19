from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NR_GRID_POINTS = 308
SLD_GRID_POINTS = 900
NR_Q_MIN = 0.0081
NR_Q_MAX = 0.1975
NR_Q_RANGE_EPS = 1e-6
MIN_POINTS_BEFORE_RESAMPLE = 10
MAX_TRUSTED_NORM_NPY_MB = 5.0

_NR_Q_GRID = np.logspace(np.log10(NR_Q_MIN), np.log10(NR_Q_MAX), num=NR_GRID_POINTS)

NR_CANONICAL_ROLES = {"nr_train", "experimental_nr"}
SLD_CANONICAL_ROLES = {"sld_train"}


@dataclass(frozen=True)
class CanonicalizationResult:
    payload: Any
    metadata: dict[str, Any]
    report: dict[str, Any]


def canonicalize_npy_payload(role: str, payload: Any) -> CanonicalizationResult:
    if role in NR_CANONICAL_ROLES:
        return _canonicalize_nr_payload(role, payload)
    if role in SLD_CANONICAL_ROLES:
        return _canonicalize_sld_payload(role, payload)
    if isinstance(payload, np.ndarray):
        shape = [int(x) for x in payload.shape]
        return CanonicalizationResult(
            payload=payload,
            metadata={"shape": shape, "original_shape": shape, "canonical_shape": shape, "canonicalized": False},
            report={
                "role": role,
                "original_shape": shape,
                "canonical_shape": shape,
                "canonicalized": False,
                "actions": [],
                "warnings": [],
            },
        )
    raise ValueError("Expected a numpy array")


def load_normalization_stats(path: Path) -> CanonicalizationResult:
    suffix = path.suffix.lower()
    warnings: list[str] = []

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            parsed = _parse_norm_stats_payload({k: np.asarray(data[k]) for k in data.files}, warnings)
    elif suffix == ".json":
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        parsed = _parse_norm_stats_payload(raw, warnings)
    elif suffix == ".npy":
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_TRUSTED_NORM_NPY_MB:
            raise ValueError(
                f"normalization_stats .npy too large ({size_mb:.2f} MB); "
                f"expected <= {MAX_TRUSTED_NORM_NPY_MB:.1f} MB"
            )
        # Trusted importer path for pyreflect bundle compatibility.
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
            raw = raw.item()
        parsed = _parse_norm_stats_payload(raw, warnings)
        warnings.append("Loaded normalization_stats from trusted .npy path")
    else:
        raise ValueError("normalization_stats must be uploaded as .npy, .npz, or .json")

    metadata = {
        "type": "normalization_stats",
        "canonicalized": bool(warnings),
        "warnings": warnings,
    }
    report = {
        "role": "normalization_stats",
        "original_format": suffix.lstrip("."),
        "canonical_format": "npy",
        "canonicalized": bool(warnings),
        "actions": ["normalized_stats_schema_validation"],
        "warnings": warnings,
        "stats_keys": sorted(parsed.keys()),
    }
    return CanonicalizationResult(payload=parsed, metadata=metadata, report=report)


def _canonicalize_nr_payload(role: str, payload: Any) -> CanonicalizationResult:
    curves, actions, warnings = _coerce_curve_array(payload)
    original_shape = [int(x) for x in curves.shape]
    canonical = np.empty((curves.shape[0], 2, NR_GRID_POINTS), dtype=np.float64)

    for idx in range(curves.shape[0]):
        q = np.asarray(curves[idx, 0, :], dtype=np.float64)
        r = np.asarray(curves[idx, 1, :], dtype=np.float64)
        q, r, sample_actions, sample_warnings = _prepare_xy_for_resample(q, r, axis_name="q")
        actions.extend(sample_actions)
        warnings.extend(sample_warnings)

        if q.shape[0] < MIN_POINTS_BEFORE_RESAMPLE:
            raise ValueError(
                f"{role} sample {idx} has only {q.shape[0]} points after cleanup; "
                f"need at least {MIN_POINTS_BEFORE_RESAMPLE}"
            )

        q_min = float(np.min(q))
        q_max = float(np.max(q))
        if q_min < NR_Q_MIN - NR_Q_RANGE_EPS or q_max > NR_Q_MAX + NR_Q_RANGE_EPS:
            raise ValueError(
                f"NR q-range out of bounds for sample {idx}: "
                f"expected [{NR_Q_MIN:.6f}, {NR_Q_MAX:.6f}], got [{q_min:.6f}, {q_max:.6f}]"
            )

        canonical[idx, 0, :] = _NR_Q_GRID
        canonical[idx, 1, :] = np.interp(_NR_Q_GRID, q, r)

    canonical_shape = [int(x) for x in canonical.shape]
    actions.append(f"resampled_to_{NR_GRID_POINTS}")
    canonicalized = original_shape != canonical_shape or len(actions) > 1 or bool(warnings)
    metadata = {
        "shape": canonical_shape,
        "original_shape": original_shape,
        "canonical_shape": canonical_shape,
        "canonicalized": canonicalized,
        "warnings": _stable_unique(warnings),
    }
    report = {
        "role": role,
        "original_shape": original_shape,
        "canonical_shape": canonical_shape,
        "canonicalized": canonicalized,
        "actions": _stable_unique(actions),
        "warnings": _stable_unique(warnings),
        "checks": {
            "min_points": MIN_POINTS_BEFORE_RESAMPLE,
            "q_range_expected": [NR_Q_MIN, NR_Q_MAX],
            "grid_points": NR_GRID_POINTS,
        },
    }
    return CanonicalizationResult(payload=canonical, metadata=metadata, report=report)


def _canonicalize_sld_payload(role: str, payload: Any) -> CanonicalizationResult:
    curves, actions, warnings = _coerce_curve_array(payload)
    original_shape = [int(x) for x in curves.shape]
    canonical = np.empty((curves.shape[0], 2, SLD_GRID_POINTS), dtype=np.float64)

    for idx in range(curves.shape[0]):
        z = np.asarray(curves[idx, 0, :], dtype=np.float64)
        y = np.asarray(curves[idx, 1, :], dtype=np.float64)
        z, y, sample_actions, sample_warnings = _prepare_xy_for_resample(z, y, axis_name="z")
        actions.extend(sample_actions)
        warnings.extend(sample_warnings)

        if z.shape[0] < MIN_POINTS_BEFORE_RESAMPLE:
            raise ValueError(
                f"{role} sample {idx} has only {z.shape[0]} points after cleanup; "
                f"need at least {MIN_POINTS_BEFORE_RESAMPLE}"
            )

        z_grid = np.linspace(float(np.min(z)), float(np.max(z)), num=SLD_GRID_POINTS)
        canonical[idx, 0, :] = z_grid
        canonical[idx, 1, :] = np.interp(z_grid, z, y)

    canonical_shape = [int(x) for x in canonical.shape]
    actions.append(f"resampled_to_{SLD_GRID_POINTS}")
    canonicalized = original_shape != canonical_shape or len(actions) > 1 or bool(warnings)
    metadata = {
        "shape": canonical_shape,
        "original_shape": original_shape,
        "canonical_shape": canonical_shape,
        "canonicalized": canonicalized,
        "warnings": _stable_unique(warnings),
    }
    report = {
        "role": role,
        "original_shape": original_shape,
        "canonical_shape": canonical_shape,
        "canonicalized": canonicalized,
        "actions": _stable_unique(actions),
        "warnings": _stable_unique(warnings),
        "checks": {"min_points": MIN_POINTS_BEFORE_RESAMPLE, "grid_points": SLD_GRID_POINTS},
    }
    return CanonicalizationResult(payload=canonical, metadata=metadata, report=report)


def _coerce_curve_array(payload: Any) -> tuple[np.ndarray, list[str], list[str]]:
    if not isinstance(payload, np.ndarray):
        raise ValueError("Expected a numpy array")

    if payload.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("Curve array must be numeric")

    arr = np.asarray(payload, dtype=np.float64)
    actions: list[str] = []
    warnings: list[str] = []

    if arr.ndim == 3:
        if arr.shape[1] in {2, 3}:
            curves = arr
        elif arr.shape[2] in {2, 3}:
            curves = np.moveaxis(arr, 2, 1)
            actions.append("transposed_last_axis_to_channel")
        else:
            raise ValueError("3D curve data must have 2 or 3 channels")
    elif arr.ndim == 2:
        if arr.shape[0] in {2, 3}:
            curve = arr
        elif arr.shape[1] in {2, 3}:
            curve = arr.T
            actions.append("transposed_2d_curve")
        else:
            raise ValueError("2D curve data must be (2|3, L) or (L, 2|3)")
        curves = curve[np.newaxis, :, :]
        actions.append("added_batch_dimension")
    else:
        raise ValueError("Curve data must be 2D or 3D")

    if curves.shape[1] == 3:
        curves = curves[:, :2, :]
        actions.append("dropped_third_channel")
        warnings.append(
            "Dropped third channel (assumed uncertainty/error). Verify column 3 is not a signal channel."
        )

    if not np.all(np.isfinite(curves)):
        raise ValueError("Curve array contains NaN/Inf values")

    return curves, actions, warnings


def _prepare_xy_for_resample(
    x: np.ndarray, y: np.ndarray, *, axis_name: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    actions: list[str] = []
    warnings: list[str] = []

    if x.shape != y.shape:
        raise ValueError(f"{axis_name} and y channels must have the same length")

    if x.ndim != 1:
        raise ValueError("Each channel must be 1D after shape normalization")

    order = np.argsort(x)
    if not np.array_equal(order, np.arange(x.shape[0])):
        x = x[order]
        y = y[order]
        actions.append(f"sorted_{axis_name}_axis")
        warnings.append(f"Sorted non-monotonic {axis_name} values")

    unique_x, unique_indices = np.unique(x, return_index=True)
    if unique_x.shape[0] != x.shape[0]:
        x = unique_x
        y = y[unique_indices]
        actions.append(f"deduplicated_{axis_name}_axis")
        warnings.append(f"Deduplicated repeated {axis_name} values")

    return x, y, actions, warnings


def _parse_norm_stats_payload(raw: Any, warnings: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        raw = raw.item()

    if not isinstance(raw, dict):
        raise ValueError("normalization_stats must be a dict-like payload")

    if "nr" in raw and "sld" in raw:
        nr = _parse_curve_stats(raw["nr"], curve_name="nr")
        sld = _parse_curve_stats(raw["sld"], curve_name="sld")
        return {"nr": nr, "sld": sld}

    if "x" in raw and "y" in raw:
        axis_x = _parse_axis_stats(raw["x"], axis_name="x")
        axis_y = _parse_axis_stats(raw["y"], axis_name="y")
        warnings.append("Legacy x/y normalization stats duplicated for nr and sld")
        return {
            "nr": {"x": dict(axis_x), "y": dict(axis_y)},
            "sld": {"x": dict(axis_x), "y": dict(axis_y)},
        }

    raise ValueError("normalization_stats must contain either 'nr'+'sld' or legacy 'x'+'y' keys")


def _parse_curve_stats(raw: Any, *, curve_name: str) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        raise ValueError(f"normalization_stats['{curve_name}'] must be a dict")
    if "x" not in raw or "y" not in raw:
        raise ValueError(f"normalization_stats['{curve_name}'] must contain 'x' and 'y'")
    return {
        "x": _parse_axis_stats(raw["x"], axis_name=f"{curve_name}.x"),
        "y": _parse_axis_stats(raw["y"], axis_name=f"{curve_name}.y"),
    }


def _parse_axis_stats(raw: Any, *, axis_name: str) -> dict[str, float]:
    if isinstance(raw, dict):
        if "min" not in raw or "max" not in raw:
            raise ValueError(f"normalization_stats axis '{axis_name}' must contain 'min' and 'max'")
        min_v = _as_float(raw["min"], field=f"{axis_name}.min")
        max_v = _as_float(raw["max"], field=f"{axis_name}.max")
    else:
        arr = np.asarray(raw)
        if arr.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"normalization_stats axis '{axis_name}' must be numeric")
        flat = arr.reshape(-1)
        if flat.shape[0] != 2:
            raise ValueError(
                f"normalization_stats axis '{axis_name}' must be [min, max] when array-like"
            )
        min_v = float(flat[0])
        max_v = float(flat[1])

    if not np.isfinite(min_v) or not np.isfinite(max_v):
        raise ValueError(f"normalization_stats axis '{axis_name}' must be finite")
    if min_v >= max_v:
        raise ValueError(
            f"normalization_stats axis '{axis_name}' must have min < max "
            f"(got {min_v} >= {max_v})"
        )
    return {"min": min_v, "max": max_v}


def _as_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"normalization_stats field '{field}' must be numeric") from exc
    return out


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
