"""Tests for Modal worker GPU selection functions.

These tests require the `modal` package to be installed.
The tests are skipped if modal is not available.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import sys
from pathlib import Path

import pytest


# Ensure modal_worker can be imported
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# Check if modal is available
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODAL_AVAILABLE, reason="modal package not installed")


class TestGetGpuWorkerFn:
    """Tests for get_gpu_worker_fn function."""

    def test_returns_t4_worker_for_t4(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_t4

        assert get_gpu_worker_fn("T4") == run_rq_worker_t4

    def test_returns_l4_worker_for_l4(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_l4

        assert get_gpu_worker_fn("L4") == run_rq_worker_l4

    def test_returns_a10g_worker_for_a10g(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_a10g

        assert get_gpu_worker_fn("A10G") == run_rq_worker_a10g

    def test_returns_l40s_worker_for_l40s(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_l40s

        assert get_gpu_worker_fn("L40S") == run_rq_worker_l40s

    def test_returns_a100_worker_for_a100(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_a100

        assert get_gpu_worker_fn("A100") == run_rq_worker_a100

    def test_returns_a100_80gb_worker_for_a100_80gb(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_a100_80gb

        assert get_gpu_worker_fn("A100-80GB") == run_rq_worker_a100_80gb

    def test_returns_h100_worker_for_h100(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_h100

        assert get_gpu_worker_fn("H100") == run_rq_worker_h100

    def test_returns_h200_worker_for_h200(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_h200

        assert get_gpu_worker_fn("H200") == run_rq_worker_h200

    def test_returns_b200_worker_for_b200(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_b200

        assert get_gpu_worker_fn("B200") == run_rq_worker_b200

    def test_case_insensitive_matching(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_t4, run_rq_worker_h100

        assert get_gpu_worker_fn("t4") == run_rq_worker_t4
        assert get_gpu_worker_fn("h100") == run_rq_worker_h100

    def test_falls_back_to_t4_for_unknown_gpu(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_t4

        assert get_gpu_worker_fn("UNKNOWN_GPU") == run_rq_worker_t4

    def test_falls_back_to_t4_for_none(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_t4

        assert get_gpu_worker_fn(None) == run_rq_worker_t4

    def test_falls_back_to_t4_for_empty_string(self) -> None:
        from modal_worker import get_gpu_worker_fn, run_rq_worker_t4

        assert get_gpu_worker_fn("") == run_rq_worker_t4


class TestGetRequestedGpuFromQueue:
    """Tests for _get_requested_gpu_from_queue function."""

    def test_returns_gpu_from_first_job(self) -> None:
        from modal_worker import _get_requested_gpu_from_queue

        mock_job = MagicMock()
        mock_job.args = [{"gpu": "H100", "layers": [], "generator": {}, "training": {}}]

        mock_queue = MagicMock()
        mock_queue.job_ids = ["job-1", "job-2"]
        mock_queue.fetch_job.return_value = mock_job

        result = _get_requested_gpu_from_queue(mock_queue)
        assert result == "H100"
        mock_queue.fetch_job.assert_called_once_with("job-1")

    def test_returns_default_when_no_gpu_in_params(self) -> None:
        from modal_worker import _get_requested_gpu_from_queue, DEFAULT_GPU

        mock_job = MagicMock()
        mock_job.args = [{"layers": [], "generator": {}, "training": {}}]

        mock_queue = MagicMock()
        mock_queue.job_ids = ["job-1"]
        mock_queue.fetch_job.return_value = mock_job

        result = _get_requested_gpu_from_queue(mock_queue)
        assert result == DEFAULT_GPU

    def test_returns_default_when_queue_empty(self) -> None:
        from modal_worker import _get_requested_gpu_from_queue, DEFAULT_GPU

        mock_queue = MagicMock()
        mock_queue.job_ids = []

        result = _get_requested_gpu_from_queue(mock_queue)
        assert result == DEFAULT_GPU

    def test_returns_default_when_job_has_no_args(self) -> None:
        from modal_worker import _get_requested_gpu_from_queue, DEFAULT_GPU

        mock_job = MagicMock()
        mock_job.args = None

        mock_queue = MagicMock()
        mock_queue.job_ids = ["job-1"]
        mock_queue.fetch_job.return_value = mock_job

        result = _get_requested_gpu_from_queue(mock_queue)
        assert result == DEFAULT_GPU

    def test_returns_default_when_job_args_not_dict(self) -> None:
        from modal_worker import _get_requested_gpu_from_queue, DEFAULT_GPU

        mock_job = MagicMock()
        mock_job.args = ["not-a-dict"]

        mock_queue = MagicMock()
        mock_queue.job_ids = ["job-1"]
        mock_queue.fetch_job.return_value = mock_job

        result = _get_requested_gpu_from_queue(mock_queue)
        assert result == DEFAULT_GPU

    def test_returns_default_when_fetch_job_returns_none(self) -> None:
        from modal_worker import _get_requested_gpu_from_queue, DEFAULT_GPU

        mock_queue = MagicMock()
        mock_queue.job_ids = ["job-1"]
        mock_queue.fetch_job.return_value = None

        result = _get_requested_gpu_from_queue(mock_queue)
        assert result == DEFAULT_GPU

    def test_returns_default_on_exception(self) -> None:
        from modal_worker import _get_requested_gpu_from_queue, DEFAULT_GPU

        mock_queue = MagicMock()
        mock_queue.job_ids = ["job-1"]
        mock_queue.fetch_job.side_effect = Exception("Redis connection error")

        result = _get_requested_gpu_from_queue(mock_queue)
        assert result == DEFAULT_GPU


class TestGpuTiersConstant:
    """Tests for GPU_TIERS constant."""

    def test_all_gpu_tiers_defined(self) -> None:
        from modal_worker import GPU_TIERS

        expected_gpus = ["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200", "B200"]
        for gpu in expected_gpus:
            assert gpu in GPU_TIERS

    def test_default_gpu_is_t4(self) -> None:
        from modal_worker import DEFAULT_GPU

        assert DEFAULT_GPU == "T4"
