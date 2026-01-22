
import sys
from unittest.mock import MagicMock, patch
import pytest
import types
import numpy as np

# --- Mock missing modules BEFORE importing service code ---
mock_rq = MagicMock()
mock_rq_job = MagicMock()
mock_rq.job = mock_rq_job
sys.modules["rq"] = mock_rq
sys.modules["rq.job"] = mock_rq_job
sys.modules["rq.command"] = MagicMock()
sys.modules["rq.registry"] = MagicMock()

mock_hf_hub = MagicMock()
sys.modules["huggingface_hub"] = mock_hf_hub

# Now we can safely import service code
# We might need to reload if they were already imported? 
# Usually pytest isolation matches, but to be safe we can use patch on sys.modules? 
# But just setting it here at top level works for this test file execution.

@pytest.fixture
def mock_hf_integration():
    mock_hf = MagicMock()
    mock_hf.available = True
    mock_hf.api = MagicMock()
    mock_hf.repo_id = "test/repo"
    return mock_hf

def test_download_training_data_success(mock_hf_integration):
    from service.integrations.huggingface import download_training_data
    
    # Mock numpy load to return dummy data
    with patch("numpy.load") as mock_load:
        
        # Configure mocked hf_hub_download
        mock_hf_hub.hf_hub_download.side_effect = ["nr_path.npy", "sld_path.npy"]
        mock_load.side_effect = [
            np.zeros((10, 2, 100)), # nr_curves
            np.zeros((10, 2, 200))  # sld_curves
        ]
        
        # Call function
        result = download_training_data(mock_hf_integration, "model-123")
        
        # Verify
        assert result is not None
        nr, sld = result
        assert nr.shape == (10, 2, 100)
        assert sld.shape == (10, 2, 200)
        
        # Check download calls
        assert mock_hf_hub.hf_hub_download.call_count == 2
        mock_hf_hub.hf_hub_download.assert_any_call(repo_id="test/repo", filename="models/model-123/nr_train.npy", repo_type="dataset")
        mock_hf_hub.hf_hub_download.assert_any_call(repo_id="test/repo", filename="models/model-123/sld_train.npy", repo_type="dataset")

def test_download_training_data_hf_not_available():
    from service.integrations.huggingface import download_training_data
    mock_hf = MagicMock()
    mock_hf.available = False
    
    assert download_training_data(mock_hf, "model-123") is None

def test_download_training_data_download_error(mock_hf_integration):
    from service.integrations.huggingface import download_training_data
    
    mock_hf_hub.hf_hub_download.side_effect = Exception("Download failed")
    
    result = download_training_data(mock_hf_integration, "model-123")
    assert result is None

# --- Mocking for run_training_job tests ---

class _Sentinel(Exception):
    pass

def _fake_pyreflect(*, calls: list[str]) -> object:
    class FakeRDG:
        def __init__(self, **kwargs):
            pass

        def generate(self, n):
            calls.append("generate")
            return np.zeros((n, 2, 100)), np.zeros((n, 2, 200))

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False
        @staticmethod
        def device(name: str):
            return name
        
        def manual_seed(self, seed): pass

    return types.SimpleNamespace(
        available=True,
        ReflectivityDataGenerator=FakeRDG,
        DataProcessor=MagicMock(),
        CNN=MagicMock(),
        DEVICE="cpu",
        torch=FakeTorch(),
        compute_nr_available=False,
        compute_nr_from_sld=None,
    )

@pytest.fixture
def fake_job_params():
    return {
        "layers": [],
        "generator": {"numCurves": 10, "numFilmLayers": 2},
        "training": {"batchSize": 5, "epochs": 1},
        "gpu": "cpu",
    }

def test_run_training_job_reuses_data(monkeypatch, fake_job_params):
    from service import jobs
    # Import the module where download_training_data is defined so we can patch it
    import service.integrations.huggingface
    
    calls = []
    # Patch PYREFLECT where it is defined/imported from
    monkeypatch.setattr("service.services.pyreflect_runtime.PYREFLECT", _fake_pyreflect(calls=calls))
    monkeypatch.setattr("rq.get_current_job", lambda: None)
    
    # Mock download_training_data to return valid data matching numCurves=10
    mock_download = MagicMock(return_value=(
        np.zeros((10, 2, 100)), 
        np.zeros((10, 2, 200))
    ))
    # Patch it at the source, because run_training_job imports it inside the function
    monkeypatch.setattr("service.integrations.huggingface.download_training_data", mock_download)
    
    # Mock HF Integration class mock to return our instance
    mock_hf_instance = MagicMock()
    mock_hf_instance.available = True
    monkeypatch.setattr("service.integrations.huggingface.HuggingFaceIntegration", MagicMock(return_value=mock_hf_instance))
    
    # To stop execution early, we'll patch DataProcessor on the FAKE pyreflect we inserted
    fake_pr = _fake_pyreflect(calls=calls)
    fake_pr.DataProcessor.normalize_xy_curves.side_effect = _Sentinel("stop_at_preprocessing")
    monkeypatch.setattr("service.services.pyreflect_runtime.PYREFLECT", fake_pr)
    
    with pytest.raises(_Sentinel, match="stop_at_preprocessing"):
        jobs.run_training_job(
            fake_job_params, 
            reuse_model_id="old-model-id",
            hf_config={"repo_id": "test/repo"}
        )
        
    # Verify generate was NOT called
    assert "generate" not in calls
    mock_download.assert_called_once()
    assert mock_download.call_args[0][1] == "old-model-id"

def test_run_training_job_generates_when_reuse_fails(monkeypatch, fake_job_params):
    from service import jobs
    import service.integrations.huggingface
    
    calls = []
    fake_pr = _fake_pyreflect(calls=calls)
    fake_pr.DataProcessor.normalize_xy_curves.side_effect = _Sentinel("stop_at_preprocessing")
    monkeypatch.setattr("service.services.pyreflect_runtime.PYREFLECT", fake_pr)
    
    monkeypatch.setattr("rq.get_current_job", lambda: None)
    
    # Mock download to fail (return None)
    mock_download = MagicMock(return_value=None)
    monkeypatch.setattr("service.integrations.huggingface.download_training_data", mock_download)
    
    mock_hf_instance = MagicMock()
    mock_hf_instance.available = True
    monkeypatch.setattr("service.integrations.huggingface.HuggingFaceIntegration", MagicMock(return_value=mock_hf_instance))
    
    with pytest.raises(_Sentinel, match="stop_at_preprocessing"):
        jobs.run_training_job(
            fake_job_params, 
            reuse_model_id="old-model-id",
            hf_config={"repo_id": "test/repo"}
        )
        
    # Verify generate WAS called
    assert "generate" in calls
    mock_download.assert_called_once()

def test_run_training_job_generates_when_shape_mismatch(monkeypatch, fake_job_params):
    from service import jobs
    import service.integrations.huggingface
    
    calls = []
    fake_pr = _fake_pyreflect(calls=calls)
    fake_pr.DataProcessor.normalize_xy_curves.side_effect = _Sentinel("stop_at_preprocessing")
    monkeypatch.setattr("service.services.pyreflect_runtime.PYREFLECT", fake_pr)
    
    monkeypatch.setattr("rq.get_current_job", lambda: None)

    
    # Mock download to return WRONG shape (5 instead of 10)
    mock_download = MagicMock(return_value=(
        np.zeros((5, 2, 100)), 
        np.zeros((5, 2, 200))
    ))
    monkeypatch.setattr("service.integrations.huggingface.download_training_data", mock_download)
    
    mock_hf_instance = MagicMock()
    mock_hf_instance.available = True
    monkeypatch.setattr("service.integrations.huggingface.HuggingFaceIntegration", MagicMock(return_value=mock_hf_instance))

    with pytest.raises(_Sentinel, match="stop_at_preprocessing"):
        jobs.run_training_job(
            fake_job_params, 
            reuse_model_id="old-model-id", # numCurves=10 in fixture
            hf_config={"repo_id": "test/repo"}
        )
        
    # Verify generate WAS called because shape mismatch
    assert "generate" in calls


# --- Mocking for retry endpoint tests ---

def test_retry_job_extracts_model_id_from_result(monkeypatch):
    from service.routers import jobs as jobs_router
    
    # Mocks
    mock_queues = MagicMock() # rq object wrapper
    mock_rq_job_inst = MagicMock()
    
    # We mock _get_rq_or_reconnect to return our mock
    mock_rq_wrapper = MagicMock()
    mock_rq_wrapper.queue.enqueue.return_value = MagicMock(id="new-job-id")
    mock_rq_wrapper.available = True
    monkeypatch.setattr(jobs_router, "_get_rq_or_reconnect", lambda req: mock_rq_wrapper)
    
    # Mock Job.fetch
    mock_old_job = MagicMock()
    mock_old_job.get_status.return_value = "failed"
    mock_old_job.kwargs = {}
    mock_old_job.latest_result.return_value.return_value = {"model_id": "found-in-result"}
    mock_old_job.result = {"model_id": "found-in-result"}
    mock_old_job.meta = {"job_params": {}}
    
    # Need to patch Job in the router AND also ensure Job.fetch works
    # Check imports in routers/jobs.py:
    # `from rq.job import Job` inside retry_job function.
    # So we must patch `rq.job.Job` which we mocked at sys.modules level.
    
    mock_rq_job.Job.fetch.return_value = mock_old_job
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    loop.run_until_complete(jobs_router.retry_job("old-job", MagicMock()))
    
    # Check enqueue args
    args, kwargs = mock_rq_wrapper.queue.enqueue.call_args
    assert kwargs["reuse_model_id"] == "found-in-result"

def test_retry_job_extracts_model_id_from_meta(monkeypatch):
    from service.routers import jobs as jobs_router
    
    mock_rq_wrapper = MagicMock()
    mock_rq_wrapper.queue.enqueue.return_value = MagicMock(id="new-job-id")
    mock_rq_wrapper.available = True
    monkeypatch.setattr(jobs_router, "_get_rq_or_reconnect", lambda req: mock_rq_wrapper)
    
    mock_old_job = MagicMock()
    mock_old_job.get_status.return_value = "failed"
    mock_old_job.kwargs = {}
    mock_old_job.latest_result.return_value = None
    mock_old_job.result = None
    mock_old_job.meta = {"job_params": {}, "model_id": "found-in-meta"}
    
    mock_rq_job.Job.fetch.return_value = mock_old_job
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    loop.run_until_complete(jobs_router.retry_job("old-job", MagicMock()))
    
    # Check enqueue args
    args, kwargs = mock_rq_wrapper.queue.enqueue.call_args
    assert kwargs["reuse_model_id"] == "found-in-meta"
