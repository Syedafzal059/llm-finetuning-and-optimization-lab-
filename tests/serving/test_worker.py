"""Integration tests for the background inference worker."""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock

import fakeredis
import pytest
import torch

from core.serving.queue import RequestQueue
from core.serving.worker import InferenceWorker


@pytest.fixture
def queue_config() -> dict:
    """Minimal queue configuration for worker tests."""
    return {
        "queue": {
            "max_queue_depth": 50,
            "job_ttl_seconds": 600,
            "result_ttl_seconds": 600,
            "worker_threads": 1,
        },
        "redis": {"host": "localhost", "port": 6379, "db": 0},
    }


@pytest.fixture
def request_queue(queue_config: dict) -> RequestQueue:
    """RequestQueue backed by fake Redis."""
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    return RequestQueue("clinical-notes", queue_config, redis_client=redis_client)


@pytest.fixture
def mock_model() -> MagicMock:
    """Serving model stub that returns deterministic generation output."""
    model = MagicMock()
    model.model_display_name = "TestModel + adapter"
    model.generate.return_value = ("Generated text", 12, 8)
    return model


@pytest.fixture
def mock_tracker() -> MagicMock:
    """Usage tracker stub."""
    return MagicMock()


def test_worker_processes_enqueued_job_and_stores_result(
    request_queue: RequestQueue,
    mock_model: MagicMock,
    mock_tracker: MagicMock,
) -> None:
    """One enqueued job is processed and its result becomes pollable."""
    job_id = str(uuid.uuid4())
    request_queue.enqueue(
        job_id,
        {
            "prompt": "Hello",
            "max_tokens": 32,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    worker = InferenceWorker()
    worker.start(
        model=mock_model,
        project_name="clinical-notes",
        tracker=mock_tracker,
        queue=request_queue,
    )

    try:
        deadline = time.time() + 5.0
        result = None
        while time.time() < deadline:
            result = request_queue.get_result(job_id)
            if result is not None:
                break
            time.sleep(0.05)

        assert result is not None
        assert result["status"] == "done"
        assert result["output"] == "Generated text"
        assert result["tokens_in"] == 12
        assert result["tokens_out"] == 8
        mock_tracker.log_request.assert_called_once()
    finally:
        worker.stop()


def test_worker_stores_error_on_oom(
    request_queue: RequestQueue,
    mock_model: MagicMock,
    mock_tracker: MagicMock,
) -> None:
    """CUDA OOM failures are surfaced as retriable error results."""
    mock_model.generate.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM")
    job_id = str(uuid.uuid4())
    request_queue.enqueue(job_id, {"prompt": "OOM test", "max_tokens": 16})

    worker = InferenceWorker()
    worker.start(
        model=mock_model,
        project_name="clinical-notes",
        tracker=mock_tracker,
        queue=request_queue,
    )

    try:
        deadline = time.time() + 5.0
        result = None
        while time.time() < deadline:
            result = request_queue.get_result(job_id)
            if result is not None:
                break
            time.sleep(0.05)

        assert result is not None
        assert result["status"] == "error"
        assert "GPU memory full" in result["error"]
    finally:
        worker.stop()
