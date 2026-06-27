"""Integration tests for the Redis-backed request queue."""

from __future__ import annotations

import json
import uuid

import fakeredis
import pytest

from core.serving.queue import RequestQueue


@pytest.fixture
def queue_config() -> dict:
    """Minimal queue configuration for tests."""
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
    """RequestQueue backed by an in-memory fake Redis."""
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    return RequestQueue("clinical-notes", queue_config, redis_client=redis_client)


def test_enqueue_returns_job_id_and_job_is_dequeueable(request_queue: RequestQueue) -> None:
    """Submitting a job makes it the next item the worker can dequeue."""
    job_id = str(uuid.uuid4())
    payload = {
        "prompt": "Summarize this note.",
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    returned_id = request_queue.enqueue(job_id, payload)

    assert returned_id == job_id
    job = request_queue.dequeue(timeout=0)
    assert job is not None
    assert job["job_id"] == job_id
    assert job["prompt"] == payload["prompt"]
    assert job["status"] == "processing"


def test_store_result_makes_result_retrievable(request_queue: RequestQueue) -> None:
    """A completed job result can be fetched by job_id."""
    result_payload = {
        "output": "Patient stable.",
        "tokens_in": 10,
        "tokens_out": 5,
        "latency_ms": 42.0,
        "status": "done",
    }

    request_queue.store_result("job-abc", result_payload)

    stored = request_queue.get_result("job-abc")
    assert stored == result_payload


def test_get_result_returns_none_while_job_is_pending(request_queue: RequestQueue) -> None:
    """Polling before completion reports not-ready without a 404."""
    job_id = str(uuid.uuid4())
    request_queue.enqueue(job_id, {"prompt": "wait", "max_tokens": 16})

    assert request_queue.get_result(job_id) is None
    assert request_queue.is_job_known(job_id) is True


def test_store_error_makes_error_result_retrievable(request_queue: RequestQueue) -> None:
    """Failed jobs expose an error status through the same result interface."""
    request_queue.store_error("job-err", "GPU memory full, please retry")

    result = request_queue.get_result("job-err")
    assert result is not None
    assert result["status"] == "error"
    assert result["error"] == "GPU memory full, please retry"


def test_get_queue_depth_reflects_waiting_jobs(request_queue: RequestQueue) -> None:
    """Queue depth increases with each enqueued job and drops after dequeue."""
    assert request_queue.get_queue_depth() == 0

    request_queue.enqueue("j1", {"prompt": "a", "max_tokens": 8})
    request_queue.enqueue("j2", {"prompt": "b", "max_tokens": 8})
    assert request_queue.get_queue_depth() == 2

    request_queue.dequeue(timeout=0)
    assert request_queue.get_queue_depth() == 1


def test_is_queue_full_when_at_max_depth(queue_config: dict) -> None:
    """New requests are rejected when depth reaches max_queue_depth."""
    queue_config["queue"]["max_queue_depth"] = 2
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    request_queue = RequestQueue("clinical-notes", queue_config, redis_client=redis_client)

    request_queue.enqueue("j1", {"prompt": "a", "max_tokens": 8})
    request_queue.enqueue("j2", {"prompt": "b", "max_tokens": 8})

    assert request_queue.is_queue_full() is True


def test_unknown_job_is_not_known(request_queue: RequestQueue) -> None:
    """Expired or invalid job ids are distinguishable from pending jobs."""
    assert request_queue.is_job_known("missing-job") is False
