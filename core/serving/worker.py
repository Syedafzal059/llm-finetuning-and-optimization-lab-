"""Background inference worker that processes one GPU job at a time."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Protocol

import torch

from core.serving.queue import RequestQueue
from core.serving.tracker import UsageTracker

logger = logging.getLogger(__name__)

_DEQUEUE_TIMEOUT_SECONDS = 1
_IDLE_SLEEP_SECONDS = 0.1


class InferenceModel(Protocol):
    """Minimal model interface required by the inference worker."""

    @property
    def model_display_name(self) -> str:
        """Human-readable model identifier for result payloads."""

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int, int]:
        """Run synchronous generation and return output plus token counts."""


class InferenceWorker:
    """Daemon thread worker that serializes GPU inference through a Redis queue."""

    def __init__(self) -> None:
        """Initialize worker state."""
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    @property
    def is_running(self) -> bool:
        """Return whether the worker thread is alive."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        model: InferenceModel,
        project_name: str,
        tracker: UsageTracker,
        queue: RequestQueue,
    ) -> None:
        """Start the background worker loop in a daemon thread."""
        if self.is_running:
            logger.warning("InferenceWorker already running for project=%s", project_name)
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(model, project_name, tracker, queue),
            name=f"inference-worker-{project_name}",
            daemon=True,
        )
        self._thread.start()
        logger.info("InferenceWorker started for project=%s", project_name)

    def stop(self, drain_timeout: float = 30.0) -> None:
        """Signal shutdown and wait for the worker thread to finish."""
        if not self._thread:
            return

        self._stop_event.set()
        self._thread.join(timeout=drain_timeout)
        if self._thread.is_alive():
            logger.warning("InferenceWorker did not stop within %.1fs", drain_timeout)
        else:
            logger.info("InferenceWorker stopped gracefully")
        self._running = False
        self._thread = None

    def _run_loop(
        self,
        model: InferenceModel,
        project_name: str,
        tracker: UsageTracker,
        queue: RequestQueue,
    ) -> None:
        """Continuously dequeue and process jobs until stop is requested."""
        while not self._stop_event.is_set():
            job = queue.dequeue(project_name, timeout=_DEQUEUE_TIMEOUT_SECONDS)
            if job is None:
                time.sleep(_IDLE_SLEEP_SECONDS)
                continue

            self._process_job(model, tracker, queue, job)

        self._drain_remaining_jobs(model, project_name, tracker, queue)

    def _drain_remaining_jobs(
        self,
        model: InferenceModel,
        project_name: str,
        tracker: UsageTracker,
        queue: RequestQueue,
    ) -> None:
        """Process any jobs still waiting when shutdown is requested."""
        while True:
            job = queue.dequeue(project_name, timeout=0)
            if job is None:
                break
            self._process_job(model, tracker, queue, job)

    def _process_job(
        self,
        model: InferenceModel,
        tracker: UsageTracker,
        queue: RequestQueue,
        job: dict[str, Any],
    ) -> None:
        """Run inference for a single dequeued job and store the outcome."""
        job_id = job["job_id"]
        started_at = time.perf_counter()

        try:
            output_text, tokens_in, tokens_out = model.generate(
                job["prompt"],
                int(job["max_tokens"]),
                float(job["temperature"]),
                float(job["top_p"]),
            )
            latency_ms = (time.perf_counter() - started_at) * 1000

            queue.store_result(
                job_id,
                {
                    "output": output_text,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "latency_ms": round(latency_ms, 2),
                    "model": model.model_display_name,
                },
            )
            tracker.log_request(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                status="success",
            )
            logger.info(
                "Job %s completed in %.0fms (tokens_in=%s tokens_out=%s)",
                job_id,
                latency_ms,
                tokens_in,
                tokens_out,
            )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            queue.store_error(job_id, "GPU memory full, please retry")
            latency_ms = (time.perf_counter() - started_at) * 1000
            tracker.log_request(
                tokens_in=0,
                tokens_out=0,
                latency_ms=latency_ms,
                status="error",
                error_msg="GPU memory full, please retry",
            )
            logger.error("Job %s failed with CUDA OOM", job_id)

        except Exception as exc:
            queue.store_error(job_id, str(exc))
            latency_ms = (time.perf_counter() - started_at) * 1000
            tracker.log_request(
                tokens_in=0,
                tokens_out=0,
                latency_ms=latency_ms,
                status="error",
                error_msg=str(exc),
            )
            logger.exception("Job %s failed: %s", job_id, exc)
