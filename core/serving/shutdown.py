"""Graceful shutdown coordination for PM2-managed serving processes."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_SHUTDOWN_RESPONSE = JSONResponse(
    status_code=503,
    content={
        "error": "server_restarting",
        "message": "Server is restarting. Please retry in 10s.",
    },
)

_DEFAULT_DRAIN_TIMEOUT_SECONDS = 60.0


class ShutdownCoordinator:
    """Tracks shutdown state and in-flight requests for graceful PM2 restarts."""

    def __init__(self, drain_timeout: float = _DEFAULT_DRAIN_TIMEOUT_SECONDS) -> None:
        """Initialize shutdown coordination state."""
        self._drain_timeout = drain_timeout
        self._lock = threading.Lock()
        self._shutting_down = False
        self._in_flight = 0
        self._app: Optional[FastAPI] = None
        self._worker: Any = None
        self._executor: Any = None
        self._tracker: Any = None
        self._serving_logger: Optional[logging.Logger] = None

    @property
    def is_shutting_down(self) -> bool:
        """Return whether the server is draining for shutdown."""
        return self._shutting_down

    @property
    def drain_timeout(self) -> float:
        """Return the maximum seconds to wait while draining."""
        return self._drain_timeout

    @property
    def in_flight_count(self) -> int:
        """Return how many HTTP requests are currently being processed."""
        with self._lock:
            return self._in_flight

    def register(
        self,
        app: FastAPI,
        *,
        worker: Any = None,
        executor: Any = None,
        tracker: Any = None,
        serving_logger: Optional[logging.Logger] = None,
        install_signal_handlers: bool = False,
    ) -> None:
        """Attach shutdown middleware to a FastAPI app."""
        self._app = app
        self._worker = worker
        self._executor = executor
        self._tracker = tracker
        self._serving_logger = serving_logger

        app.state.shutdown_coordinator = self
        app.state.shutting_down = False

        @app.middleware("http")
        async def check_shutdown(request, call_next):
            if self._shutting_down:
                return _SHUTDOWN_RESPONSE
            self.track_request_started()
            try:
                return await call_next(request)
            finally:
                self.track_request_finished()

        if install_signal_handlers:
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
            signal.signal(signal.SIGINT, self._handle_shutdown_signal)

    def begin_shutdown(self) -> None:
        """Mark the server as shutting down and reject new requests."""
        self._shutting_down = True
        if self._app is not None:
            self._app.state.shutting_down = True

    def track_request_started(self) -> None:
        """Increment the in-flight request counter."""
        with self._lock:
            self._in_flight += 1

    def track_request_finished(self) -> None:
        """Decrement the in-flight request counter."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def wait_for_drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until all in-flight requests finish. Returns True if drained."""
        deadline = time.monotonic() + (timeout if timeout is not None else self._drain_timeout)
        while time.monotonic() < deadline:
            if self.in_flight_count == 0:
                return True
            time.sleep(0.1)
        return self.in_flight_count == 0

    def shutdown(self) -> None:
        """Drain in-flight work during uvicorn lifespan shutdown."""
        log = self._serving_logger or logger
        log.info("Shutdown signal received")
        self.begin_shutdown()
        log.info("Draining in-flight requests and queue...")

        if self._worker is not None:
            self._worker.stop(drain_timeout=self._drain_timeout)

        drained = self.wait_for_drain(timeout=self._drain_timeout)
        if not drained:
            log.warning(
                "Shutdown timed out with %s in-flight request(s)",
                self.in_flight_count,
            )

        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)

        if self._tracker is not None and hasattr(self._tracker, "close"):
            self._tracker.close()

        log.info("Shutdown complete")

    def drain_and_exit(self) -> None:
        """Drain queue and in-flight work, then exit cleanly."""
        self.shutdown()
        sys.exit(0)

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Handle SIGTERM/SIGINT from PM2 or manual stop."""
        del frame
        signal_name = signal.Signals(signum).name
        log = self._serving_logger or logger
        log.info("Received %s", signal_name)
        self.drain_and_exit()
