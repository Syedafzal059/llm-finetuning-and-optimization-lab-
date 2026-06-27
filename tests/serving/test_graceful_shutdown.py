"""Tests for graceful shutdown behavior during PM2 restarts."""

from __future__ import annotations

import threading
import time

from core.serving.shutdown import ShutdownCoordinator


def test_rejects_requests_with_503_when_shutting_down() -> None:
    """New requests receive 503 while the server is draining for restart."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    coordinator = ShutdownCoordinator()
    coordinator.register(app)
    coordinator.begin_shutdown()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "server_restarting"
    assert "retry" in body["message"].lower()


def test_shutdown_coordinator_waits_for_in_flight_requests() -> None:
    """Shutdown waits until active requests finish before reporting complete."""
    coordinator = ShutdownCoordinator()
    coordinator.track_request_started()
    coordinator.track_request_started()

    import threading
    import time

    def finish_one_request() -> None:
        time.sleep(0.05)
        coordinator.track_request_finished()

    threading.Thread(target=finish_one_request, daemon=True).start()

    completed = coordinator.wait_for_drain(timeout=2.0)
    assert completed is False

    coordinator.track_request_finished()
    completed = coordinator.wait_for_drain(timeout=2.0)
    assert completed is True
