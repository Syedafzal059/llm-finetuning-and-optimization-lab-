"""Tests for the internal health monitoring system."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from scripts.alert_engine import format_alert_body, format_summary_body, should_alert
from scripts.monitor import (
    HealthCheck,
    MonitorConfig,
    check_backup_freshness,
    check_disk_space,
    check_queue_depth,
    worst_status,
)


@pytest.fixture
def monitor_config(tmp_path) -> MonitorConfig:
    """Return a monitor config pointed at a temporary workspace."""
    return MonitorConfig(
        workspace=tmp_path,
        projects=({"name": "clinical-notes", "port": 8001},),
        redis_host="localhost",
        redis_port=6379,
        api_latency_warning_ms=5000,
        gpu_memory_warning_pct=85,
        gpu_memory_critical_pct=95,
        queue_warning_depth=10,
        queue_critical_depth=20,
        disk_warning_pct=80,
        disk_critical_pct=90,
        backup_warning_hours=26,
        backup_critical_hours=48,
        pm2_restart_warning=5,
        subprocess_timeout_seconds=5,
        http_timeout_seconds=5,
    )


def test_worst_status_prefers_critical() -> None:
    checks = [
        HealthCheck("a", "ok", "fine"),
        HealthCheck("b", "warning", "slow"),
        HealthCheck("c", "critical", "down"),
    ]
    assert worst_status(checks) == "critical"


def test_format_alert_body_includes_critical_details() -> None:
    critical = HealthCheck(
        "clinical-notes_api",
        "critical",
        "API unreachable",
        value=1.0,
        threshold=0.0,
    )
    body = format_alert_body([critical], [])
    assert "clinical-notes_api" in body
    assert "API unreachable" in body


def test_format_summary_body_counts_statuses() -> None:
    checks = [
        HealthCheck("a", "ok", "fine"),
        HealthCheck("b", "warning", "slow"),
    ]
    body = format_summary_body(checks)
    assert "Warnings: 1" in body
    assert "OK:       1" in body


def test_check_disk_space_ok(monitor_config: MonitorConfig) -> None:
    check = check_disk_space(monitor_config)
    assert check.status in {"ok", "warning", "critical"}
    assert check.value is not None


def test_check_queue_depth_critical(monitor_config: MonitorConfig) -> None:
    mock_redis = MagicMock()
    mock_redis.llen.return_value = 25

    with patch("scripts.monitor._redis_client", return_value=mock_redis):
        check = check_queue_depth("clinical-notes", monitor_config)

    assert check.status == "critical"
    assert check.value == 25.0


def test_check_backup_freshness_critical(monitor_config: MonitorConfig) -> None:
    log_dir = monitor_config.workspace / "logs"
    log_dir.mkdir(parents=True)
    old_time = datetime.now() - timedelta(hours=72)
    log_dir.joinpath("backup.log").write_text(
        f"{old_time.strftime('%Y-%m-%d %H:%M:%S')} | === Backup complete: 2026-01-01 ===\n",
        encoding="utf-8",
    )

    check = check_backup_freshness(monitor_config)
    assert check.status == "critical"


def test_should_alert_respects_cooldown(monitor_config: MonitorConfig) -> None:
    mock_redis = MagicMock()
    mock_redis.exists.return_value = True

    with patch("scripts.alert_engine._redis_client", return_value=mock_redis):
        assert should_alert("gpu_memory", "critical", monitor_config) is False


def test_should_alert_bypasses_cooldown(monitor_config: MonitorConfig) -> None:
    assert should_alert(
        "gpu_memory",
        "critical",
        monitor_config,
        bypass_cooldown=True,
    ) is True
