"""Tests for backup alert email helper."""

from __future__ import annotations

from unittest.mock import patch

from scripts.backup_alert import send_backup_summary


def test_skips_alert_when_email_env_not_configured(capsys) -> None:
    """No email is sent when alert credentials are missing."""
    with patch.dict("os.environ", {}, clear=True):
        send_backup_summary("SUCCESS", "2026-06-27", "120M", "42")

    captured = capsys.readouterr()
    assert "Skipping backup alert" in captured.out
