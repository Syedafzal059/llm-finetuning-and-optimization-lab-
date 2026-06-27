"""Tests for PM2 crash alert email helper."""

from __future__ import annotations

from unittest.mock import patch

from scripts.crash_alert import send_crash_alert


def test_skips_alert_when_email_env_not_configured(capsys) -> None:
    """No email is sent when alert credentials are missing."""
    with patch.dict("os.environ", {}, clear=True):
        send_crash_alert("clinical-notes", 3, "missing.log")

    captured = capsys.readouterr()
    assert "Skipping crash alert" in captured.out
