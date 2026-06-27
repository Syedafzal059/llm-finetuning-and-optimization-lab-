"""Usage alert notifications (email when configured, always logged)."""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Any

logger = logging.getLogger(__name__)

_sent_warnings: set[str] = set()


def _format_token_count(tokens: int) -> str:
    """Format token counts for human-readable display."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.0f}K"
    return str(tokens)


def _send_email(to_address: str, subject: str, body: str) -> None:
    """Send an email via SMTP when server credentials are configured."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_user).strip()

    if not smtp_host or not smtp_user or not smtp_password:
        logger.warning(
            "Alert email skipped (SMTP not configured): subject=%s to=%s",
            subject,
            to_address,
        )
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_from
    message["To"] = to_address
    message.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(message)

    logger.info("Alert email sent: subject=%s to=%s", subject, to_address)


def maybe_send_usage_warning(
    project_name: str,
    alert_type: str,
    current: int,
    limit: int,
    config: dict[str, Any],
) -> None:
    """Send a one-time warning email when usage crosses the configured threshold."""
    alerts_cfg = config.get("alerts", {})
    warning_pct = alerts_cfg.get(f"{alert_type}_warning_pct", 80)
    email = alerts_cfg.get("email", "")

    if limit <= 0:
        return

    usage_pct = (current / limit) * 100
    if usage_pct < warning_pct:
        return

    warning_key = f"{project_name}:{alert_type}:{warning_pct}"
    if warning_key in _sent_warnings:
        return

    _sent_warnings.add(warning_key)
    label = "token" if alert_type == "token" else "monthly request"
    body = (
        f"Project '{project_name}' has reached {usage_pct:.0f}% of its {label} limit.\n"
        f"Current usage: {current:,} / {limit:,}\n"
    )
    subject = f"[{project_name}] {label.title()} usage at {usage_pct:.0f}%"

    logger.warning(body.replace("\n", " | "))
    if email:
        _send_email(email, subject, body)


def build_limit_status(
    *,
    rate_count: int,
    rate_limit: int,
    month_count: int,
    monthly_limit: int,
    month_tokens: int,
    token_limit: int,
) -> dict[str, str]:
    """Build limit_status dict for the /metrics response."""
    return {
        "rate_limit": f"{rate_count}/{rate_limit} per min",
        "monthly_limit": f"{month_count}/{monthly_limit}",
        "token_limit": f"{_format_token_count(month_tokens)}/{_format_token_count(token_limit)}",
    }
