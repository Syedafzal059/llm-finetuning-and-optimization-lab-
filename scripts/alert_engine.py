"""Alert delivery and cooldown logic for the internal monitor."""

from __future__ import annotations

import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Literal

import redis
from dotenv import load_dotenv

from core.utils.config_loader import PROJECT_ROOT
from scripts.monitor import HealthCheck, MonitorConfig, worst_status

AlertType = Literal["immediate", "summary", "test"]


def _load_alert_settings() -> dict[str, str | int]:
    """Load SMTP alert settings from the repository .env file."""
    root_env = PROJECT_ROOT / ".env"
    if root_env.is_file():
        load_dotenv(root_env)

    return {
        "from_email": os.getenv("ALERT_FROM_EMAIL", "").strip(),
        "to_email": os.getenv("ALERT_TO_EMAIL", "").strip(),
        "password": os.getenv("ALERT_EMAIL_PASSWORD", "").strip(),
        "smtp_host": os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com").strip(),
        "smtp_port": int(os.getenv("ALERT_SMTP_PORT", "465")),
        "cooldown_minutes": int(os.getenv("MONITOR_ALERT_COOLDOWN_MINUTES", "30")),
        "summary_hour": int(os.getenv("MONITOR_DAILY_SUMMARY_HOUR", "8")),
    }


def _redis_client(config: MonitorConfig) -> redis.Redis:
    """Return a Redis client for alert cooldown keys."""
    return redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        decode_responses=True,
        socket_connect_timeout=config.http_timeout_seconds,
        socket_timeout=config.http_timeout_seconds,
    )


def should_alert(
    check_name: str,
    status: str,
    config: MonitorConfig,
    *,
    bypass_cooldown: bool = False,
) -> bool:
    """Return True when an alert for this check should be sent now."""
    if bypass_cooldown:
        return True

    settings = _load_alert_settings()
    cooldown_minutes = int(settings["cooldown_minutes"])
    key = f"alert_sent:{check_name}:{status}"

    try:
        client = _redis_client(config)
        if client.exists(key):
            return False
        client.setex(key, cooldown_minutes * 60, "1")
        return True
    except redis.RedisError:
        # If Redis is unavailable, prefer alerting over silence.
        return True


def send_email(subject: str, body: str) -> bool:
    """Send an alert email using configured SMTP credentials."""
    settings = _load_alert_settings()
    from_email = str(settings["from_email"])
    to_email = str(settings["to_email"])
    password = str(settings["password"])
    smtp_host = str(settings["smtp_host"])
    smtp_port = int(settings["smtp_port"])

    if not from_email or not to_email or not password:
        print(
            "Skipping alert email: ALERT_FROM_EMAIL, ALERT_TO_EMAIL, "
            "or ALERT_EMAIL_PASSWORD not set."
        )
        return False

    smtp_timeout = float(os.getenv("MONITOR_SMTP_TIMEOUT_SECONDS", "15"))

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email

    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=smtp_timeout) as server:
        server.login(from_email, password)
        server.send_message(message)

    print(f"Alert sent: {subject}")
    return True


def format_alert_body(
    criticals: list[HealthCheck],
    warnings: list[HealthCheck],
) -> str:
    """Format a critical alert email body."""
    lines = [
        "CRITICAL ISSUES DETECTED",
        "=" * 40,
        "",
    ]

    for check in criticals:
        lines.append(f"X {check.name}")
        lines.append(f"   {check.message}")
        if check.value is not None and check.threshold is not None:
            lines.append(
                f"   Value: {check.value} (threshold: {check.threshold})"
            )
        lines.append("")

    if warnings:
        lines.append("WARNINGS:")
        for warning in warnings:
            lines.append(f"! {warning.name}: {warning.message}")

    lines.extend(
        [
            "",
            "Quick actions:",
            "Check status: pm2 status",
            "View logs:    pm2 logs",
            "Health JSON:  curl -H \"X-API-Key: $ADMIN_KEY\" http://localhost:8001/admin/health",
            "",
            f"Time: {datetime.now(timezone.utc).isoformat()}",
        ]
    )
    return "\n".join(lines)


def format_summary_body(checks: list[HealthCheck]) -> str:
    """Format the daily health summary email body."""
    ok_checks = [check for check in checks if check.status == "ok"]
    warn_checks = [check for check in checks if check.status == "warning"]
    crit_checks = [check for check in checks if check.status == "critical"]

    lines = [
        "DAILY HEALTH SUMMARY",
        "=" * 40,
        f"OK:       {len(ok_checks)}",
        f"Warnings: {len(warn_checks)}",
        f"Critical: {len(crit_checks)}",
        "",
        "DETAILS:",
        "",
    ]

    status_prefix = {
        "ok": "[OK]",
        "warning": "[WARN]",
        "critical": "[CRIT]",
    }
    for check in checks:
        prefix = status_prefix[check.status]
        lines.append(f"{prefix} {check.name}: {check.message}")

    lines.append("")
    lines.append(f"Overall: {worst_status(checks)}")
    lines.append(f"Time: {datetime.now(timezone.utc).isoformat()}")
    return "\n".join(lines)


def format_test_body(checks: list[HealthCheck]) -> str:
    """Format a test alert email body."""
    return "\n".join(
        [
            "MONITOR TEST ALERT",
            "=" * 40,
            "",
            "This is a test alert from scripts/run_monitor.py --test.",
            "If you received this email, SMTP alerting is configured correctly.",
            "",
            f"Checks sampled: {len(checks)}",
            f"Overall status: {worst_status(checks)}",
            "",
            f"Time: {datetime.now(timezone.utc).isoformat()}",
        ]
    )


def send_alert(
    checks: list[HealthCheck],
    alert_type: AlertType,
    config: MonitorConfig | None = None,
    *,
    bypass_cooldown: bool = False,
) -> bool:
    """Send immediate, summary, or test alerts based on check results."""
    cfg = config or MonitorConfig.from_env()
    criticals = [check for check in checks if check.status == "critical"]
    warnings = [check for check in checks if check.status == "warning"]

    if alert_type == "immediate":
        new_criticals = [
            check
            for check in criticals
            if should_alert(
                check.name,
                "critical",
                cfg,
                bypass_cooldown=bypass_cooldown,
            )
        ]
        if not new_criticals:
            return False

        subject = f"CRITICAL: {len(new_criticals)} issues detected"
        body = format_alert_body(new_criticals, warnings)
        return send_email(subject, body)

    if alert_type == "summary":
        subject = (
            "Daily health summary — "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        )
        body = format_summary_body(checks)
        return send_email(subject, body)

    if alert_type == "test":
        subject = "MONITOR TEST: alerting pipeline OK"
        body = format_test_body(checks)
        return send_email(subject, body)

    raise ValueError(f"Unsupported alert type: {alert_type}")


def should_send_daily_summary(config: MonitorConfig | None = None) -> bool:
    """Return True once per day during the configured summary hour."""
    cfg = config or MonitorConfig.from_env()
    settings = _load_alert_settings()
    summary_hour = int(settings["summary_hour"])
    now = datetime.now()

    if now.hour != summary_hour or now.minute != 0:
        return False

    date_key = now.strftime("%Y-%m-%d")
    redis_key = f"daily_summary_sent:{date_key}"

    try:
        client = _redis_client(cfg)
        if client.exists(redis_key):
            return False
        client.setex(redis_key, 25 * 3600, "1")
        return True
    except redis.RedisError:
        return True
