"""Send email alerts when a PM2-managed client crashes and auto-restarts."""

from __future__ import annotations

import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path


def _read_last_error_lines(error_log_path: str, line_count: int = 20) -> str:
    log_path = Path(error_log_path)
    if not log_path.is_file():
        return "Could not read error log (file not found)."

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "Could not read error log."

    if not lines:
        return "(error log is empty)"
    return "\n".join(lines[-line_count:])


def send_crash_alert(
    process_name: str,
    restart_count: int,
    error_log_path: str,
) -> None:
    """Email operators when a client process crashed and PM2 restarted it."""
    from_email = os.getenv("ALERT_FROM_EMAIL", "").strip()
    to_email = os.getenv("ALERT_TO_EMAIL", "").strip()
    password = os.getenv("ALERT_EMAIL_PASSWORD", "").strip()
    smtp_host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.getenv("ALERT_SMTP_PORT", "465"))

    if not from_email or not to_email or not password:
        print(
            f"Skipping crash alert for {process_name}: "
            "ALERT_FROM_EMAIL, ALERT_TO_EMAIL, or ALERT_EMAIL_PASSWORD not set."
        )
        return

    last_lines = _read_last_error_lines(error_log_path)
    subject = f"CRASH: {process_name} restarted ({restart_count} times)"
    body = f"""Your LLM API crashed and was auto-restarted.

Process  : {process_name}
Time     : {datetime.now(timezone.utc).isoformat()}
Restarts : {restart_count}

Last error log:
{last_lines}

Action needed if restart count > 5:
SSH into server and check manually.
"""

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(from_email, password)
        server.send_message(message)

    print(f"Alert sent for {process_name}")


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python scripts/crash_alert.py "
            "<process_name> <restart_count> <error_log_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    process_name = sys.argv[1]
    restart_count = int(sys.argv[2])
    error_log_path = sys.argv[3]
    send_crash_alert(process_name, restart_count, error_log_path)


if __name__ == "__main__":
    main()
