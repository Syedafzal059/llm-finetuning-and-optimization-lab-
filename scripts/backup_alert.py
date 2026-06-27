"""Send email alerts after backup runs (success, failure, or corruption)."""

from __future__ import annotations

import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.text import MIMEText


def send_backup_summary(
    status: str,
    date: str,
    archive_size: str = "unknown",
    file_count: str = "unknown",
) -> None:
    """Email operators with backup run results."""
    from_email = os.getenv("ALERT_FROM_EMAIL", "").strip()
    to_email = os.getenv("ALERT_TO_EMAIL", "").strip()
    password = os.getenv("ALERT_EMAIL_PASSWORD", "").strip()
    smtp_host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.getenv("ALERT_SMTP_PORT", "465"))

    if not from_email or not to_email or not password:
        print(
            f"Skipping backup alert ({status}): "
            "ALERT_FROM_EMAIL, ALERT_TO_EMAIL, or ALERT_EMAIL_PASSWORD not set."
        )
        return

    if status == "SUCCESS":
        subject = f"Backup OK — {date}"
        body = f"""Daily backup completed successfully.

Date         : {date}
Archive size : {archive_size}
GDrive files : {file_count}

Destinations:
- Local archive (/backups/daily/)
- Google Drive synced
- Backblaze B2 (adapters only)

No action needed.
"""
    elif status == "CORRUPT":
        subject = f"BACKUP CORRUPT — {date}"
        body = f"""URGENT: Backup archive is corrupt.

File: {archive_size}
Time: {datetime.now(timezone.utc).isoformat()}

Action needed:
1. SSH into server immediately
2. Run: bash scripts/backup.sh manually
3. Check disk health: df -h
4. Check: dmesg | grep -i error
"""
    elif status == "FAILED":
        subject = f"BACKUP FAILED — {date}"
        body = f"""URGENT: Backup did not complete.

Time  : {datetime.now(timezone.utc).isoformat()}
Error : {archive_size}

Action needed:
1. SSH into server
2. Check: tail -n 100 logs/backup.log
3. Run backup manually: bash scripts/backup.sh
"""
    else:
        subject = f"BACKUP ALERT — {date}"
        body = f"Backup status: {status}\nDetails: {archive_size}\nFiles: {file_count}"

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(from_email, password)
        server.send_message(message)

    print(f"Alert sent: {subject}")


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/backup_alert.py "
            "<SUCCESS|CORRUPT|FAILED> <date> [detail] [file_count]",
            file=sys.stderr,
        )
        sys.exit(1)

    status = sys.argv[1]
    date = sys.argv[2]
    detail = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    file_count = sys.argv[4] if len(sys.argv) > 4 else "unknown"
    send_backup_summary(status, date, detail, file_count)


if __name__ == "__main__":
    main()
