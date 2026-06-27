"""Main orchestrator for the internal health monitoring system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.alert_engine import (  # noqa: E402
    send_alert,
    should_send_daily_summary,
)
from scripts.monitor import (  # noqa: E402
    MonitorConfig,
    log_checks,
    run_all_checks,
    worst_status,
)


def _print_checks(checks: list) -> None:
    """Print health checks to stdout for operator visibility."""
    print(f"Overall: {worst_status(checks)}", flush=True)
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.message}", flush=True)


def main() -> None:
    """Run all health checks, log results, and dispatch alerts."""
    parser = argparse.ArgumentParser(
        description="Run internal health checks for all client APIs.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a test alert email and print all check results.",
    )
    args = parser.parse_args()

    config = MonitorConfig.from_env()
    checks = run_all_checks(config)
    log_checks(checks, config)

    if args.test:
        _print_checks(checks)
        sent = send_alert(
            checks,
            "test",
            config,
            bypass_cooldown=True,
        )
        if not sent:
            print(
                "Test alert was not sent. Configure ALERT_FROM_EMAIL, "
                "ALERT_TO_EMAIL, and ALERT_EMAIL_PASSWORD in .env."
            )
        return

    send_alert(checks, "immediate", config)

    if should_send_daily_summary(config):
        send_alert(checks, "summary", config, bypass_cooldown=True)


if __name__ == "__main__":
    main()
