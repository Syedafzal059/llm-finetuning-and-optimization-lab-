"""Health check functions for the internal monitoring system."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
import redis
from dotenv import load_dotenv

from core.serving.auth import load_project_env
from core.serving.tracker import UsageTracker
from core.utils.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

StatusLevel = Literal["ok", "warning", "critical"]

DEFAULT_PROJECTS: list[dict[str, Any]] = [
    {"name": "clinical-notes", "port": 8001},
    {"name": "medical-coding", "port": 8002},
    {"name": "patient-support", "port": 8003},
]


@dataclass(frozen=True)
class MonitorConfig:
    """Thresholds and connection settings loaded from environment variables."""

    workspace: Path
    projects: tuple[dict[str, Any], ...]
    redis_host: str
    redis_port: int
    api_latency_warning_ms: float
    gpu_memory_warning_pct: float
    gpu_memory_critical_pct: float
    queue_warning_depth: int
    queue_critical_depth: int
    disk_warning_pct: int
    disk_critical_pct: int
    backup_warning_hours: float
    backup_critical_hours: float
    pm2_restart_warning: int
    subprocess_timeout_seconds: float
    http_timeout_seconds: float

    @classmethod
    def from_env(cls) -> MonitorConfig:
        """Build monitor configuration from environment variables with defaults."""
        root_env = PROJECT_ROOT / ".env"
        if root_env.is_file():
            load_dotenv(root_env)

        workspace = Path(
            os.getenv("WORKSPACE", str(PROJECT_ROOT))
        ).resolve()

        projects_raw = os.getenv("MONITOR_PROJECTS", "").strip()
        if projects_raw:
            projects = tuple(json.loads(projects_raw))
        else:
            projects = tuple(DEFAULT_PROJECTS)

        return cls(
            workspace=workspace,
            projects=projects,
            redis_host=os.getenv("MONITOR_REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("MONITOR_REDIS_PORT", "6379")),
            api_latency_warning_ms=float(
                os.getenv("MONITOR_API_LATENCY_WARNING_MS", "5000")
            ),
            gpu_memory_warning_pct=float(
                os.getenv("MONITOR_GPU_WARNING_PCT", "85")
            ),
            gpu_memory_critical_pct=float(
                os.getenv("MONITOR_GPU_CRITICAL_PCT", "95")
            ),
            queue_warning_depth=int(os.getenv("MONITOR_QUEUE_WARNING_DEPTH", "10")),
            queue_critical_depth=int(os.getenv("MONITOR_QUEUE_CRITICAL_DEPTH", "20")),
            disk_warning_pct=int(os.getenv("MONITOR_DISK_WARNING_PCT", "80")),
            disk_critical_pct=int(os.getenv("MONITOR_DISK_CRITICAL_PCT", "90")),
            backup_warning_hours=float(
                os.getenv("MONITOR_BACKUP_WARNING_HOURS", "26")
            ),
            backup_critical_hours=float(
                os.getenv("MONITOR_BACKUP_CRITICAL_HOURS", "48")
            ),
            pm2_restart_warning=int(os.getenv("MONITOR_PM2_RESTART_WARNING", "5")),
            subprocess_timeout_seconds=float(
                os.getenv("MONITOR_SUBPROCESS_TIMEOUT_SECONDS", "15")
            ),
            http_timeout_seconds=float(
                os.getenv("MONITOR_HTTP_TIMEOUT_SECONDS", "10")
            ),
        )


@dataclass
class HealthCheck:
    """Result of a single health probe."""

    name: str
    status: StatusLevel
    message: str
    value: float | None = None
    threshold: float | None = None


def _redis_client(config: MonitorConfig) -> redis.Redis:
    """Return a Redis client using monitor configuration."""
    return redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        decode_responses=True,
        socket_connect_timeout=config.http_timeout_seconds,
        socket_timeout=config.http_timeout_seconds,
    )


def _load_admin_api_key(project_name: str) -> str:
    """Load the admin API key for a project from its .env file."""
    load_project_env(project_name)
    return os.getenv("ADMIN_API_KEY", "").strip()


def _run_subprocess(
    command: list[str],
    config: MonitorConfig,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with a configured timeout."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=config.subprocess_timeout_seconds,
        check=False,
    )


def check_api_health(
    project: str,
    port: int,
    api_key: str,
    config: MonitorConfig,
) -> HealthCheck:
    """Probe a project's /health endpoint and evaluate latency."""
    check_name = f"{project}_api"
    headers = {"X-API-Key": api_key} if api_key else {}

    try:
        start = time.perf_counter()
        with httpx.Client(timeout=config.http_timeout_seconds) as client:
            response = client.get(
                f"http://localhost:{port}/health",
                headers=headers,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            return HealthCheck(
                name=check_name,
                status="critical",
                message=f"API returned HTTP {response.status_code}",
                value=float(response.status_code),
            )

        payload = response.json()
        api_status = str(payload.get("status", "unknown"))

        if api_status != "ok":
            return HealthCheck(
                name=check_name,
                status="warning",
                message=f"API degraded: status={api_status}, latency={latency_ms:.0f}ms",
                value=latency_ms,
            )

        if latency_ms > config.api_latency_warning_ms:
            return HealthCheck(
                name=check_name,
                status="warning",
                message=f"Slow response: {latency_ms:.0f}ms",
                value=latency_ms,
                threshold=config.api_latency_warning_ms,
            )

        return HealthCheck(
            name=check_name,
            status="ok",
            message=f"API healthy: {latency_ms:.0f}ms",
            value=latency_ms,
        )

    except httpx.ConnectError:
        return HealthCheck(
            name=check_name,
            status="critical",
            message="API unreachable — port not responding",
        )
    except httpx.TimeoutException:
        return HealthCheck(
            name=check_name,
            status="critical",
            message=f"API timeout — no response in {config.http_timeout_seconds:.0f}s",
        )
    except Exception as exc:
        return HealthCheck(
            name=check_name,
            status="warning",
            message=f"Health check failed: {exc}",
        )


def check_gpu_memory(config: MonitorConfig) -> HealthCheck:
    """Read GPU memory utilization via nvidia-smi."""
    try:
        result = _run_subprocess(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            config,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")

        used_str, total_str = result.stdout.strip().split(", ")
        used = int(used_str)
        total = int(total_str)
        if total <= 0:
            raise RuntimeError("Invalid GPU memory total")

        pct = (used / total) * 100

        if pct >= config.gpu_memory_critical_pct:
            return HealthCheck(
                name="gpu_memory",
                status="critical",
                message=f"GPU memory critical: {pct:.1f}%",
                value=pct,
                threshold=config.gpu_memory_critical_pct,
            )
        if pct >= config.gpu_memory_warning_pct:
            return HealthCheck(
                name="gpu_memory",
                status="warning",
                message=f"GPU memory high: {pct:.1f}%",
                value=pct,
                threshold=config.gpu_memory_warning_pct,
            )
        return HealthCheck(
            name="gpu_memory",
            status="ok",
            message=f"GPU memory OK: {pct:.1f}%",
            value=pct,
        )
    except Exception as exc:
        return HealthCheck(
            name="gpu_memory",
            status="warning",
            message=f"Could not read GPU: {exc}",
        )


def check_queue_depth(project: str, config: MonitorConfig) -> HealthCheck:
    """Measure pending inference jobs in Redis for a project."""
    check_name = f"{project}_queue"
    try:
        client = _redis_client(config)
        depth = int(client.llen(f"queue:{project}"))

        if depth >= config.queue_critical_depth:
            return HealthCheck(
                name=check_name,
                status="critical",
                message=f"Queue backed up: {depth} jobs waiting",
                value=float(depth),
                threshold=float(config.queue_critical_depth),
            )
        if depth >= config.queue_warning_depth:
            return HealthCheck(
                name=check_name,
                status="warning",
                message=f"Queue growing: {depth} jobs waiting",
                value=float(depth),
                threshold=float(config.queue_warning_depth),
            )
        return HealthCheck(
            name=check_name,
            status="ok",
            message=f"Queue normal: {depth} jobs",
            value=float(depth),
        )
    except Exception as exc:
        return HealthCheck(
            name=check_name,
            status="warning",
            message=f"Redis unreachable: {exc}",
        )


def check_disk_space(config: MonitorConfig) -> HealthCheck:
    """Report root filesystem utilization."""
    try:
        if shutil.which("df"):
            result = _run_subprocess(["df", "-h", "/"], config)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "df failed")
            lines = result.stdout.strip().split("\n")
            parts = lines[1].split()
            pct = int(parts[4].replace("%", ""))
        else:
            usage = shutil.disk_usage("/")
            pct = int((usage.used / usage.total) * 100)

        if pct >= config.disk_critical_pct:
            return HealthCheck(
                name="disk_space",
                status="critical",
                message=f"Disk almost full: {pct}% used",
                value=float(pct),
                threshold=float(config.disk_critical_pct),
            )
        if pct >= config.disk_warning_pct:
            return HealthCheck(
                name="disk_space",
                status="warning",
                message=f"Disk filling up: {pct}% used",
                value=float(pct),
                threshold=float(config.disk_warning_pct),
            )
        return HealthCheck(
            name="disk_space",
            status="ok",
            message=f"Disk OK: {pct}% used",
            value=float(pct),
        )
    except Exception as exc:
        return HealthCheck(
            name="disk_space",
            status="warning",
            message=f"Could not check disk: {exc}",
        )


def _latency_cutoff_iso(minutes: int) -> str:
    """Return an ISO timestamp cutoff compatible with usage.db rows."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%S")


def check_response_trend(project: str, config: MonitorConfig) -> HealthCheck:
    """Detect latency degradation by comparing recent vs hourly averages."""
    check_name = f"{project}_latency"
    db_path = config.workspace / "projects" / project / "usage.db"

    try:
        if not db_path.is_file():
            return HealthCheck(
                name=check_name,
                status="ok",
                message="No usage database yet",
            )

        recent_cutoff = _latency_cutoff_iso(10)
        hourly_cutoff = _latency_cutoff_iso(60)

        with sqlite3.connect(str(db_path)) as conn:
            recent = conn.execute(
                """
                SELECT AVG(latency_ms)
                FROM usage
                WHERE timestamp >= ? AND status = 'success'
                """,
                (recent_cutoff,),
            ).fetchone()[0]
            hourly = conn.execute(
                """
                SELECT AVG(latency_ms)
                FROM usage
                WHERE timestamp >= ? AND status = 'success'
                """,
                (hourly_cutoff,),
            ).fetchone()[0]

        if recent is None:
            return HealthCheck(
                name=check_name,
                status="ok",
                message="No recent requests",
            )

        recent_value = float(recent)
        if hourly and recent_value > (float(hourly) * 2):
            return HealthCheck(
                name=check_name,
                status="warning",
                message=(
                    f"Latency degrading: {recent_value:.0f}ms vs "
                    f"{float(hourly):.0f}ms hourly avg"
                ),
                value=recent_value,
                threshold=float(hourly) * 2,
            )

        return HealthCheck(
            name=check_name,
            status="ok",
            message=f"Latency OK: {recent_value:.0f}ms",
            value=recent_value,
        )
    except Exception as exc:
        return HealthCheck(
            name=check_name,
            status="warning",
            message=f"Could not check latency: {exc}",
        )


def check_pm2_processes(config: MonitorConfig) -> list[HealthCheck]:
    """Inspect PM2 process list for stopped processes and restart storms."""
    checks: list[HealthCheck] = []
    try:
        if not shutil.which("pm2"):
            return [
                HealthCheck(
                    name="pm2",
                    status="warning",
                    message="pm2 command not found",
                )
            ]

        result = _run_subprocess(["pm2", "jlist"], config)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "pm2 jlist failed")

        processes = json.loads(result.stdout or "[]")
        if not processes:
            return [
                HealthCheck(
                    name="pm2",
                    status="warning",
                    message="No PM2 processes registered",
                )
            ]

        for proc in processes:
            name = str(proc.get("name", "unknown"))
            env = proc.get("pm2_env") or {}
            proc_status = str(env.get("status", "unknown"))
            restarts = int(env.get("restart_time") or 0)

            if proc_status != "online":
                checks.append(
                    HealthCheck(
                        name=f"pm2_{name}",
                        status="critical",
                        message=f"Process {name} is {proc_status}",
                        value=float(restarts),
                    )
                )
            elif restarts > config.pm2_restart_warning:
                checks.append(
                    HealthCheck(
                        name=f"pm2_{name}",
                        status="warning",
                        message=f"Process {name} restarted {restarts}x",
                        value=float(restarts),
                        threshold=float(config.pm2_restart_warning),
                    )
                )
            else:
                checks.append(
                    HealthCheck(
                        name=f"pm2_{name}",
                        status="ok",
                        message=f"Process {name} online",
                        value=float(restarts),
                    )
                )
    except Exception as exc:
        checks.append(
            HealthCheck(
                name="pm2",
                status="warning",
                message=f"Could not inspect PM2: {exc}",
            )
        )
    return checks


def check_redis(config: MonitorConfig) -> HealthCheck:
    """Ping Redis and report memory usage."""
    try:
        client = _redis_client(config)
        client.ping()
        info = client.info()
        memory_mb = float(info.get("used_memory", 0)) / 1024 / 1024
        return HealthCheck(
            name="redis",
            status="ok",
            message=f"Redis OK: {memory_mb:.1f}MB used",
            value=memory_mb,
        )
    except Exception as exc:
        return HealthCheck(
            name="redis",
            status="critical",
            message=f"Redis down: {exc}",
        )


def check_backup_freshness(config: MonitorConfig) -> HealthCheck:
    """Verify the latest successful backup timestamp from backup.log."""
    backup_log = config.workspace / "logs" / "backup.log"
    try:
        if not backup_log.is_file():
            return HealthCheck(
                name="backup",
                status="warning",
                message="Backup log not found",
            )

        last_success: datetime | None = None
        for line in reversed(backup_log.read_text(encoding="utf-8").splitlines()):
            if "Backup complete" in line:
                timestamp_part = line.split(" | ", 1)[0].strip()
                last_success = datetime.strptime(
                    timestamp_part,
                    "%Y-%m-%d %H:%M:%S",
                )
                break

        if last_success is None:
            return HealthCheck(
                name="backup",
                status="warning",
                message="No successful backup found in logs",
            )

        hours_ago = (datetime.now() - last_success).total_seconds() / 3600

        if hours_ago > config.backup_critical_hours:
            return HealthCheck(
                name="backup",
                status="critical",
                message=f"Last backup was {hours_ago:.0f}h ago",
                value=hours_ago,
                threshold=config.backup_critical_hours,
            )
        if hours_ago > config.backup_warning_hours:
            return HealthCheck(
                name="backup",
                status="warning",
                message=f"Backup overdue: {hours_ago:.0f}h ago",
                value=hours_ago,
                threshold=config.backup_warning_hours,
            )
        return HealthCheck(
            name="backup",
            status="ok",
            message=f"Backup fresh: {hours_ago:.0f}h ago",
            value=hours_ago,
        )
    except Exception as exc:
        return HealthCheck(
            name="backup",
            status="warning",
            message=f"Cannot check backup: {exc}",
        )


def run_all_checks(config: Optional[MonitorConfig] = None) -> list[HealthCheck]:
    """Execute every configured health probe and return aggregated results."""
    cfg = config or MonitorConfig.from_env()
    checks: list[HealthCheck] = []

    checks.append(check_gpu_memory(cfg))
    checks.append(check_disk_space(cfg))
    checks.append(check_redis(cfg))
    checks.append(check_backup_freshness(cfg))

    for project in cfg.projects:
        project_name = str(project["name"])
        port = int(project["port"])
        api_key = _load_admin_api_key(project_name)

        checks.append(check_api_health(project_name, port, api_key, cfg))
        checks.append(check_queue_depth(project_name, cfg))
        checks.append(check_response_trend(project_name, cfg))

    checks.extend(check_pm2_processes(cfg))
    return checks


def worst_status(checks: list[HealthCheck]) -> StatusLevel:
    """Return the most severe status level from a list of checks."""
    if any(check.status == "critical" for check in checks):
        return "critical"
    if any(check.status == "warning" for check in checks):
        return "warning"
    return "ok"


def log_checks(checks: list[HealthCheck], config: Optional[MonitorConfig] = None) -> None:
    """Append formatted check results to logs/monitor.log."""
    cfg = config or MonitorConfig.from_env()
    log_path = cfg.workspace / "logs" / "monitor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"{timestamp} | monitor run | overall={worst_status(checks)}"]
    for check in checks:
        lines.append(f"  [{check.status}] {check.name}: {check.message}")

    entry = "\n".join(lines) + "\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def _gpu_memory_pct(config: MonitorConfig) -> float | None:
    """Return current GPU memory utilization percentage, if available."""
    check = check_gpu_memory(config)
    return check.value


def _backup_hours_ago(config: MonitorConfig) -> float | None:
    """Return hours since the last successful backup, if known."""
    check = check_backup_freshness(config)
    return check.value


def _requests_today(project_name: str) -> int:
    """Return successful request count for today from the usage tracker."""
    try:
        return UsageTracker(project_name).get_today_count()
    except Exception:
        return 0


def build_admin_health_snapshot(
    config: Optional[MonitorConfig] = None,
) -> dict[str, Any]:
    """Build the aggregated /admin/health response payload."""
    cfg = config or MonitorConfig.from_env()
    gpu_pct = _gpu_memory_pct(cfg)
    disk_check = check_disk_space(cfg)
    redis_check = check_redis(cfg)
    backup_hours = _backup_hours_ago(cfg)

    clients: dict[str, Any] = {}
    client_checks: list[HealthCheck] = []

    for project in cfg.projects:
        project_name = str(project["name"])
        port = int(project["port"])
        api_key = _load_admin_api_key(project_name)

        api_check = check_api_health(project_name, port, api_key, cfg)
        queue_check = check_queue_depth(project_name, cfg)
        client_checks.extend([api_check, queue_check])

        clients[project_name] = {
            "api": api_check.status,
            "latency_ms": api_check.value,
            "queue_depth": int(queue_check.value or 0),
            "requests_today": _requests_today(project_name),
            "gpu_memory_pct": gpu_pct,
        }

    system_checks = [
        check_gpu_memory(cfg),
        disk_check,
        redis_check,
        check_backup_freshness(cfg),
    ]
    overall = worst_status(client_checks + system_checks)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "clients": clients,
        "system": {
            "gpu_memory_pct": gpu_pct,
            "disk_pct": int(disk_check.value) if disk_check.value is not None else None,
            "redis": redis_check.status if redis_check.status == "ok" else redis_check.status,
            "backup_hours_ago": backup_hours,
        },
    }
