"""Centralized logging for training and serving."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from core.utils.config_loader import PROJECT_ROOT, get_logs_dir


def _build_formatter(include_name: bool = True) -> logging.Formatter:
    """Build a log formatter matching the serving spec."""
    if include_name:
        fmt = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
    else:
        fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _add_rotating_handler(
    logger: logging.Logger,
    log_file: Path,
    level: int,
    formatter: logging.Formatter,
) -> None:
    """Attach a daily-rotating file handler (30-day retention)."""
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(level)
    logger.addHandler(file_handler)


def setup_logger(
    project_name: str,
    log_type: str = "training",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure console + rotating file logging under projects/{name}/logs/."""
    log_dir = PROJECT_ROOT / "projects" / project_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_type}.log"

    logger_name = f"{project_name}.{log_type}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = _build_formatter(include_name=True)
    _add_rotating_handler(logger, log_file, level, formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)
    return logger


def setup_serving_loggers(project_name: str) -> tuple[logging.Logger, logging.Logger]:
    """Configure serving.log (all requests) and error.log (errors only)."""
    log_dir = PROJECT_ROOT / "projects" / project_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    serving_logger = logging.getLogger(f"{project_name}.serving")
    error_logger = logging.getLogger(f"{project_name}.serving.error")

    if serving_logger.handlers and error_logger.handlers:
        return serving_logger, error_logger

    formatter = _build_formatter(include_name=False)

    for logger, log_file, level in (
        (serving_logger, log_dir / "serving.log", logging.INFO),
        (error_logger, log_dir / "error.log", logging.ERROR),
    ):
        logger.setLevel(level)
        logger.propagate = False
        _add_rotating_handler(logger, log_file, level, formatter)

        if logger is serving_logger:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(logging.INFO)
            logger.addHandler(stream_handler)

    return serving_logger, error_logger


def setup_security_loggers(
    project_name: str,
) -> tuple[logging.Logger, logging.Logger]:
    """Configure pii_audit.log and security.log for HIPAA compliance auditing."""
    log_dir = PROJECT_ROOT / "projects" / project_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pii_logger = logging.getLogger(f"{project_name}.pii_audit")
    security_logger = logging.getLogger(f"{project_name}.security")

    if pii_logger.handlers and security_logger.handlers:
        return pii_logger, security_logger

    formatter = _build_formatter(include_name=False)

    for audit_logger, log_file, level in (
        (pii_logger, log_dir / "pii_audit.log", logging.INFO),
        (security_logger, log_dir / "security.log", logging.WARNING),
    ):
        audit_logger.setLevel(level)
        audit_logger.propagate = False
        _add_rotating_handler(audit_logger, log_file, level, formatter)

    return pii_logger, security_logger


def log_serving_request(
    logger: logging.Logger,
    *,
    method: str,
    path: str,
    api_key_masked: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: float = 0.0,
    status: int = 200,
) -> None:
    """Log a serving request without prompt text or full API keys."""
    logger.info(
        "%s %s | api_key=%s | tokens_in=%d | tokens_out=%d | latency=%.0fms | status=%d",
        method,
        path,
        api_key_masked,
        tokens_in,
        tokens_out,
        latency_ms,
        status,
    )


def log_serving_error(
    error_logger: logging.Logger,
    *,
    method: str,
    path: str,
    api_key_masked: str,
    error: str,
    status: int,
) -> None:
    """Log a serving error to error.log and console via the error logger."""
    error_logger.error(
        "%s %s | api_key=%s | error=%s | status=%d",
        method,
        path,
        api_key_masked,
        error,
        status,
    )


def get_project_log_path(config: dict, log_type: str) -> Path:
    """Return the log file path for a project without configuring handlers."""
    return get_logs_dir(config) / f"{log_type}.log"
