"""API key violation tracking and auto-revocation."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import redis

from core.serving.alerts import _send_email
from core.serving.auth import mask_api_key

logger = logging.getLogger(__name__)

_VIOLATION_TTL_SECONDS = 86_400
_REVOKED_KEYS_SET = "revoked_keys"


class _InMemoryViolationStore:
    """Thread-safe violation counter used when Redis is unavailable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._revoked: set[str] = set()

    def increment(self, api_key: str) -> int:
        """Increment and return the violation count for an API key."""
        with self._lock:
            self._counts[api_key] = self._counts.get(api_key, 0) + 1
            return self._counts[api_key]

    def reset(self, api_key: str) -> None:
        """Clear the violation counter for an API key."""
        with self._lock:
            self._counts.pop(api_key, None)

    def is_revoked(self, api_key: str) -> bool:
        """Return whether the API key has been revoked."""
        with self._lock:
            return api_key in self._revoked

    def revoke(self, api_key: str) -> None:
        """Mark an API key as revoked."""
        with self._lock:
            self._revoked.add(api_key)

    def reinstate(self, api_key: str) -> None:
        """Remove an API key from the revoked set and reset its counter."""
        with self._lock:
            self._revoked.discard(api_key)
            self._counts.pop(api_key, None)


class APIKeyManager:
    """Track policy violations and auto-revoke abusive API keys."""

    def __init__(
        self,
        project_name: str,
        config: dict[str, Any],
        *,
        redis_client: Optional[redis.Redis] = None,
        security_logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize Redis-backed violation tracking for a project."""
        self._project_name = project_name
        self._config = config
        self._sanitization = config.get("sanitization", {})
        self._alerts = config.get("alerts", {})
        self._security_logger = security_logger
        self._memory_store = _InMemoryViolationStore()
        self._redis: Optional[redis.Redis] = redis_client
        if self._redis is None:
            self._connect_redis()

    def _connect_redis(self) -> None:
        """Connect to Redis using project configuration."""
        redis_cfg = self._config.get("redis", {})
        host = redis_cfg.get("host", "localhost")
        port = int(redis_cfg.get("port", 6379))
        db = int(redis_cfg.get("db", 0))
        password = redis_cfg.get("password") or None

        try:
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            client.ping()
            self._redis = client
        except redis.RedisError as exc:
            logger.warning(
                "Redis unavailable for key manager; using in-memory fallback: %s",
                exc,
            )
            self._redis = None

    def _violation_key(self, api_key: str) -> str:
        """Return the Redis key for a violation counter."""
        return f"violations:{self._project_name}:{api_key}"

    def check_revoked(self, api_key: str) -> bool:
        """Return True when the API key is on the revoked list."""
        if self._redis is not None:
            try:
                return bool(self._redis.sismember(_REVOKED_KEYS_SET, api_key))
            except redis.RedisError as exc:
                logger.warning("Redis revoked check failed: %s", exc)
        return self._memory_store.is_revoked(api_key)

    def track_violation(self, api_key: str, violation_type: str) -> int:
        """Increment the violation counter and return the new total."""
        if self._redis is not None:
            try:
                key = self._violation_key(api_key)
                pipe = self._redis.pipeline()
                pipe.incr(key)
                pipe.expire(key, _VIOLATION_TTL_SECONDS)
                count, _ = pipe.execute()
                return int(count)
            except redis.RedisError as exc:
                logger.warning("Redis violation tracking failed: %s", exc)

        return self._memory_store.increment(api_key)

    def auto_revoke(self, api_key: str, reason: str) -> None:
        """Add an API key to the revoked set and notify administrators."""
        masked = mask_api_key(api_key)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if self._redis is not None:
            try:
                self._redis.sadd(_REVOKED_KEYS_SET, api_key)
            except redis.RedisError as exc:
                logger.warning("Redis auto-revoke failed: %s", exc)
                self._memory_store.revoke(api_key)
        else:
            self._memory_store.revoke(api_key)

        if self._security_logger is not None:
            self._security_logger.error(
                "REVOKED | api_key=%s | reason=%s | client=%s | time=%s",
                masked,
                reason,
                self._project_name,
                timestamp,
            )

        email = self._alerts.get("email", "")
        if email:
            subject = "API Key Auto-Revoked"
            body = (
                f"Key: {masked}\n"
                f"Reason: {reason}\n"
                f"Client: {self._project_name}\n"
                f"Time: {timestamp}\n"
                "Action: Key added to revoked list\n"
            )
            _send_email(email, subject, body)

    def maybe_auto_revoke_after_violation(
        self,
        api_key: str,
        violation_type: str,
        *,
        matched_patterns: list[str] | None = None,
    ) -> int:
        """Track a violation and revoke the key when the threshold is exceeded."""
        attempt_count = self.track_violation(api_key, violation_type)
        max_attempts = int(self._sanitization.get("max_injection_attempts", 3))
        auto_revoke = bool(self._sanitization.get("auto_revoke_key", True))

        if self._security_logger is not None:
            pattern_label = (
                matched_patterns[0] if matched_patterns else violation_type
            )
            self._security_logger.warning(
                "BLOCKED | api_key=%s | reason=%s | risk_score=1.00 | "
                "pattern_matched=%s | attempt_count=%d",
                mask_api_key(api_key),
                violation_type,
                pattern_label,
                attempt_count,
            )

        if auto_revoke and attempt_count >= max_attempts:
            self.auto_revoke(
                api_key,
                reason=f"{attempt_count} {violation_type} attempts",
            )
        return attempt_count

    def reinstate_key(self, api_key: str) -> None:
        """Remove an API key from the revoked set and reset its counter."""
        if self._redis is not None:
            try:
                self._redis.srem(_REVOKED_KEYS_SET, api_key)
                self._redis.delete(self._violation_key(api_key))
            except redis.RedisError as exc:
                logger.warning("Redis reinstate failed: %s", exc)
        self._memory_store.reinstate(api_key)

        if self._security_logger is not None:
            self._security_logger.info(
                "REINSTATED | api_key=%s | client=%s",
                mask_api_key(api_key),
                self._project_name,
            )


def generate_monthly_security_report(
    project_name: str,
    logs_dir: Path,
    *,
    email: str = "",
) -> dict[str, int]:
    """Summarize security and PII audit logs for compliance reporting."""
    security_log = logs_dir / "security.log"
    pii_log = logs_dir / "pii_audit.log"
    serving_log = logs_dir / "serving.log"

    stats = {
        "total_requests": 0,
        "blocked_injection": 0,
        "blocked_jailbreak": 0,
        "blocked_training_data": 0,
        "pii_redactions": 0,
        "auto_revoked_keys": 0,
    }

    if serving_log.is_file():
        stats["total_requests"] = sum(
            1 for line in serving_log.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    if security_log.is_file():
        for line in security_log.read_text(encoding="utf-8").splitlines():
            if "prompt_injection_detected" in line:
                stats["blocked_injection"] += 1
            elif "jailbreak_detected" in line:
                stats["blocked_jailbreak"] += 1
            elif "training_data_extraction" in line:
                stats["blocked_training_data"] += 1
            elif "REVOKED" in line:
                stats["auto_revoked_keys"] += 1

    if pii_log.is_file():
        stats["pii_redactions"] = sum(
            1
            for line in pii_log.read_text(encoding="utf-8").splitlines()
            if "REDACT" in line
        )

    if email:
        subject = f"[{project_name}] Monthly Security Report"
        body = (
            f"Monthly security report for {project_name}\n\n"
            f"Total requests: {stats['total_requests']}\n"
            f"Blocked (injection): {stats['blocked_injection']}\n"
            f"Blocked (jailbreak): {stats['blocked_jailbreak']}\n"
            f"Blocked (training data): {stats['blocked_training_data']}\n"
            f"PII redactions: {stats['pii_redactions']}\n"
            f"Auto-revoked keys: {stats['auto_revoked_keys']}\n"
        )
        _send_email(email, subject, body)

    return stats
