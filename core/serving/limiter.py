"""Redis-backed rate limiting with in-memory fallback and quota enforcement."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import redis
from fastapi import HTTPException, status

from core.serving.alerts import maybe_send_usage_warning
from core.serving.tracker import UsageTracker

logger = logging.getLogger(__name__)

_RATE_WINDOW_SECONDS = 60


class _InMemoryRateCounter:
    """Fixed-window request counter used when Redis is unavailable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._window_start = time.monotonic()
        self._count = 0

    def _reset_if_expired(self, now: float) -> None:
        if now - self._window_start >= _RATE_WINDOW_SECONDS:
            self._window_start = now
            self._count = 0

    def increment_and_get(self) -> int:
        """Increment the current window counter and return the new total."""
        with self._lock:
            now = time.monotonic()
            self._reset_if_expired(now)
            self._count += 1
            return self._count

    def get_count(self) -> int:
        """Return the current window count without incrementing."""
        with self._lock:
            now = time.monotonic()
            self._reset_if_expired(now)
            return self._count


class RateLimiter:
    """Per-client rate, monthly request, and token quota enforcement."""

    def __init__(
        self,
        project_name: str,
        config: dict[str, Any],
        tracker: UsageTracker,
    ) -> None:
        """Initialize Redis client and serving limits from config."""
        self._project_name = project_name
        self._config = config
        self._tracker = tracker
        self._serving = config.get("serving", {})
        self._alerts = config.get("alerts", {})
        self._redis: Optional[redis.Redis] = None
        self._memory_counter = _InMemoryRateCounter()
        self._connect_redis()

    @property
    def rate_limit(self) -> int:
        """Return configured requests-per-minute limit."""
        return int(self._serving.get("rate_limit", 10))

    @property
    def monthly_limit(self) -> int:
        """Return configured monthly request limit."""
        return int(self._serving.get("monthly_limit", 10000))

    @property
    def token_limit(self) -> int:
        """Return configured monthly token limit."""
        return int(self._serving.get("token_limit", 1_000_000))

    @property
    def upgrade_email(self) -> str:
        """Return the upgrade contact email for quota-exceeded responses."""
        return self._alerts.get("email", "support@example.com")

    def _connect_redis(self) -> None:
        """Establish a Redis connection using config or environment defaults."""
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
                socket_connect_timeout=5,
            )
            client.ping()
            self._redis = client
            logger.info("Connected to Redis at %s:%s db=%s", host, port, db)
        except redis.RedisError as exc:
            logger.warning("Redis unavailable (%s). Rate limiting will use in-memory fallback.", exc)
            self._redis = None

    @property
    def is_redis_connected(self) -> bool:
        """Return whether Redis is available."""
        return self._redis is not None

    def ping(self) -> bool:
        """Check Redis connectivity."""
        if self._redis is None:
            return False
        try:
            return bool(self._redis.ping())
        except redis.RedisError:
            return False

    def _redis_key(self) -> str:
        """Return the Redis counter key for this client."""
        return f"rate:{self._project_name}"

    def get_current_rate_count(self) -> int:
        """Return the current per-minute request count."""
        if self._redis is not None:
            try:
                value = self._redis.get(self._redis_key())
                return int(value) if value else 0
            except redis.RedisError as exc:
                logger.warning("Redis read failed (%s); using in-memory rate count.", exc)
        return self._memory_counter.get_count()

    def _raise_if_over_limit(self, count: int) -> None:
        """Raise 429 when the per-minute limit is exceeded."""
        if count > self.rate_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"Rate limit exceeded: {self.rate_limit} requests per minute. "
                        "Please retry shortly."
                    ),
                },
            )

    def _check_redis_rate_limit(self) -> None:
        """Increment Redis counter and raise 429 when over the per-minute limit."""
        key = self._redis_key()
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = pipe.execute()

        if ttl == -1:
            self._redis.expire(key, _RATE_WINDOW_SECONDS)

        self._raise_if_over_limit(int(count))

    def _check_memory_rate_limit(self) -> None:
        """Increment in-memory counter and raise 429 when over the per-minute limit."""
        count = self._memory_counter.increment_and_get()
        self._raise_if_over_limit(count)

    def check_rate_limit(self) -> None:
        """Increment rate counter via Redis, falling back to in-memory storage."""
        if self._redis is not None:
            try:
                self._check_redis_rate_limit()
                return
            except HTTPException:
                raise
            except redis.RedisError as exc:
                logger.warning(
                    "Redis rate limit error (%s); falling back to in-memory counter.",
                    exc,
                )

        self._check_memory_rate_limit()

    def check_monthly_limit(self) -> None:
        """Raise 402 when the monthly request quota is exhausted."""
        month_count = self._tracker.get_month_count()
        maybe_send_usage_warning(
            self._project_name,
            "monthly",
            month_count,
            self.monthly_limit,
            self._config,
        )

        if month_count >= self.monthly_limit:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "monthly_limit_reached",
                    "message": (
                        f"You have used {month_count}/{self.monthly_limit} "
                        "requests this month. Please upgrade your plan."
                    ),
                    "upgrade_email": self.upgrade_email,
                },
            )

    def check_token_limit(self, estimated_tokens: int = 0) -> None:
        """Raise 402 when the monthly token quota is exhausted."""
        month_tokens = self._tracker.get_month_tokens()
        maybe_send_usage_warning(
            self._project_name,
            "token",
            month_tokens,
            self.token_limit,
            self._config,
        )

        if month_tokens + estimated_tokens >= self.token_limit:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "token_limit_reached",
                    "message": (
                        f"You have used {month_tokens:,}/{self.token_limit:,} "
                        "tokens this month. Please upgrade your plan."
                    ),
                    "upgrade_email": self.upgrade_email,
                },
            )

    def enforce_all_limits(self, estimated_tokens: int = 0) -> None:
        """Run rate, monthly, and token limit checks before inference."""
        self.check_rate_limit()
        self.check_monthly_limit()
        self.check_token_limit(estimated_tokens)
