"""Redis-backed request queue for serialized GPU inference."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import redis

logger = logging.getLogger(__name__)


class RequestQueue:
    """Thread-safe Redis queue for inference jobs and results."""

    def __init__(
        self,
        project_name: str,
        config: dict[str, Any],
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        """Initialize queue settings and Redis connection."""
        self._project_name = project_name
        self._config = config
        queue_cfg = config.get("queue", {})
        self._max_queue_depth = int(queue_cfg.get("max_queue_depth", 50))
        self._job_ttl_seconds = int(queue_cfg.get("job_ttl_seconds", 600))
        self._result_ttl_seconds = int(queue_cfg.get("result_ttl_seconds", 600))
        self._lock = threading.Lock()
        self._redis: Optional[redis.Redis] = redis_client
        if self._redis is None:
            self._connect_redis()

    @property
    def max_queue_depth(self) -> int:
        """Return configured maximum waiting jobs before rejecting new requests."""
        return self._max_queue_depth

    @property
    def is_redis_connected(self) -> bool:
        """Return whether Redis is available."""
        return self._redis is not None

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
            logger.info("RequestQueue connected to Redis at %s:%s db=%s", host, port, db)
        except redis.RedisError as exc:
            logger.error("RequestQueue requires Redis but connection failed: %s", exc)
            raise RuntimeError(
                "Redis is required for the inference queue but is unavailable."
            ) from exc

    def _queue_key(self, project_name: Optional[str] = None) -> str:
        """Return the Redis list key for pending jobs."""
        name = project_name or self._project_name
        return f"queue:{name}"

    def _job_key(self, job_id: str) -> str:
        """Return the Redis key for job metadata."""
        return f"job:{job_id}"

    def _result_key(self, job_id: str) -> str:
        """Return the Redis key for a completed or failed result."""
        return f"result:{job_id}"

    def enqueue(
        self,
        job_id: str,
        payload: dict[str, Any],
        project_name: Optional[str] = None,
    ) -> str:
        """Push a job onto the queue and store its metadata with TTL."""
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized.")

        project = project_name or self._project_name
        job_record = {
            "job_id": job_id,
            "prompt": payload["prompt"],
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "top_p": float(payload.get("top_p", 0.9)),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "pending",
        }

        with self._lock:
            pipe = self._redis.pipeline()
            pipe.set(
                self._job_key(job_id),
                json.dumps(job_record),
                ex=self._job_ttl_seconds,
            )
            pipe.rpush(self._queue_key(project), job_id)
            pipe.execute()

        return job_id

    def dequeue(
        self,
        project_name: Optional[str] = None,
        timeout: int = 1,
    ) -> Optional[dict[str, Any]]:
        """Block-pop the next job from the queue, or return None when empty."""
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized.")

        project = project_name or self._project_name
        popped = self._redis.blpop(self._queue_key(project), timeout=timeout)
        if popped is None:
            return None

        _, job_id = popped
        raw = self._redis.get(self._job_key(job_id))
        if raw is None:
            logger.warning("Job %s was queued but metadata expired or missing.", job_id)
            return None

        job = json.loads(raw)
        job["status"] = "processing"
        self._redis.set(
            self._job_key(job_id),
            json.dumps(job),
            ex=self._job_ttl_seconds,
        )
        return job

    def store_result(self, job_id: str, result: dict[str, Any]) -> None:
        """Persist a successful inference result with TTL."""
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized.")

        payload = dict(result)
        payload["status"] = "done"
        self._redis.set(
            self._result_key(job_id),
            json.dumps(payload),
            ex=self._result_ttl_seconds,
        )
        self._redis.delete(self._job_key(job_id))

    def store_error(self, job_id: str, error_msg: str) -> None:
        """Persist a failed job result with TTL."""
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized.")

        payload = {"status": "error", "error": error_msg}
        self._redis.set(
            self._result_key(job_id),
            json.dumps(payload),
            ex=self._result_ttl_seconds,
        )
        self._redis.delete(self._job_key(job_id))

    def get_result(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return the completed result, None if still pending, absent if unknown."""
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized.")

        raw = self._redis.get(self._result_key(job_id))
        if raw is not None:
            return json.loads(raw)

        if self._redis.exists(self._job_key(job_id)):
            return None

        return None

    def is_job_known(self, job_id: str) -> bool:
        """Return whether a job is pending or has a stored result."""
        if self._redis is None:
            return False
        return bool(
            self._redis.exists(self._result_key(job_id))
            or self._redis.exists(self._job_key(job_id))
        )

    def get_queue_depth(self, project_name: Optional[str] = None) -> int:
        """Return how many jobs are waiting in the queue."""
        if self._redis is None:
            return 0
        return int(self._redis.llen(self._queue_key(project_name)))

    def is_queue_full(self, project_name: Optional[str] = None) -> bool:
        """Return True when the queue has reached max_queue_depth."""
        return self.get_queue_depth(project_name) >= self._max_queue_depth
