"""FastAPI serving layer for fine-tuned LLM adapters via vLLM."""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from core.serving.alerts import build_limit_status
from core.serving.auth import (
    ApiKeys,
    load_api_keys,
    load_project_env,
    mask_api_key,
    validate_api_key,
)
from core.serving.key_manager import APIKeyManager
from core.serving.limiter import RateLimiter
from core.serving.model import ServingModel
from core.serving.sanitizer import InputSanitizer, get_user_message
from core.serving.queue import RequestQueue
from core.serving.schemas import (
    AdminHealthResponse,
    ClientHealthSnapshot,
    CompletedResultResponse,
    FailedResultResponse,
    FineTuneStatusResponse,
    GenerateRequest,
    HealthResponse,
    JobNotFoundResponse,
    LimitStatus,
    MetricsResponse,
    PendingResultResponse,
    QueuedGenerateResponse,
    SystemHealthSnapshot,
)
from core.serving.shutdown import ShutdownCoordinator
from core.serving.tracker import UsageTracker
from core.serving.worker import InferenceWorker
from core.utils.config_loader import build_serving_config, get_logs_dir, load_project_config
from core.utils.logger import (
    log_serving_error,
    log_serving_request,
    setup_security_loggers,
    setup_serving_loggers,
)

_serving_config: dict[str, Any] = {}
_serving_model: Optional[ServingModel] = None
_startup_time: float = 0.0
_executor: Optional[ThreadPoolExecutor] = None
_api_keys: Optional[ApiKeys] = None
_tracker: Optional[UsageTracker] = None
_limiter: Optional[RateLimiter] = None
_request_queue: Optional[RequestQueue] = None
_inference_worker: Optional[InferenceWorker] = None
_project_name: str = ""
_serving_logger: Optional[logging.Logger] = None
_error_logger: Optional[logging.Logger] = None
_shutdown_coordinator: Optional[ShutdownCoordinator] = None
_sanitizer: Optional[InputSanitizer] = None
_key_manager: Optional[APIKeyManager] = None
_pii_logger: Optional[logging.Logger] = None
_security_logger: Optional[logging.Logger] = None


def _error_response(status_code: int, error: str, message: str, **extra: Any) -> JSONResponse:
    """Build a JSON error response matching the serving spec."""
    content: dict[str, Any] = {"error": error, "message": message}
    content.update(extra)
    return JSONResponse(status_code=status_code, content=content)


def _request_log_target(request: Request) -> str:
    """Return host + path for logs (e.g. clinical-notes.example.com/generate)."""
    host = request.headers.get("host", "unknown")
    return f"{host}{request.url.path}"


def _ensure_model_ready() -> None:
    """Raise 503 when the serving model is not loaded."""
    if _serving_model is None or not _serving_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_not_ready",
                "message": "Server is warming up",
            },
        )


def _enforce_generation_limits(tokens_in: int, max_tokens: int) -> None:
    """Apply rate, monthly, and token quotas before generation."""
    if _limiter is not None:
        _limiter.enforce_all_limits(estimated_tokens=tokens_in + max_tokens)


def _validate_prompt_tokens(prompt: str) -> int:
    """Return prompt token count or raise 422/503."""
    _ensure_model_ready()
    max_tokens = _serving_config.get("validation", {}).get("max_prompt_tokens", 2048)
    token_count = _serving_model.count_prompt_tokens(prompt)
    if token_count > max_tokens:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "validation_error",
                "message": (
                    f"Prompt exceeds maximum length of {max_tokens} tokens "
                    f"(got {token_count})."
                ),
            },
        )
    return token_count


def _ensure_api_key_active(api_key: str) -> None:
    """Reject revoked API keys before processing a request."""
    if _key_manager is not None and _key_manager.check_revoked(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "api_key_revoked",
                "message": (
                    "Your API key has been revoked due to policy violations. "
                    "Contact support@yourdomain.com"
                ),
            },
        )


def _apply_sanitization(prompt: str, api_key: str) -> tuple[str, int]:
    """Sanitize a prompt and return the cleaned text plus token count."""
    if _sanitizer is None:
        return prompt, _validate_prompt_tokens(prompt)

    _ensure_api_key_active(api_key)
    result = _sanitizer.sanitize(prompt, api_key=api_key)

    if not result.passed:
        masked_key = mask_api_key(api_key)
        if _serving_logger is not None:
            _serving_logger.warning(
                "Request blocked | reason=%s | api_key=%s | risk_score=%.2f",
                result.blocked_reason,
                masked_key,
                result.risk_score,
            )

        if result.violation_type and _key_manager is not None:
            _key_manager.maybe_auto_revoke_after_violation(
                api_key,
                result.violation_type,
                matched_patterns=result.matched_patterns,
            )

        detail: dict[str, Any] = {
            "error": result.blocked_reason or "request_blocked",
            "message": get_user_message(result.blocked_reason or "request_blocked"),
            "risk_score": result.risk_score,
        }
        detail.update(result.block_detail)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    if result.warnings and _serving_logger is not None:
        _serving_logger.warning(
            "Sanitizer warnings: %s | api_key=%s",
            result.warnings,
            mask_api_key(api_key),
        )

    if result.redactions and _pii_logger is not None:
        _pii_logger.info(
            "Redacted: %s | api_key=%s",
            result.redactions,
            mask_api_key(api_key),
        )

    return result.cleaned_prompt, result.token_count


def create_app(project_name: str) -> FastAPI:
    """Build a FastAPI app configured for a specific project."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load configuration, Redis, SQLite, and model once at startup."""
        global _serving_config, _serving_model, _startup_time, _executor
        global _api_keys, _tracker, _limiter, _request_queue, _inference_worker, _project_name
        global _serving_logger, _error_logger, _shutdown_coordinator
        global _sanitizer, _key_manager, _pii_logger, _security_logger

        _project_name = project_name
        load_project_env(project_name)
        project_config = load_project_config(project_name)
        _serving_config = build_serving_config(project_config)

        _serving_logger, _error_logger = setup_serving_loggers(project_name)
        _pii_logger, _security_logger = setup_security_loggers(project_name)
        _serving_logger.info("Starting serving layer for project=%s", project_name)

        _api_keys = load_api_keys(project_name)
        _tracker = UsageTracker(project_name)
        _limiter = RateLimiter(project_name, _serving_config, _tracker)
        _request_queue = RequestQueue(project_name, _serving_config)

        _serving_model = ServingModel(_serving_config)
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vllm-gen")

        try:
            _serving_model.load()
        except Exception as exc:
            _serving_logger.exception("Failed to load model at startup: %s", exc)
            if _error_logger:
                log_serving_error(
                    _error_logger,
                    method="STARTUP",
                    path="/model/load",
                    api_key_masked="xxxx",
                    error=str(exc),
                    status=503,
                )
            raise

        _key_manager = APIKeyManager(
            project_name,
            _serving_config,
            security_logger=_security_logger,
        )
        _sanitizer = InputSanitizer(
            _serving_config,
            _serving_model.count_prompt_tokens,
            pii_audit_logger=_pii_logger,
            security_logger=_security_logger,
        )

        _inference_worker = InferenceWorker()
        _inference_worker.start(
            model=_serving_model,
            project_name=project_name,
            tracker=_tracker,
            queue=_request_queue,
        )

        _startup_time = time.time()
        port = _serving_config.get("serving", {}).get("port", 8000)
        queue_depth = _request_queue.get_queue_depth()
        _serving_logger.info(
            "%s API ready on port %s. Worker started. Queue depth: %s",
            project_name,
            port,
            queue_depth,
        )

        _shutdown_coordinator = ShutdownCoordinator(
            drain_timeout=float(_serving_config.get("serving", {}).get("shutdown_timeout", 60)),
        )
        _shutdown_coordinator.register(
            app,
            worker=_inference_worker,
            executor=_executor,
            tracker=_tracker,
            serving_logger=_serving_logger,
        )

        yield

        _serving_logger.info("Shutting down serving layer for project=%s", project_name)
        if _shutdown_coordinator is not None:
            _shutdown_coordinator.shutdown()

    app = FastAPI(
        title=f"LLM Serving — {project_name}",
        description="Production FastAPI + vLLM layer for LoRA/QLoRA adapters.",
        version="3.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log requests with the public hostname when behind Cloudflare Tunnel."""
        start = time.time()
        response = await call_next(request)
        if _serving_logger is not None and request.url.path != "/health":
            log_serving_request(
                _serving_logger,
                method=request.method,
                path=_request_log_target(request),
                api_key_masked=mask_api_key(request.headers.get("X-API-Key", "")),
                latency_ms=(time.time() - start) * 1000,
                status=response.status_code,
            )
        return response

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add HTTPS security headers for traffic terminated at Cloudflare edge."""
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if "server" in response.headers:
            del response.headers["server"]
        return response

    async def verify_client_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
        """Validate client or admin API key for /generate."""
        if _api_keys is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "model_not_ready", "message": "Server is warming up"},
            )
        return validate_api_key(x_api_key, _api_keys, require_admin=False)

    async def verify_admin_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
        """Validate admin API key for /metrics."""
        if _api_keys is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "model_not_ready", "message": "Server is warming up"},
            )
        return validate_api_key(x_api_key, _api_keys, require_admin=True)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return 422 with a specific field error message."""
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(part) for part in first.get("loc", []) if part != "body")
            message = first.get("msg", "Invalid request payload.")
            if loc:
                message = f"{loc}: {message}"
        else:
            message = "Invalid request payload."

        if _error_logger:
            log_serving_error(
                _error_logger,
                method=request.method,
                path=_request_log_target(request),
                api_key_masked=mask_api_key(request.headers.get("X-API-Key", "")),
                error=message,
                status=422,
            )
        return _error_response(422, "validation_error", message)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Normalize HTTPException payloads to {error, message} format."""
        detail = exc.detail
        if isinstance(detail, dict):
            content = detail
        else:
            content = {"error": "request_error", "message": str(detail)}

        if exc.status_code >= 500 and _error_logger:
            log_serving_error(
                _error_logger,
                method=request.method,
                path=_request_log_target(request),
                api_key_masked=mask_api_key(request.headers.get("X-API-Key", "")),
                error=content.get("message", str(detail)),
                status=exc.status_code,
            )
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.get("/health", response_model=HealthResponse, tags=["monitoring"])
    async def health_check() -> HealthResponse:
        """Return model readiness, GPU memory, queue depth, and uptime."""
        queue_depth = _request_queue.get_queue_depth() if _request_queue else 0
        worker_status = (
            "running"
            if _inference_worker is not None and _inference_worker.is_running
            else "stopped"
        )

        if _serving_model is None or not _serving_model.is_loaded:
            return HealthResponse(
                status="degraded",
                model=_serving_model.base_model_name if _serving_model else "unknown",
                adapter=_project_name or None,
                gpu_memory_used="0.0GB",
                gpu_memory_total="0GB",
                uptime_seconds=round(time.time() - _startup_time, 2) if _startup_time else 0.0,
                queue_depth=queue_depth,
                worker_status=worker_status,
            )

        used_mb, total_mb = ServingModel.get_gpu_memory_mb()
        used_gb, total_gb = ServingModel.format_gpu_memory_gb(used_mb, total_mb)
        short_name = ServingModel._short_model_name(_serving_model.base_model_name)

        return HealthResponse(
            status="ok",
            model=short_name,
            adapter=_project_name,
            gpu_memory_used=used_gb,
            gpu_memory_total=total_gb,
            uptime_seconds=round(time.time() - _startup_time, 2) if _startup_time else 0.0,
            queue_depth=queue_depth,
            worker_status=worker_status,
        )

    @app.get("/admin/health", response_model=AdminHealthResponse, tags=["monitoring"])
    async def admin_health(
        api_key: str = Depends(verify_admin_key),
    ) -> AdminHealthResponse:
        """Return aggregated health for all client APIs (admin API key required)."""
        del api_key
        from scripts.monitor import build_admin_health_snapshot

        snapshot = build_admin_health_snapshot()
        clients = {
            name: ClientHealthSnapshot(**payload)
            for name, payload in snapshot["clients"].items()
        }
        return AdminHealthResponse(
            timestamp=snapshot["timestamp"],
            overall=snapshot["overall"],
            clients=clients,
            system=SystemHealthSnapshot(**snapshot["system"]),
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
    async def metrics(
        api_key: str = Depends(verify_admin_key),
    ) -> MetricsResponse:
        """Return aggregate usage metrics (admin API key required)."""
        if _tracker is None or _limiter is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "model_not_ready",
                    "message": "Server is warming up",
                },
            )

        rate_count = _limiter.get_current_rate_count()
        month_count = _tracker.get_month_count()
        month_tokens = _tracker.get_month_tokens()

        limit_status = build_limit_status(
            rate_count=rate_count,
            rate_limit=_limiter.rate_limit,
            month_count=month_count,
            monthly_limit=_limiter.monthly_limit,
            month_tokens=month_tokens,
            token_limit=_limiter.token_limit,
        )

        return MetricsResponse(
            total_requests_today=_tracker.get_today_count(),
            total_requests_month=month_count,
            total_tokens_month=month_tokens,
            avg_latency_ms=_tracker.get_avg_latency(),
            requests_per_minute=_tracker.get_rpm(),
            limit_status=LimitStatus(**limit_status),
        )

    @app.post("/fine-tune-status", response_model=FineTuneStatusResponse, tags=["monitoring"])
    async def fine_tune_status(
        api_key: str = Depends(verify_client_key),
    ) -> FineTuneStatusResponse:
        """Return model and adapter readiness (client or admin API key)."""
        del api_key
        if _serving_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "model_not_ready",
                    "message": "Server is warming up",
                },
            )

        status_payload = _serving_model.get_fine_tune_status()
        status_payload["uptime_seconds"] = (
            round(time.time() - _startup_time, 2) if _startup_time else 0.0
        )
        return FineTuneStatusResponse(**status_payload)

    @app.post(
        "/generate",
        response_model=QueuedGenerateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["inference"],
        responses={
            202: {"description": "Job accepted and queued for inference"},
            503: {"description": "Queue full or model not ready"},
        },
    )
    async def generate(
        body: GenerateRequest,
        api_key: str = Depends(verify_client_key),
    ) -> QueuedGenerateResponse:
        """Enqueue a generation job and return a job_id for polling."""
        if body.stream:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "streaming_not_supported",
                    "message": (
                        "Streaming is not supported with the queued inference API. "
                        "Submit with stream=false and poll GET /result/{job_id}."
                    ),
                },
            )

        if _request_queue is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "model_not_ready",
                    "message": "Server is warming up",
                },
            )

        masked_key = mask_api_key(api_key)
        del masked_key  # auth verified; usage logged by worker on completion

        cleaned_prompt, tokens_in = _apply_sanitization(body.prompt, api_key)
        _enforce_generation_limits(tokens_in, body.max_tokens)

        if _request_queue.is_queue_full():
            depth = _request_queue.get_queue_depth()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "queue_full",
                    "message": "Server is busy. Please retry in 60 seconds.",
                    "queue_depth": depth,
                },
            )

        job_id = str(uuid.uuid4())
        _request_queue.enqueue(
            job_id,
            {
                "prompt": cleaned_prompt,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
                "top_p": body.top_p,
            },
        )
        position = _request_queue.get_queue_depth()

        return QueuedGenerateResponse(
            job_id=job_id,
            status="queued",
            position=position,
            poll_url=f"/result/{job_id}",
            message=(
                "Your request is queued. Poll /result/{job_id} for the result."
            ),
        )

    @app.get(
        "/result/{job_id}",
        tags=["inference"],
        responses={
            200: {
                "description": "Pending, completed, or failed job result",
                "model": CompletedResultResponse | PendingResultResponse | FailedResultResponse,
            },
            404: {"description": "Job not found or expired", "model": JobNotFoundResponse},
        },
    )
    async def get_result(
        job_id: str,
        api_key: str = Depends(verify_client_key),
    ) -> (
        PendingResultResponse
        | CompletedResultResponse
        | FailedResultResponse
        | JSONResponse
    ):
        """Poll for the result of a queued generation job."""
        del api_key

        if _request_queue is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "model_not_ready",
                    "message": "Server is warming up",
                },
            )

        if not _request_queue.is_job_known(job_id):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "job_not_found",
                    "message": "Job expired or invalid job_id",
                },
            )

        result = _request_queue.get_result(job_id)
        if result is None:
            return PendingResultResponse(
                status="pending",
                message="Still processing, poll again in 1 second",
            )

        if result.get("status") == "error":
            return FailedResultResponse(
                status="error",
                error=str(result.get("error", "Unknown error")),
            )

        return CompletedResultResponse(
            status="done",
            output=str(result["output"]),
            tokens_in=int(result["tokens_in"]),
            tokens_out=int(result["tokens_out"]),
            latency_ms=float(result["latency_ms"]),
            model=str(result.get("model", "unknown")),
        )

    def _load_eval_report_files() -> tuple[dict[str, Any], str | None]:
        """Load eval_report.json and eval_report.md from the project logs directory."""
        import json

        config = load_project_config(_project_name)
        logs_dir = get_logs_dir(config)
        json_path = logs_dir / "eval_report.json"
        md_path = logs_dir / "eval_report.md"

        if not json_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "eval_report_not_found",
                    "message": (
                        f"No evaluation report found for {_project_name}. "
                        "Run: python run.py --project "
                        f"{_project_name} --mode eval"
                    ),
                },
            )

        with json_path.open("r", encoding="utf-8") as handle:
            report_data = json.load(handle)

        markdown_text: str | None = None
        if md_path.is_file():
            markdown_text = md_path.read_text(encoding="utf-8")

        return report_data, markdown_text

    @app.get("/eval-report", tags=["monitoring"])
    async def eval_report(
        api_key: str = Depends(verify_admin_key),
    ) -> dict[str, Any]:
        """Return the latest evaluation report summary (admin API key required)."""
        del api_key
        from core.evaluation.report_generator import ReportGenerator

        report_data, _ = _load_eval_report_files()
        verdict = ReportGenerator.verdict_from_dict(report_data)

        return {
            "project": report_data.get("project_name", _project_name),
            "eval_date": report_data.get("eval_date"),
            "metrics": {
                "rouge1_improvement_pct": round(
                    report_data.get("rouge1_improvement", 0.0),
                ),
                "bleu_improvement_pct": round(
                    report_data.get("bleu_improvement", 0.0),
                ),
                "perplexity_reduction_pct": round(
                    report_data.get("perplexity_reduction", 0.0),
                ),
                "base_rouge1": report_data.get("base_rouge1"),
                "ft_rouge1": report_data.get("ft_rouge1"),
                "base_bleu": report_data.get("base_bleu"),
                "ft_bleu": report_data.get("ft_bleu"),
            },
            "verdict": verdict,
            "report_url": "/eval-report/markdown",
        }

    @app.get("/eval-report/markdown", tags=["monitoring"])
    async def eval_report_markdown(
        api_key: str = Depends(verify_admin_key),
    ) -> PlainTextResponse:
        """Return the full markdown evaluation report (admin API key required)."""
        del api_key
        _, markdown_text = _load_eval_report_files()
        if markdown_text is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "eval_report_markdown_not_found",
                    "message": "Markdown report not found. Re-run evaluation.",
                },
            )
        return PlainTextResponse(
            content=markdown_text,
            media_type="text/markdown; charset=utf-8",
        )

    return app
