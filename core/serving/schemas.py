"""Pydantic request and response models for the serving API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Input payload for text generation."""

    prompt: str = Field(..., min_length=1, description="Input prompt text.")
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=512,
        description="Maximum number of new tokens to generate (1–512).",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (0.1–2.0).",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Nucleus sampling top-p (0.1–1.0).",
    )
    stream: bool = Field(
        default=False,
        description="When true, stream output tokens via Server-Sent Events.",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        """Reject whitespace-only prompts."""
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace only.")
        return value


class GenerateResponse(BaseModel):
    """Successful generation response (direct or polled)."""

    output: str = Field(..., description="Generated text.")
    tokens_in: int = Field(..., ge=0, description="Input prompt token count.")
    tokens_out: int = Field(..., ge=0, description="Generated output token count.")
    latency_ms: float = Field(..., ge=0, description="End-to-end latency in milliseconds.")
    model: str = Field(..., description="Model and adapter identifier.")


class QueuedGenerateResponse(BaseModel):
    """Response when a generation job is accepted into the queue."""

    job_id: str = Field(..., description="Unique job identifier for polling.")
    status: str = Field(..., description="Initial job status, typically 'queued'.")
    position: int = Field(..., ge=0, description="Queue position after enqueue.")
    poll_url: str = Field(..., description="Relative URL to poll for the result.")
    message: str = Field(..., description="Human-readable polling instructions.")


class PendingResultResponse(BaseModel):
    """Response while a queued job is still processing."""

    status: str = Field(..., description="Job status, 'pending'.")
    message: str = Field(..., description="Human-readable wait message.")


class CompletedResultResponse(BaseModel):
    """Successful result from a completed queued job."""

    status: str = Field(..., description="Job status, 'done'.")
    output: str = Field(..., description="Generated text.")
    tokens_in: int = Field(..., ge=0, description="Input prompt token count.")
    tokens_out: int = Field(..., ge=0, description="Generated output token count.")
    latency_ms: float = Field(..., ge=0, description="Inference latency in milliseconds.")
    model: str = Field(..., description="Model and adapter identifier.")


class FailedResultResponse(BaseModel):
    """Failed result from a queued job."""

    status: str = Field(..., description="Job status, 'error'.")
    error: str = Field(..., description="Error message describing the failure.")


class JobNotFoundResponse(BaseModel):
    """Response when a job id is invalid or expired."""

    error: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str = Field(..., description="Health status string.")
    model: str = Field(..., description="Base model display name.")
    adapter: Optional[str] = Field(None, description="Adapter project name when loaded.")
    gpu_memory_used: str = Field(..., description="GPU memory currently allocated.")
    gpu_memory_total: str = Field(..., description="Total GPU memory available.")
    uptime_seconds: float = Field(..., ge=0, description="Process uptime in seconds.")
    queue_depth: int = Field(default=0, ge=0, description="Jobs waiting in the inference queue.")
    worker_status: str = Field(
        default="stopped",
        description="Background worker state: running or stopped.",
    )


class LimitStatus(BaseModel):
    """Current usage against configured limits."""

    rate_limit: str = Field(..., description="Current/minute rate limit usage.")
    monthly_limit: str = Field(..., description="Current/monthly request usage.")
    token_limit: str = Field(..., description="Current/monthly token usage.")


class MetricsResponse(BaseModel):
    """Aggregate serving metrics (admin only)."""

    total_requests_today: int = Field(..., ge=0, description="Successful requests today (UTC).")
    total_requests_month: int = Field(..., ge=0, description="Successful requests this month (UTC).")
    total_tokens_month: int = Field(..., ge=0, description="Total tokens consumed this month (UTC).")
    avg_latency_ms: float = Field(..., ge=0, description="Average latency this month (ms).")
    requests_per_minute: float = Field(..., ge=0, description="Requests in the last 60 seconds.")
    limit_status: LimitStatus = Field(..., description="Usage against configured limits.")


class ErrorResponse(BaseModel):
    """Structured error payload."""

    error: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")


class QuotaErrorResponse(ErrorResponse):
    """Quota exceeded error with upgrade contact."""

    upgrade_email: str = Field(..., description="Contact email for plan upgrades.")


class ClientHealthSnapshot(BaseModel):
    """Per-client health metrics for the admin dashboard."""

    api: str = Field(..., description="API health status: ok, warning, or critical.")
    latency_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Most recent /health probe latency in milliseconds.",
    )
    queue_depth: int = Field(..., ge=0, description="Jobs waiting in the inference queue.")
    requests_today: int = Field(..., ge=0, description="Successful requests logged today (UTC).")
    gpu_memory_pct: Optional[float] = Field(
        None,
        ge=0,
        description="Shared GPU memory utilization percentage.",
    )


class SystemHealthSnapshot(BaseModel):
    """Host-level health metrics for the admin dashboard."""

    gpu_memory_pct: Optional[float] = Field(
        None,
        ge=0,
        description="GPU memory utilization percentage.",
    )
    disk_pct: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Root filesystem utilization percentage.",
    )
    redis: str = Field(..., description="Redis health status.")
    backup_hours_ago: Optional[float] = Field(
        None,
        ge=0,
        description="Hours since the last successful backup.",
    )


class AdminHealthResponse(BaseModel):
    """Aggregated health snapshot across all client APIs."""

    timestamp: str = Field(..., description="UTC timestamp when the snapshot was generated.")
    overall: str = Field(..., description="Worst status across all checks: ok, warning, or critical.")
    clients: dict[str, ClientHealthSnapshot] = Field(
        ...,
        description="Per-project health metrics keyed by project name.",
    )
    system: SystemHealthSnapshot = Field(..., description="Shared system health metrics.")


class FineTuneStatusResponse(BaseModel):
    """Model and adapter readiness for a project."""

    status: str = Field(..., description="Overall readiness: ok, degraded, or not_ready.")
    model: str = Field(..., description="Base model identifier.")
    adapter: Optional[str] = Field(None, description="Project or adapter label when configured.")
    adapter_path: Optional[str] = Field(None, description="Resolved adapter directory path.")
    adapter_on_disk: bool = Field(..., description="Whether the adapter directory exists.")
    adapter_config_present: bool = Field(
        ...,
        description="Whether adapter_config.json is present on disk.",
    )
    model_loaded: bool = Field(..., description="Whether vLLM finished loading the model.")
    gpu_available: bool = Field(..., description="Whether a CUDA device is available.")
    uptime_seconds: float = Field(..., ge=0, description="Process uptime in seconds.")
