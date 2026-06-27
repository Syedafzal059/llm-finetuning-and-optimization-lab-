"""API key authentication for client and admin routes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

from core.utils.config_loader import PROJECT_ROOT


@dataclass(frozen=True)
class ApiKeys:
    """Validated API keys loaded from a project environment file."""

    client_key: str
    admin_key: str


def mask_api_key(api_key: str) -> str:
    """Mask an API key for safe logging, showing only the last four characters."""
    if len(api_key) <= 4:
        return "xxxx"
    return f"xxxx{api_key[-4:]}"


def load_project_env(project_name: str) -> None:
    """Load environment variables from projects/{project_name}/.env if present."""
    env_path = PROJECT_ROOT / "projects" / project_name / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)


def load_api_keys(project_name: str) -> ApiKeys:
    """Load CLIENT_API_KEY and ADMIN_API_KEY from the project .env file."""
    load_project_env(project_name)

    client_key = os.getenv("CLIENT_API_KEY", "").strip()
    admin_key = os.getenv("ADMIN_API_KEY", "").strip()

    if not client_key:
        raise ValueError(
            f"CLIENT_API_KEY is not set in projects/{project_name}/.env"
        )
    if not admin_key:
        raise ValueError(
            f"ADMIN_API_KEY is not set in projects/{project_name}/.env"
        )

    return ApiKeys(client_key=client_key, admin_key=admin_key)


def validate_api_key(
    api_key: Optional[str],
    keys: ApiKeys,
    *,
    require_admin: bool = False,
) -> str:
    """Validate the provided API key against client or admin credentials."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "Missing X-API-Key header."},
        )

    if require_admin:
        if api_key == keys.admin_key:
            return api_key
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "Invalid or insufficient API key."},
        )

    if api_key in (keys.client_key, keys.admin_key):
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "unauthorized", "message": "Invalid or missing API key."},
    )


def require_client_key(keys: ApiKeys):
    """FastAPI dependency factory for /generate (client or admin key)."""

    async def verify(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
        return validate_api_key(x_api_key, keys, require_admin=False)

    return verify


def require_admin_key(keys: ApiKeys):
    """FastAPI dependency factory for /metrics (admin key only)."""

    async def verify(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
        return validate_api_key(x_api_key, keys, require_admin=True)

    return verify
