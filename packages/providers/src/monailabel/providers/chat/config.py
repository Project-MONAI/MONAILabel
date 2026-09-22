"""Startup configuration. Credentials are environment references, never arguments."""

import os
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from monailabel.core.models import Contract

from .local_models import LOCAL_MODELS


class CoordinatorConfig(Contract):
    provider: Literal["local", "openai", "anthropic", "gemini", "compatible"] = "local"
    variant: Literal["4b", "9b", "lightning"] = "lightning"
    model: str | None = None
    base_url: str | None = None
    key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    timeout: float = Field(default=90, gt=0, le=300)
    max_tokens: int = Field(default=4096, ge=256, le=8192)
    thinking: bool | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    gpu: str = Field(default="0", pattern=r"^(?:[0-9]+|GPU-[a-fA-F0-9-]+)$")
    gpu_memory_utilization: float | None = Field(default=None, ge=0.05, le=0.95)

    @model_validator(mode="before")
    @classmethod
    def default_budget(cls, value: Any) -> Any:
        if isinstance(value, dict) and "max_tokens" not in value:
            lightning = (
                value.get("provider", "local") == "local"
                and value.get("variant", "lightning") == "lightning"
            )
            return {**value, "max_tokens": 8192 if lightning else 4096}
        return value

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        if self.provider != "local" and not self.model:
            raise ValueError("Set --assistant-model for a hosted or existing endpoint.")
        if self.provider == "compatible" and not self.base_url:
            raise ValueError("Set --assistant-base-url for an OpenAI-compatible endpoint.")
        if self.provider == "local" and (self.base_url or self.model or self.key_env):
            raise ValueError("Use --assistant compatible for an existing model server.")
        if self.base_url:
            url = urlsplit(self.base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "Use an HTTP(S) API base URL without credentials or query parameters."
                )
            if url.scheme != "https" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Remote coordinator endpoints require HTTPS.")
        return self

    @property
    def model_name(self) -> str:
        return self.model or LOCAL_MODELS[self.variant].name

    @property
    def endpoint(self) -> str:
        return (
            self.base_url
            or {
                "local": f"http://127.0.0.1:{LOCAL_MODELS[self.variant].port}/v1",
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com/v1",
                "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
            }.get(self.provider, "")
        ).rstrip("/")

    @property
    def credential_env(self) -> str | None:
        return self.key_env or {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }.get(self.provider)

    @classmethod
    def from_env(cls, **overrides: object) -> "CoordinatorConfig":
        values = {
            field: os.environ[name]
            for field in cls.model_fields
            if (name := "MONAILABEL_ASSISTANT_" + field.upper()) in os.environ
        }
        return cls.model_validate({**values, **overrides})
