"""Environment-backed configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import MODEL_DEFAULTS


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    candidate_models: tuple[str, ...]
    enabled: bool
    extra_headers: dict[str, str]
    inline_image_limit_kb: int
    poll_timeout_seconds: int


@dataclass
class AppConfig:
    repo_root: Path
    code_root: Path
    cache_dir: Path
    prompt_version: str
    primary_provider: ProviderConfig
    fallback_provider: ProviderConfig
    enable_live_models: bool

    @classmethod
    def from_repo(cls, repo_root: Path) -> "AppConfig":
        code_root = repo_root / "code"
        cache_dir = code_root / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        nim_key = os.getenv("NVIDIA_API_KEY", "").strip()
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        primary = ProviderConfig(
            name="nim",
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=nim_key,
            model=os.getenv("PRIMARY_MODEL", MODEL_DEFAULTS["nim"]),
            candidate_models=tuple(
                model.strip()
                for model in os.getenv(
                    "NIM_FALLBACK_MODELS",
                    "meta/llama-4-maverick-17b-128e-instruct,nvidia/nemotron-nano-12b-v2-vl",
                ).split(",")
                if model.strip()
            ),
            enabled=bool(nim_key),
            extra_headers={},
            inline_image_limit_kb=int(os.getenv("NIM_INLINE_IMAGE_LIMIT_KB", "180")),
            poll_timeout_seconds=int(os.getenv("NIM_POLL_TIMEOUT_SECONDS", "45")),
        )
        fallback = ProviderConfig(
            name="openrouter",
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=openrouter_key,
            model=os.getenv("FALLBACK_MODEL", MODEL_DEFAULTS["openrouter"]),
            candidate_models=tuple(),
            enabled=bool(openrouter_key),
            extra_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/PRADDZY/orch-0626"),
                "X-OpenRouter-Title": os.getenv("OPENROUTER_TITLE", "orch-0626"),
            },
            inline_image_limit_kb=0,
            poll_timeout_seconds=45,
        )
        return cls(
            repo_root=repo_root,
            code_root=code_root,
            cache_dir=cache_dir,
            prompt_version=os.getenv("PROMPT_VERSION", "v8"),
            primary_provider=primary,
            fallback_provider=fallback,
            enable_live_models=primary.enabled or fallback.enabled,
        )
