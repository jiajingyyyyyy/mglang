"""OpenAI-compatible client helpers for local SGLang MotifAgent calls."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:30003/v1"
DEFAULT_API_KEY = "EMPTY"


Message = Mapping[str, Any]


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def default_base_url() -> str:
    """Return the OpenAI-compatible endpoint for the local SGLang server."""

    return _normalize_base_url(
        _env("SGLANG_OPENAI_BASE_URL")
        or _env("OPENAI_BASE_URL")
        or _env("SGLANG_BASE_URL")
        or DEFAULT_BASE_URL
    )


def default_api_key() -> str:
    """Return the API key accepted by a local SGLang server."""

    return _env("SGLANG_API_KEY") or _env("OPENAI_API_KEY") or DEFAULT_API_KEY


def default_model() -> str:
    """Return the local SGLang model name used by MotifAgent."""

    return _env("SGLANG_MODEL") or _env("MOTIF_SGLANG_MODEL") or "qwen3-14b"


def qwen_thinking_extra_body(enable_thinking: bool = False) -> dict[str, Any]:
    """Build SGLang/Qwen chat template options for thinking on/off control."""

    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}


def merge_extra_body(*parts: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Merge shallow OpenAI ``extra_body`` dictionaries, ignoring empty parts."""

    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(dict(part))
    return merged or None


@dataclass(slots=True)
class MotifSGLangConfig:
    """Configuration for local OpenAI-compatible SGLang calls."""

    base_url: str = field(default_factory=default_base_url)
    api_key: str = field(default_factory=default_api_key)
    model: str = field(default_factory=default_model)
    timeout: float = 120.0
    max_retries: int = 0
    temperature: float = 0.0
    max_tokens: int = 1024
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(self.base_url)

    @property
    def litellm_model(self) -> str:
        """Model name that routes LiteLLM to an OpenAI-compatible endpoint."""

        if self.model.startswith("openai/"):
            return self.model
        return f"openai/{self.model}"

    def litellm_kwargs(self, messages: Iterable[Message], **overrides: Any) -> dict[str, Any]:
        """Build kwargs for ``litellm.completion`` against the local SGLang server."""

        extra_body = merge_extra_body(
            qwen_thinking_extra_body(self.enable_thinking),
            overrides.pop("extra_body", None),
        )
        kwargs = {
            "model": self.litellm_model,
            "custom_llm_provider": "openai",
            "api_base": self.base_url,
            "api_key": self.api_key,
            "messages": list(messages),
            "temperature": overrides.pop("temperature", self.temperature),
            "max_tokens": overrides.pop("max_tokens", self.max_tokens),
            "timeout": overrides.pop("timeout", self.timeout),
            "num_retries": overrides.pop("num_retries", self.max_retries),
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        kwargs.update(overrides)
        return kwargs


class MotifSGLangOpenAIClient:
    """Small OpenAI SDK wrapper for MotifAgent algorithm-layer inference."""

    def __init__(self, config: MotifSGLangConfig | None = None, **overrides: Any) -> None:
        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both.")
        self.config = config or MotifSGLangConfig(**overrides)
        self._client = None

    @property
    def client(self) -> Any:
        """Lazily construct and return ``openai.OpenAI``."""

        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def chat(
        self,
        messages: Iterable[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        stream: bool = False,
        extra_body: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call ``/v1/chat/completions`` through the OpenAI SDK."""

        thinking_enabled = self.config.enable_thinking if enable_thinking is None else enable_thinking
        request_extra_body = merge_extra_body(
            qwen_thinking_extra_body(thinking_enabled),
            extra_body,
        )
        return self.client.chat.completions.create(
            model=model or self.config.model,
            messages=list(messages),
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stream=stream,
            extra_body=request_extra_body,
            **kwargs,
        )

    def complete_text(self, messages: Iterable[Message], **kwargs: Any) -> str:
        """Return the first assistant message content as plain text."""

        response = self.chat(messages, stream=False, **kwargs)
        if not getattr(response, "choices", None):
            return ""
        message = response.choices[0].message
        return getattr(message, "content", None) or ""

    def stream_text(self, messages: Iterable[Message], **kwargs: Any) -> Iterator[str]:
        """Yield text deltas from a streaming chat completion."""

        stream = self.chat(messages, stream=True, **kwargs)
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield text

    def completion(self, messages: Iterable[Message], **kwargs: Any) -> str:
        """MotifAgent-style injected ``llm_client.completion`` adapter."""

        return self.complete_text(messages, **kwargs)

    def litellm_kwargs(self, messages: Iterable[Message], **overrides: Any) -> dict[str, Any]:
        """Expose LiteLLM kwargs with the same local SGLang config."""

        return self.config.litellm_kwargs(messages, **overrides)


def build_client(**overrides: Any) -> MotifSGLangOpenAIClient:
    """Convenience factory for MotifAgent code paths."""

    return MotifSGLangOpenAIClient(**overrides)


__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "MotifSGLangConfig",
    "MotifSGLangOpenAIClient",
    "build_client",
    "default_api_key",
    "default_base_url",
    "default_model",
    "merge_extra_body",
    "qwen_thinking_extra_body",
]
