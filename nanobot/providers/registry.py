"""
Provider Registry — single source of truth for LLM provider metadata.

Adding a new provider:
  1. Add a ProviderSpec to PROVIDERS below.
  2. Add a field to ProvidersConfig in config/schema.py.
  Done. Env vars, prefixing, config matching, status display all derive from here.

Order matters — it controls match priority and fallback. Custom first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Model catalog — context windows and pricing
# ---------------------------------------------------------------------------

# Default context window sizes for known models (in tokens).
# Used to trigger memory consolidation when prompt_tokens exceeds 80% of the window.
# Keys are substring-matched against model name (lowercase).
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Cloudsway proxy models (actual limits differ from official)
    "cloudsway-claude-opus-4.6-cache-1m": 1_000_000,
    "cloudsway-claude-opus-4.6-cache": 200_000,
    "cloudsway-claude-sonnet-4.6-cache": 200_000,
    # Claude models
    "-1m": 1_000_000,               # Any model with -1M suffix
    "claude-opus-4.6-cache": 1_000_000,
    "claude-sonnet-4.6-cache": 200_000,
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku": 200_000,
    # OpenAI models
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    # DeepSeek models
    "deepseek": 64_000,
    # Zhipu models
    "glm-5": 128_000,
    "glm-4": 128_000,
    # Gemini models
    "gemini-2": 1_000_000,
    "gemini-1.5": 1_000_000,
    # Qwen models
    "qwen": 128_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000  # Fallback for unknown models

# Model pricing per million tokens (USD).
# Keys are substring-matched against model name (lowercase).
# Format: {key: (input, output, cache_write, cache_read)}
# cache_write/cache_read are None if caching is not supported.
MODEL_PRICING: dict[str, tuple[float, float, float | None, float | None]] = {
    # Claude models (Anthropic pricing)
    "claude-opus-4.6-cache-1m": (10.0,  37.5,  12.5,  1.0),
    "claude-opus-4.6":          (5.0,   25.0,  6.25,  0.50),
    "claude-opus-4.5":          (5.0,   25.0,  6.25,  0.50),
    "claude-opus-4":            (15.0,  75.0,  18.75, 1.50),
    "claude-sonnet-4":          (3.0,   15.0,  3.75,  0.30),
    "claude-haiku":             (0.80,  4.0,   1.0,   0.08),
    # OpenAI models
    "gpt-4o":                   (2.50,  10.0,  None,  None),
    "gpt-4o-mini":              (0.15,  0.60,  None,  None),
    "o1":                       (15.0,  60.0,  None,  None),
    "o3":                       (10.0,  40.0,  None,  None),
    # DeepSeek
    "deepseek":                 (0.27,  1.10,  None,  None),
    # Zhipu
    "glm-5":                    (5.0,   20.0,  None,  None),
    "glm-4":                    (1.0,   5.0,   None,  None),
}


def get_context_window(model: str) -> int:
    """Return context window size for a model (substring match, case-insensitive)."""
    model_lower = model.lower()
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return window
    return DEFAULT_CONTEXT_WINDOW


def get_pricing(model: str) -> tuple[float, float, float | None, float | None] | None:
    """Return (input, output, cache_write, cache_read) per MTok, or None if unknown."""
    model_lower = model.lower()
    for key, pricing in MODEL_PRICING.items():
        if key in model_lower:
            return pricing
    return None


# ---------------------------------------------------------------------------
# Model catalog — context windows and pricing
# ---------------------------------------------------------------------------

# Default context window sizes for known models (in tokens).
# Keys are substring-matched against model name (lowercase).
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Cloudsway proxy models (actual limits differ from official)
    "cloudsway-claude-opus-4.6-cache-1m": 1_000_000,
    "cloudsway-claude-opus-4.6-cache": 200_000,
    "cloudsway-claude-sonnet-4.6-cache": 200_000,
    # Claude models
    "-1m": 1_000_000,
    "claude-opus-4.6-cache": 1_000_000,
    "claude-sonnet-4.6-cache": 200_000,
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku": 200_000,
    # OpenAI models
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    # DeepSeek
    "deepseek": 64_000,
    # Zhipu
    "glm-5": 128_000,
    "glm-4": 128_000,
    # Gemini
    "gemini-2": 1_000_000,
    "gemini-1.5": 1_000_000,
    # Qwen
    "qwen": 128_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000

# Model pricing per million tokens (USD).
# Format: {key: (input, output, cache_write, cache_read)}
# cache_write/cache_read are None if caching is not supported.
MODEL_PRICING: dict[str, tuple[float, float, float | None, float | None]] = {
    "claude-opus-4.6-cache-1m": (10.0,  37.5, 12.5,  1.0),
    "claude-opus-4.6":          (5.0,   25.0, 6.25,  0.50),
    "claude-opus-4.5":          (5.0,   25.0, 6.25,  0.50),
    "claude-opus-4":            (15.0,  75.0, 18.75, 1.50),
    "claude-sonnet-4":          (3.0,   15.0, 3.75,  0.30),
    "claude-haiku":             (0.80,  4.0,  1.0,   0.08),
    "gpt-4o":                   (2.50,  10.0, None,  None),
    "gpt-4o-mini":              (0.15,  0.60, None,  None),
    "o1":                       (15.0,  60.0, None,  None),
    "o3":                       (10.0,  40.0, None,  None),
    "deepseek":                 (0.27,  1.10, None,  None),
    "glm-5":                    (5.0,   20.0, None,  None),
    "glm-4":                    (1.0,   5.0,  None,  None),
}


def get_context_window(model: str) -> int:
    """Return context window size for the given model name (substring match)."""
    model_lower = model.lower()
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return window
    return DEFAULT_CONTEXT_WINDOW


def get_pricing(model: str) -> tuple[float, float, float | None, float | None] | None:
    """Return (input, output, cache_write, cache_read) per MTok, or None if unknown."""
    model_lower = model.lower()
    for key, pricing in MODEL_PRICING.items():
        if key in model_lower:
            return pricing
    return None


@dataclass(frozen=True)
class ProviderSpec:
    """One LLM provider's metadata."""

    # identity
    name: str                       # config field name, e.g. "deepseek"
    keywords: tuple[str, ...]       # model-name keywords for matching (lowercase)
    env_key: str                    # LiteLLM env var
    display_name: str = ""          # shown in `nanobot status`

    # model prefixing
    litellm_prefix: str = ""
    skip_prefixes: tuple[str, ...] = ()

    # extra env vars, e.g. (("ZHIPUAI_API_KEY", "{api_key}"),)
    env_extras: tuple[tuple[str, str], ...] = ()

    # gateway / local detection
    is_gateway: bool = False
    is_local: bool = False
    detect_by_key_prefix: str = ""
    detect_by_base_keyword: str = ""
    default_api_base: str = ""

    # gateway behavior
    strip_model_prefix: bool = False

    # per-model param overrides
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()

    # OAuth-based providers
    is_oauth: bool = False

    # Direct providers bypass LiteLLM entirely
    is_direct: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()


# ---------------------------------------------------------------------------
# PROVIDERS — the registry. Order = priority.
# ---------------------------------------------------------------------------

PROVIDERS: tuple[ProviderSpec, ...] = (

    # === Custom (direct OpenAI-compatible endpoint, bypasses LiteLLM) ======
    ProviderSpec(
        name="custom",
        keywords=(),
        env_key="",
        display_name="Custom",
        litellm_prefix="",
        is_direct=True,
    ),

    # === Standard providers =================================================

    # DeepSeek
    ProviderSpec(
        name="deepseek",
        keywords=("deepseek",),
        env_key="DEEPSEEK_API_KEY",
        display_name="DeepSeek",
        litellm_prefix="deepseek",
        skip_prefixes=("deepseek/",),
    ),

    # Zhipu GLM
    ProviderSpec(
        name="zhipu",
        keywords=("zhipu", "glm", "zai"),
        env_key="ZAI_API_KEY",
        display_name="Zhipu AI",
        litellm_prefix="zai",
        skip_prefixes=("zhipu/", "zai/"),
        env_extras=(
            ("ZHIPUAI_API_KEY", "{api_key}"),
        ),
    ),

    # Moonshot / Kimi
    ProviderSpec(
        name="moonshot",
        keywords=("moonshot", "kimi"),
        env_key="MOONSHOT_API_KEY",
        display_name="Moonshot",
        litellm_prefix="moonshot",
        skip_prefixes=("moonshot/",),
        env_extras=(
            ("MOONSHOT_API_BASE", "{api_base}"),
        ),
        default_api_base="https://api.moonshot.ai/v1",
        model_overrides=(
            ("kimi-k2.5", {"temperature": 1.0}),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def find_by_model(model: str) -> ProviderSpec | None:
    """Match a provider by model-name keyword (case-insensitive)."""
    model_lower = model.lower()
    for spec in PROVIDERS:
        if spec.is_gateway or spec.is_local:
            continue
        if any(kw in model_lower for kw in spec.keywords):
            return spec
    return None


def find_gateway(
    provider_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderSpec | None:
    """Detect gateway/local provider."""
    if provider_name:
        spec = find_by_name(provider_name)
        if spec and (spec.is_gateway or spec.is_local):
            return spec

    for spec in PROVIDERS:
        if spec.detect_by_key_prefix and api_key and api_key.startswith(spec.detect_by_key_prefix):
            return spec
        if spec.detect_by_base_keyword and api_base and spec.detect_by_base_keyword in api_base:
            return spec

    return None


def find_by_name(name: str) -> ProviderSpec | None:
    """Find a provider spec by config field name."""
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None
