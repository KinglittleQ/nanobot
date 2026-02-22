"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import asyncio
from typing import Any

import json_repair
from loguru import logger
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2  # seconds, exponential backoff: 2, 4, 8


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self._cache_enabled = "cache" in default_model.lower()

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7) -> LLMResponse:
        effective_model = model or self.default_model

        # Inject cache_control breakpoints for Anthropic prompt caching
        if self._cache_enabled or "cache" in (effective_model or "").lower():
            messages = self._inject_cache_control(messages)

        kwargs: dict[str, Any] = {"model": effective_model, "messages": messages,
                                  "max_tokens": max(1, max_tokens), "temperature": temperature}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._parse(await self._client.chat.completions.create(**kwargs))
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{_MAX_RETRIES}): {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"LLM call failed after {_MAX_RETRIES} attempts: {e}")

        return LLMResponse(content=f"Error: {last_error}", finish_reason="error")

    @staticmethod
    def _inject_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject Anthropic cache_control breakpoints for prompt caching.

        Adds ``cache_control: {type: "ephemeral"}`` to:
        1. The system message — caches the static system prompt.
        2. The second-to-last message — caches the conversation history prefix.

        Converts string content to array-of-blocks format as required by
        the Anthropic cache_control API.
        """
        if not messages:
            return messages

        messages = [m.copy() for m in messages]
        cache_marker = {"type": "ephemeral"}

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, "cache_control": cache_marker}
                ]
            elif isinstance(content, list) and content:
                last = content[-1].copy() if isinstance(content[-1], dict) else {"type": "text", "text": str(content[-1])}
                last["cache_control"] = cache_marker
                msg["content"] = content[:-1] + [last]
            return msg

        # Breakpoint 1: system message
        if messages[0].get("role") == "system":
            messages[0] = _mark(messages[0])

        # Breakpoint 2: second-to-last message (history prefix boundary)
        if len(messages) >= 3:
            messages[-2] = _mark(messages[-2])

        return messages

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        usage: dict[str, Any] = {}
        if u:
            usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }
            # Parse Anthropic prompt cache stats from response
            cache_created = getattr(u, "cache_creation_input_tokens", None)
            cache_read = None
            ptd = getattr(u, "prompt_tokens_details", None)
            if ptd:
                cache_read = getattr(ptd, "cached_tokens", None)
            if cache_created or cache_read:
                usage["cache_creation_input_tokens"] = cache_created or 0
                usage["cache_read_input_tokens"] = cache_read or 0
                logger.info(
                    f"Prompt cache: created={cache_created or 0}, "
                    f"read={cache_read or 0}, prompt={u.prompt_tokens}"
                )

        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=getattr(msg, "reasoning_content", None),
        )

    def get_default_model(self) -> str:
        return self.default_model
