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

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": model or self.default_model, "messages": messages,
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

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None),
        )

    def get_default_model(self) -> str:
        return self.default_model
