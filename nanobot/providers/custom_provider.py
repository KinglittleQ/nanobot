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

        # Sanitize tool call IDs in message history for Claude compatibility
        messages = self._sanitize_tool_call_ids_in_messages(messages)

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

        Adds ``cache_control: {type: "ephemeral"}`` to up to 3 positions:
        1. The system message — caches the static system prompt.
        2. The message just before the last user message — caches the
           conversation history prefix so it survives across user turns.
        3. The second-to-last message — during tool-call loops, this caches
           everything up to the previous round so only the latest tool result
           is uncached.

        During a tool-call loop the message list grows:
          Round 1: [sys*] [history...] [prev*] [user_new] → BP3 on user_new
          Round 2: [sys*] [history...] [prev*] [user_new] [asst+tools] [tool_result*] → BP3 on asst+tools
          Round 3: [sys*] [history...] [prev*] [user_new] [asst+tools] [tool_result] [asst+tools] [tool_result*]

        BP3 moves with each round, maximizing cache hits during tool loops.
        Anthropic allows up to 4 breakpoints; we use 3.
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
        bp1_idx = None
        if messages[0].get("role") == "system":
            messages[0] = _mark(messages[0])
            bp1_idx = 0

        # Breakpoint 2: the message just before the last user message.
        # Find the last user message, then mark the one before it.
        bp2_idx = None
        last_user_idx = None
        for i in range(len(messages) - 1, 0, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx and last_user_idx >= 2:
            bp2_idx = last_user_idx - 1
            messages[bp2_idx] = _mark(messages[bp2_idx])

        # Breakpoint 3: second-to-last message (for tool-call loop caching).
        # Only add if it's a different position from BP1 and BP2, and there
        # are at least 2 messages after the last user message (i.e. we're
        # in a tool-call loop).
        if len(messages) >= 3:
            bp3_idx = len(messages) - 2
            if bp3_idx != bp1_idx and bp3_idx != bp2_idx:
                messages[bp3_idx] = _mark(messages[bp3_idx])

        return messages

    @staticmethod
    def _sanitize_tool_call_id(tool_call_id: str) -> str:
        """Sanitize tool call ID to match Claude's required pattern: ^[a-zA-Z0-9_-]+$
        
        Claude requires tool call IDs to only contain alphanumeric characters,
        underscores, and hyphens. This method replaces invalid characters.
        """
        import re
        # Replace any character that doesn't match the allowed pattern with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(tool_call_id))
        return sanitized

    @classmethod
    def _sanitize_tool_call_ids_in_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize all tool call IDs in message history.
        
        This ensures that any tool_calls in assistant messages and tool_call_id
        in tool messages conform to Claude's ID pattern requirements.
        """
        sanitized_messages = []
        for msg in messages:
            msg_copy = msg.copy()
            role = msg_copy.get("role")
            
            # Sanitize tool_calls in assistant messages
            if role == "assistant" and msg_copy.get("tool_calls"):
                msg_copy["tool_calls"] = [
                    {
                        **tc,
                        "id": cls._sanitize_tool_call_id(tc.get("id", "")),
                        "function": {
                            **tc.get("function", {}),
                            "arguments": tc.get("function", {}).get("arguments", "{}")
                        }
                    }
                    for tc in msg_copy["tool_calls"]
                ]
            
            # Sanitize tool_call_id in tool messages
            if role == "tool" and msg_copy.get("tool_call_id"):
                msg_copy["tool_call_id"] = cls._sanitize_tool_call_id(msg_copy["tool_call_id"])
            
            sanitized_messages.append(msg_copy)
        
        return sanitized_messages

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=self._sanitize_tool_call_id(tc.id), name=tc.function.name,
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
