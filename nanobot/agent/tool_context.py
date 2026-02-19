"""Per-coroutine tool context using contextvars.

When processing messages concurrently, each coroutine needs its own
channel/chat_id context so that tools like message, spawn, and cron
route responses to the correct session.
"""

import contextvars

_tool_channel: contextvars.ContextVar[str] = contextvars.ContextVar("tool_channel", default="")
_tool_chat_id: contextvars.ContextVar[str] = contextvars.ContextVar("tool_chat_id", default="")


def set_tool_context(channel: str, chat_id: str) -> None:
    """Set the tool context for the current coroutine."""
    _tool_channel.set(channel)
    _tool_chat_id.set(chat_id)


def get_tool_channel() -> str:
    """Get the current tool channel."""
    return _tool_channel.get()


def get_tool_chat_id() -> str:
    """Get the current tool chat_id."""
    return _tool_chat_id.get()
