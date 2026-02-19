"""Per-coroutine tool context using contextvars.

When processing messages concurrently, each coroutine needs its own
channel/chat_id/sender_id context so that tools like message, spawn,
and cron route responses to the correct session, and protected file
operations can check the caller's identity.
"""

import json
import contextvars
from pathlib import Path

_tool_channel: contextvars.ContextVar[str] = contextvars.ContextVar("tool_channel", default="")
_tool_chat_id: contextvars.ContextVar[str] = contextvars.ContextVar("tool_chat_id", default="")
_tool_sender_id: contextvars.ContextVar[str] = contextvars.ContextVar("tool_sender_id", default="")


def set_tool_context(channel: str, chat_id: str, sender_id: str = "") -> None:
    """Set the tool context for the current coroutine."""
    _tool_channel.set(channel)
    _tool_chat_id.set(chat_id)
    _tool_sender_id.set(sender_id)


def get_tool_channel() -> str:
    """Get the current tool channel."""
    return _tool_channel.get()


def get_tool_chat_id() -> str:
    """Get the current tool chat_id."""
    return _tool_chat_id.get()


def get_tool_sender_id() -> str:
    """Get the current tool sender_id."""
    return _tool_sender_id.get()


def get_sender_display_name() -> str:
    """Get the display name for the current sender from admin.json mapping.

    Falls back to the raw sender_id if no mapping is found.
    """
    sender_id = get_tool_sender_id()
    if not sender_id:
        return ""
    try:
        config_path = Path.home() / ".nanobot" / "workspace" / "admin.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            name = config.get("user_names", {}).get(sender_id)
            if name:
                return name
    except Exception:
        pass
    return sender_id
