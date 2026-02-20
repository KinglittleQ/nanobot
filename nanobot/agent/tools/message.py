"""Message tool for sending messages to users."""

from typing import Any, Callable, Awaitable

from nanobot.agent.tools.base import Tool
from nanobot.agent.tool_context import get_tool_channel, get_tool_chat_id, get_tool_reply_to
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels.

    When sending multiple messages in a single turn, the first message
    creates a thread and subsequent messages automatically reply under it.
    """

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        send_and_wait: Callable[[OutboundMessage, float], Awaitable[dict]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._send_and_wait = send_and_wait
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        # Tracks the first message sent in a proactive (non-reply) context
        # so that follow-up messages go into the same thread.
        self._thread_root_id: str | None = None

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context (legacy, prefer contextvars)."""
        self._default_channel = channel
        self._default_chat_id = chat_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def reset_thread(self) -> None:
        """Reset the thread root so the next message starts a new thread."""
        self._thread_root_id = None

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send",
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)",
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID",
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        channel = channel or get_tool_channel() or self._default_channel
        chat_id = chat_id or get_tool_chat_id() or self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        # Determine reply_to:
        # 1. From tool context (replying to user's message) — highest priority
        # 2. From self._thread_root_id (follow-up in a proactive thread)
        reply_to = get_tool_reply_to() or self._thread_root_id or None

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            reply_to=reply_to,
            media=media or [],
        )

        try:
            # If we have send_and_wait and this is the first proactive message,
            # use it to capture the sent_message_id for threading.
            if self._send_and_wait and not reply_to:
                metadata = await self._send_and_wait(msg, 15.0)
                sent_id = metadata.get("sent_message_id")
                if sent_id:
                    self._thread_root_id = sent_id
            else:
                await self._send_callback(msg)
                # If msg was sent via send_callback and metadata was populated
                sent_id = msg.metadata.get("sent_message_id")
                if sent_id and not self._thread_root_id and not get_tool_reply_to():
                    self._thread_root_id = sent_id

            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
