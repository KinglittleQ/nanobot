"""Event types for the message bus."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from a chat channel."""
    
    channel: str  # telegram, discord, slack, whatsapp
    sender_id: str  # User identifier
    chat_id: str  # Chat/channel identifier
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    
    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""
    
    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional future resolved after the channel finishes sending.
    # Callers can set this and await it to get metadata back (e.g. sent_message_id).
    _done: asyncio.Future | None = field(default=None, repr=False)

    def set_done_future(self, future: asyncio.Future) -> None:
        """Attach a future that will be resolved when send completes."""
        self._done = future

    def resolve(self) -> None:
        """Mark the message as sent (resolves the _done future)."""
        if self._done and not self._done.done():
            self._done.set_result(self.metadata)
