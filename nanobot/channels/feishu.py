"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import re
import threading
from collections import OrderedDict
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateFileRequest,
        CreateFileRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        Emoji,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


def _extract_post_text(content_json: dict) -> str:
    """Extract plain text from Feishu post (rich text) message content.
    
    Supports two formats:
    1. Direct format: {"title": "...", "content": [...]}
    2. Localized format: {"zh_cn": {"title": "...", "content": [...]}}
    """
    def extract_from_lang(lang_content: dict) -> str | None:
        if not isinstance(lang_content, dict):
            return None
        title = lang_content.get("title", "")
        content_blocks = lang_content.get("content", [])
        if not isinstance(content_blocks, list):
            return None
        text_parts = []
        if title:
            text_parts.append(title)
        for block in content_blocks:
            if not isinstance(block, list):
                continue
            for element in block:
                if isinstance(element, dict):
                    tag = element.get("tag")
                    if tag == "text":
                        text_parts.append(element.get("text", ""))
                    elif tag == "a":
                        text_parts.append(element.get("text", ""))
                    elif tag == "at":
                        text_parts.append(f"@{element.get('user_name', 'user')}")
        return " ".join(text_parts).strip() if text_parts else None
    
    # Try direct format first
    if "content" in content_json:
        result = extract_from_lang(content_json)
        if result:
            return result
    
    # Try localized format
    for lang_key in ("zh_cn", "en_us", "ja_jp"):
        lang_content = content_json.get(lang_key)
        result = extract_from_lang(lang_content)
        if result:
            return result
    
    return ""


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.
    
    Uses WebSocket to receive events - no public IP or webhook required.
    
    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """
    
    name = "feishu"
    
    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._loop: asyncio.AbstractEventLoop | None = None
        self._bot_open_id: str | None = None  # Bot's own open_id for mention detection
    
    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return
        
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        
        # Fetch bot's own open_id for mention detection in group chats
        await self._fetch_bot_open_id()
        
        # Create event handler (only register message receive, ignore other events)
        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        ).build()
        
        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )
        
        # Start WebSocket client in a separate thread with reconnect loop
        def run_ws():
            while self._running:
                try:
                    self._ws_client.start()
                except Exception as e:
                    logger.warning(f"Feishu WebSocket error: {e}")
                if self._running:
                    import time; time.sleep(5)
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def _fetch_bot_open_id(self) -> None:
        """Fetch the bot's own open_id via Feishu API for mention detection."""
        import requests
        try:
            loop = asyncio.get_running_loop()
            def _fetch():
                # Get tenant access token
                resp = requests.post(
                    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                    json={"app_id": self.config.app_id, "app_secret": self.config.app_secret},
                    timeout=10,
                )
                token = resp.json().get("tenant_access_token")
                if not token:
                    return None
                # Get bot info
                resp2 = requests.get(
                    "https://open.feishu.cn/open-apis/bot/v3/info",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10,
                )
                data = resp2.json()
                if data.get("code") == 0:
                    return data.get("bot", {}).get("open_id")
                return None
            
            self._bot_open_id = await loop.run_in_executor(None, _fetch)
            if self._bot_open_id:
                logger.info(f"Bot open_id: {self._bot_open_id}")
            else:
                logger.warning("Failed to fetch bot open_id; group mention filtering disabled")
        except Exception as e:
            logger.warning(f"Error fetching bot open_id: {e}")

    def _is_bot_mentioned(self, message) -> bool:
        """Check if the bot is mentioned in the message."""
        mentions = getattr(message, "mentions", None)
        if not mentions:
            return False
        for mention in mentions:
            mid = getattr(mention, "id", None)
            if mid:
                open_id = getattr(mid, "open_id", None)
                if open_id and open_id == self._bot_open_id:
                    return True
        return False

    def _strip_bot_mention(self, content: str, message) -> str:
        """Remove the bot @mention placeholder from message content."""
        mentions = getattr(message, "mentions", None)
        if not mentions:
            return content
        for mention in mentions:
            mid = getattr(mention, "id", None)
            if mid:
                open_id = getattr(mid, "open_id", None)
                if open_id and open_id == self._bot_open_id:
                    key = getattr(mention, "key", None)
                    if key:
                        content = content.replace(key, "").strip()
        return content

    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket client: {e}")
        logger.info("Feishu bot stopped")
    
    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()
            
            response = self._client.im.v1.message_reaction.create(request)
            
            if not response.success():
                logger.warning(f"Failed to add reaction: code={response.code}, msg={response.msg}")
            else:
                logger.debug(f"Added {emoji_type} reaction to message {message_id}")
        except Exception as e:
            logger.warning(f"Error adding reaction: {e}")

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> None:
        """
        Add a reaction emoji to a message (non-blocking).
        
        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client or not Emoji:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)
    
    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    _CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            return None
        split = lambda l: [c.strip() for c in l.strip("|").split("|")]
        headers = split(lines[0])
        rows = [split(l) for l in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into div/markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()]
            if before.strip():
                elements.extend(self._split_headings(before))
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:]
        if remaining.strip():
            elements.extend(self._split_headings(remaining))
        return elements or [{"tag": "markdown", "content": content}]

    def _split_headings(self, content: str) -> list[dict]:
        """Split content by headings, converting headings to div elements."""
        protected = content
        code_blocks = []
        for m in self._CODE_BLOCK_RE.finditer(content):
            code_blocks.append(m.group(1))
            protected = protected.replace(m.group(1), f"\x00CODE{len(code_blocks)-1}\x00", 1)

        elements = []
        last_end = 0
        for m in self._HEADING_RE.finditer(protected):
            before = protected[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            level = len(m.group(1))
            text = m.group(2).strip()
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**{text}**",
                },
            })
            last_end = m.end()
        remaining = protected[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})

        for i, cb in enumerate(code_blocks):
            for el in elements:
                if el.get("tag") == "markdown":
                    el["content"] = el["content"].replace(f"\x00CODE{i}\x00", cb)

        return elements or [{"tag": "markdown", "content": content}]

    # Image file extensions
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif"}

    # Audio file extensions (Feishu only supports opus for audio messages)
    _AUDIO_EXTS = {".opus"}

    # Video file extensions
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}

    # File type mapping for Feishu file upload API
    _FILE_TYPE_MAP = {
        ".opus": "opus", ".mp4": "mp4", ".pdf": "pdf", ".doc": "doc", ".docx": "doc",
        ".xls": "xls", ".xlsx": "xls", ".ppt": "ppt", ".pptx": "ppt",
    }

    def _upload_image_sync(self, file_path: str) -> str | None:
        """Upload an image to Feishu and return the image_key."""
        import os
        try:
            with open(file_path, "rb") as f:
                request = CreateImageRequest.builder() \
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    ).build()
                response = self._client.im.v1.image.create(request)
                if response.success():
                    image_key = response.data.image_key
                    logger.debug(f"Uploaded image {os.path.basename(file_path)}: {image_key}")
                    return image_key
                else:
                    logger.error(f"Failed to upload image: code={response.code}, msg={response.msg}")
                    return None
        except Exception as e:
            logger.error(f"Error uploading image {file_path}: {e}")
            return None

    def _upload_file_sync(self, file_path: str) -> str | None:
        """Upload a file to Feishu and return the file_key."""
        import os
        ext = os.path.splitext(file_path)[1].lower()
        file_type = self._FILE_TYPE_MAP.get(ext, "stream")
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                request = CreateFileRequest.builder() \
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(file_name)
                        .file(f)
                        .build()
                    ).build()
                response = self._client.im.v1.file.create(request)
                if response.success():
                    file_key = response.data.file_key
                    logger.debug(f"Uploaded file {file_name}: {file_key}")
                    return file_key
                else:
                    logger.error(f"Failed to upload file: code={response.code}, msg={response.msg}")
                    return None
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            return None

    def _convert_to_mp4(self, file_path: str) -> str:
        """Convert a video file to mp4 format if needed. Returns path to mp4 file."""
        import os
        import subprocess
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".mp4":
            return file_path
        mp4_path = os.path.splitext(file_path)[0] + ".mp4"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", file_path, "-c:v", "libx264", "-c:a", "aac", mp4_path],
                capture_output=True, check=True, timeout=120,
            )
            logger.debug(f"Converted {file_path} to {mp4_path}")
            return mp4_path
        except Exception as e:
            logger.error(f"Failed to convert video to mp4: {e}")
            return file_path

    def _extract_video_thumbnail(self, file_path: str) -> str | None:
        """Extract a thumbnail from a video file. Returns path to thumbnail image."""
        import os
        import subprocess
        thumb_path = os.path.splitext(file_path)[0] + "_thumb.png"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", file_path, "-ss", "00:00:01", "-vframes", "1", thumb_path],
                capture_output=True, check=True, timeout=30,
            )
            if os.path.isfile(thumb_path) and os.path.getsize(thumb_path) > 0:
                logger.debug(f"Extracted thumbnail: {thumb_path}")
                return thumb_path
            return None
        except Exception as e:
            logger.warning(f"Failed to extract video thumbnail: {e}")
            return None

    def _send_message_sync(self, receive_id_type: str, receive_id: str, msg_type: str, content: str) -> bool:
        """Send a single message (text/image/file/interactive) synchronously."""
        try:
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    f"Failed to send Feishu {msg_type} message: code={response.code}, "
                    f"msg={response.msg}, log_id={response.get_log_id()}"
                )
                return False
            logger.debug(f"Feishu {msg_type} message sent to {receive_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending Feishu {msg_type} message: {e}")
            return False

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu, including media (images/files) if present."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        try:
            import os

            # Determine receive_id_type based on chat_id format
            # open_id starts with "ou_", chat_id starts with "oc_"
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"

            loop = asyncio.get_running_loop()

            # --- Send media attachments first ---
            if msg.media:
                for file_path in msg.media:
                    if not os.path.isfile(file_path):
                        logger.warning(f"Media file not found: {file_path}")
                        continue

                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in self._IMAGE_EXTS:
                        # Upload and send as image
                        image_key = await loop.run_in_executor(None, self._upload_image_sync, file_path)
                        if image_key:
                            content = json.dumps({"image_key": image_key})
                            await loop.run_in_executor(
                                None, self._send_message_sync,
                                receive_id_type, msg.chat_id, "image", content,
                            )
                    elif ext in self._AUDIO_EXTS:
                        # Upload and send as audio (voice message)
                        file_key = await loop.run_in_executor(None, self._upload_file_sync, file_path)
                        if file_key:
                            content = json.dumps({"file_key": file_key})
                            await loop.run_in_executor(
                                None, self._send_message_sync,
                                receive_id_type, msg.chat_id, "audio", content,
                            )
                    elif ext in self._VIDEO_EXTS:
                        # Convert to mp4 if needed, extract thumbnail, upload and send as media (video)
                        mp4_path = await loop.run_in_executor(None, self._convert_to_mp4, file_path)
                        # Extract thumbnail for video cover
                        thumb_path = await loop.run_in_executor(None, self._extract_video_thumbnail, mp4_path)
                        image_key = ""
                        if thumb_path:
                            image_key = await loop.run_in_executor(None, self._upload_image_sync, thumb_path) or ""
                            # Clean up thumbnail
                            try:
                                os.remove(thumb_path)
                            except OSError:
                                pass
                        # Upload video file
                        file_key = await loop.run_in_executor(None, self._upload_file_sync, mp4_path)
                        if file_key:
                            media_content = {"file_key": file_key}
                            if image_key:
                                media_content["image_key"] = image_key
                            content = json.dumps(media_content)
                            await loop.run_in_executor(
                                None, self._send_message_sync,
                                receive_id_type, msg.chat_id, "media", content,
                            )
                        # Clean up converted file
                        if mp4_path != file_path:
                            try:
                                os.remove(mp4_path)
                            except OSError:
                                pass
                    else:
                        # Upload and send as file
                        file_key = await loop.run_in_executor(None, self._upload_file_sync, file_path)
                        if file_key:
                            content = json.dumps({"file_key": file_key})
                            await loop.run_in_executor(
                                None, self._send_message_sync,
                                receive_id_type, msg.chat_id, "file", content,
                            )

            # --- Send text content (if any) ---
            if msg.content and msg.content.strip():
                # Build card with markdown + table support
                elements = self._build_card_elements(msg.content)
                card = {
                    "config": {"wide_screen_mode": True},
                    "elements": elements,
                }
                content = json.dumps(card, ensure_ascii=False)
                await loop.run_in_executor(
                    None, self._send_message_sync,
                    receive_id_type, msg.chat_id, "interactive", content,
                )

        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")
    
    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)
    
    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            
            # Trim cache: keep most recent 500 when exceeds 1000
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)
            
            # Skip bot messages
            sender_type = sender.sender_type
            if sender_type == "bot":
                return
            
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type
            
            # In group chats, only respond when the bot is @mentioned
            if chat_type == "group" and self._bot_open_id:
                if not self._is_bot_mentioned(message):
                    logger.debug(f"Ignoring group message without bot mention: {message_id}")
                    return
            
            # Add reaction to indicate "seen"
            await self._add_reaction(message_id, "THUMBSUP")
            
            # Parse message content
            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except json.JSONDecodeError:
                    content = message.content or ""
            elif msg_type == "post":
                try:
                    content_json = json.loads(message.content)
                    content = _extract_post_text(content_json)
                except (json.JSONDecodeError, TypeError):
                    content = message.content or ""
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")
            
            if not content:
                return
            
            # Strip bot @mention placeholder from content
            if chat_type == "group":
                content = self._strip_bot_mention(content, message)
                if not content:
                    return
            
            # Forward to message bus
            reply_to = chat_id if chat_type == "group" else sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
