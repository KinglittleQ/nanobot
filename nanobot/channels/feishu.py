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
    
    Content blocks can be:
    - Nested lists: [[{tag: "text", text: "..."}, ...], ...]
    - Direct dicts: [{tag: "code_block", ...}, ...]  (e.g. code blocks)
    """
    def _extract_element(element: dict) -> str | None:
        """Extract text from a single element dict."""
        if not isinstance(element, dict):
            return None
        tag = element.get("tag")
        if tag == "text":
            return element.get("text", "")
        elif tag == "a":
            return element.get("text", "")
        elif tag == "at":
            return f"@{element.get('user_name', 'user')}"
        elif tag == "code_block":
            lang = element.get("language", "")
            code = element.get("text", "")
            return f"\n```{lang}\n{code}\n```\n"
        elif tag == "emotion":
            return element.get("emoji_type", "")
        elif tag == "img":
            return "[image]"
        elif tag == "media":
            return "[media]"
        return None

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
            if isinstance(block, list):
                # Nested list of elements: [[{tag: "text"}, ...], ...]
                for element in block:
                    text = _extract_element(element)
                    if text:
                        text_parts.append(text)
            elif isinstance(block, dict):
                # Direct element dict: {tag: "code_block", ...}
                text = _extract_element(block)
                if text:
                    text_parts.append(text)
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
        # Track message IDs sent by the bot, so we can detect thread replies
        self._bot_sent_message_ids: OrderedDict[str, None] = OrderedDict()
    
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

    def _track_bot_message(self, message_id: str) -> None:
        """Record a message sent by the bot for thread detection."""
        self._bot_sent_message_ids[message_id] = None
        # Trim to keep memory bounded
        while len(self._bot_sent_message_ids) > 500:
            self._bot_sent_message_ids.popitem(last=False)

    def _is_bot_thread(self, message) -> bool:
        """Check if the message is a reply in a thread started by (or involving) the bot."""
        root_id = getattr(message, "root_id", None)
        parent_id = getattr(message, "parent_id", None)
        if root_id and root_id in self._bot_sent_message_ids:
            return True
        if parent_id and parent_id in self._bot_sent_message_ids:
            return True
        return False

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

    def _download_image_sync(self, image_key: str) -> str | None:
        """Download an image from Feishu by image_key. Returns local file path or None."""
        import tempfile
        try:
            from lark_oapi.api.im.v1 import GetImageRequest
            request = (
                GetImageRequest.builder()
                .image_key(image_key)
                .build()
            )
            response = self._client.im.v1.image.get(request)
            if not response.success():
                logger.error(f"Failed to download image {image_key}: code={response.code}, msg={response.msg}")
                return None

            # Determine extension from file_name or default to .png
            ext = ".png"
            if response.file_name:
                import os
                _, fext = os.path.splitext(response.file_name)
                if fext:
                    ext = fext

            tmp = tempfile.NamedTemporaryFile(
                suffix=ext, prefix="feishu_img_", dir="/tmp", delete=False
            )
            tmp.write(response.file.read())
            tmp.close()
            logger.debug(f"Downloaded image {image_key} → {tmp.name}")
            return tmp.name
        except Exception as e:
            logger.error(f"Error downloading image {image_key}: {e}")
            return None

    def _download_resource_sync(self, message_id: str, file_key: str, resource_type: str = "image") -> str | None:
        """Download a message resource (image/file) by message_id and file_key."""
        import tempfile
        try:
            from lark_oapi.api.im.v1 import GetMessageResourceRequest
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type(resource_type)
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if not response.success():
                logger.error(f"Failed to download resource {file_key}: code={response.code}, msg={response.msg}")
                return None

            ext = ".png" if resource_type == "image" else ".bin"
            if response.file_name:
                import os
                _, fext = os.path.splitext(response.file_name)
                if fext:
                    ext = fext

            tmp = tempfile.NamedTemporaryFile(
                suffix=ext, prefix="feishu_res_", dir="/tmp", delete=False
            )
            tmp.write(response.file.read())
            tmp.close()
            logger.debug(f"Downloaded resource {file_key} → {tmp.name}")
            return tmp.name
        except Exception as e:
            logger.error(f"Error downloading resource {file_key}: {e}")
            return None

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

    def _cleanup_stale_session(self, chat_id: str) -> None:
        """Remove session file for a chat the bot is no longer in."""
        try:
            from pathlib import Path
            sessions_dir = Path.home() / ".nanobot" / "workspace" / "sessions"
            session_file = sessions_dir / f"feishu_{chat_id}.jsonl"
            if session_file.exists():
                session_file.unlink()
                logger.info(f"Removed stale session for chat {chat_id} (bot no longer in chat)")
        except Exception as e:
            logger.warning(f"Failed to clean up stale session for {chat_id}: {e}")

    def _send_message_sync(self, receive_id_type: str, receive_id: str, msg_type: str, content: str) -> str | None:
        """Send a single message (text/image/file/interactive) synchronously.

        Returns the message_id on success, or None on failure.
        """
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
                # Auto-clean stale session if bot is no longer in the chat
                if response.code == 230002:
                    self._cleanup_stale_session(receive_id)
                return None
            message_id = getattr(response.data, "message_id", None)
            logger.debug(f"Feishu {msg_type} message sent to {receive_id}, message_id={message_id}")
            if message_id:
                self._track_bot_message(message_id)
            return message_id
        except Exception as e:
            logger.error(f"Error sending Feishu {msg_type} message: {e}")
            return None

    def _reply_message_sync(self, parent_message_id: str, msg_type: str, content: str) -> str | None:
        """Reply to a message in a thread (话题). Returns message_id or None."""
        try:
            from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody
            request = ReplyMessageRequest.builder() \
                .message_id(parent_message_id) \
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type(msg_type)
                    .content(content)
                    .reply_in_thread(True)
                    .build()
                ).build()
            response = self._client.im.v1.message.reply(request)
            if not response.success():
                logger.error(
                    f"Failed to reply Feishu {msg_type} message: code={response.code}, "
                    f"msg={response.msg}, log_id={response.get_log_id()}"
                )
                return None
            message_id = getattr(response.data, "message_id", None)
            logger.debug(f"Feishu {msg_type} reply sent to thread {parent_message_id}, message_id={message_id}")
            if message_id:
                self._track_bot_message(message_id)
            return message_id
        except Exception as e:
            logger.error(f"Error replying Feishu {msg_type} message: {e}")
            return None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu, including media (images/files) if present.

        When both text and media are present, the text is sent first and
        media items are threaded under it so they appear in a single topic.
        """
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        try:
            import os

            # Determine receive_id_type based on chat_id format
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"

            loop = asyncio.get_running_loop()
            reply_to = msg.reply_to or msg.metadata.get("reply_to")

            # Helper: send a new message or reply in thread
            async def _send(msg_type: str, content: str, thread_id: str | None = None) -> str | None:
                target = thread_id or reply_to
                if target:
                    return await loop.run_in_executor(
                        None, self._reply_message_sync, target, msg_type, content,
                    )
                else:
                    return await loop.run_in_executor(
                        None, self._send_message_sync,
                        receive_id_type, msg.chat_id, msg_type, content,
                    )

            # --- Send text content first (if any) to establish thread root ---
            thread_root: str | None = None
            if msg.content and msg.content.strip():
                text = msg.content.strip()
                # Short plain messages (≤200 chars, no markdown) → plain text
                _has_md = re.search(r"\*\*|```|^#{1,6}\s|\|", text, re.MULTILINE)
                if len(text) <= 200 and not _has_md:
                    content = json.dumps({"text": text})
                    sent_msg_id = await _send("text", content)
                else:
                    elements = self._build_card_elements(text)
                    card = {
                        "config": {"wide_screen_mode": True},
                        "elements": elements,
                    }
                    content = json.dumps(card, ensure_ascii=False)
                    sent_msg_id = await _send("interactive", content)
                if sent_msg_id:
                    msg.metadata["sent_message_id"] = sent_msg_id
                    # Use this message as thread root for subsequent media
                    if not reply_to:
                        thread_root = sent_msg_id

            # --- Send media attachments (threaded under text if applicable) ---
            if msg.media:
                for file_path in msg.media:
                    if not os.path.isfile(file_path):
                        logger.warning(f"Media file not found: {file_path}")
                        continue

                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in self._IMAGE_EXTS:
                        image_key = await loop.run_in_executor(None, self._upload_image_sync, file_path)
                        if image_key:
                            mid = await _send("image", json.dumps({"image_key": image_key}), thread_root)
                            if mid and not thread_root and not reply_to:
                                thread_root = mid
                    elif ext in self._AUDIO_EXTS:
                        file_key = await loop.run_in_executor(None, self._upload_file_sync, file_path)
                        if file_key:
                            mid = await _send("audio", json.dumps({"file_key": file_key}), thread_root)
                            if mid and not thread_root and not reply_to:
                                thread_root = mid
                    elif ext in self._VIDEO_EXTS:
                        mp4_path = await loop.run_in_executor(None, self._convert_to_mp4, file_path)
                        thumb_path = await loop.run_in_executor(None, self._extract_video_thumbnail, mp4_path)
                        image_key = ""
                        if thumb_path:
                            image_key = await loop.run_in_executor(None, self._upload_image_sync, thumb_path) or ""
                            try:
                                os.remove(thumb_path)
                            except OSError:
                                pass
                        file_key = await loop.run_in_executor(None, self._upload_file_sync, mp4_path)
                        if file_key:
                            media_content = {"file_key": file_key}
                            if image_key:
                                media_content["image_key"] = image_key
                            mid = await _send("media", json.dumps(media_content), thread_root)
                            if mid and not thread_root and not reply_to:
                                thread_root = mid
                        if mp4_path != file_path:
                            try:
                                os.remove(mp4_path)
                            except OSError:
                                pass
                    else:
                        file_key = await loop.run_in_executor(None, self._upload_file_sync, file_path)
                        if file_key:
                            mid = await _send("file", json.dumps({"file_key": file_key}), thread_root)
                            if mid and not thread_root and not reply_to:
                                thread_root = mid

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
            
            # In group chats, respond when:
            # 1. Bot is @mentioned, OR
            # 2. Message is a reply in a thread started/involving the bot
            if chat_type == "group" and self._bot_open_id:
                is_mentioned = self._is_bot_mentioned(message)
                is_thread = self._is_bot_thread(message)
                if not is_mentioned and not is_thread:
                    logger.debug(f"Ignoring group message without bot mention or thread: {message_id}")
                    return
            
            # Add reaction to indicate "seen"
            await self._add_reaction(message_id, "Get")
            
            # Parse message content
            logger.info(f"Feishu message type={msg_type}, raw content={message.content[:500] if message.content else ''}")
            media_files: list[str] = []  # Downloaded media file paths

            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except json.JSONDecodeError:
                    content = message.content or ""
            elif msg_type == "post":
                try:
                    content_json = json.loads(message.content)
                    logger.info(f"Post content JSON: {json.dumps(content_json, ensure_ascii=False)[:1000]}")
                    content = _extract_post_text(content_json)
                except (json.JSONDecodeError, TypeError):
                    content = message.content or ""
            elif msg_type == "image":
                # Download image and pass to agent as media
                content = "用户发送了一张图片，请描述或回应。"
                try:
                    image_key = json.loads(message.content).get("image_key", "")
                    if image_key:
                        loop = asyncio.get_running_loop()
                        # Use GetMessageResourceRequest (message_id + file_key) to download
                        # user-sent images. GetImageRequest only works for bot-uploaded images.
                        local_path = await loop.run_in_executor(
                            None, self._download_resource_sync,
                            message_id, image_key, "image"
                        )
                        if not local_path:
                            # Fallback: try GetImageRequest
                            local_path = await loop.run_in_executor(
                                None, self._download_image_sync, image_key
                            )
                        if local_path:
                            media_files.append(local_path)
                            logger.info(f"Downloaded image {image_key} → {local_path}")
                        else:
                            content = "[图片下载失败]"
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"Failed to parse/download image: {e}")
                    content = "[图片处理失败]"
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

            # If this message is in a thread, pass the root_id so replies go to the same thread
            root_id = getattr(message, "root_id", None) or getattr(message, "parent_id", None)
            msg_metadata: dict[str, Any] = {
                "message_id": message_id,
                "chat_type": chat_type,
                "msg_type": msg_type,
            }
            if root_id:
                msg_metadata["reply_to"] = root_id

            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                media=media_files if media_files else None,
                metadata=msg_metadata,
            )
            
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
