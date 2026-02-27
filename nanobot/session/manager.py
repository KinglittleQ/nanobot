"""Session management for conversation history."""


import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no class dependency)
# ---------------------------------------------------------------------------

def _strip_base64_images(msg: dict[str, Any]) -> dict[str, Any]:
    """Replace base64 image data in a message with a placeholder.

    Prevents huge base64 strings from being persisted and avoids
    MIME-type mismatch errors when the session is reloaded.
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    new_content = []
    had_images = False
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                had_images = True
                new_content.append({"type": "text", "text": "[image was attached]"})
            else:
                new_content.append(part)
        else:
            new_content.append(part)
    if had_images:
        msg = msg.copy()
        msg["content"] = new_content
    return msg


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fix broken tool_use / tool_result pairing in a message list.

    After a crash mid-tool-call the saved messages may contain orphaned
    tool_result or tool_use entries.  Claude requires every tool_result to
    have a matching tool_use; this removes orphans to prevent 400 errors.
    """
    tool_use_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc_id := tc.get("id"):
                    tool_use_ids.add(tc_id)
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            tool_result_ids.add(msg["tool_call_id"])

    orphan_results = tool_result_ids - tool_use_ids
    orphan_uses = tool_use_ids - tool_result_ids
    if not orphan_results and not orphan_uses:
        return messages

    logger.warning(
        f"Sanitizing session: {len(orphan_results)} orphan tool_results, "
        f"{len(orphan_uses)} orphan tool_uses"
    )

    sanitized: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") in orphan_results:
            logger.warning(f"Dropping orphan tool_result: {msg.get('tool_call_id')}")
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls") and orphan_uses:
            cleaned = [tc for tc in msg["tool_calls"] if tc.get("id") not in orphan_uses]
            if not cleaned:
                if msg.get("content"):
                    sanitized.append({k: v for k, v in msg.items() if k != "tool_calls"})
                else:
                    logger.warning("Dropping assistant message with only orphan tool_calls")
                continue
            elif len(cleaned) < len(msg["tool_calls"]):
                msg = {**msg, "tool_calls": cleaned}
        sanitized.append(msg)
    return sanitized


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files
    _disk_message_count: int = field(default=0, repr=False)  # Messages already on disk (for incremental save)
    
    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def extend_messages(self, messages: list[dict[str, Any]]) -> None:
        """Append raw LLM messages (including tool calls/results) to the session.
        
        Each message gets a timestamp added if not already present.
        This preserves the full conversation context including tool_calls,
        tool_call_id, and tool results for context persistence across restarts.
        
        Base64 image data in user messages is replaced with a placeholder
        to keep session files small and avoid mime-type mismatch errors on reload.
        """
        now = datetime.now().isoformat()
        for msg in messages:
            if "timestamp" not in msg:
                msg = {**msg, "timestamp": now}
            self.messages.append(_strip_base64_images(msg))
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int | None = None, max_tokens: int | None = None) -> list[dict[str, Any]]:
        """Get messages in LLM format, preserving tool metadata.

        If *max_messages* is given, only the last N messages are considered.
        If *max_tokens* is given, further trim from the front so that the
        estimated token count (chars // 4) stays within that budget.

        The cut-off point may land in the middle of a tool_use / tool_result
        group, leaving orphaned messages that would cause a 400 error from
        Claude. We sanitize the result to remove any such orphans.
        """
        recent = self.messages[-max_messages:] if max_messages else list(self.messages)

        # Token-budget trimming: drop oldest messages until estimated tokens fit
        if max_tokens is not None:
            budget_chars = max_tokens * 4
            total_chars = sum(len(json.dumps(m)) for m in recent)
            while recent and total_chars > budget_chars:
                removed = recent[0]
                total_chars -= len(json.dumps(removed))
                recent = recent[1:]

        # --- smart truncation: avoid cutting inside a tool-call group ---
        # Walk forward from the start of `recent` and skip any leading
        # tool-result messages whose tool_use is not in this window.
        # Collect tool_use ids present in this window first.
        tool_use_ids: set[str] = set()
        for m in recent:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc_id := tc.get("id"):
                        tool_use_ids.add(tc_id)

        # Drop leading orphan tool-result messages (their tool_use was truncated)
        start = 0
        for i, m in enumerate(recent):
            if m.get("role") == "tool" and m.get("tool_call_id") not in tool_use_ids:
                start = i + 1
            else:
                break
        recent = recent[start:]

        # Build output, stripping internal fields like timestamp
        out: list[dict[str, Any]] = []
        for m in recent:
            entry: dict[str, Any] = {"role": m["role"]}
            # Only include content if present (some assistant msgs with tool_calls have no content)
            if "content" in m:
                entry["content"] = m["content"]
            for k in ("tool_calls", "tool_call_id", "name"):
                if k in m:
                    entry[k] = m[k]
            # Strip any leftover base64 images from old sessions
            entry = _strip_base64_images(entry)
            out.append(entry)

        # Final safety net: run full sanitize to catch any remaining orphans
        # (e.g. assistant with tool_calls at the end without results)
        return _sanitize_messages(out)
    
    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self._disk_message_count = 0
        self.updated_at = datetime.now()

    def trim(self, keep: int = 16) -> int:
        """Trim session to the last *keep* messages.

        Avoids cutting inside a tool-call group (assistant with tool_calls
        followed by tool results).  Returns the number of messages removed.

        After trimming, the session must be fully re-saved (not incremental)
        because messages were removed from the front.
        """
        if len(self.messages) <= keep:
            return 0

        cut = len(self.messages) - keep

        # Don't cut inside a tool-call group: walk forward from the cut
        # point until we're past any orphan tool results.
        while cut < len(self.messages) and self.messages[cut].get("role") == "tool":
            cut += 1

        if cut >= len(self.messages):
            return 0  # safety: don't remove everything

        removed = cut
        self.messages = self.messages[cut:]
        # Reset disk count so next save does a full rewrite
        self._disk_message_count = 0
        self.last_consolidated = max(0, self.last_consolidated - removed)
        self.updated_at = datetime.now()
        return removed


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = Path.home() / ".nanobot" / "sessions"
        self._cache: dict[str, Session] = {}
    
    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"
    
    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            key: Session key (usually channel:chat_id).
        
        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]
        
        session = self._load(key)
        if session is None:
            session = Session(key=key)
        
        self._cache[key] = session
        return session
    
    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                import shutil
                shutil.move(str(legacy_path), str(path))
                logger.info(f"Migrated session {key} from legacy path")

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            last_consolidated = 0

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed lines (e.g. from interrupted incremental append)
                        logger.warning(f"Skipping malformed line in session {key}: {line[:80]}...")
                        continue

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            # Sanitize to fix any broken tool_use/tool_result pairing
            messages = _sanitize_messages(messages)

            session = Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
            session._disk_message_count = len(messages)
            return session
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to disk atomically (full rewrite).

        Writes to a temporary file first, then renames (atomic on POSIX)
        to avoid data loss if the process crashes mid-write.

        Use ``save_incremental()`` when only new messages were appended
        (no metadata change) to avoid rewriting the entire file.
        """
        import os
        import tempfile

        path = self._get_session_path(session.key)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.sessions_dir), suffix=".tmp"
        )
        try:
            # Parse channel and chat_id from session key for reliable resume
            key_parts = session.key.split(":", 1)
            channel = key_parts[0] if len(key_parts) == 2 else ""
            chat_id = key_parts[1] if len(key_parts) == 2 else session.key

            with os.fdopen(fd, "w") as f:
                metadata_line = {
                    "_type": "metadata",
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                    "last_consolidated": session.last_consolidated,
                    "channel": channel,
                    "chat_id": chat_id,
                }
                f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            os.replace(tmp_path, str(path))  # atomic on POSIX
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        session._disk_message_count = len(session.messages)
        self._cache[session.key] = session

    def save_incremental(self, session: Session) -> None:
        """Append only new messages to the session file on disk.

        This is much faster than ``save()`` for large sessions during
        tool-call loops where many messages are added in quick succession.
        Falls back to full ``save()`` if the file doesn't exist yet or if
        messages were removed (truncation).

        Safety: appends are NOT atomic — a crash mid-append could leave a
        partial JSON line.  The ``_load()`` method already handles this
        gracefully by skipping malformed lines.
        """
        path = self._get_session_path(session.key)

        # Fall back to full save if file doesn't exist or messages were truncated
        if not path.exists() or len(session.messages) < session._disk_message_count:
            self.save(session)
            return

        new_messages = session.messages[session._disk_message_count:]
        if not new_messages:
            return  # Nothing to append

        try:
            with open(path, "a") as f:
                for msg in new_messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            session._disk_message_count = len(session.messages)
            self._cache[session.key] = session
        except Exception as e:
            logger.warning(f"Incremental save failed for {session.key}, falling back to full save: {e}")
            self.save(session)
    
    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session info dicts.
        """
        sessions = []
        
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            # Use channel:chat_id from metadata if available
                            channel = data.get("channel", "")
                            chat_id = data.get("chat_id", "")
                            if channel and chat_id:
                                key = f"{channel}:{chat_id}"
                            else:
                                key = path.stem.replace("_", ":", 1)
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
