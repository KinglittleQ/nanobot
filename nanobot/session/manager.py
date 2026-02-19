"""Session management for conversation history."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename


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
        """
        now = datetime.now().isoformat()
        for msg in messages:
            if "timestamp" not in msg:
                msg["timestamp"] = now
            self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Get recent messages in LLM format, preserving tool metadata."""
        out: list[dict[str, Any]] = []
        for m in self.messages[-max_messages:]:
            entry: dict[str, Any] = {"role": m["role"]}
            # Only include content if present (some assistant msgs with tool_calls have no content)
            if "content" in m:
                entry["content"] = m["content"]
            for k in ("tool_calls", "tool_call_id", "name"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
        return out
    
    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


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
    
    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize messages to fix broken tool_use/tool_result pairing.

        After a crash or restart mid-tool-call, the saved messages may contain:
        - tool_result messages whose corresponding tool_use (in an assistant msg)
          was never saved, or vice versa.

        Claude API requires every tool_result to have a matching tool_use in the
        preceding assistant message.  This method removes orphaned messages to
        prevent 400 errors.
        """
        # Pass 1: collect all tool_use ids and all tool_result ids
        tool_use_ids: set[str] = set()
        tool_result_ids: set[str] = set()

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        tool_use_ids.add(tc_id)
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                tool_result_ids.add(msg["tool_call_id"])

        # Find orphans
        orphan_tool_results = tool_result_ids - tool_use_ids  # results without a use
        orphan_tool_uses = tool_use_ids - tool_result_ids      # uses without a result

        if not orphan_tool_results and not orphan_tool_uses:
            return messages

        logger.warning(
            f"Sanitizing session: {len(orphan_tool_results)} orphan tool_results, "
            f"{len(orphan_tool_uses)} orphan tool_uses"
        )

        # Pass 2: rebuild messages, dropping orphans
        sanitized: list[dict[str, Any]] = []
        for msg in messages:
            # Drop tool_result messages with no matching tool_use
            if msg.get("role") == "tool" and msg.get("tool_call_id") in orphan_tool_results:
                logger.warning(f"Dropping orphan tool_result: {msg.get('tool_call_id')}")
                continue

            # For assistant messages with tool_calls, remove orphan tool_use entries
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                if orphan_tool_uses:
                    cleaned_tcs = [
                        tc for tc in msg["tool_calls"]
                        if tc.get("id") not in orphan_tool_uses
                    ]
                    if not cleaned_tcs:
                        # All tool_calls were orphans; convert to plain assistant msg
                        if msg.get("content"):
                            sanitized.append({
                                k: v for k, v in msg.items() if k != "tool_calls"
                            })
                        else:
                            logger.warning("Dropping assistant message with only orphan tool_calls")
                        continue
                    elif len(cleaned_tcs) < len(msg["tool_calls"]):
                        msg = {**msg, "tool_calls": cleaned_tcs}

            sanitized.append(msg)

        return sanitized

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

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            # Sanitize to fix any broken tool_use/tool_result pairing
            messages = self._sanitize_messages(messages)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        with open(path, "w") as f:
            metadata_line = {
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated
            }
            f.write(json.dumps(metadata_line) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg) + "\n")

        self._cache[session.key] = session
    
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
                            sessions.append({
                                "key": path.stem.replace("_", ":"),
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
