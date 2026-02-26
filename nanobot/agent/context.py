"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        # System prompt cache: (cache_key -> prompt_str)
        # cache_key is derived from file mtimes so changes are detected.
        self._system_prompt_cache: dict[str, str] = {}

    def _system_prompt_cache_key(self) -> str:
        """Compute a cache key based on mtimes of all files that affect the system prompt."""
        import os
        mtimes = []
        # Bootstrap files
        for filename in self.BOOTSTRAP_FILES:
            p = self.workspace / filename
            if p.exists():
                mtimes.append(f"{filename}:{p.stat().st_mtime_ns}")
        # Memory file
        mem_p = self.workspace / "memory" / "MEMORY.md"
        if mem_p.exists():
            mtimes.append(f"MEMORY.md:{mem_p.stat().st_mtime_ns}")
        # Skills directory (check mtime of skills dir itself as a proxy)
        skills_dir = self.workspace / "skills"
        if skills_dir.exists():
            mtimes.append(f"skills:{skills_dir.stat().st_mtime_ns}")
        return "|".join(mtimes)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        The system prompt is kept STATIC (no timestamps or per-request dynamic
        content) so that Anthropic prompt caching can match the prefix across
        calls.  Dynamic content like current time is injected into the user
        message instead (see build_messages).

        Results are cached in-process keyed by file mtimes, so repeated calls
        within the same turn (or across turns when files haven't changed) skip
        disk I/O and return the same string — maximising Anthropic cache hits.
        
        Args:
            skill_names: Optional list of skills to include.
        
        Returns:
            Complete system prompt.
        """
        # Check in-process cache first
        cache_key = self._system_prompt_cache_key()
        if cache_key in self._system_prompt_cache:
            return self._system_prompt_cache[cache_key]

        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Memory context
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        result = "\n\n---\n\n".join(parts)

        # Store in in-process cache (keyed by file mtimes)
        # Keep cache small: evict all old entries when files change
        self._system_prompt_cache.clear()
        self._system_prompt_cache[cache_key] = result
        return result
    
    def _get_identity(self) -> str:
        """Get the core identity section.
        
        NOTE: Dynamic content (Current Time) is placed at the END of the
        system prompt (appended by build_system_prompt) so that the static
        prefix can benefit from Anthropic prompt caching.
        """
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
For normal conversation, just respond with text - do not call the message tool.

Always be helpful, accurate, and concise. Before calling tools, briefly tell the user what you're about to do (one short sentence in the user's language).
When remembering something important, write to {workspace_path}/memory/MEMORY.md
To recall past events, grep {workspace_path}/memory/HISTORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        The system prompt is kept static for prompt cache efficiency.
        Dynamic per-request context (current time, session info) is
        prepended to the user message so that [system] + [history]
        forms a stable, cacheable prefix.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt (static — no timestamps for cache friendliness)
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History (unchanged — forms cacheable prefix with system prompt)
        messages.extend(history)

        # Current message with dynamic context (time) injected
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        timestamped_message = f"[{now} ({tz})]\n{current_message}"

        user_content = self._build_user_content(timestamped_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            # Detect actual MIME type from file content, not extension
            mime = self._detect_image_mime(str(p))
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return [{"type": "text", "text": text}] + images
    
    def _detect_image_mime(self, path: str) -> str | None:
        """Detect MIME type from file content, not extension."""
        # Try using python-magic if available
        try:
            import magic
            mime = magic.from_file(path, mime=True)
            if mime and mime.startswith("image/"):
                return mime
        except Exception:
            pass
        
        # Fallback: use mimetypes but validate with file header
        mime, _ = mimetypes.guess_type(path)
        
        # Read file header to validate/fix MIME type
        try:
            with open(path, "rb") as f:
                header = f.read(12)
            
            # JPEG: FF D8 FF
            if header[:3] == b"\xff\xd8\xff":
                return "image/jpeg"
            # PNG: 89 50 4E 47 0D 0A 1A 0A
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            # GIF: GIF87a or GIF89a
            if header[:6] in (b"GIF87a", b"GIF89a"):
                return "image/gif"
            # WebP: RIFF....WEBP
            if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                return "image/webp"
        except Exception:
            pass
        
        return mime
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant"}

        # Always include content — some providers (e.g. StepFun) reject
        # assistant messages that omit the key entirely.
        msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Include reasoning content when provided (required by some thinking models)
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content

        messages.append(msg)
        return messages
