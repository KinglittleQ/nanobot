"""Agent loop: the core processing engine."""

import asyncio
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
import re
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tool_context import set_tool_context
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        max_concurrent_sessions: int = 5,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self.verbose_tool_output = False  # Default: brief mode
        self._session_locks: dict[str, asyncio.Lock] = {}  # Per-session locks
        self._concurrency_sem = asyncio.Semaphore(max_concurrent_sessions)
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info.

        Sets both the coroutine-local contextvars (used by concurrent
        sessions) and the legacy instance-level defaults.
        """
        # Coroutine-local context (safe for concurrent sessions)
        set_tool_context(channel, chat_id)

        # Legacy instance-level defaults (kept for backward compatibility)
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _format_tool_detail(name: str, arguments: dict, result: str, max_lines: int = 5, verbose: bool = True) -> str:
        """Format a complete tool call with arguments and result for user display.
        
        When verbose=False, returns a single-line summary.
        """
        if not verbose:
            # Brief mode: single line
            result_preview = result.replace("\n", " ").strip()
            if len(result_preview) > 80:
                result_preview = result_preview[:77] + "..."
            # Show key argument (first string value or first key)
            arg_hint = ""
            if arguments:
                for k, v in arguments.items():
                    if isinstance(v, str) and len(v) < 80:
                        arg_hint = f" {k}={v}"
                        break
                if not arg_hint:
                    first_key = next(iter(arguments))
                    val = str(arguments[first_key])
                    if len(val) > 40:
                        val = val[:37] + "..."
                    arg_hint = f" {first_key}={val}"
            return f"🔧 {name}{arg_hint} → {result_preview}"

        args_str = json.dumps(arguments, ensure_ascii=False, indent=2) if arguments else "{}"
        lines = result.splitlines()
        if len(lines) > max_lines:
            result_display = "\n".join(lines[:max_lines]) + f"\n... ({len(lines)} lines total, showing first {max_lines})"
        else:
            result_display = result
        return f"🔧 **{name}**\n```\n{args_str}\n```\n📤 Result:\n```\n{result_display}\n```"

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        session: Session | None = None,
    ) -> tuple[str | None, list[str], list[dict], bool]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.
            on_progress: Optional callback to push intermediate content to the user.
            session: Optional session for incremental persistence. When provided,
                     new messages are saved to disk after each LLM round so that
                     progress is not lost on restart.

        Returns:
            Tuple of (final_content, list_of_tools_used, full_messages, hit_max).
            full_messages includes all messages from the loop (excluding system prompt).
            hit_max is True if the loop exited because max_iterations was reached.
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        # Track how many non-system messages have already been persisted to the session.
        # This lets us incrementally append only the *new* messages each round.
        persisted_count = len(session.messages) if session else 0

        def _flush_to_session() -> None:
            """Incrementally persist any new messages to the session on disk."""
            nonlocal persisted_count
            if session is None:
                return
            non_system = [m for m in messages if m.get("role") != "system"]
            new_msgs = non_system[persisted_count:]
            if new_msgs:
                session.extend_messages(new_msgs)
                self.sessions.save(session)
                persisted_count = len(session.messages)

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    if on_progress:
                        detail = self._format_tool_detail(
                            tool_call.name, tool_call.arguments, result,
                            verbose=self.verbose_tool_output,
                        )
                        await on_progress(detail)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                # Persist after each tool-call round so progress survives restarts
                _flush_to_session()
            else:
                final_content = self._strip_think(response.content)
                break

        # Extract non-system messages for persistence
        new_messages = [m for m in messages if m.get("role") != "system"]
        hit_max = (iteration >= self.max_iterations and final_content is None)

        return final_content, tools_used, new_messages, hit_max

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """Get or create a per-session lock to serialize messages within the same session."""
        if session_key not in self._session_locks:
            self._session_locks[session_key] = asyncio.Lock()
        return self._session_locks[session_key]

    async def _handle_message(self, msg: InboundMessage) -> None:
        """Handle a single inbound message with per-session serialization.

        Messages from different sessions run concurrently, but messages
        targeting the same session are serialized via a per-session lock.
        The session lock is acquired *before* the concurrency semaphore so
        that queued messages for the same session don't waste semaphore slots.
        """
        session_key = msg.session_key
        lock = self._get_session_lock(session_key)

        async with lock:
            async with self._concurrency_sem:
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus.

        Messages from different sessions are processed concurrently (up to
        max_concurrent_sessions).  Messages within the same session are
        serialized to avoid race conditions on shared session state.
        """
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started (concurrent session processing enabled)")

        tasks: set[asyncio.Task] = set()

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                task = asyncio.create_task(self._handle_message(msg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            except asyncio.TimeoutError:
                continue

        # Wait for in-flight tasks on shutdown
        if tasks:
            logger.info(f"Waiting for {len(tasks)} in-flight tasks to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
            on_progress: Optional callback for intermediate output (defaults to bus publish).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            mode = "verbose 📝" if self.verbose_tool_output else "brief 📎"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"🐈 nanobot commands:\n/new — Start a new conversation\n/verbose — Verbose tool output\n/brief — Brief tool output (one line)\n/help — Show available commands\n\nCurrent tool output mode: {mode}")
        if cmd == "/verbose":
            self.verbose_tool_output = True
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="📝 Tool output mode: **verbose** (full details)")
        if cmd == "/brief":
            self.verbose_tool_output = False
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="📎 Tool output mode: **brief** (one line per tool call)")
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str) -> None:
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content,
                metadata=msg.metadata or {},
            ))

        final_content, tools_used, full_messages, hit_max = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
            session=session,
        )

        if final_content is None:
            if hit_max:
                final_content = "⚠️ 达到最大迭代次数，任务可能未完成。"
            else:
                # Already responded via tool calls (e.g. message tool), no extra reply needed
                # Intermediate messages were already saved incrementally by _run_agent_loop.
                # Save any remaining unsaved messages (e.g. final assistant message).
                self._save_remaining(session, full_messages)
                return None
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        # Save any remaining messages not yet persisted (e.g. the final assistant reply).
        # Most tool-call rounds were already saved incrementally by _run_agent_loop.
        self._save_remaining(session, full_messages)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    def _save_remaining(self, session: Session, full_messages: list[dict]) -> None:
        """Persist any messages from full_messages that are not yet in the session.

        _run_agent_loop incrementally saves after each tool-call round, but the
        final assistant reply (or user message in a no-tool-call turn) may still
        be unsaved.  This method calculates the diff and saves only the new ones.
        """
        history_count = len(session.messages)
        new_turn_messages = full_messages[history_count:]
        if new_turn_messages:
            session.extend_messages(new_turn_messages)
            self.sessions.save(session)

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=f"[System: {msg.sender_id}] {msg.content}",
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _, full_messages, hit_max = await self._run_agent_loop(
            initial_messages, session=session,
        )

        if final_content is None:
            if hit_max:
                final_content = "⚠️ 达到最大迭代次数，后台任务可能未完成。"
            else:
                final_content = "Background task completed."
        
        # Save any remaining messages (final reply, etc.)
        self._save_remaining(session, full_messages)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            ts = m.get("timestamp", "?")[:16]
            
            # Skip tool results in consolidation summary (too verbose)
            if role == "tool":
                tool_name = m.get("name", "unknown")
                lines.append(f"[{ts}] TOOL({tool_name}): [result omitted]")
                continue
            
            # Assistant messages with tool_calls but no content
            if role == "assistant" and not content and m.get("tool_calls"):
                tool_names = [tc.get("function", {}).get("name", "?") for tc in m.get("tool_calls", [])]
                lines.append(f"[{ts}] ASSISTANT: [called tools: {', '.join(tool_names)}]")
                continue
            
            if not content:
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{ts}] {role.upper()}{tools}: {content}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
            on_progress: Optional callback for intermediate output.
        
        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
