"""Subagent manager for background task execution with persistence."""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.message import MessageTool
from nanobot.utils.helpers import ensure_dir


class SubagentManager:
    """
    Manages background subagent execution with disk persistence.

    Each subagent task is registered in a JSON file on disk so that:
    - Status can be queried at any time (``/tasks`` command)
    - Progress is logged to per-task log files
    - After a restart, incomplete tasks can be manually resumed
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace

        # In-memory tracking of running asyncio tasks
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

        # Persistent storage
        self._store_dir = ensure_dir(workspace / "subagents")
        self._registry_path = self._store_dir / "registry.json"
        self._registry: dict[str, dict[str, Any]] = self._load_registry()

    # ------------------------------------------------------------------
    # Registry persistence
    # ------------------------------------------------------------------

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        """Load the task registry from disk."""
        if self._registry_path.exists():
            try:
                data = json.loads(self._registry_path.read_text())
                return data.get("tasks", {})
            except Exception as e:
                logger.warning(f"Failed to load subagent registry: {e}")
        return {}

    def _save_registry(self) -> None:
        """Save the task registry to disk atomically."""
        import os
        import tempfile
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._store_dir), suffix=".tmp"
            )
            with os.fdopen(fd, "w") as f:
                json.dump({"tasks": self._registry}, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(self._registry_path))
        except BaseException as e:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            logger.warning(f"Failed to save subagent registry: {e}")

    def _update_task(self, task_id: str, persist: bool = True, **fields: Any) -> None:
        """Update fields for a task and optionally persist.

        Args:
            task_id: The task to update.
            persist: If False, update in-memory only (caller must save later).
                     Use this in hot loops to avoid excessive disk writes.
        """
        if task_id in self._registry:
            self._registry[task_id].update(fields)
            if persist:
                self._save_registry()

    # ------------------------------------------------------------------
    # Task log files
    # ------------------------------------------------------------------

    def _log_path(self, task_id: str) -> Path:
        return self._store_dir / f"{task_id}.log"

    def _append_log(self, task_id: str, line: str) -> None:
        """Append a line to the task's log file."""
        try:
            with open(self._log_path(task_id), "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{ts}] {line}\n")
        except Exception:
            pass

    def read_log(self, task_id: str, tail: int = 50) -> str:
        """Read the last *tail* lines of a task log."""
        path = self._log_path(task_id)
        if not path.exists():
            return "(no log)"
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > tail:
            return "\n".join(lines[-tail:])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        reply_to: str = "",
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        # Register task on disk
        self._registry[task_id] = {
            "id": task_id,
            "label": display_label,
            "task": task,
            "origin": origin,
            "reply_to": reply_to,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "iterations": 0,
            "tool_calls": 0,
            "error": None,
        }
        self._save_registry()

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, reply_to=reply_to)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        self._append_log(task_id, f"Task started: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def resume(self, task_id: str) -> str:
        """Resume an incomplete task after restart.

        Re-spawns the subagent with the original task description.
        The previous conversation context is lost, but the task prompt
        includes a hint that this is a resumed task.
        """
        entry = self._registry.get(task_id)
        if not entry:
            return f"Error: Task '{task_id}' not found in registry."
        if task_id in self._running_tasks:
            return f"Task '{task_id}' is already running."

        # Mark as running again
        original_task = entry["task"]
        label = entry.get("label", task_id)
        origin = entry.get("origin", {"channel": "cli", "chat_id": "direct"})

        resume_task = (
            f"[RESUMED TASK — this task was interrupted by a restart. "
            f"Check the workspace for any partial progress before starting over.]\n\n"
            f"{original_task}"
        )

        reply_to = entry.get("reply_to", "")

        self._update_task(task_id, status="running", updated_at=datetime.now().isoformat(), error=None)
        self._append_log(task_id, "Task resumed after restart")

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, resume_task, label, origin, reply_to=reply_to)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Resumed subagent [{task_id}]: {label}")
        return f"Subagent [{label}] resumed (id: {task_id})."

    def cancel(self, task_id: str) -> str:
        """Cancel a running subagent task."""
        bg_task = self._running_tasks.get(task_id)
        if not bg_task:
            return f"Error: Task '{task_id}' is not running."
        bg_task.cancel()
        self._running_tasks.pop(task_id, None)
        self._update_task(task_id, status="cancelled", updated_at=datetime.now().isoformat())
        self._append_log(task_id, "Task cancelled by user")
        logger.info(f"Cancelled subagent [{task_id}]")
        return f"Task '{task_id}' cancelled."

    def list_tasks(self, include_completed: bool = False) -> list[dict[str, Any]]:
        """List tasks from the registry.

        By default only returns running/interrupted tasks.
        """
        tasks = []
        for entry in self._registry.values():
            status = entry.get("status", "unknown")
            # Check if task claims to be running but has no asyncio task (interrupted)
            if status == "running" and entry["id"] not in self._running_tasks:
                entry["status"] = "interrupted"
                self._save_registry()
            if include_completed or entry["status"] in ("running", "interrupted"):
                tasks.append(entry)
        return sorted(tasks, key=lambda t: t.get("created_at", ""), reverse=True)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get a single task entry."""
        return self._registry.get(task_id)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    async def auto_resume(self) -> int:
        """Resume all tasks that were running when the process last exited.

        Called at gateway startup.  Returns the number of tasks resumed.
        """
        resumed = 0
        for task_id, entry in list(self._registry.items()):
            status = entry.get("status", "")
            if status == "running":
                # Was running when we crashed — mark interrupted then resume
                self._update_task(task_id, status="interrupted")
            if status in ("running", "interrupted"):
                try:
                    result = await self.resume(task_id)
                    logger.info(f"Auto-resumed subagent [{task_id}]: {result}")
                    resumed += 1
                except Exception as e:
                    logger.error(f"Failed to auto-resume subagent [{task_id}]: {e}")
        return resumed

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        reply_to: str = "",
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        try:
            # Build subagent tools via shared factory (no bus/cron — subagent adds its own MessageTool below)
            from nanobot.agent.loop import _build_tools
            tools = _build_tools(
                workspace=self.workspace,
                brave_api_key=self.brave_api_key,
                exec_config=self.exec_config,
                restrict_to_workspace=self.restrict_to_workspace,
            )

            # Determine thread target:
            # - If reply_to is set (user's message_id), reply under user's thread
            # - If reply_to is None (not set), create a new thread root
            # - If reply_to is "" (explicitly no thread), don't use threading
            if reply_to:
                thread_root_id = reply_to
                # Send a start notification in the user's thread
                from nanobot.bus.events import OutboundMessage as _OutMsg
                start_msg = _OutMsg(
                    channel=origin["channel"],
                    chat_id=origin["chat_id"],
                    content=f"🔄 **子任务启动: {label}** (`{task_id}`)",
                    reply_to=thread_root_id,
                )
                await self.bus.publish_outbound(start_msg)
            elif reply_to is None:
                # Create a new thread root only when reply_to is not explicitly set
                thread_root_id = await self._send_thread_root(task_id, label, origin)
            else:
                # reply_to is "" (explicitly no thread mode)
                thread_root_id = None

            # Message tool: wraps send to reply in the thread
            async def _threaded_send(msg: "OutboundMessage") -> None:
                if thread_root_id and not msg.reply_to:
                    msg.reply_to = thread_root_id
                await self.bus.publish_outbound(msg)

            message_tool = MessageTool(send_callback=_threaded_send)
            message_tool.set_context(origin["channel"], origin["chat_id"])
            tools.register(message_tool)

            # Set tool context for this subagent coroutine so contextvars
            # (used by message tool, filesystem protected file checks, etc.)
            # point to the correct origin session, not the main agent's last session.
            from nanobot.agent.tool_context import set_tool_context
            set_tool_context(origin["channel"], origin["chat_id"], reply_to=reply_to)

            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop
            max_iterations = 200
            max_duration_s = 30 * 60  # 30 minute timeout
            iteration = 0
            start_time = time.time()
            final_result: str | None = None

            while iteration < max_iterations:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > max_duration_s:
                    final_result = f"⚠️ Task timed out after {int(elapsed // 60)} minutes."
                    self._append_log(task_id, f"TIMEOUT after {int(elapsed)}s")
                    break

                iteration += 1
                # Don't persist on every iteration — save after tool calls or at completion
                self._update_task(task_id, persist=False, iterations=iteration, updated_at=datetime.now().isoformat())

                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })

                    # Log assistant thinking
                    if response.content:
                        self._append_log(task_id, f"THINK: {response.content[:200]}")

                    # Execute tools
                    for tool_call in response.tool_calls:
                        tc_count = self._registry.get(task_id, {}).get("tool_calls", 0) + 1
                        self._update_task(task_id, persist=False, tool_calls=tc_count)

                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        self._append_log(task_id, f"TOOL: {tool_call.name}({args_str[:200]})")
                        logger.debug(f"Subagent [{task_id}] executing: {tool_call.name}")

                        result = await tools.execute(tool_call.name, tool_call.arguments)

                        # Log result (truncated)
                        result_preview = result[:200].replace("\n", " ")
                        self._append_log(task_id, f"  → {result_preview}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                    # Persist registry once after all tool calls in this round
                    self._save_registry()
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            self._update_task(
                task_id, status="completed", updated_at=datetime.now().isoformat()
            )
            self._append_log(task_id, f"COMPLETED: {final_result[:200]}")
            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except asyncio.CancelledError:
            self._update_task(task_id, status="cancelled", updated_at=datetime.now().isoformat())
            self._append_log(task_id, "CANCELLED")
            logger.info(f"Subagent [{task_id}] was cancelled")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self._update_task(
                task_id, status="error", error=str(e), updated_at=datetime.now().isoformat()
            )
            self._append_log(task_id, f"ERROR: {e}")
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _send_thread_root(
        self, task_id: str, label: str, origin: dict[str, str]
    ) -> str | None:
        """Send a root message that starts a thread for subagent progress.

        Returns the message_id of the root message (for reply threading),
        or None if the channel doesn't support it.
        """
        from nanobot.bus.events import OutboundMessage

        root_msg = OutboundMessage(
            channel=origin["channel"],
            chat_id=origin["chat_id"],
            content=f"🔄 **Task started: {label}** (`{task_id}`)\nProgress updates will appear in this thread.",
        )
        try:
            metadata = await self.bus.send_and_wait(root_msg, timeout=10.0)
            thread_id = metadata.get("sent_message_id")
            if thread_id:
                self._append_log(task_id, f"Thread root message: {thread_id}")
                self._update_task(task_id, thread_id=thread_id)
            return thread_id
        except Exception as e:
            logger.warning(f"Failed to send thread root for subagent [{task_id}]: {e}")
            return None

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        # Send completion notice to the thread
        entry = self._registry.get(task_id, {})
        thread_id = entry.get("reply_to") or entry.get("thread_id")
        if thread_id:
            icon = "✅" if status == "ok" else "❌"
            from nanobot.bus.events import OutboundMessage
            thread_msg = OutboundMessage(
                channel=origin["channel"],
                chat_id=origin["chat_id"],
                content=f"{icon} **Task {status_text}: {label}**",
                reply_to=thread_id,
            )
            await self.bus.publish_outbound(thread_msg)

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")

    def _build_subagent_prompt(self, task: str) -> str:
        """Build a focused system prompt for the subagent."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings
5. For long tasks, use the message tool to send progress updates to the user

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Send progress updates or results to the user via the message tool
- Complete the task thoroughly

## What You Cannot Do
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

When you have completed the task, provide a clear summary of your findings or actions."""
