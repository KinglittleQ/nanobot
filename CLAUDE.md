# CLAUDE.md — Nanobot Project Guide

This file provides context for AI coding assistants working on the nanobot codebase.

## Project Overview

Nanobot is a lightweight personal AI assistant framework. It connects to multiple chat channels (Feishu, Telegram, Discord, etc.), processes messages through an LLM-powered agent loop, and responds using a set of built-in tools.

## Architecture

```
User ←→ Channel (feishu/telegram/...) ←→ MessageBus ←→ AgentLoop ←→ LLM Provider
                                              ↕
                                          ToolRegistry
                                     (exec, read_file, web_search, message, spawn, cron, ...)
```

### Key Components

| Component | Path | Description |
|-----------|------|-------------|
| **AgentLoop** | `nanobot/agent/loop.py` | Core engine: receives messages, builds context, runs LLM + tool call loop, manages sessions |
| **ContextBuilder** | `nanobot/agent/context.py` | Assembles system prompt from bootstrap files (AGENTS.md, SOUL.md, USER.md), memory, and skills |
| **ToolRegistry** | `nanobot/agent/tools/registry.py` | Registers and dispatches tool calls |
| **Tool Context** | `nanobot/agent/tool_context.py` | Per-coroutine contextvars for channel/chat_id/sender_id (concurrency-safe) |
| **MessageBus** | `nanobot/bus/queue.py` | Async queue decoupling channels from agent |
| **SessionManager** | `nanobot/session/manager.py` | JSONL-based session persistence with incremental saves |
| **Channels** | `nanobot/channels/` | Chat platform adapters (feishu.py, telegram.py, discord.py, etc.) |
| **Providers** | `nanobot/providers/` | LLM backends (custom_provider.py for OpenAI-compatible, litellm_provider.py) |
| **Config** | `nanobot/config/schema.py` | Pydantic config schema |
| **CLI** | `nanobot/cli/commands.py` | CLI commands including `gateway` (main entry point) |
| **Skills** | `nanobot/skills/` | Built-in skill docs; user skills in `~/.nanobot/workspace/skills/` |

### Message Flow

1. Channel receives message → publishes `InboundMessage` to `MessageBus.inbound`
2. `AgentLoop.run()` consumes from bus, dispatches to `_handle_message()` via `asyncio.create_task()`
3. Per-session lock ensures messages within the same session are serialized
4. `_process_message()` builds context, calls `_run_agent_loop()` (LLM + tool calls)
5. Tool calls execute, results feed back to LLM; session saved incrementally after each round
6. Final response published to `MessageBus.outbound` → dispatched to channel

### Concurrency Model

- Different sessions are processed **concurrently** (up to `max_concurrent_sessions`, default 5)
- Same session messages are **serialized** via per-session `asyncio.Lock`
- Tool context (channel/chat_id/sender_id) uses `contextvars` for coroutine isolation
- Semaphore acquired **inside** session lock to avoid wasting slots

## Configuration

- Config file: `~/.nanobot/config.json`
- Workspace: `~/.nanobot/workspace/`
- Bootstrap files loaded into system prompt: `AGENTS.md`, `SOUL.md`, `USER.md`
- Memory: `workspace/memory/MEMORY.md` (long-term), `workspace/memory/HISTORY.md` (event log)
- Admin config: `workspace/admin.json` (admin_ids, protected_files, user_names)

## Key Design Decisions

### Session Persistence
- Sessions stored as JSONL files in `workspace/sessions/`
- **Incremental save**: after each tool-call round, new messages are appended to disk
- On load, `_sanitize_messages()` removes orphaned tool_use/tool_result pairs (crash recovery)

### Protected Files
- `SOUL.md`, `AGENTS.md`, `USER.md` are protected by default
- Only admin users (defined in `admin.json`) can modify them via write_file/edit_file
- Checked via `_check_protected_file()` in `filesystem.py` using `get_tool_sender_id()` from contextvars

### Tool Output Modes
- `/brief` (default): single-line summary per tool call
- `/verbose`: full arguments + result display
- Controlled by `self.verbose_tool_output` on AgentLoop

### Web Tools Proxy Fallback
- `web_search` and `web_fetch` try direct connection first
- On failure, automatically retry with fallback proxy (`127.0.0.1:7892`)

## Development

### Setup
```bash
cd /home/ubuntu/work/agents/nanobot
source /home/ubuntu/work/nanobot_env/bin/activate
pip install -e .
```

### Running
```bash
nanobot gateway                    # Start the gateway (main process)
nanobot chat                       # Interactive CLI chat
nanobot chat -m "hello"            # One-shot message
```

### Restart Procedure
1. Install changes: `pip install -e .`
2. **Always confirm with user before restarting**
3. Restart via: `nohup bash restart.sh &`
4. The restart script kills the old process and starts a new one in tmux session "nanobot"

### Git Workflow
- Upstream: `HKUDS/nanobot` (no direct push access)
- Fork: `KinglittleQ/nanobot`
- Push to fork, create PR to upstream
- Use proxy for git push: `export http_proxy=http://127.0.0.1:7892 https_proxy=http://127.0.0.1:7892`

### Testing Changes
```bash
python -c "from nanobot.agent.loop import AgentLoop; print('OK')"  # Quick import check
```

## Common Patterns

### Adding a New Tool
1. Create class in `nanobot/agent/tools/` extending `Tool` base class
2. Implement `name`, `description`, `parameters` properties and `execute()` method
3. Register in `AgentLoop._register_default_tools()`
4. If tool needs session context, use `get_tool_channel()` / `get_tool_chat_id()` / `get_tool_sender_id()` from `tool_context.py`

### Adding a Slash Command
Add handling in `AgentLoop._process_message()` after the existing `/help`, `/verbose`, `/brief` checks.

### Modifying Feishu Channel
- Main file: `nanobot/channels/feishu.py`
- Uses lark-oapi SDK with WebSocket long connection
- Media sending uses `loop.run_in_executor()` for sync SDK calls
- Group chat: only responds when bot is @mentioned

## Important Notes

- ⚠️ Never restart without user confirmation
- ⚠️ The agent runs inside its own process — cannot `kill` itself directly; use `restart.sh` via `nohup`
- ⚠️ After restart, context is lost; startup notification is configured via `config.json` → `gateway.startupNotify`
- ⚠️ Network proxy (`127.0.0.1:7892`) needed for GitHub, external APIs; web tools handle this automatically
- ⚠️ `git commit` with inline message may be blocked by safety guard; write message to file first, then `git commit -F /tmp/commit_msg.txt`
