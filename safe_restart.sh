#!/bin/bash
# safe_restart.sh — 安全重启 nanobot，失败自动回滚
# 用法: ./safe_restart.sh [--stash]

set -euo pipefail

TIMEOUT=40
CHECK_INTERVAL=2
SERVICE=nanobot
REPO=/home/ubuntu/work/agents/nanobot
RESTART_LOG=/home/ubuntu/.nanobot/logs/safe_restart.log

log() {
    echo "[safe_restart] $*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$RESTART_LOG"
}

# --- 1. 语法检查 ---
log "检查语法..."
cd "$REPO"
CHANGED=$(git diff --name-only HEAD | grep '\.py$' || true)
for f in $CHANGED; do
    [ -f "$f" ] || continue
    if ! python3 -m py_compile "$f" 2>&1; then
        log "❌ 语法错误: $f，中止重启"
        exit 1
    fi
    log "✓ $f"
done

# --- 2. 记录回滚点 ---
ROLLBACK_HASH=$(git rev-parse HEAD)
log "回滚点: $ROLLBACK_HASH"

if [[ "${1:-}" == "--stash" ]]; then
    git stash push -m "safe_restart_$(date +%s)" || true
    log "已 git stash 当前改动"
fi

# --- 3. 如果是在 nanobot 进程内调用，用 systemd-run 脱离 cgroup 后重新执行 ---
if [ -z "${_SAFE_RESTART_DETACHED:-}" ]; then
    log "通过 systemd-run 脱离 nanobot cgroup..."
    export _SAFE_RESTART_DETACHED=1
    exec systemd-run --user --scope \
        -E DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}" \
        -E XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}" \
        -E _SAFE_RESTART_DETACHED=1 \
        bash "$0" "$@"
fi

# --- 4. 重启服务 ---
log "重启 nanobot..."
systemctl --user restart "$SERVICE" || true

# --- 5. 等待启动，超时则回滚 ---
log "等待服务启动（最多 ${TIMEOUT}s）..."
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
    STATUS=$(systemctl --user is-active "$SERVICE" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "active" ]; then
        log "✅ nanobot 启动成功（${ELAPSED}s）"
        exit 0
    elif [ "$STATUS" = "failed" ]; then
        log "❌ 服务 failed，立即回滚..."
        break
    fi
    log "  等待中... ${ELAPSED}s (status=$STATUS)"
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    log "❌ 启动超时（${TIMEOUT}s），回滚..."
fi

# --- 6. 回滚 ---
log "回滚到 $ROLLBACK_HASH..."
git -C "$REPO" checkout -- .
systemctl --user restart "$SERVICE" || true

sleep 5
STATUS=$(systemctl --user is-active "$SERVICE" 2>/dev/null || echo "unknown")
if [ "$STATUS" = "active" ]; then
    log "✅ 回滚成功，nanobot 已恢复运行"
else
    log "💀 回滚后仍然失败！请手动检查: journalctl --user -u nanobot -n 50"
    exit 2
fi
exit 1
