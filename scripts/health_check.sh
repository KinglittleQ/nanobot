#!/bin/bash
# nanobot health check script
# 检查 nanobot 服务状态、磁盘、日志、session 大小

set -e
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "🐈 nanobot 健康检查 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# 1. systemd 服务状态
echo ""
echo "📦 服务状态："
if systemctl --user is-active --quiet nanobot; then
    echo -e "  ${GREEN}✅ nanobot.service 运行中${NC}"
    UPTIME=$(systemctl --user show nanobot --property=ActiveEnterTimestamp | cut -d= -f2)
    echo "     启动时间: $UPTIME"
else
    echo -e "  ${RED}❌ nanobot.service 未运行！${NC}"
fi

# 2. 进程检查
PID=$(pgrep -f "nanobot" | head -1)
if [ -n "$PID" ]; then
    MEM=$(ps -o rss= -p $PID 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
    CPU=$(ps -o %cpu= -p $PID 2>/dev/null | tr -d ' ')
    echo "  PID: $PID | 内存: $MEM | CPU: ${CPU}%"
fi

# 3. 磁盘空间
echo ""
echo "💾 磁盘空间："
df -h /home/ubuntu/.nanobot/ | tail -1 | awk '{printf "  已用: %s / %s (%s)\n", $3, $2, $5}'

# 4. Session 文件大小
echo ""
echo "📁 Session 文件："
WORKSPACE="/home/ubuntu/.nanobot/workspace"
SESSION_DIR="$WORKSPACE/sessions"
if [ -d "$SESSION_DIR" ]; then
    ls -lhS "$SESSION_DIR"/*.jsonl 2>/dev/null | awk '{printf "  %-50s %s\n", $NF, $5}' | head -10
    TOTAL=$(du -sh "$SESSION_DIR" 2>/dev/null | cut -f1)
    echo "  总大小: $TOTAL"
fi

# 5. 日志最近错误
echo ""
echo "📋 最近日志 (最后10行)："
LOG_FILE="/home/ubuntu/.nanobot/logs/gateway.log"
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE" | sed 's/^/  /'
    ERROR_COUNT=$(grep -c "ERROR\|CRITICAL" "$LOG_FILE" 2>/dev/null || echo 0)
    echo "  总错误数: $ERROR_COUNT"
else
    echo "  日志文件不存在: $LOG_FILE"
fi

# 6. Memory 文件大小
echo ""
echo "🧠 Memory 文件："
MEMORY_FILE="$WORKSPACE/memory/MEMORY.md"
HISTORY_FILE="$WORKSPACE/memory/HISTORY.md"
if [ -f "$MEMORY_FILE" ]; then
    echo "  MEMORY.md: $(wc -l < "$MEMORY_FILE") 行 / $(wc -c < "$MEMORY_FILE") 字节"
fi
if [ -f "$HISTORY_FILE" ]; then
    echo "  HISTORY.md: $(wc -l < "$HISTORY_FILE") 行 / $(wc -c < "$HISTORY_FILE") 字节"
fi

# 7. Git 状态
echo ""
echo "🔧 代码状态："
cd /home/ubuntu/work/agents/nanobot
BRANCH=$(git branch --show-current 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached HEAD")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DIRTY=$(git status --porcelain 2>/dev/null | wc -l)
echo "  分支: $BRANCH | 提交: $COMMIT | 未提交改动: $DIRTY 个文件"

# 上游差距
BEHIND=$(git rev-list HEAD..origin/main --count 2>/dev/null || echo "?")
echo "  落后上游: $BEHIND commits"

echo ""
echo "============================================"
echo "✅ 健康检查完成"
