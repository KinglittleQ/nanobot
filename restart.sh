#!/bin/bash
# Restart nanobot gateway
# Usage: nohup bash /home/ubuntu/work/agents/nanobot/restart.sh > /tmp/nanobot_restart.log 2>&1 &

echo "[$(date)] Stopping nanobot gateway..."
pkill -f "nanobot gateway"
sleep 3

echo "[$(date)] Starting nanobot gateway in tmux..."
# Start nanobot gateway inside the tmux session
tmux send-keys -t nanobot "source /home/ubuntu/work/nanobot_env/bin/activate && nanobot gateway" Enter

echo "[$(date)] Restart command sent to tmux."
