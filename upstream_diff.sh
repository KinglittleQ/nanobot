#!/bin/bash
# upstream_diff.sh - Quick comparison between local nanobot and upstream HKUDS/nanobot
# Usage: ./upstream_diff.sh [--fetch] [--files] [--prs]

set -e
cd /home/ubuntu/work/agents/nanobot

PROXY="ALL_PROXY=socks5://127.0.0.1:7891"

if [[ "$1" == "--fetch" || "$1" == "" ]]; then
    echo "📡 Fetching upstream..."
    eval $PROXY git fetch https://github.com/HKUDS/nanobot.git main 2>&1 | tail -3
    echo ""
fi

echo "📊 Upstream Status"
echo "===================="
AHEAD=$(git log --oneline FETCH_HEAD..HEAD | wc -l)
BEHIND=$(git log --oneline HEAD..FETCH_HEAD | wc -l)
echo "Local is $AHEAD commits ahead, $BEHIND commits behind upstream"
echo ""

if [[ "$1" == "--prs" || "$1" == "" ]]; then
    echo "🔀 Upstream Merged PRs (not in local):"
    echo "---"
    git log --oneline HEAD..FETCH_HEAD | grep -i "Merge\|feat\|fix" | head -20
    echo ""
fi

if [[ "$1" == "--files" || "$1" == "" ]]; then
    echo "📁 Changed Files (upstream vs local):"
    echo "---"
    git diff --stat HEAD..FETCH_HEAD | tail -5
    echo ""
    
    echo "🔑 Key File Changes:"
    for f in nanobot/agent/loop.py nanobot/channels/feishu.py nanobot/session/manager.py nanobot/agent/tools/shell.py nanobot/agent/tools/filesystem.py; do
        changes=$(git diff HEAD..FETCH_HEAD -- $f | grep "^[+-]" | grep -v "^[+-][+-][+-]" | wc -l)
        if [ "$changes" -gt 0 ]; then
            echo "  $f: $changes line changes"
        fi
    done
fi

echo ""
echo "✅ Done. Run 'git diff HEAD..FETCH_HEAD -- <file>' for details."
