#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-.}"
CLAUDE_MD="$TARGET_DIR/CLAUDE.md"

MEMORY_BLOCK='## Claude AI Memory

本项目已接入 claude-ai-memory 上下文管理系统。

### 会话启动
- 每次会话开始时，调用 context_load 获取项目上下文概览
- 根据概览中的待办事项和最近进展，了解当前项目状态

### 工作中
- 当遇到需要历史背景的问题时，调用 context_search 搜索相关记忆
- 不要频繁搜索，只在确实需要历史上下文时才调用

### 会话结束
- 会话结束前，用 context_save 保存：
  1. 本次会话摘要（做了什么、关键决策、遇到的问题）
  2. 新产生的待办事项
  3. 任何值得记录的架构决策或技术选型
- 一次调用批量保存，不要拆成多次'

# Check if already injected
if [ -f "$CLAUDE_MD" ] && grep -q "Claude AI Memory" "$CLAUDE_MD"; then
    echo "CLAUDE.md already contains memory instructions. Skipping."
    exit 0
fi

# Append
echo "" >> "$CLAUDE_MD"
echo "$MEMORY_BLOCK" >> "$CLAUDE_MD"
echo "Injected memory instructions into $CLAUDE_MD"
