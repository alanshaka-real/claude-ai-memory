# claude-ai-memory 设计文档

## 概述

claude-ai-memory 是一个 MCP Server，为 Claude Code 提供项目级的上下文存储、管理和语义搜索能力。它基于 OpenViking 上下文数据库，使 Claude Code 能够跨会话记住项目的对话摘要、架构决策、变更记录和开发资料。

## 核心问题

Claude Code 的会话是无状态的——关闭终端后，对话中的决策、讨论、上下文全部丢失。项目文件能记录代码变更，但无法记录：

- 为什么做出某个技术选型
- 上次会话讨论了什么、还有什么没做完
- 客户端交流中的关键信息
- 跨多次会话积累的项目理解

## 架构

### 组件拓扑

```
┌──────────────────────────────────────────────────────┐
│                   Docker Compose                      │
│                                                       │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ openviking   │  │ embedding  │  │ ollama       │  │
│  │ (Rust)       │  │ (FastAPI + │  │ Qwen3-0.6B   │  │
│  │ :1933        │←─│ MiniLM-L6) │  │ :11434       │  │
│  │              │  │ :8100      │  │ (fallback)   │  │
│  └──────┬───────┘  └────────────┘  └──────────────┘  │
│         │  openviking_data (named volume, 持久化)      │
└─────────┼────────────────────────────────────────────┘
          ↑ HTTP
┌─────────┴────────────────┐
│  claude-ai-memory MCP    │
│  (Python, stdio)         │
└─────────┬────────────────┘
          ↑ MCP stdio
┌─────────┴────────┐
│  Claude Code CLI  │
│  + CLAUDE.md 提示  │
└──────────────────┘
```

### 技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| MCP Server | Python + `mcp` SDK | stdio 传输 |
| 上下文数据库 | OpenViking (Docker) | 文件系统范式，L0/L1/L2 分层 |
| Embedding | all-MiniLM-L6-v2 (本地) | FastAPI 服务，OpenAI 兼容 /v1/embeddings 接口，维度 384 |
| VLM（主力） | MiniMax M2.5 | Coding Plan Key，Anthropic SDK 兼容，~¥6/月 |
| VLM（备用） | Qwen3-0.6B-FP8 (Ollama) | 本地 CPU 推理，API 不可用时自动 fallback |
| 容器编排 | Docker Compose v2 | 命名卷持久化 |

### 费用

| 项目 | 费用 |
|------|------|
| OpenViking | 免费（Apache 2.0） |
| Embedding (本地) | 免费 |
| Ollama (本地) | 免费 |
| MiniMax M2.5 API | ~¥6/月（$0.30/M input + $1.20/M output） |
| **总计** | **~¥6/月** |

## 多项目隔离

利用 OpenViking 的文件系统范式，每个项目映射为独立的 URI 命名空间：

```
viking://
├── projects/
│   ├── <project-hash-1>/        # 项目 A
│   │   ├── sessions/            # 对话摘要
│   │   ├── decisions/           # 架构决策
│   │   ├── changes/             # 重要变更记录
│   │   └── knowledge/           # 开发资料、笔记
│   ├── <project-hash-2>/        # 项目 B
│   │   └── ...
│   └── ...
└── global/                      # 跨项目通用知识
```

项目标识：项目根目录绝对路径的 SHA256 前 12 位。

## MCP Tools 设计

只暴露 4 个粗粒度 tool，最小化调用次数：

### 1. context_load

会话启动时调用一次，返回项目上下文概览。

```json
{
  "name": "context_load",
  "inputSchema": {
    "type": "object",
    "properties": {
      "project_path": { "type": "string", "description": "项目根目录绝对路径" }
    },
    "required": ["project_path"]
  }
}
```

**返回**：

```json
{
  "project_id": "a1b2c3d4e5f6",
  "registered": true,
  "last_session": "2026-02-28T15:30:00Z",
  "summary": "项目概述...",
  "recent_sessions": [
    {"date": "2026-02-28", "summary": "完成了购物车功能..."}
  ],
  "pending_items": ["支付接口对接"],
  "active_decisions": ["数据库方案已确认"]
}
```

如果项目未注册，返回 `{"registered": false}`。

### 2. context_search

语义搜索项目记忆，按需调用。

```json
{
  "name": "context_search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "project_path": { "type": "string" },
      "query": { "type": "string", "description": "自然语言搜索查询" },
      "scope": {
        "type": "string",
        "enum": ["all", "sessions", "decisions", "changes", "knowledge"],
        "default": "all"
      },
      "limit": { "type": "integer", "default": 5 }
    },
    "required": ["project_path", "query"]
  }
}
```

**返回**：直接包含 L1 级别内容，避免二次调用。

```json
{
  "results": [
    {
      "type": "session",
      "date": "2026-02-25",
      "relevance": 0.87,
      "content": "讨论了支付接口选型：最终选 Stripe..."
    }
  ]
}
```

### 3. context_save

批量写入记忆条目。

```json
{
  "name": "context_save",
  "inputSchema": {
    "type": "object",
    "properties": {
      "project_path": { "type": "string" },
      "entries": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "type": {
              "type": "string",
              "enum": ["session_summary", "decision", "change", "knowledge", "todo"]
            },
            "title": { "type": "string" },
            "content": { "type": "string" },
            "tags": { "type": "array", "items": { "type": "string" } }
          },
          "required": ["type", "content"]
        }
      }
    },
    "required": ["project_path", "entries"]
  }
}
```

### 4. context_manage

低频管理操作。

```json
{
  "name": "context_manage",
  "inputSchema": {
    "type": "object",
    "properties": {
      "project_path": { "type": "string" },
      "action": {
        "type": "string",
        "enum": ["list", "delete", "update", "init_project"]
      },
      "target": { "type": "string", "description": "操作目标 URI 或 ID" },
      "content": { "type": "string", "description": "更新时的新内容" }
    },
    "required": ["project_path", "action"]
  }
}
```

### 调用频率预期

| 场景 | 调用次数 |
|------|---------|
| 会话启动 | 1 次 context_load |
| 正常工作（30 分钟会话） | 0-3 次 context_search |
| 会话结束 | 1 次 context_save（批量） |
| 手动 /remember | 1 次 context_save |
| **典型会话总计** | **2-5 次** |

## 数据流

### 会话生命周期

```
会话启动
  ├─ Claude 读取 CLAUDE.md → 发现 "调用 context_load"
  ├─ context_load(project_path) → 返回项目概览
  ├─ Claude 获得上下文，开始工作
  │   ├─（遇到相关问题时）context_search(query)
  │   └─（用户 /remember）context_save(entries)
  └─ 会话结束前
      └─ context_save([session_summary, decisions, todos])
```

### CLAUDE.md 注入内容

项目初始化时追加到 CLAUDE.md：

```markdown
## Claude AI Memory

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
- 一次调用批量保存，不要拆成多次
```

## Docker 部署

### docker-compose.yml

```yaml
version: '3.8'
services:
  openviking:
    image: ghcr.io/volcengine/openviking:main   # 官方 GHCR 镜像
    container_name: openviking
    ports: ["1933:1933"]
    volumes:
      # 数据和配置都在宿主机上，容器销毁不影响数据
      - /var/lib/openviking/data:/app/data
      - ./config/ov.conf:/app/ov.conf:ro
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:1933/health || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    restart: unless-stopped

  embedding:
    build: ./embedding-service
    ports: ["8100:8100"]
    volumes:
      - model_cache:/root/.cache/torch
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  model_cache:
    name: claude-ai-memory-models
  ollama_data:
    name: claude-ai-memory-ollama
```

### 升级流程（数据零影响）

```bash
docker compose pull openviking   # 拉取最新 ghcr.io 镜像
docker compose up -d openviking  # 重建容器，宿主机 /var/lib/openviking/data 不受影响
docker compose ps                # 确认 healthy 状态
```

数据安全保障：
- `/var/lib/openviking/data` 是宿主机 bind mount，容器删除数据仍在
- `ov.conf` 只读挂载，也在宿主机上
- OpenViking 更新活跃（1-3 天一版），当前 v0.2.1

### OpenViking 配置 (config/ov.conf)

```json
{
  "storage": {
    "workspace": "/app/data"
  },
  "embedding": {
    "dense": {
      "api_base": "http://embedding:8100/v1",
      "api_key": "local",
      "provider": "openai",
      "dimension": 384,
      "model": "all-MiniLM-L6-v2"
    }
  },
  "vlm": {
    "api_base": "https://api.minimaxi.com/anthropic",
    "api_key": "${MINIMAX_API_KEY}",
    "provider": "litellm",
    "model": "MiniMax-M2.5",
    "max_concurrent": 10
  }
}
```

## Embedding Service

轻量 FastAPI 服务，暴露 OpenAI 兼容接口：

```python
# embedding-service/main.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    vectors = model.encode(request.input).tolist()
    return {
        "data": [{"embedding": v, "index": i} for i, v in enumerate(vectors)],
        "model": "all-MiniLM-L6-v2",
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }
```

## VLM Fallback 策略

```
context_save(原文)
  ├─ 立即: 原文写入 OpenViking → 返回成功
  └─ 异步后台:
      ├─ 尝试 MiniMax M2.5 API 生成摘要
      ├─ 失败 → 切换到本地 Ollama Qwen3-0.6B
      ├─ 仍失败 → 进入重试队列（最多 3 次，间隔递增）
      └─ 全部失败 → 仅保留原文，不生成摘要
```

## 错误处理

| 故障场景 | 处理方式 |
|---------|---------|
| OpenViking 不可用 | context_load 返回 `{"available": false}`，Claude 正常工作 |
| Embedding 不可用 | 写入正常，搜索降级为目录浏览 |
| MiniMax API 不可用 | 自动 fallback 到 Ollama Qwen3-0.6B |
| Ollama 也不可用 | 原文存储，摘要异步重试 |
| 项目未注册 | 返回 `{"registered": false}`，提示初始化 |

核心原则：**所有后端故障都不阻塞 Claude Code 的正常工作。**

## 项目结构

```
claude-ai-memory/
├── docker-compose.yml
├── config/
│   └── ov.conf
├── embedding-service/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── mcp-server/
│   ├── __init__.py
│   ├── server.py                  # MCP Server 入口，4 个 tools
│   ├── viking_client.py           # OpenViking SDK 封装
│   ├── project_manager.py         # 项目隔离、URI 管理
│   └── models.py                  # 数据模型
├── tests/
│   ├── unit/
│   │   ├── test_project_manager.py
│   │   ├── test_models.py
│   │   └── test_viking_client.py
│   ├── integration/
│   │   ├── test_mcp_tools.py
│   │   ├── test_embedding.py
│   │   └── test_ollama.py
│   └── e2e/
│       └── test_full_workflow.py
├── scripts/
│   ├── setup.sh                   # 一键部署
│   └── inject-claude-md.sh        # CLAUDE.md 注入
├── pyproject.toml
└── README.md
```

## 测试策略

1. **单元测试**：项目 ID 生成、URI 映射、数据模型序列化
2. **集成测试**：4 个 MCP tool 端到端、embedding/ollama 健康检查
3. **E2E 测试**：init → save → search → load 完整流程
4. **关键场景**：多项目隔离、后端故障优雅降级
