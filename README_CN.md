# claude-ai-memory

一个 MCP (Model Context Protocol) Server，为 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 提供持久化的项目上下文存储、管理和语义搜索能力。基于 [OpenViking](https://github.com/volcengine/OpenViking) 上下文数据库构建。

**[English Documentation](./README.md)**

## 要解决的问题

Claude Code 的会话是无状态的——关闭终端后，对话中的一切都会丢失：

- 为什么做出某个技术选型
- 上次会话讨论了什么、还有什么没做完
- 客户端交流中的关键信息
- 跨多次会话积累的项目理解

项目文件能记录代码变更，但记不住变更背后的思考、讨论和上下文。

## 解决方案

claude-ai-memory 为 Claude Code 提供**持久化的项目记忆层**：

- **自动会话摘要** — 每次会话结束时，自动保存关键决策、变更和待办事项
- **手动记忆捕获** — 通过 `/remember` 命令主动存储重要信息
- **语义搜索** — 用自然语言查询历史上下文
- **智能上下文加载** — 会话启动时加载项目概览，工作中按需检索详情
- **多项目隔离** — 每个项目拥有独立的上下文空间，互不干扰

## 架构

```
┌──────────────────────────────────────────────────────┐
│                   Docker Compose                      │
│                                                       │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ openviking   │  │ embedding  │  │ ollama       │  │
│  │ (Rust)       │  │ (FastAPI + │  │ Qwen3-0.6B   │  │
│  │ :1933        │←─│ MiniLM-L6) │  │ :11434       │  │
│  │              │  │ :8100      │  │ (备用)        │  │
│  └──────┬───────┘  └────────────┘  └──────────────┘  │
│         │  /var/lib/openviking/data (持久化存储)       │
└─────────┼────────────────────────────────────────────┘
          ↑ HTTP
┌─────────┴────────────────┐
│  claude-ai-memory MCP    │
│  (Python, stdio)         │
└─────────┬────────────────┘
          ↑ MCP stdio
┌─────────┴────────┐
│  Claude Code CLI  │
│  + CLAUDE.md      │
└──────────────────┘
```

### 技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| MCP Server | Python + `mcp` SDK | stdio 传输，暴露 4 个 tools |
| 上下文数据库 | [OpenViking](https://github.com/volcengine/OpenViking) (Docker) | 文件系统范式，L0/L1/L2 分层检索 |
| Embedding | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (本地) | 384 维语义向量，零 API 费用 |
| VLM（主力） | [MiniMax M2.5](https://platform.minimaxi.com) | Anthropic SDK 兼容，用于生成摘要 |
| VLM（备用） | [Qwen3-0.6B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-FP8) via Ollama | 本地 CPU 推理，API 不可用时自动切换 |

### 费用

| 项目 | 费用 |
|------|------|
| OpenViking | 免费（Apache 2.0） |
| Embedding（本地） | 免费 |
| Ollama（本地） | 免费 |
| MiniMax M2.5 API | 约 ¥6/月（$0.30/M input + $1.20/M output） |
| **总计** | **约 ¥6/月** |

## MCP Tools

只暴露 4 个粗粒度 tool，最小化 API 调用次数：

### `context_load` — 加载项目上下文

会话启动时调用一次，返回项目概览、最近会话摘要、待办事项和活跃决策。

```json
{
  "project_path": "/home/user/my-project"
}
```

返回示例：

```json
{
  "project_id": "a1b2c3d4e5f6",
  "registered": true,
  "last_session": "2026-02-28T15:30:00Z",
  "summary": "这是一个 Next.js 电商项目，使用 Supabase 后端...",
  "recent_sessions": [
    {"date": "2026-02-28", "summary": "完成了购物车功能，待解决：支付接口对接"},
    {"date": "2026-02-27", "summary": "重构了认证模块，改用 JWT"}
  ],
  "pending_items": ["支付接口对接", "商品图片 CDN 迁移"],
  "active_decisions": ["数据库从 PostgreSQL 迁移到 Supabase 的方案已确认"]
}
```

### `context_search` — 语义搜索

在项目记忆中进行语义搜索，直接返回 L1 级别内容，无需二次调用。

```json
{
  "project_path": "/home/user/my-project",
  "query": "为什么选择 Stripe 做支付",
  "scope": "decisions",
  "limit": 5
}
```

返回示例：

```json
{
  "results": [
    {
      "type": "decision",
      "date": "2026-02-25",
      "relevance": 0.87,
      "content": "讨论了支付接口选型：最终选 Stripe，因为文档好、国际化支持好。排除了支付宝国际版（SDK 质量差）。"
    }
  ]
}
```

### `context_save` — 批量保存

一次调用保存多条记忆条目，支持的类型：

| 类型 | 说明 | 示例 |
|------|------|------|
| `session_summary` | 会话摘要 | 本次做了什么、遇到什么问题 |
| `decision` | 架构/技术决策 | 为什么选 A 不选 B |
| `change` | 重要变更 | 重构了某模块、改了 API 设计 |
| `knowledge` | 开发资料 | 第三方库使用心得、调试经验 |
| `todo` | 待办事项 | 下次要做的事 |

```json
{
  "project_path": "/home/user/my-project",
  "entries": [
    {
      "type": "session_summary",
      "title": "实现用户注册流程",
      "content": "完成了邮箱注册、验证码发送、密码强度校验。使用 Resend 发邮件。待做：OAuth 登录。"
    },
    {
      "type": "decision",
      "title": "邮件服务选型",
      "content": "选择 Resend 而非 SendGrid，原因：API 更简洁，免费额度够用。"
    },
    {
      "type": "todo",
      "content": "实现 Google OAuth 登录"
    }
  ]
}
```

### `context_manage` — 管理操作

低频管理操作：浏览目录、删除条目、更新内容、初始化项目。

### 典型会话调用频率

| 场景 | 调用次数 |
|------|---------|
| 会话启动 | 1 次 `context_load` |
| 正常工作（30 分钟会话） | 0-3 次 `context_search` |
| 会话结束 | 1 次 `context_save`（批量） |
| 手动 /remember | 1 次 `context_save` |
| **典型会话总计** | **2-5 次** |

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

项目标识：项目根目录绝对路径的 SHA256 前 12 位，确保唯一且稳定。

## 部署

### 前置条件

- Docker & Docker Compose v2
- Python 3.10+
- MiniMax API key（[在此获取](https://platform.minimaxi.com)）

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/alanshaka-real/claude-ai-memory.git
cd claude-ai-memory

# 2. 配置
cp config/ov.conf.example config/ov.conf
# 编辑 config/ov.conf，填入你的 MiniMax API key

# 3. 启动服务
docker compose up -d

# 4. 拉取 Ollama 备用模型
docker compose exec ollama ollama pull qwen3:0.6b

# 5. 将 MCP Server 添加到 Claude Code
# 在 ~/.claude/settings.json 中添加：
# {
#   "mcpServers": {
#     "claude-ai-memory": {
#       "command": "python",
#       "args": ["/path/to/claude-ai-memory/mcp-server/server.py"]
#     }
#   }
# }
```

### 升级（数据零影响）

```bash
docker compose pull openviking   # 拉取最新镜像
docker compose up -d openviking  # 重建容器，宿主机数据不受影响
docker compose ps                # 确认 healthy 状态
```

数据存储在宿主机 `/var/lib/openviking/data`，通过 bind mount 挂载——容器升级永远不会影响你的数据。

### Docker Compose 服务

| 服务 | 镜像 | 端口 | 用途 |
|------|------|------|------|
| openviking | `ghcr.io/volcengine/openviking:main` | 1933 | 上下文数据库 |
| embedding | 自建（FastAPI + sentence-transformers） | 8100 | 本地 embedding 服务 |
| ollama | `ollama/ollama:latest` | 11434 | 备用本地 LLM |

## 错误处理

所有后端故障都优雅处理——**Claude Code 始终正常工作**，无论记忆系统是否可用：

| 故障场景 | 行为 |
|---------|------|
| OpenViking 不可用 | `context_load` 返回 `{"available": false}`，Claude 正常工作但无历史上下文 |
| Embedding 不可用 | 写入正常，搜索降级为目录浏览 |
| MiniMax API 不可用 | 自动切换到本地 Ollama Qwen3-0.6B |
| Ollama 也不可用 | 保存原文，摘要进入重试队列 |
| 项目未注册 | 返回 `{"registered": false}`，提示初始化 |

## 工作原理

项目初始化后，claude-ai-memory 会在项目的 `CLAUDE.md` 中注入使用说明：

- **会话启动**：Claude 调用 `context_load` 了解项目状态
- **工作中**：Claude 在需要历史背景时调用 `context_search`
- **会话结束**：Claude 调用 `context_save` 保存本次会话知识
- **手动保存**：用户触发 `/remember` 主动存储信息

这形成了一个自然的记忆循环，Claude 跨会话逐步建立对项目的理解。

## VLM Fallback 策略

```
context_save(原文)
  ├─ 立即：原文写入 OpenViking → 返回成功
  └─ 异步后台：
      ├─ 尝试 MiniMax M2.5 API 生成摘要
      ├─ 失败 → 自动切换到本地 Ollama Qwen3-0.6B
      ├─ 仍失败 → 进入重试队列（最多 3 次，间隔递增）
      └─ 全部失败 → 仅保留原文，不生成摘要
```

## 项目结构

```
claude-ai-memory/
├── docker-compose.yml              # 三个服务：openviking, embedding, ollama
├── config/
│   └── ov.conf                     # OpenViking 配置
├── embedding-service/
│   ├── Dockerfile
│   ├── main.py                     # FastAPI embedding 服务
│   └── requirements.txt
├── mcp-server/
│   ├── __init__.py
│   ├── server.py                   # MCP Server 入口，4 个 tools
│   ├── viking_client.py            # OpenViking SDK 封装
│   ├── project_manager.py          # 项目隔离、URI 管理
│   └── models.py                   # 数据模型
├── tests/
│   ├── unit/                       # 单元测试
│   ├── integration/                # 集成测试
│   └── e2e/                        # 端到端测试
├── scripts/
│   ├── setup.sh                    # 一键部署脚本
│   └── inject-claude-md.sh         # CLAUDE.md 注入脚本
├── docs/
│   └── plans/
│       └── 2026-03-01-claude-ai-memory-design.md
├── pyproject.toml
└── README.md
```

## 测试

```bash
# 单元测试
pytest tests/unit/

# 集成测试（需要 Docker 服务运行）
pytest tests/integration/

# 端到端测试
pytest tests/e2e/
```

关键测试场景：
1. 项目初始化 → 保存 → 搜索 → 加载 完整流程
2. 批量保存多种类型条目
3. 后端故障时的优雅降级
4. 多项目隔离（项目 A 的数据搜不到项目 B 的内容）

## 许可证

[Apache License 2.0](./LICENSE)

## 致谢

- [OpenViking](https://github.com/volcengine/OpenViking) — AI Agent 上下文数据库
- [MiniMax](https://www.minimax.io) — 高性价比 LLM API
- [sentence-transformers](https://www.sbert.net) — 本地 embedding 模型
- [Ollama](https://ollama.ai) — 本地 LLM 推理
