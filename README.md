# claude-ai-memory

An MCP (Model Context Protocol) Server that provides persistent project context storage, management, and semantic search for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Powered by [OpenViking](https://github.com/volcengine/OpenViking) context database.

**[中文文档](./README_CN.md)**

## The Problem

Claude Code sessions are stateless. When you close the terminal, everything discussed is lost:

- Why a certain technical decision was made
- What was discussed in the last session and what's still pending
- Key information from client communications
- Project understanding accumulated across multiple sessions

Project files capture code changes, but not the reasoning, discussions, and context behind them.

## The Solution

claude-ai-memory gives Claude Code a **persistent memory layer** for each project:

- **Automatic session summaries** — Key decisions, changes, and TODOs are saved at the end of each session
- **Manual memory capture** — Use `/remember` to explicitly store important information
- **Semantic search** — Find relevant historical context using natural language queries
- **Smart context loading** — Project overview is loaded at session startup, detailed context is retrieved on-demand
- **Multi-project isolation** — Each project has its own independent context space

## Architecture

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
│         │  /var/lib/openviking/data (persistent)      │
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

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| MCP Server | Python + `mcp` SDK | Exposes 4 tools to Claude Code via stdio transport |
| Context Database | [OpenViking](https://github.com/volcengine/OpenViking) (Docker) | Filesystem-paradigm context DB with L0/L1/L2 tiered retrieval |
| Embedding | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (local) | Semantic vector encoding, 384 dimensions, zero API cost |
| VLM (primary) | [MiniMax M2.5](https://platform.minimaxi.com) | Summary generation via Anthropic-compatible API |
| VLM (fallback) | [Qwen3-0.6B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-FP8) via Ollama | Local CPU inference when API is unavailable |

### Cost

| Item | Cost |
|------|------|
| OpenViking | Free (Apache 2.0) |
| Embedding (local) | Free |
| Ollama (local) | Free |
| MiniMax M2.5 API | ~$1/month ($0.30/M input + $1.20/M output) |
| **Total** | **~$1/month** |

## MCP Tools

Only 4 coarse-grained tools are exposed to minimize API calls:

### `context_load`

Called once at session startup. Returns project overview, recent session summaries, pending items, and active decisions in a single response.

```json
{
  "project_path": "/home/user/my-project"
}
```

### `context_search`

Semantic search across project memory. Returns L1-level content directly — no follow-up calls needed.

```json
{
  "project_path": "/home/user/my-project",
  "query": "why did we choose Stripe for payments",
  "scope": "decisions",
  "limit": 5
}
```

### `context_save`

Batch save multiple memory entries in a single call.

```json
{
  "project_path": "/home/user/my-project",
  "entries": [
    {
      "type": "session_summary",
      "title": "Implemented user registration",
      "content": "Completed email registration, verification, password strength check. Used Resend for email. TODO: OAuth login."
    },
    {
      "type": "decision",
      "title": "Email service selection",
      "content": "Chose Resend over SendGrid: simpler API, sufficient free tier."
    }
  ]
}
```

### `context_manage`

Low-frequency management operations: list, delete, update, and project initialization.

### Typical Session Call Pattern

| Event | Calls |
|-------|-------|
| Session start | 1x `context_load` |
| During work (30 min session) | 0-3x `context_search` |
| Session end | 1x `context_save` (batch) |
| Manual `/remember` | 1x `context_save` |
| **Typical total** | **2-5 calls** |

## Multi-Project Isolation

Each project maps to an independent URI namespace in OpenViking's filesystem paradigm:

```
viking://
├── projects/
│   ├── <project-hash-1>/        # Project A
│   │   ├── sessions/            # Session summaries
│   │   ├── decisions/           # Architecture decisions
│   │   ├── changes/             # Change records
│   │   └── knowledge/           # Dev resources, notes
│   ├── <project-hash-2>/        # Project B
│   └── ...
└── global/                      # Cross-project knowledge
```

Project ID: first 12 characters of SHA256 hash of the project root absolute path.

## Deployment

### Prerequisites

- Docker & Docker Compose v2
- Python 3.10+
- MiniMax API key ([get one here](https://platform.minimaxi.com))

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/alanshaka-real/claude-ai-memory.git
cd claude-ai-memory

# 2. Configure
cp config/ov.conf.example config/ov.conf
# Edit config/ov.conf with your MiniMax API key

# 3. Start services
docker compose up -d

# 4. Pull Ollama fallback model
docker compose exec ollama ollama pull qwen3:0.6b

# 5. Add MCP server to Claude Code
# Add to ~/.claude/settings.json:
# {
#   "mcpServers": {
#     "claude-ai-memory": {
#       "command": "python",
#       "args": ["/path/to/claude-ai-memory/mcp-server/server.py"]
#     }
#   }
# }
```

### Upgrading (Zero Data Loss)

```bash
docker compose pull openviking   # Pull latest image
docker compose up -d openviking  # Recreate container, data untouched
docker compose ps                # Verify healthy status
```

Data is stored on the host at `/var/lib/openviking/data` via bind mount — container upgrades never affect your data.

### Docker Compose Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| openviking | `ghcr.io/volcengine/openviking:main` | 1933 | Context database |
| embedding | Custom (FastAPI + sentence-transformers) | 8100 | Local embedding service |
| ollama | `ollama/ollama:latest` | 11434 | Fallback LLM |

## Error Handling

All backend failures are gracefully handled — **Claude Code always works normally**, with or without memory:

| Failure | Behavior |
|---------|----------|
| OpenViking unavailable | `context_load` returns `{"available": false}`, Claude works without history |
| Embedding unavailable | Writes succeed, search degrades to directory browsing |
| MiniMax API unavailable | Auto-fallback to local Ollama Qwen3-0.6B |
| Ollama also unavailable | Raw text saved, summaries queued for retry |
| Project not registered | Returns `{"registered": false}`, prompts initialization |

## How It Works with Claude Code

When a project is initialized, claude-ai-memory injects instructions into the project's `CLAUDE.md`:

- **Session start**: Claude calls `context_load` to understand project state
- **During work**: Claude calls `context_search` when historical context is needed
- **Session end**: Claude calls `context_save` to preserve session knowledge
- **Manual save**: User triggers `/remember` to explicitly store information

This creates a natural memory loop where Claude progressively builds project understanding across sessions.

## Project Structure

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
│   ├── server.py
│   ├── viking_client.py
│   ├── project_manager.py
│   └── models.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
│   ├── setup.sh
│   └── inject-claude-md.sh
├── docs/
│   └── plans/
│       └── 2026-03-01-claude-ai-memory-design.md
├── pyproject.toml
└── README.md
```

## License

[Apache License 2.0](./LICENSE)

## Acknowledgments

- [OpenViking](https://github.com/volcengine/OpenViking) — Context database for AI Agents
- [MiniMax](https://www.minimax.io) — Cost-effective LLM API
- [sentence-transformers](https://www.sbert.net) — Local embedding models
- [Ollama](https://ollama.ai) — Local LLM inference
