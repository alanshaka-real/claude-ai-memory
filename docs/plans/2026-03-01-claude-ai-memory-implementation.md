# claude-ai-memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an MCP Server that gives Claude Code persistent project memory using OpenViking, with local embedding and MiniMax M2.5 / Ollama fallback for summarization.

**Architecture:** Python MCP Server (FastMCP, stdio transport) wraps OpenViking Python SDK. Docker Compose runs OpenViking + embedding service + Ollama. Each project gets an isolated URI namespace in OpenViking's filesystem.

**Tech Stack:** Python 3.10+, FastMCP (`mcp` SDK), OpenViking Python SDK (`openviking`), FastAPI + sentence-transformers (embedding), Docker Compose v2, Ollama (Qwen3-0.6B fallback)

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `mcp-server/__init__.py`
- Create: `tests/__init__.py`
- Create: `config/ov.conf.example`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "claude-ai-memory"
version = "0.1.0"
description = "MCP Server for Claude Code project context memory"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.0.0",
    "openviking>=0.2.0",
    "pydantic>=2.0.0",
    "httpx>=0.25.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 2: Create empty init files**

```bash
mkdir -p mcp-server tests/unit tests/integration tests/e2e
touch mcp-server/__init__.py tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/e2e/__init__.py
```

**Step 3: Create config example**

Create `config/ov.conf.example`:
```json
{
  "storage": {
    "workspace": "/app/data"
  },
  "log": {
    "level": "INFO",
    "output": "stdout"
  },
  "embedding": {
    "dense": {
      "api_base": "http://embedding:8100/v1",
      "api_key": "local",
      "provider": "openai",
      "dimension": 384,
      "model": "all-MiniLM-L6-v2"
    },
    "max_concurrent": 10
  },
  "vlm": {
    "api_base": "https://api.minimaxi.com/anthropic",
    "api_key": "YOUR_MINIMAX_API_KEY_HERE",
    "provider": "litellm",
    "model": "MiniMax-M2.5",
    "max_concurrent": 10
  }
}
```

**Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

**Step 5: Commit**

```bash
git add pyproject.toml mcp-server/ tests/ config/ov.conf.example
git commit -m "feat: project scaffolding and dependencies"
```

---

### Task 2: Docker infrastructure — embedding service

**Files:**
- Create: `embedding-service/main.py`
- Create: `embedding-service/requirements.txt`
- Create: `embedding-service/Dockerfile`

**Step 1: Write the embedding service test**

Create `tests/integration/test_embedding.py`:
```python
import httpx
import pytest

EMBEDDING_URL = "http://localhost:8100"

@pytest.mark.integration
def test_health():
    r = httpx.get(f"{EMBEDDING_URL}/health")
    assert r.status_code == 200

@pytest.mark.integration
def test_single_embedding():
    r = httpx.post(f"{EMBEDDING_URL}/v1/embeddings", json={
        "input": "hello world",
        "model": "all-MiniLM-L6-v2"
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 384

@pytest.mark.integration
def test_batch_embedding():
    r = httpx.post(f"{EMBEDDING_URL}/v1/embeddings", json={
        "input": ["hello", "world"],
        "model": "all-MiniLM-L6-v2"
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 2
```

**Step 2: Write embedding-service/requirements.txt**

```
fastapi==0.115.*
uvicorn==0.34.*
sentence-transformers==3.4.*
pydantic==2.*
```

**Step 3: Write embedding-service/main.py**

```python
from __future__ import annotations
import time
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Embedding Service")
model = SentenceTransformer("all-MiniLM-L6-v2")

class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = "all-MiniLM-L6-v2"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    vectors = model.encode(texts).tolist()
    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=v, index=i)
            for i, v in enumerate(vectors)
        ],
        model=request.model,
        usage={"prompt_tokens": 0, "total_tokens": 0},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
```

**Step 4: Write embedding-service/Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY main.py .

EXPOSE 8100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
```

**Step 5: Commit**

```bash
git add embedding-service/ tests/integration/test_embedding.py
git commit -m "feat: embedding service with OpenAI-compatible API"
```

---

### Task 3: Docker Compose — all services

**Files:**
- Create: `docker-compose.yml`

**Step 1: Write docker-compose.yml**

```yaml
version: '3.8'

services:
  openviking:
    image: ghcr.io/volcengine/openviking:main
    container_name: openviking
    ports:
      - "1933:1933"
    volumes:
      - /var/lib/openviking/data:/app/data
      - ./config/ov.conf:/app/ov.conf:ro
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:1933/health || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    depends_on:
      embedding:
        condition: service_started

  embedding:
    build: ./embedding-service
    container_name: embedding
    ports:
      - "8100:8100"
    volumes:
      - model_cache:/root/.cache/torch
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:8100/health || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  model_cache:
    name: claude-ai-memory-models
  ollama_data:
    name: claude-ai-memory-ollama
```

**Step 2: Create host data directory**

```bash
sudo mkdir -p /var/lib/openviking/data
sudo chown $USER:$USER /var/lib/openviking/data
```

**Step 3: Copy config**

```bash
cp config/ov.conf.example config/ov.conf
# Edit config/ov.conf: replace YOUR_MINIMAX_API_KEY_HERE with real key
```

**Step 4: Build and start services**

```bash
docker compose build embedding
docker compose up -d
```

**Step 5: Verify all services healthy**

```bash
docker compose ps
# Expected: all 3 services running
curl -s http://localhost:1933/health
# Expected: 200 OK
curl -s http://localhost:8100/health
# Expected: {"status":"ok"}
curl -s http://localhost:11434/api/tags
# Expected: 200 with JSON
```

**Step 6: Pull Ollama fallback model**

```bash
docker compose exec ollama ollama pull qwen3:0.6b
```

**Step 7: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: docker-compose with openviking, embedding, and ollama"
```

---

### Task 4: Data models

**Files:**
- Create: `mcp-server/models.py`
- Create: `tests/unit/test_models.py`

**Step 1: Write tests for data models**

Create `tests/unit/test_models.py`:
```python
import pytest
from mcp_server.models import MemoryEntry, EntryType, ProjectContext, SearchResult

def test_memory_entry_required_fields():
    entry = MemoryEntry(type=EntryType.SESSION_SUMMARY, content="test content")
    assert entry.type == EntryType.SESSION_SUMMARY
    assert entry.content == "test content"
    assert entry.title is None
    assert entry.tags == []

def test_memory_entry_all_fields():
    entry = MemoryEntry(
        type=EntryType.DECISION,
        title="Use Stripe",
        content="Chose Stripe for payments",
        tags=["payments", "stripe"],
    )
    assert entry.title == "Use Stripe"
    assert entry.tags == ["payments", "stripe"]

def test_entry_type_values():
    assert EntryType.SESSION_SUMMARY == "session_summary"
    assert EntryType.DECISION == "decision"
    assert EntryType.CHANGE == "change"
    assert EntryType.KNOWLEDGE == "knowledge"
    assert EntryType.TODO == "todo"

def test_project_context_not_registered():
    ctx = ProjectContext(registered=False)
    assert ctx.registered is False
    assert ctx.project_id is None

def test_project_context_registered():
    ctx = ProjectContext(
        registered=True,
        project_id="abc123def456",
        summary="A web project",
        recent_sessions=[{"date": "2026-03-01", "summary": "init"}],
        pending_items=["deploy"],
        active_decisions=["use postgres"],
    )
    assert ctx.project_id == "abc123def456"
    assert len(ctx.recent_sessions) == 1

def test_search_result():
    r = SearchResult(type="decision", date="2026-03-01", relevance=0.87, content="chose X")
    assert r.relevance == 0.87
```

**Step 2: Run tests — expect failure**

```bash
pytest tests/unit/test_models.py -v
```
Expected: ModuleNotFoundError

**Step 3: Write mcp-server/models.py**

```python
from __future__ import annotations
from enum import StrEnum
from typing import Any, Optional
from pydantic import BaseModel, Field


class EntryType(StrEnum):
    SESSION_SUMMARY = "session_summary"
    DECISION = "decision"
    CHANGE = "change"
    KNOWLEDGE = "knowledge"
    TODO = "todo"


class MemoryEntry(BaseModel):
    type: EntryType
    title: Optional[str] = None
    content: str
    tags: list[str] = Field(default_factory=list)


class ProjectContext(BaseModel):
    registered: bool = False
    available: bool = True
    project_id: Optional[str] = None
    last_session: Optional[str] = None
    summary: Optional[str] = None
    recent_sessions: list[dict[str, Any]] = Field(default_factory=list)
    pending_items: list[str] = Field(default_factory=list)
    active_decisions: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    type: str
    date: Optional[str] = None
    relevance: float
    content: str
```

**Step 4: Run tests — expect pass**

```bash
pytest tests/unit/test_models.py -v
```
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add mcp-server/models.py tests/unit/test_models.py
git commit -m "feat: data models for memory entries and project context"
```

---

### Task 5: Project manager — project ID and URI mapping

**Files:**
- Create: `mcp-server/project_manager.py`
- Create: `tests/unit/test_project_manager.py`

**Step 1: Write tests**

Create `tests/unit/test_project_manager.py`:
```python
import pytest
from mcp_server.project_manager import ProjectManager

def test_project_id_deterministic():
    pm = ProjectManager()
    id1 = pm.get_project_id("/home/user/project-a")
    id2 = pm.get_project_id("/home/user/project-a")
    assert id1 == id2

def test_project_id_length():
    pm = ProjectManager()
    pid = pm.get_project_id("/home/user/project")
    assert len(pid) == 12

def test_project_id_unique():
    pm = ProjectManager()
    id_a = pm.get_project_id("/home/user/project-a")
    id_b = pm.get_project_id("/home/user/project-b")
    assert id_a != id_b

def test_project_uri():
    pm = ProjectManager()
    uri = pm.get_project_uri("/home/user/myapp")
    pid = pm.get_project_id("/home/user/myapp")
    assert uri == f"viking://projects/{pid}"

def test_entry_uri():
    pm = ProjectManager()
    uri = pm.get_entry_uri("/home/user/myapp", "sessions")
    pid = pm.get_project_id("/home/user/myapp")
    assert uri == f"viking://projects/{pid}/sessions"

def test_entry_uri_types():
    pm = ProjectManager()
    path = "/home/user/myapp"
    for subdir in ["sessions", "decisions", "changes", "knowledge"]:
        uri = pm.get_entry_uri(path, subdir)
        assert subdir in uri
```

**Step 2: Run tests — expect failure**

```bash
pytest tests/unit/test_project_manager.py -v
```
Expected: ModuleNotFoundError

**Step 3: Write mcp-server/project_manager.py**

```python
from __future__ import annotations
import hashlib


class ProjectManager:
    """Manages project identification and URI mapping for OpenViking."""

    ENTRY_TYPE_TO_DIR = {
        "session_summary": "sessions",
        "decision": "decisions",
        "change": "changes",
        "knowledge": "knowledge",
        "todo": "sessions",  # todos stored alongside sessions
    }

    def get_project_id(self, project_path: str) -> str:
        """Generate a deterministic 12-char project ID from absolute path."""
        return hashlib.sha256(project_path.encode()).hexdigest()[:12]

    def get_project_uri(self, project_path: str) -> str:
        """Get the OpenViking URI root for a project."""
        return f"viking://projects/{self.get_project_id(project_path)}"

    def get_entry_uri(self, project_path: str, subdir: str) -> str:
        """Get the OpenViking URI for a specific entry type directory."""
        return f"{self.get_project_uri(project_path)}/{subdir}"

    def get_dir_for_entry_type(self, entry_type: str) -> str:
        """Map an entry type to its storage directory name."""
        return self.ENTRY_TYPE_TO_DIR.get(entry_type, "knowledge")
```

**Step 4: Run tests — expect pass**

```bash
pytest tests/unit/test_project_manager.py -v
```
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add mcp-server/project_manager.py tests/unit/test_project_manager.py
git commit -m "feat: project manager for ID generation and URI mapping"
```

---

### Task 6: Viking client wrapper

**Files:**
- Create: `mcp-server/viking_client.py`
- Create: `tests/unit/test_viking_client.py`

**Step 1: Write tests (mocked OpenViking)**

Create `tests/unit/test_viking_client.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from mcp_server.viking_client import VikingClient
from mcp_server.models import MemoryEntry, EntryType

@pytest.fixture
def mock_ov():
    with patch("mcp_server.viking_client.SyncOpenViking") as MockOV:
        client_instance = MagicMock()
        client_instance.is_healthy.return_value = True
        MockOV.return_value = client_instance
        yield client_instance

@pytest.fixture
def client(mock_ov):
    return VikingClient(openviking_url="http://localhost:1933")

def test_is_available_healthy(client, mock_ov):
    assert client.is_available() is True
    mock_ov.is_healthy.assert_called_once()

def test_is_available_unhealthy(client, mock_ov):
    mock_ov.is_healthy.return_value = False
    assert client.is_available() is False

def test_project_exists_true(client, mock_ov):
    mock_ov.ls.return_value = [{"name": "sessions"}]
    assert client.project_exists("/home/user/proj") is True

def test_project_exists_false(client, mock_ov):
    mock_ov.ls.side_effect = Exception("not found")
    assert client.project_exists("/home/user/proj") is False

def test_init_project(client, mock_ov):
    client.init_project("/home/user/proj")
    assert mock_ov.mkdir.call_count >= 4  # sessions, decisions, changes, knowledge

def test_save_entries(client, mock_ov):
    entries = [
        MemoryEntry(type=EntryType.DECISION, title="Use Stripe", content="Chose Stripe"),
    ]
    client.save_entries("/home/user/proj", entries)
    mock_ov.add_resource.assert_called_once()

def test_search(client, mock_ov):
    mock_ov.find.return_value = {"results": []}
    results = client.search("/home/user/proj", "payment")
    assert isinstance(results, list)
```

**Step 2: Run tests — expect failure**

```bash
pytest tests/unit/test_viking_client.py -v
```
Expected: ModuleNotFoundError

**Step 3: Write mcp-server/viking_client.py**

```python
from __future__ import annotations
import json
import tempfile
import time
import logging
from pathlib import Path
from typing import Any, Optional

from mcp_server.models import MemoryEntry, SearchResult
from mcp_server.project_manager import ProjectManager

logger = logging.getLogger(__name__)

try:
    from openviking import SyncOpenViking
except ImportError:
    SyncOpenViking = None


class VikingClient:
    """Wraps OpenViking SDK with project-aware operations."""

    def __init__(self, openviking_url: str = "http://localhost:1933"):
        self.url = openviking_url
        self.pm = ProjectManager()
        self._ov: Optional[Any] = None
        self._connect()

    def _connect(self):
        """Initialize OpenViking client."""
        if SyncOpenViking is None:
            logger.warning("openviking package not installed")
            return
        try:
            self._ov = SyncOpenViking(url=self.url)
        except Exception as e:
            logger.error(f"Failed to connect to OpenViking: {e}")
            self._ov = None

    def is_available(self) -> bool:
        """Check if OpenViking is reachable and healthy."""
        if self._ov is None:
            return False
        try:
            return self._ov.is_healthy()
        except Exception:
            return False

    def project_exists(self, project_path: str) -> bool:
        """Check if a project is already registered."""
        if not self.is_available():
            return False
        try:
            uri = self.pm.get_project_uri(project_path)
            self._ov.ls(uri)
            return True
        except Exception:
            return False

    def init_project(self, project_path: str) -> str:
        """Initialize a new project in OpenViking. Returns project_id."""
        pid = self.pm.get_project_id(project_path)
        base_uri = self.pm.get_project_uri(project_path)
        for subdir in ["sessions", "decisions", "changes", "knowledge"]:
            self._ov.mkdir(f"{base_uri}/{subdir}")
        return pid

    def save_entries(self, project_path: str, entries: list[MemoryEntry]) -> int:
        """Save memory entries to OpenViking. Returns count saved."""
        saved = 0
        for entry in entries:
            try:
                subdir = self.pm.get_dir_for_entry_type(entry.type)
                target_uri = self.pm.get_entry_uri(project_path, subdir)
                # Write entry as a temporary file, then add as resource
                timestamp = int(time.time() * 1000)
                title_slug = (entry.title or entry.type).replace(" ", "-")[:30]
                filename = f"{timestamp}-{title_slug}.md"
                content = self._format_entry(entry)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False
                ) as f:
                    f.write(content)
                    tmp_path = f.name
                self._ov.add_resource(tmp_path, target=target_uri)
                Path(tmp_path).unlink(missing_ok=True)
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save entry: {e}")
        return saved

    def _format_entry(self, entry: MemoryEntry) -> str:
        """Format a MemoryEntry as markdown for storage."""
        lines = []
        if entry.title:
            lines.append(f"# {entry.title}")
            lines.append("")
        lines.append(f"**Type:** {entry.type}")
        if entry.tags:
            lines.append(f"**Tags:** {', '.join(entry.tags)}")
        lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append(entry.content)
        return "\n".join(lines)

    def load_context(self, project_path: str) -> dict[str, Any]:
        """Load project context overview for session startup."""
        if not self.is_available():
            return {"available": False, "reason": "OpenViking not reachable"}
        if not self.project_exists(project_path):
            return {"registered": False}

        pid = self.pm.get_project_id(project_path)
        base_uri = self.pm.get_project_uri(project_path)
        result: dict[str, Any] = {
            "registered": True,
            "available": True,
            "project_id": pid,
        }

        # Get project overview (L0/L1)
        try:
            result["summary"] = self._ov.abstract(base_uri)
        except Exception:
            result["summary"] = None

        # Recent sessions
        try:
            sessions_uri = f"{base_uri}/sessions"
            items = self._ov.ls(sessions_uri, simple=True)
            recent = sorted(items, reverse=True)[:5]
            result["recent_sessions"] = []
            for item in recent:
                try:
                    abstract = self._ov.abstract(f"{sessions_uri}/{item}")
                    result["recent_sessions"].append({
                        "name": item,
                        "summary": abstract,
                    })
                except Exception:
                    pass
        except Exception:
            result["recent_sessions"] = []

        # Pending items (search for todos)
        try:
            todo_results = self._ov.find(
                "todo pending next", target_uri=f"{base_uri}/sessions", limit=5
            )
            result["pending_items"] = [
                r.get("content", r.get("abstract", ""))
                for r in todo_results.get("results", [])
            ]
        except Exception:
            result["pending_items"] = []

        # Active decisions
        try:
            decisions_uri = f"{base_uri}/decisions"
            items = self._ov.ls(decisions_uri, simple=True)
            recent_decisions = sorted(items, reverse=True)[:3]
            result["active_decisions"] = []
            for item in recent_decisions:
                try:
                    abstract = self._ov.abstract(f"{decisions_uri}/{item}")
                    result["active_decisions"].append(abstract)
                except Exception:
                    pass
        except Exception:
            result["active_decisions"] = []

        return result

    def search(
        self,
        project_path: str,
        query: str,
        scope: str = "all",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic search across project memory."""
        base_uri = self.pm.get_project_uri(project_path)
        target_uri = base_uri
        if scope != "all":
            dir_name = self.pm.ENTRY_TYPE_TO_DIR.get(scope, scope)
            # scope might be a directory name directly
            if dir_name in ["sessions", "decisions", "changes", "knowledge"]:
                target_uri = f"{base_uri}/{dir_name}"

        try:
            results = self._ov.find(query, target_uri=target_uri, limit=limit)
            output = []
            for r in results.get("results", []):
                output.append({
                    "type": r.get("type", "unknown"),
                    "uri": r.get("uri", ""),
                    "relevance": r.get("score", 0.0),
                    "content": r.get("overview", r.get("abstract", r.get("content", ""))),
                })
            return output
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def list_entries(self, project_path: str, subdir: str = "") -> list[dict[str, Any]]:
        """List entries in a project directory."""
        base_uri = self.pm.get_project_uri(project_path)
        target = f"{base_uri}/{subdir}" if subdir else base_uri
        try:
            items = self._ov.ls(target)
            return items if isinstance(items, list) else []
        except Exception:
            return []

    def delete_entry(self, project_path: str, target_uri: str) -> bool:
        """Delete a specific entry."""
        try:
            self._ov.rm(target_uri)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def update_entry(self, project_path: str, target_uri: str, content: str) -> bool:
        """Update content of an entry by replacing the resource."""
        try:
            self._ov.rm(target_uri)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False
            ) as f:
                f.write(content)
                tmp_path = f.name
            parent_uri = "/".join(target_uri.rsplit("/", 1)[:-1])
            self._ov.add_resource(tmp_path, target=parent_uri)
            Path(tmp_path).unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
```

**Step 4: Run tests — expect pass**

```bash
pytest tests/unit/test_viking_client.py -v
```
Expected: all 7 tests PASS

**Step 5: Commit**

```bash
git add mcp-server/viking_client.py tests/unit/test_viking_client.py
git commit -m "feat: viking client wrapper with project-aware operations"
```

---

### Task 7: MCP Server — 4 tools

**Files:**
- Create: `mcp-server/server.py`

**Step 1: Write MCP server integration test**

Create `tests/integration/test_mcp_tools.py`:
```python
import json
import pytest
from unittest.mock import MagicMock, patch

# Test that the MCP server can be imported and tools are registered
def test_server_imports():
    from mcp_server.server import mcp
    assert mcp is not None

def test_server_has_four_tools():
    from mcp_server.server import mcp
    # FastMCP stores tools internally
    tools = mcp._tool_manager._tools
    assert "context_load" in tools
    assert "context_search" in tools
    assert "context_save" in tools
    assert "context_manage" in tools
    assert len(tools) == 4
```

**Step 2: Run test — expect failure**

```bash
pytest tests/integration/test_mcp_tools.py -v
```
Expected: ModuleNotFoundError

**Step 3: Write mcp-server/server.py**

```python
from __future__ import annotations
import json
import logging
import os
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from mcp_server.models import EntryType, MemoryEntry, ProjectContext
from mcp_server.viking_client import VikingClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude-ai-memory")

# Initialize MCP server
mcp = FastMCP(
    "claude-ai-memory",
    version="0.1.0",
)

# Initialize Viking client (lazy — connects on first use)
_client: Optional[VikingClient] = None


def get_client() -> VikingClient:
    global _client
    if _client is None:
        url = os.environ.get("OPENVIKING_URL", "http://localhost:1933")
        _client = VikingClient(openviking_url=url)
    return _client


# --- Tool 1: context_load ---

@mcp.tool()
def context_load(project_path: str) -> dict[str, Any]:
    """Load project context overview. Call once at session start.

    Returns project summary, recent session history, pending items,
    and active decisions. If the project is not registered, returns
    {"registered": false}.
    """
    client = get_client()
    return client.load_context(project_path)


# --- Tool 2: context_search ---

@mcp.tool()
def context_search(
    project_path: str,
    query: str,
    scope: str = "all",
    limit: int = 5,
) -> dict[str, Any]:
    """Semantic search across project memory.

    Search for relevant historical context using natural language.
    Results include L1-level content directly — no follow-up needed.

    Args:
        project_path: Absolute path to project root.
        query: Natural language search query.
        scope: Search scope — "all", "sessions", "decisions", "changes", "knowledge".
        limit: Maximum number of results (default 5).
    """
    client = get_client()
    if not client.is_available():
        return {"error": "OpenViking not available", "results": []}
    results = client.search(project_path, query, scope=scope, limit=limit)
    return {"results": results}


# --- Tool 3: context_save ---

class SaveEntry(BaseModel):
    type: str = Field(description="Entry type: session_summary, decision, change, knowledge, todo")
    title: Optional[str] = Field(default=None, description="Optional title")
    content: str = Field(description="Entry content")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


@mcp.tool()
def context_save(
    project_path: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Batch save memory entries to project context.

    Save session summaries, decisions, changes, knowledge, and TODOs.
    Multiple entries can be saved in a single call.

    Args:
        project_path: Absolute path to project root.
        entries: List of entries, each with "type", "content", and optionally "title" and "tags".
    """
    client = get_client()
    if not client.is_available():
        return {"error": "OpenViking not available", "saved": 0}

    # Auto-init project if not exists
    if not client.project_exists(project_path):
        client.init_project(project_path)

    parsed = []
    for e in entries:
        try:
            parsed.append(MemoryEntry(
                type=EntryType(e["type"]),
                title=e.get("title"),
                content=e["content"],
                tags=e.get("tags", []),
            ))
        except (KeyError, ValueError) as err:
            logger.warning(f"Skipping invalid entry: {err}")

    saved = client.save_entries(project_path, parsed)
    return {"saved": saved, "total": len(entries)}


# --- Tool 4: context_manage ---

@mcp.tool()
def context_manage(
    project_path: str,
    action: str,
    target: Optional[str] = None,
    content: Optional[str] = None,
) -> dict[str, Any]:
    """Manage project memory: list, delete, update, or initialize.

    Args:
        project_path: Absolute path to project root.
        action: One of "list", "delete", "update", "init_project".
        target: Target URI or subdirectory (for list/delete/update).
        content: New content (for update action).
    """
    client = get_client()
    if not client.is_available():
        return {"error": "OpenViking not available"}

    if action == "init_project":
        pid = client.init_project(project_path)
        return {"success": True, "project_id": pid}

    elif action == "list":
        items = client.list_entries(project_path, subdir=target or "")
        return {"items": items}

    elif action == "delete":
        if not target:
            return {"error": "target is required for delete"}
        ok = client.delete_entry(project_path, target)
        return {"success": ok}

    elif action == "update":
        if not target or not content:
            return {"error": "target and content are required for update"}
        ok = client.update_entry(project_path, target, content)
        return {"success": ok}

    else:
        return {"error": f"Unknown action: {action}"}


# --- Entry point ---

def main():
    mcp.run()


if __name__ == "__main__":
    main()
```

**Step 4: Run tests — expect pass**

```bash
pytest tests/integration/test_mcp_tools.py -v
```
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add mcp-server/server.py tests/integration/test_mcp_tools.py
git commit -m "feat: MCP server with 4 tools — context_load, search, save, manage"
```

---

### Task 8: Setup and CLAUDE.md injection scripts

**Files:**
- Create: `scripts/setup.sh`
- Create: `scripts/inject-claude-md.sh`

**Step 1: Write scripts/setup.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== claude-ai-memory setup ==="

# 1. Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# 2. Create host data dir
echo "Creating /var/lib/openviking/data..."
sudo mkdir -p /var/lib/openviking/data
sudo chown "$USER:$USER" /var/lib/openviking/data

# 3. Check config
if [ ! -f "$PROJECT_DIR/config/ov.conf" ]; then
    echo "Creating config from example..."
    cp "$PROJECT_DIR/config/ov.conf.example" "$PROJECT_DIR/config/ov.conf"
    echo "IMPORTANT: Edit config/ov.conf and set your MiniMax API key"
fi

# 4. Build and start
echo "Building and starting services..."
cd "$PROJECT_DIR"
docker compose build embedding
docker compose up -d

# 5. Wait for health
echo "Waiting for services..."
for i in {1..30}; do
    if curl -fsS http://localhost:1933/health &>/dev/null; then
        echo "OpenViking: healthy"
        break
    fi
    sleep 2
done

for i in {1..30}; do
    if curl -fsS http://localhost:8100/health &>/dev/null; then
        echo "Embedding: healthy"
        break
    fi
    sleep 2
done

# 6. Pull Ollama model
echo "Pulling Ollama fallback model..."
docker compose exec ollama ollama pull qwen3:0.6b || echo "WARNING: Ollama model pull failed"

# 7. Install Python deps
echo "Installing Python dependencies..."
pip install -e ".[dev]"

echo ""
echo "=== Setup complete ==="
echo "Add to ~/.claude/settings.json:"
echo '  "mcpServers": {'
echo '    "claude-ai-memory": {'
echo "      \"command\": \"python\","
echo "      \"args\": [\"$PROJECT_DIR/mcp-server/server.py\"]"
echo '    }'
echo '  }'
```

**Step 2: Write scripts/inject-claude-md.sh**

```bash
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
```

**Step 3: Make executable**

```bash
chmod +x scripts/setup.sh scripts/inject-claude-md.sh
```

**Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: setup and CLAUDE.md injection scripts"
```

---

### Task 9: End-to-end test

**Files:**
- Create: `tests/e2e/test_full_workflow.py`

**Step 1: Write E2E test**

This test requires Docker services running.

Create `tests/e2e/test_full_workflow.py`:
```python
"""
End-to-end test: init → save → search → load.
Requires: docker compose services running.
Run: pytest tests/e2e/ -v -m e2e
"""
import pytest
import time

from mcp_server.viking_client import VikingClient
from mcp_server.models import MemoryEntry, EntryType

TEST_PROJECT = "/tmp/test-project-e2e"

@pytest.fixture(scope="module")
def client():
    c = VikingClient(openviking_url="http://localhost:1933")
    if not c.is_available():
        pytest.skip("OpenViking not available")
    return c

@pytest.mark.e2e
def test_01_init_project(client):
    pid = client.init_project(TEST_PROJECT)
    assert len(pid) == 12
    assert client.project_exists(TEST_PROJECT)

@pytest.mark.e2e
def test_02_save_entries(client):
    entries = [
        MemoryEntry(
            type=EntryType.SESSION_SUMMARY,
            title="Initial setup",
            content="Set up the project with Next.js and Supabase. Configured auth.",
        ),
        MemoryEntry(
            type=EntryType.DECISION,
            title="Database choice",
            content="Chose Supabase over raw PostgreSQL for built-in auth and realtime.",
            tags=["database", "supabase"],
        ),
        MemoryEntry(
            type=EntryType.TODO,
            content="Implement user profile page",
        ),
    ]
    saved = client.save_entries(TEST_PROJECT, entries)
    assert saved == 3

@pytest.mark.e2e
def test_03_wait_for_indexing(client):
    """Wait for OpenViking to process and index the entries."""
    time.sleep(5)  # Allow async processing

@pytest.mark.e2e
def test_04_search(client):
    results = client.search(TEST_PROJECT, "database choice")
    assert len(results) > 0

@pytest.mark.e2e
def test_05_load_context(client):
    ctx = client.load_context(TEST_PROJECT)
    assert ctx["registered"] is True
    assert ctx["project_id"] is not None

@pytest.mark.e2e
def test_06_list_entries(client):
    items = client.list_entries(TEST_PROJECT, "sessions")
    assert isinstance(items, list)

@pytest.mark.e2e
def test_99_cleanup(client):
    """Clean up test project."""
    try:
        from mcp_server.project_manager import ProjectManager
        pm = ProjectManager()
        uri = pm.get_project_uri(TEST_PROJECT)
        client._ov.rm(uri, recursive=True)
    except Exception:
        pass  # Best effort
```

**Step 2: Run E2E test (requires Docker services)**

```bash
pytest tests/e2e/ -v -m e2e
```
Expected: all tests PASS if Docker services are running

**Step 3: Commit**

```bash
git add tests/e2e/
git commit -m "test: end-to-end workflow test"
```

---

### Task 10: Configure pytest and final wiring

**Files:**
- Create: `pytest.ini`
- Modify: `pyproject.toml` (add entry point)

**Step 1: Create pytest.ini**

```ini
[pytest]
markers =
    integration: Integration tests (require external services)
    e2e: End-to-end tests (require full Docker stack)
testpaths = tests
pythonpath = .
```

Note: `pythonpath = .` allows `from mcp_server.xxx import` to work. The package is `mcp-server/` directory but Python needs it importable as `mcp_server`. We need to handle this:

**Step 2: Rename directory for Python imports**

```bash
git mv mcp-server mcp_server
```

Update all references in the codebase (imports already use `mcp_server`).

**Step 3: Add entry point to pyproject.toml**

Append to `pyproject.toml`:
```toml
[project.scripts]
claude-ai-memory = "mcp_server.server:main"
```

**Step 4: Run all unit tests**

```bash
pytest tests/unit/ -v
```
Expected: all tests PASS

**Step 5: Commit**

```bash
git add pytest.ini pyproject.toml mcp_server/
git commit -m "chore: pytest config, rename to mcp_server, add entry point"
```

---

### Task 11: Push to GitHub

**Step 1: Verify all files**

```bash
git status
git log --oneline
```

**Step 2: Push**

```bash
git push origin main
```

---

## Task Dependency Summary

```
Task 1 (scaffolding)
  ├─→ Task 2 (embedding service)
  ├─→ Task 4 (data models)
  │     └─→ Task 5 (project manager)
  │           └─→ Task 6 (viking client)
  │                 └─→ Task 7 (MCP server)
  │                       └─→ Task 9 (E2E test)
  ├─→ Task 3 (Docker Compose) ──→ Task 9 (E2E test)
  └─→ Task 8 (scripts)

Task 10 (final wiring) depends on all above
Task 11 (push) depends on Task 10
```

Independent tasks that can run in parallel:
- Task 2 + Task 4 + Task 8
- Task 3 can start after Task 2
