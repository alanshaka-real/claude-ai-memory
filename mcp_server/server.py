from __future__ import annotations
import logging
import os
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.models import EntryType, MemoryEntry
from mcp_server.viking_client import VikingClient

_log_file = os.path.join(os.path.dirname(__file__), "..", "logs", "mcp-memory.log")
os.makedirs(os.path.dirname(_log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    filename=_log_file,
)
logger = logging.getLogger("claude-ai-memory")

mcp = FastMCP(
    "claude-ai-memory",
)

_client: Optional[VikingClient] = None


def get_client() -> VikingClient:
    global _client
    if _client is None:
        url = os.environ.get("OPENVIKING_URL", "http://localhost:1933")
        api_key = os.environ.get("OPENVIKING_API_KEY", "local-dev-key")
        _client = VikingClient(openviking_url=url, api_key=api_key)
    return _client


@mcp.tool()
async def context_load(project_path: str) -> dict[str, Any]:
    """Load project context overview. Call once at session start.

    Returns project summary, recent session history, pending items,
    and active decisions. If the project is not registered, returns
    {"registered": false}.
    """
    client = get_client()
    return await client.load_context(project_path)


@mcp.tool()
async def context_search(
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
    if not await client.is_available():
        return {"error": "OpenViking not available", "results": []}
    results = await client.search(project_path, query, scope=scope, limit=limit)
    return {"results": results}


@mcp.tool()
async def context_save(
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
    if not await client.is_available():
        return {"error": "OpenViking not available", "saved": 0}

    if not await client.project_exists(project_path):
        await client.init_project(project_path)

    parsed = []
    errors = []
    for i, e in enumerate(entries):
        try:
            parsed.append(MemoryEntry(
                type=EntryType(e["type"]),
                title=e.get("title"),
                content=e["content"],
                tags=e.get("tags", []),
            ))
        except (KeyError, ValueError) as err:
            logger.warning(f"Skipping invalid entry {i}: {err}")
            errors.append(f"Entry {i}: {err}")

    saved = await client.save_entries(project_path, parsed)
    result: dict[str, Any] = {"saved": saved, "total": len(entries)}
    if errors:
        result["skipped"] = len(errors)
        result["errors"] = errors
    return result


@mcp.tool()
async def context_manage(
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
    if not await client.is_available():
        return {"error": "OpenViking not available"}

    if action == "init_project":
        pid = await client.init_project(project_path)
        return {"success": True, "project_id": pid}
    elif action == "list":
        items = await client.list_entries(project_path, subdir=target or "")
        return {"items": items}
    elif action == "delete":
        if not target:
            return {"error": "target is required for delete"}
        ok = await client.delete_entry(project_path, target)
        return {"success": ok}
    elif action == "update":
        if not target or not content:
            return {"error": "target and content are required for update"}
        ok = await client.update_entry(project_path, target, content)
        return {"success": ok}
    else:
        return {"error": f"Unknown action: {action}"}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
