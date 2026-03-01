from __future__ import annotations
import tempfile
import time
import logging
from pathlib import Path
from typing import Any, Optional

from mcp_server.models import MemoryEntry
from mcp_server.project_manager import ProjectManager

logger = logging.getLogger(__name__)

try:
    from openviking import SyncHTTPClient
except ImportError:
    SyncHTTPClient = None


class VikingClient:
    """Wraps OpenViking SDK with project-aware operations."""

    def __init__(
        self,
        openviking_url: str = "http://localhost:1933",
        api_key: str = "local-dev-key",
    ):
        self.url = openviking_url
        self.api_key = api_key
        self.pm = ProjectManager()
        self._ov: Optional[Any] = None
        self._connect()

    def _connect(self):
        if SyncHTTPClient is None:
            logger.warning("openviking package not installed")
            return
        try:
            self._ov = SyncHTTPClient(url=self.url, api_key=self.api_key)
            self._ov.initialize()
            # Force file upload even for localhost (needed when server runs in Docker)
            self._ov._async_client._is_local_server = lambda: False
        except Exception as e:
            logger.error(f"Failed to connect to OpenViking: {e}")
            self._ov = None

    def is_available(self) -> bool:
        if self._ov is None:
            # Try to reconnect if previously failed
            self._connect()
            if self._ov is None:
                return False
        try:
            healthy = self._ov.is_healthy()
            if not healthy:
                # Try reconnect on unhealthy
                self._connect()
                if self._ov is None:
                    return False
                return self._ov.is_healthy()
            return True
        except Exception:
            self._ov = None
            return False

    def project_exists(self, project_path: str) -> bool:
        if not self.is_available():
            return False
        try:
            uri = self.pm.get_project_uri(project_path)
            result = self._ov.ls(uri)
            # ls returns items; if we get here without error, project exists
            return True
        except Exception as e:
            logger.debug(f"project_exists check failed for {project_path}: {e}")
            return False

    def init_project(self, project_path: str) -> str:
        pid = self.pm.get_project_id(project_path)
        base_uri = self.pm.get_project_uri(project_path)
        for subdir in ["sessions", "decisions", "changes", "knowledge"]:
            try:
                self._ov.mkdir(f"{base_uri}/{subdir}")
            except Exception:
                pass  # Directory may already exist
        return pid

    def save_entries(self, project_path: str, entries: list[MemoryEntry]) -> int:
        saved = 0
        for entry in entries:
            try:
                subdir = self.pm.get_dir_for_entry_type(entry.type)
                target_uri = self.pm.get_entry_uri(project_path, subdir)
                content = self._format_entry(entry)
                timestamp = int(time.time() * 1000)
                title_slug = (entry.title or entry.type).replace(" ", "-")[:30]
                suffix = f"-{timestamp}-{title_slug}.md"
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=suffix, delete=False
                ) as f:
                    f.write(content)
                    tmp_path = f.name
                try:
                    self._ov.add_resource(tmp_path, target=target_uri)
                    saved += 1
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to save entry: {e}")
        return saved

    def _format_entry(self, entry: MemoryEntry) -> str:
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

        try:
            result["summary"] = self._ov.abstract(base_uri)
        except Exception:
            result["summary"] = None

        try:
            sessions_uri = f"{base_uri}/sessions"
            items = self._ov.ls(sessions_uri, simple=True)
            recent = sorted(items, reverse=True)[:5]
            result["recent_sessions"] = []
            for item in recent:
                try:
                    abstract = self._ov.abstract(f"{sessions_uri}/{item}")
                    result["recent_sessions"].append({"name": item, "summary": abstract})
                except Exception:
                    pass
        except Exception:
            result["recent_sessions"] = []

        try:
            todo_find = self._ov.find(
                "todo pending next", target_uri=f"{base_uri}/sessions", limit=5
            )
            todo_contexts = getattr(todo_find, "resources", []) or []
            result["pending_items"] = []
            for ctx in todo_contexts:
                content = getattr(ctx, "overview", None) or getattr(ctx, "abstract", "")
                if not content:
                    try:
                        content = self._ov.read(getattr(ctx, "uri", ""))
                    except Exception:
                        pass
                if content:
                    result["pending_items"].append(content)
        except Exception:
            result["pending_items"] = []

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
        self, project_path: str, query: str, scope: str = "all", limit: int = 5,
    ) -> list[dict[str, Any]]:
        base_uri = self.pm.get_project_uri(project_path)
        target_uri = base_uri
        if scope != "all":
            dir_name = self.pm.ENTRY_TYPE_TO_DIR.get(scope, scope)
            if dir_name in ["sessions", "decisions", "changes", "knowledge"]:
                target_uri = f"{base_uri}/{dir_name}"

        try:
            find_result = self._ov.find(query, target_uri=target_uri, limit=limit)
            # FindResult has .resources, .memories, .skills lists
            contexts = getattr(find_result, "resources", []) or []
            output = []
            for ctx in contexts:
                uri = getattr(ctx, "uri", "")
                content = getattr(ctx, "overview", None) or getattr(ctx, "abstract", "")
                # If content is empty, read the actual file
                if not content and uri:
                    try:
                        content = self._ov.read(uri)
                    except Exception:
                        pass
                output.append({
                    "type": str(getattr(ctx, "context_type", "unknown")),
                    "uri": uri,
                    "relevance": getattr(ctx, "score", 0.0),
                    "content": content or "",
                })
            return output
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def list_entries(self, project_path: str, subdir: str = "") -> list[dict[str, Any]]:
        base_uri = self.pm.get_project_uri(project_path)
        target = f"{base_uri}/{subdir}" if subdir else base_uri
        try:
            items = self._ov.ls(target)
            return items if isinstance(items, list) else []
        except Exception:
            return []

    def _validate_project_uri(self, project_path: str, target_uri: str) -> bool:
        base_uri = self.pm.get_project_uri(project_path)
        if not target_uri.startswith(base_uri):
            logger.error(f"URI {target_uri} does not belong to project {project_path}")
            return False
        return True

    def delete_entry(self, project_path: str, target_uri: str) -> bool:
        if not self._validate_project_uri(project_path, target_uri):
            return False
        try:
            self._ov.rm(target_uri)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def update_entry(self, project_path: str, target_uri: str, content: str) -> bool:
        if not self._validate_project_uri(project_path, target_uri):
            return False
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(content)
                tmp_path = f.name
            try:
                self._ov.rm(target_uri)
                parent_uri = "/".join(target_uri.rsplit("/", 1)[:-1])
                self._ov.add_resource(tmp_path, target=parent_uri)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
