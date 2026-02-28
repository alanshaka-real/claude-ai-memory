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
        if SyncOpenViking is None:
            logger.warning("openviking package not installed")
            return
        try:
            self._ov = SyncOpenViking(url=self.url)
        except Exception as e:
            logger.error(f"Failed to connect to OpenViking: {e}")
            self._ov = None

    def is_available(self) -> bool:
        if self._ov is None:
            return False
        try:
            return self._ov.is_healthy()
        except Exception:
            return False

    def project_exists(self, project_path: str) -> bool:
        if not self.is_available():
            return False
        try:
            uri = self.pm.get_project_uri(project_path)
            self._ov.ls(uri)
            return True
        except Exception:
            return False

    def init_project(self, project_path: str) -> str:
        pid = self.pm.get_project_id(project_path)
        base_uri = self.pm.get_project_uri(project_path)
        for subdir in ["sessions", "decisions", "changes", "knowledge"]:
            self._ov.mkdir(f"{base_uri}/{subdir}")
        return pid

    def save_entries(self, project_path: str, entries: list[MemoryEntry]) -> int:
        saved = 0
        for entry in entries:
            try:
                subdir = self.pm.get_dir_for_entry_type(entry.type)
                target_uri = self.pm.get_entry_uri(project_path, subdir)
                timestamp = int(time.time() * 1000)
                title_slug = (entry.title or entry.type).replace(" ", "-")[:30]
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
            todo_results = self._ov.find(
                "todo pending next", target_uri=f"{base_uri}/sessions", limit=5
            )
            result["pending_items"] = [
                r.get("content", r.get("abstract", ""))
                for r in todo_results.get("results", [])
            ]
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
        base_uri = self.pm.get_project_uri(project_path)
        target = f"{base_uri}/{subdir}" if subdir else base_uri
        try:
            items = self._ov.ls(target)
            return items if isinstance(items, list) else []
        except Exception:
            return []

    def delete_entry(self, project_path: str, target_uri: str) -> bool:
        try:
            self._ov.rm(target_uri)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def update_entry(self, project_path: str, target_uri: str, content: str) -> bool:
        try:
            self._ov.rm(target_uri)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(content)
                tmp_path = f.name
            parent_uri = "/".join(target_uri.rsplit("/", 1)[:-1])
            self._ov.add_resource(tmp_path, target=parent_uri)
            Path(tmp_path).unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
