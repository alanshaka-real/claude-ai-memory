from __future__ import annotations
import hashlib


class ProjectManager:
    """Manages project identification and URI mapping for OpenViking."""

    ENTRY_TYPE_TO_DIR = {
        "session_summary": "sessions",
        "decision": "decisions",
        "change": "changes",
        "knowledge": "knowledge",
        "todo": "sessions",
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
