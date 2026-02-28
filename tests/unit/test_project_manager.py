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

def test_get_dir_for_entry_type():
    pm = ProjectManager()
    assert pm.get_dir_for_entry_type("session_summary") == "sessions"
    assert pm.get_dir_for_entry_type("decision") == "decisions"
    assert pm.get_dir_for_entry_type("change") == "changes"
    assert pm.get_dir_for_entry_type("knowledge") == "knowledge"
    assert pm.get_dir_for_entry_type("todo") == "sessions"
    assert pm.get_dir_for_entry_type("unknown") == "knowledge"
