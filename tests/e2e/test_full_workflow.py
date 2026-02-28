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
