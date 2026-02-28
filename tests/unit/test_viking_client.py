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

def test_is_available_exception(client, mock_ov):
    mock_ov.is_healthy.side_effect = Exception("connection refused")
    assert client.is_available() is False

def test_project_exists_true(client, mock_ov):
    mock_ov.ls.return_value = [{"name": "sessions"}]
    assert client.project_exists("/home/user/proj") is True

def test_project_exists_false(client, mock_ov):
    mock_ov.ls.side_effect = Exception("not found")
    assert client.project_exists("/home/user/proj") is False

def test_init_project(client, mock_ov):
    pid = client.init_project("/home/user/proj")
    assert len(pid) == 12
    assert mock_ov.mkdir.call_count == 4  # sessions, decisions, changes, knowledge

def test_save_entries(client, mock_ov):
    entries = [
        MemoryEntry(type=EntryType.DECISION, title="Use Stripe", content="Chose Stripe"),
    ]
    saved = client.save_entries("/home/user/proj", entries)
    assert saved == 1
    mock_ov.add_resource.assert_called_once()

def test_save_entries_multiple(client, mock_ov):
    entries = [
        MemoryEntry(type=EntryType.SESSION_SUMMARY, content="Did stuff"),
        MemoryEntry(type=EntryType.TODO, content="Fix bug"),
        MemoryEntry(type=EntryType.KNOWLEDGE, title="Tips", content="Use X for Y"),
    ]
    saved = client.save_entries("/home/user/proj", entries)
    assert saved == 3
    assert mock_ov.add_resource.call_count == 3

def test_load_context_not_available(client, mock_ov):
    mock_ov.is_healthy.return_value = False
    ctx = client.load_context("/home/user/proj")
    assert ctx["available"] is False

def test_load_context_not_registered(client, mock_ov):
    mock_ov.ls.side_effect = Exception("not found")
    ctx = client.load_context("/home/user/proj")
    assert ctx["registered"] is False

def test_search(client, mock_ov):
    mock_ov.find.return_value = {"results": [
        {"type": "decision", "score": 0.9, "overview": "Chose Stripe", "uri": "viking://x"}
    ]}
    results = client.search("/home/user/proj", "payment")
    assert len(results) == 1
    assert results[0]["relevance"] == 0.9

def test_search_empty(client, mock_ov):
    mock_ov.find.return_value = {"results": []}
    results = client.search("/home/user/proj", "nonexistent")
    assert results == []

def test_search_failure(client, mock_ov):
    mock_ov.find.side_effect = Exception("search error")
    results = client.search("/home/user/proj", "payment")
    assert results == []

def test_list_entries(client, mock_ov):
    mock_ov.ls.return_value = [{"name": "file1.md"}, {"name": "file2.md"}]
    items = client.list_entries("/home/user/proj", "sessions")
    assert len(items) == 2

def test_delete_entry(client, mock_ov):
    # Use a valid URI that belongs to the project
    project_path = "/home/user/proj"
    project_uri = client.pm.get_project_uri(project_path)
    target = f"{project_uri}/sessions/file1.md"
    assert client.delete_entry(project_path, target) is True
    mock_ov.rm.assert_called_once_with(target)

def test_delete_entry_failure(client, mock_ov):
    project_path = "/home/user/proj"
    project_uri = client.pm.get_project_uri(project_path)
    target = f"{project_uri}/sessions/file1.md"
    mock_ov.rm.side_effect = Exception("not found")
    assert client.delete_entry(project_path, target) is False

def test_delete_entry_wrong_project(client, mock_ov):
    # URI from a different project should be rejected
    assert client.delete_entry("/home/user/proj", "viking://projects/other123/sessions/x") is False
    mock_ov.rm.assert_not_called()

def test_update_entry_wrong_project(client, mock_ov):
    assert client.update_entry("/home/user/proj", "viking://projects/other123/x", "new") is False
    mock_ov.rm.assert_not_called()
