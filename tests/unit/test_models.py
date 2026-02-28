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
