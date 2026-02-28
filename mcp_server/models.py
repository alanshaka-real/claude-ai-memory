from __future__ import annotations
import sys
from typing import Any, Optional
from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


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
