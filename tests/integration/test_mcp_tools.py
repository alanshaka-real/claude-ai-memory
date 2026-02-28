import pytest
from unittest.mock import patch, MagicMock

def test_server_imports():
    from mcp_server.server import mcp
    assert mcp is not None

def test_server_has_four_tools():
    from mcp_server.server import mcp
    tools = mcp._tool_manager._tools
    assert "context_load" in tools
    assert "context_search" in tools
    assert "context_save" in tools
    assert "context_manage" in tools
    assert len(tools) == 4
