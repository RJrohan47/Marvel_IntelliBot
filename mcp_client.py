from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from venv import logger

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_DIR = Path(r"D:\MCP_server\Expense-tracker-MCP-server")
SERVER_PYTHON = SERVER_DIR / ".venv" / "Scripts" / "python.exe"

_client = MultiServerMCPClient(
    {
        "expense_tracker": {
            "transport": "stdio",
            "command": str(SERVER_PYTHON),
            "args": ["-c", "from expense_tracker_mcp_server import main; main()"],
            "cwd": str(SERVER_DIR),
        }
    }
)

# One long-lived event loop in a background thread. The graph stays sync;
# every MCP call is submitted to this loop and we wait for the result.
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="mcp-loop").start()


def _run(coro, timeout: int = 60):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout=timeout)


def _make_sync(tool: BaseTool) -> StructuredTool:
    """MCP tools are async-only; wrap each one so ToolNode can call it synchronously."""

    def _call(**kwargs):
        return _run(tool.ainvoke(kwargs))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_call,
    )


def expense_tracker_tool() -> list[BaseTool]:
    """Return the expense tools, or [] so Marvel still starts if the server is unavailable."""
    try:
        return [_make_sync(t) for t in _run(_client.get_tools())]
    except Exception as exc:
        logger.warning("Expense MCP server unavailable: %s", exc)
        return []