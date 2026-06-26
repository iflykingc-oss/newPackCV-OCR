"""
MCP Server 包入口

对外暴露 MCPServer 和工具集。
"""
from mcp_server.server import MCPServer, main
from mcp_server.protocol import MCPRequest, MCPResponse, MCPError
from mcp_server.tools import get_all_tools, ToolRegistry, Tool

__version__ = "1.0.0"

__all__ = [
    "MCPServer",
    "main",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "get_all_tools",
    "ToolRegistry",
    "Tool",
    "__version__",
]
