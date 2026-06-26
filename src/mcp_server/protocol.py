"""
MCP 协议定义（JSON-RPC 2.0）
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 请求"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPError(BaseModel):
    """JSON-RPC 2.0 错误"""
    code: int
    message: str
    data: Optional[Any] = None


class MCPResponse(BaseModel):
    """JSON-RPC 2.0 响应"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[MCPError] = None
