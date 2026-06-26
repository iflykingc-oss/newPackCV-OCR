"""
MCP Server - Model Context Protocol 实现

允许 AI Agent（如 Claude Desktop、Cursor）通过标准化协议调用 PackCV 的能力。

支持传输：
- stdio（CLI 工具直连）
- SSE（HTTP 长连接）

工具集：
- extract_document: 提取文档结构化数据
- answer_question: 基于文档回答问题
- batch_extract: 批量提取
- list_scenarios: 列出支持场景
- health_check: 服务健康检查
"""
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Callable, Awaitable

from mcp_server.protocol import MCPRequest, MCPResponse, MCPError
from mcp_server.tools import get_all_tools, ToolRegistry

logger = logging.getLogger("mcp_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)

__version__ = "1.0.0"


class MCPServer:
    """MCP 服务器核心

    协议: JSON-RPC 2.0 over stdio / SSE
    """

    PROTOCOL_VERSION = "2024-11-05"
    SERVER_NAME = "packcv-mcp-server"
    SERVER_VERSION = __version__

    def __init__(self) -> None:
        self.tools: ToolRegistry = get_all_tools()
        self._initialized = False

    async def handle_request(self, req: MCPRequest) -> Optional[MCPResponse]:
        """处理 JSON-RPC 请求"""
        try:
            if req.method == "initialize":
                return self._handle_initialize(req)
            elif req.method == "tools/list":
                return self._handle_list_tools(req)
            elif req.method == "tools/call":
                return await self._handle_call_tool(req)
            elif req.method == "ping":
                return MCPResponse(id=req.id, result={"pong": True})
            else:
                return MCPResponse(
                    id=req.id,
                    error=MCPError(
                        code=-32601,
                        message=f"Method not found: {req.method}",
                    ),
                )
        except Exception as e:
            logger.exception("处理请求失败: %s", e)
            return MCPResponse(
                id=req.id,
                error=MCPError(code=-32603, message=f"Internal error: {type(e).__name__}: {e}"),
            )

    def _handle_initialize(self, req: MCPRequest) -> MCPResponse:
        self._initialized = True
        return MCPResponse(
            id=req.id,
            result={
                "protocolVersion": self.PROTOCOL_VERSION,
                "serverInfo": {
                    "name": self.SERVER_NAME,
                    "version": self.SERVER_VERSION,
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                },
            },
        )

    def _handle_list_tools(self, req: MCPRequest) -> MCPResponse:
        tools = self.tools.list_definitions()
        return MCPResponse(id=req.id, result={"tools": tools})

    async def _handle_call_tool(self, req: MCPRequest) -> MCPResponse:
        params = req.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {}) or {}

        tool = self.tools.get(tool_name)
        if tool is None:
            return MCPResponse(
                id=req.id,
                error=MCPError(code=-32602, message=f"Unknown tool: {tool_name}"),
            )

        try:
            result = await tool.invoke(arguments)
            return MCPResponse(
                id=req.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, ensure_ascii=False, indent=2),
                        }
                    ],
                    "isError": False,
                },
            )
        except Exception as e:
            logger.exception("工具 %s 执行失败", tool_name)
            return MCPResponse(
                id=req.id,
                result={
                    "content": [
                        {"type": "text", "text": f"Error: {type(e).__name__}: {e}"}
                    ],
                    "isError": True,
                },
            )

    # ========== 传输层 ==========

    async def run_stdio(self) -> None:
        """stdin/stdout JSON-RPC 循环"""
        logger.info("MCP Server starting on stdio...")
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    req = MCPRequest.model_validate_json(line_str)
                except Exception as e:
                    err = MCPResponse(
                        id=None,
                        error=MCPError(code=-32700, message=f"Parse error: {e}"),
                    )
                    sys.stdout.write(err.model_dump_json() + "\n")
                    sys.stdout.flush()
                    continue

                resp = await self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(resp.model_dump_json() + "\n")
                    sys.stdout.flush()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.exception("stdio loop error: %s", e)

    async def run_sse(self, host: str = "0.0.0.0", port: int = 9002) -> None:
        """SSE/HTTP 传输（用于 Web 集成）"""
        from aiohttp import web
        from mcp_server.transports.sse import create_sse_app
        app = create_sse_app(self)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info("MCP SSE server started on http://%s:%d", host, port)
        # 永久运行
        await asyncio.Event().wait()


def main() -> None:
    """CLI 入口"""
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    server = MCPServer()

    if transport == "stdio":
        asyncio.run(server.run_stdio())
    elif transport == "sse":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "9002"))
        asyncio.run(server.run_sse(host=host, port=port))
    else:
        print(f"Unknown transport: {transport}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
