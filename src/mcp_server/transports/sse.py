"""
MCP SSE 传输层

使用 aiohttp 提供 Server-Sent Events 端点。
"""
import asyncio
import json
import logging
import queue
from typing import Any, Dict, Optional

from aiohttp import web

from mcp_server.protocol import MCPRequest, MCPResponse

logger = logging.getLogger("mcp_server.sse")


async def sse_endpoint(request: web.Request) -> web.StreamResponse:
    """SSE 端点 - 长连接接收 JSON-RPC 请求"""
    server = request.app["mcp_server"]
    resp = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)

    msg_queue: asyncio.Queue = asyncio.Queue()

    # 接收 POST 消息的端点
    async def receive_post() -> web.Response:
        try:
            data = await request.json() if request.method == "POST" else None
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)
        if data is None:
            return web.json_response({"error": "Empty body"}, status=400)

        try:
            mcp_req = MCPRequest.model_validate(data)
        except Exception as e:
            return web.json_response(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}},
                status=400,
            )

        # 处理并推送到 SSE 流
        mcp_resp = await server.handle_request(mcp_req)
        if mcp_resp is not None:
            await msg_queue.put(mcp_resp.model_dump())
        return web.json_response({"status": "queued"})

    # 启动 SSE 流
    async def sse_writer():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(msg_queue.get(), timeout=30.0)
                    payload = f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                    await resp.write(payload.encode("utf-8"))
                except asyncio.TimeoutError:
                    # 心跳
                    await resp.write(b": heartbeat\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            logger.info("SSE 客户端断开")

    # 接收消息并放入队列
    async def message_collector():
        # 简单实现：从 query string 接收 JSON
        # 实际生产可使用 WebSocket 或独立 POST 端点
        pass

    # 注册 POST 处理
    if request.method == "POST":
        return await receive_post()

    # SSE 模式
    writer_task = asyncio.create_task(sse_writer())
    try:
        await writer_task
    except asyncio.CancelledError:
        writer_task.cancel()
    return resp


async def message_endpoint(request: web.Request) -> web.Response:
    """HTTP POST 端点 - 发送 JSON-RPC 消息"""
    server = request.app["mcp_server"]
    try:
        data = await request.json()
    except Exception as e:
        return web.json_response(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}},
            status=400,
        )

    try:
        mcp_req = MCPRequest.model_validate(data)
    except Exception as e:
        return web.json_response(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": str(e)}},
            status=400,
        )

    mcp_resp = await server.handle_request(mcp_req)
    if mcp_resp is None:
        return web.json_response({"status": "no_response"})
    return web.json_response(mcp_resp.model_dump())


def create_sse_app(server) -> web.Application:
    """创建 aiohttp 应用"""
    app = web.Application()
    app["mcp_server"] = server
    app.router.add_get("/sse", sse_endpoint)
    app.router.add_post("/message", message_endpoint)
    app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))
    return app
