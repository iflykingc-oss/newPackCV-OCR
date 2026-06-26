"""
MCP 工具集

每个工具都是独立的异步函数，遵循统一签名 (arguments: dict) -> dict。
"""
import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger("mcp_server.tools")

PACKCV_BASE_URL = os.getenv("PACKCV_BASE_URL", "http://localhost:9001")
PACKCV_API_KEY = os.getenv("PACKCV_API_KEY", "pk_mcp_default")


class Tool:
    """MCP 工具定义"""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    async def invoke(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self.handler(arguments)


class ToolRegistry:
    """工具注册表"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]


# ============= 工具实现 =============

async def _call_packcv(
    method: str, path: str, body: Dict[str, Any], timeout: float = 60.0
) -> Dict[str, Any]:
    """统一调用 PackCV HTTP API"""
    headers = {
        "Authorization": f"Bearer {PACKCV_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(
            method, f"{PACKCV_BASE_URL}{path}", json=body, headers=headers
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"PackCV API {method} {path} failed: {resp.status_code} {resp.text[:200]}"
            )
        return resp.json() if resp.content else {}


async def _extract_document(args: Dict[str, Any]) -> Dict[str, Any]:
    """从图片/PDF 中提取结构化数据"""
    file_url = args.get("file_url")
    if not file_url:
        raise ValueError("file_url 必填")
    scenario = args.get("scenario", "auto")
    user_question = args.get("user_question", "提取这张图片的关键信息")

    return await _call_packcv(
        "POST",
        "/api/v1/extract",
        {
            "input_file": {"url": file_url, "file_type": "image"},
            "scenario": scenario,
            "user_question": user_question,
        },
        timeout=120.0,
    )


async def _answer_question(args: Dict[str, Any]) -> Dict[str, Any]:
    """基于图片回答问题"""
    file_url = args.get("file_url")
    question = args.get("question")
    if not file_url or not question:
        raise ValueError("file_url 和 question 必填")

    return await _call_packcv(
        "POST",
        "/api/v1/qa",
        {
            "input_file": {"url": file_url, "file_type": "image"},
            "user_question": question,
        },
        timeout=60.0,
    )


async def _batch_extract(args: Dict[str, Any]) -> Dict[str, Any]:
    """批量提取"""
    items = args.get("items")
    if not items or not isinstance(items, list):
        raise ValueError("items 必填，且必须为数组")
    return await _call_packcv(
        "POST",
        "/api/v1/batch",
        {"items": items},
        timeout=300.0,
    )


async def _list_scenarios(args: Dict[str, Any]) -> Dict[str, Any]:
    """列出所有支持场景"""
    return await _call_packcv("GET", "/api/v1/scenarios", {}, timeout=10.0)


async def _health_check(args: Dict[str, Any]) -> Dict[str, Any]:
    """健康检查"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{PACKCV_BASE_URL}/api/v1/system/health")
    return {
        "status_code": resp.status_code,
        "upstream_ok": resp.status_code == 200,
        "body": resp.json() if resp.content else None,
    }


# ============= 工具注册 =============

def get_all_tools() -> ToolRegistry:
    """获取完整工具集"""
    reg = ToolRegistry()

    reg.register(Tool(
        name="extract_document",
        description=(
            "从图片或 PDF 中提取结构化数据。"
            "支持 12 种场景（身份证/营业执照/合同/发票/简历/银行卡等）。"
            "返回 JSON 结构化数据 + confidence 评分。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "图片/PDF 的公开 URL",
                },
                "scenario": {
                    "type": "string",
                    "description": "场景 ID，如 id_card/business_license/auto",
                    "default": "auto",
                },
                "user_question": {
                    "type": "string",
                    "description": "用户的具体问题（可选）",
                    "default": "提取这张图片的关键信息",
                },
            },
            "required": ["file_url"],
        },
        handler=_extract_document,
    ))

    reg.register(Tool(
        name="answer_question",
        description=(
            "基于图片内容回答用户提问。"
            "适用于：内容理解、信息查询、字段识别、文本分析。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "file_url": {"type": "string", "description": "图片 URL"},
                "question": {"type": "string", "description": "要问的问题"},
            },
            "required": ["file_url", "question"],
        },
        handler=_answer_question,
    ))

    reg.register(Tool(
        name="batch_extract",
        description=(
            "批量提取多个文件。"
            "支持并发处理，最多 100 个/批。返回每项的提取结果。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_url": {"type": "string"},
                            "scenario": {"type": "string"},
                            "user_question": {"type": "string"},
                        },
                        "required": ["file_url"],
                    },
                    "description": "待提取项列表",
                },
            },
            "required": ["items"],
        },
        handler=_batch_extract,
    ))

    reg.register(Tool(
        name="list_scenarios",
        description="列出 PackCV 支持的所有提取场景。",
        input_schema={"type": "object", "properties": {}},
        handler=_list_scenarios,
    ))

    reg.register(Tool(
        name="health_check",
        description="检查 PackCV 服务的健康状态。",
        input_schema={"type": "object", "properties": {}},
        handler=_health_check,
    ))

    return reg
