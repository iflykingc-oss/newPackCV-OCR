# -*- coding: utf-8 -*-
"""
PackCV-OCR Web API Server
提供 HTTP API 接口，支持通过飞书机器人回调调用工作流

用法: python -m src.web_server
访问: http://localhost:8000/docs (Swagger文档)
"""

import os
import sys
import json
import logging
import uvicorn
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 设置工作目录和PYTHONPATH（必须在导入项目模块前）
_workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["COZE_WORKSPACE_PATH"] = _workspace
_src_path = os.path.join(_workspace, "src")
for _path in [_src_path, _workspace]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.file.file import File

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置工作目录
os.environ["COZE_WORKSPACE_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="PackCV-OCR API",
    description="包装OCR识别与结构化信息提取服务\n支持通过API或飞书机器人回调调用",
    version="3.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================

class OCRRequest(BaseModel):
    """OCR识别请求"""
    image_url: str = Field(..., description="图片URL（支持http/https）")
    question: str = Field(default="请提取标签中所有产品信息", description="分析问题")
    platform: str = Field(default="none", description="推送平台：feishu/wechat/none")


class FeishuEventRequest(BaseModel):
    """飞书事件回调请求（标准飞书事件格式）"""
    challenge: Optional[str] = Field(default=None, description="飞书事件验证challenge")
    token: Optional[str] = Field(default=None, description="飞书事件验证token")
    type: Optional[str] = Field(default=None, description="事件类型")
    event: Optional[Dict[str, Any]] = Field(default=None, description="事件内容")
    header: Optional[Dict[str, Any]] = Field(default=None, description="事件头信息")


class OCRResponse(BaseModel):
    """OCR识别响应"""
    success: bool = Field(..., description="是否成功")
    product_type: str = Field(default="", description="产品类型")
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="结构化提取数据")
    raw_text: str = Field(default="", description="OCR原始文本")
    corrected_text: str = Field(default="", description="纠错后文本")
    answer: str = Field(default="", description="智能问答结果")
    export_url: str = Field(default="", description="导出文件URL")


# ==================== 核心工作流引擎 ====================

def run_ocr_workflow(image_url: str, question: str = "请提取标签中所有产品信息", platform: str = "none") -> Dict[str, Any]:
    """
    运行OCR工作流的核心函数
    同步执行：图片预处理 → OCR识别 → 文本纠错 → 结构化提取 → 智能问答 → 结果输出
    """
    try:
        from graphs.graph import main_graph

        # 构建输入
        graph_input = {
            "package_image": File(url=image_url, file_type="image"),
            "question": question,
            "platform": platform,
        }

        # 执行工作流
        result = main_graph.invoke(graph_input)

        # 提取结果
        structured_data = {}
        raw_text = ""
        corrected_text = ""
        answer = ""
        export_url = ""
        product_type = ""

        if isinstance(result, dict):
            structured_data = result.get("structured_data", {})
            raw_text = result.get("raw_text", "")
            corrected_text = result.get("corrected_text", "")
            answer = result.get("answer", "")
            export_url = result.get("export_file_url", "")
        elif hasattr(result, 'structured_data'):
            structured_data = result.structured_data
            raw_text = getattr(result, 'raw_text', '') or getattr(result, 'corrected_text', '')
            corrected_text = getattr(result, 'corrected_text', '')
            answer = getattr(result, 'answer', '')
            export_url = getattr(result, 'export_file_url', '')

        # 从structured_data中提取product_type
        if isinstance(structured_data, dict):
            product_type = structured_data.pop("product_type", "") if isinstance(structured_data, dict) else ""
        else:
            product_type = ""

        return {
            "success": True,
            "product_type": product_type,
            "structured_data": structured_data,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "answer": answer,
            "export_url": export_url,
        }

    except Exception as e:
        logger.error(f"工作流执行失败: {str(e)}")
        return {
            "success": False,
            "product_type": "",
            "structured_data": {},
            "raw_text": "",
            "corrected_text": "",
            "answer": f"处理失败: {str(e)}",
            "export_url": "",
        }


def format_feishu_card(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    将OCR结果格式化为飞书交互式卡片
    适用于飞书自定义机器人webhook消息格式
    """
    structured_data = ocr_result.get("structured_data", {})
    raw_text = ocr_result.get("corrected_text", "") or ocr_result.get("raw_text", "")
    answer = ocr_result.get("answer", "")
    product_type = ocr_result.get("product_type", "")
    export_url = ocr_result.get("export_url", "")

    # 统计提取字段
    if isinstance(structured_data, dict):
        valid_fields = {k: v for k, v in structured_data.items() if v and v != "N/A"}
        field_count = len(valid_fields)
    else:
        valid_fields = {}
        field_count = 0

    # 构建字段展示文本
    field_lines = []
    field_display_map = {
        "brand": "🏷 品牌", "product_name": "📦 品名", "specification": "📐 规格",
        "production_date": "📅 生产日期", "shelf_life": "⏳ 保质期",
        "manufacturer": "🏭 生产商", "ingredients": "🧪 配料/成分",
        "standard": "📋 执行标准", "batch_number": "🔢 批号",
        "license_number": "📜 许可证", "storage_condition": "🌡 贮存条件",
        "features": "✨ 产品特点", "usage": "📖 使用方法",
        "other_info": "📝 其他信息",
    }

    for key, value in valid_fields.items():
        label = field_display_map.get(key, key)
        if isinstance(value, str) and len(value) > 60:
            value = value[:60] + "..."
        field_lines.append(f"**{label}**：{value}")

    fields_text = "\n".join(field_lines) if field_lines else "暂无识别数据"

    # 构建卡片
    elements = []

    # 头部摘要
    if product_type:
        summary = f"📌 **产品类型**：{product_type}\n📊 **识别字段**：{field_count}项"
    else:
        summary = f"📊 **识别字段**：{field_count}项"

    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md", "content": summary}
    })

    # 分隔线
    elements.append({"tag": "hr"})

    # 字段数据
    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md", "content": fields_text}
    })

    # 智能问答
    if answer:
        elements.append({"tag": "hr"})
        answer_short = answer[:200] + "..." if len(answer) > 200 else answer
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"💡 **智能分析**\n{answer_short}"}
        })

    # 操作按钮
    actions = []
    if export_url:
        actions.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "📄 查看完整JSON"},
            "type": "default",
            "url": export_url
        })

    if actions:
        elements.append({"tag": "action", "actions": actions})

    # OCR原文（折叠展示）
    if raw_text:
        raw_short = raw_text[:150] + "..." if len(raw_text) > 150 else raw_text
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"📝 OCR原文\n```\n{raw_short}\n```"}
        })

    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"🔍 包装识别结果 ({field_count}项)"},
                "template": "blue"
            },
            "elements": elements
        }
    }


# ==================== API接口 ====================

@app.get("/")
async def root():
    """服务状态"""
    return {
        "service": "PackCV-OCR API",
        "version": "3.5.0",
        "docs": "/docs",
        "endpoints": {
            "POST /api/ocr": "OCR识别（传入图片URL，返回结构化数据）",
            "POST /api/feishu/callback": "飞书事件回调（接收图片消息触发OCR）",
            "GET /health": "健康检查"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "PackCV-OCR"}


@app.post("/api/ocr", response_model=OCRResponse)
async def ocr_recognize(request: OCRRequest):
    """
    OCR识别接口
    
    上传图片URL，返回结构化提取结果。
    支持食品、日化、饮品等各类产品包装。
    
    请求示例:
    ```json
    {
        "image_url": "https://example.com/packaging.jpg",
        "question": "提取所有产品信息",
        "platform": "none"
    }
    ```
    """
    logger.info(f"收到OCR请求: image_url={request.image_url[:60]}...")
    result = run_ocr_workflow(request.image_url, request.question, request.platform)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["answer"])

    return OCRResponse(**result)


@app.post("/api/feishu/callback")
async def feishu_callback(request: FeishuEventRequest, raw_request: Request):
    """
    飞书事件回调接口（用于飞书机器人接收图片消息）
    
    处理流程：
    1. 飞书事件订阅验证（challenge）
    2. 接收图片消息事件
    3. 下载图片 → 运行OCR工作流
    4. 通过飞书机器人发送结果卡片
    
    配置方式：
    1. 在飞书开放平台创建机器人应用
    2. 订阅 `im.message.receive_v1` 事件
    3. 配置事件回调地址为此接口
    """
    # Step 1: 飞书事件验证
    if request.challenge:
        logger.info("飞书事件验证请求")
        return {
            "challenge": request.challenge
        }

    # Step 2: 处理消息事件
    try:
        event_type = request.type or (request.header or {}).get("event_type", "")
        logger.info(f"收到飞书事件: {event_type}")

        # 解析消息内容
        event_data = request.event or {}
        message = event_data.get("message", {})
        message_type = message.get("message_type", "")

        # 仅处理图片消息
        if message_type == "image":
            image_key = message.get("content", "")
            # 获取图片URL（通过飞书API）
            # 注意：实际部署时需要配置飞书应用凭证调用飞书API获取图片
            logger.info(f"收到飞书图片消息: image_key={image_key}")
            
            # 返回处理中状态
            # 实际逻辑：下载图片 → 调用OCR工作流 → 发结果回飞书
            response_text = f"已收到图片消息(image_key: {image_key})，正在识别中..."
            
            return format_feishu_card_ack(response_text)
        
        return {"msg": f"事件 {event_type} 已接收，非图片消息跳过"}
    
    except Exception as e:
        logger.error(f"处理飞书事件失败: {str(e)}")
        return {"msg": f"处理失败: {str(e)}"}


def format_feishu_card_ack(text: str) -> Dict[str, Any]:
    """返回处理中的飞书卡片确认"""
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": "⏳ 正在处理"},
                "template": "grey"
            },
            "elements": [{
                "tag": "div",
                "text": {"tag": "lark_md", "content": text}
            }]
        }
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    logger.info(f"启动 PackCV-OCR API 服务 (端口: {port})")
    logger.info(f"API文档: http://localhost:{port}/docs")
    uvicorn.run(
        "src.web_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )