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
import time
import hashlib
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, Header, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles


# ==================== API Key & 用户鉴权系统 ====================

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "packcv_users.db")


def _init_db():
    """初始化用户数据库（SQLite）"""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    # 用户表
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT DEFAULT '',
            role TEXT DEFAULT 'free',
            created_at TEXT NOT NULL,
            is_active INTEGER DEFAULT 1
        )
    """)
    # API Key表
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            api_key TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT DEFAULT 'default',
            created_at TEXT NOT NULL,
            expires_at TEXT,
            is_active INTEGER DEFAULT 1,
            rate_limit INTEGER DEFAULT 100,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    # 用量统计表
    c.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            timestamp REAL NOT NULL,
            response_time_ms INTEGER DEFAULT 0,
            success INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()


def _get_user_db() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _generate_api_key() -> str:
    return "pck-" + secrets.token_hex(24)


def _register_user(username: str, password: str, email: str = "") -> Dict[str, Any]:
    conn = _get_user_db()
    c = conn.cursor()
    try:
        user_id = str(uuid.uuid4())[:12]
        c.execute(
            "INSERT INTO users (user_id, username, password_hash, email, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, _hash_password(password), email, datetime.now().isoformat())
        )
        conn.commit()
        # 自动生成默认API Key
        api_key = _generate_api_key()
        c.execute(
            "INSERT INTO api_keys (api_key, user_id, name, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (api_key, user_id, "default", datetime.now().isoformat(),
             (datetime.now() + timedelta(days=365)).isoformat())
        )
        conn.commit()
        conn.close()
        return {"user_id": user_id, "username": username, "api_key": api_key}
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="用户名已存在")


def _authenticate(username: str, password: str) -> Dict[str, Any]:
    conn = _get_user_db()
    c = conn.cursor()
    c.execute("SELECT user_id, username, role, is_active FROM users WHERE username=? AND password_hash=?",
              (username, _hash_password(password)))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not row[3]:
        raise HTTPException(status_code=403, detail="账户已停用")
    return {"user_id": row[0], "username": row[1], "role": row[2]}


def _validate_api_key(api_key: str) -> Dict[str, Any]:
    conn = _get_user_db()
    c = conn.cursor()
    c.execute("""
        SELECT k.api_key, k.user_id, k.is_active, k.expires_at, k.rate_limit, u.role
        FROM api_keys k JOIN users u ON k.user_id = u.user_id
        WHERE k.api_key=?
    """, (api_key,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="无效的API Key")
    if not row[2]:
        raise HTTPException(status_code=403, detail="API Key已停用")
    if row[3] and row[3] < datetime.now().isoformat():
        raise HTTPException(status_code=403, detail="API Key已过期")
    return {"api_key": row[0], "user_id": row[1], "rate_limit": row[4], "role": row[5]}


def _check_rate_limit(api_key: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
    """检查API Key的速率限制（滑动窗口）"""
    conn = _get_user_db()
    c = conn.cursor()
    cutoff = time.time() - window_seconds
    c.execute("SELECT COUNT(*) FROM usage_log WHERE api_key=? AND timestamp>?", (api_key, cutoff))
    count = c.fetchone()[0]
    conn.close()
    return count < max_requests


def _log_usage(api_key: str, endpoint: str, response_time_ms: int, success: bool = True):
    conn = _get_user_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO usage_log (api_key, endpoint, timestamp, response_time_ms, success) VALUES (?, ?, ?, ?, ?)",
        (api_key, endpoint, time.time(), response_time_ms, 1 if success else 0)
    )
    conn.commit()
    conn.close()


# 初始化数据库
_init_db()

security = HTTPBearer(auto_error=False)

# 设置工作目录和PYTHONPATH（必须在导入项目模块前）
_workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["COZE_WORKSPACE_PATH"] = _workspace
_src_path = os.path.join(_workspace, "src")
for _path in [_src_path, _workspace]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.file.file import File
from utils.im_platform import get_dispatcher
from utils.config_manager import ConfigManager
from graphs.nodes.call_audit_node import AuditStore

# 全局配置管理器（三级配置中心）
_config_manager = ConfigManager()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置工作目录
os.environ["COZE_WORKSPACE_PATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="PackCV-OCR API",
    description="包装OCR识别与结构化信息提取服务\n支持API、飞书、钉钉、企业微信多平台接入\nV5.5: API Key鉴权 + 用户管理 + 图像质量路由 + 多语言路由",
    version="5.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（Web Demo）
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


# ==================== 数据模型 ====================

class OCRRequest(BaseModel):
    """OCR识别请求"""
    image_url: str = Field(..., description="图片URL（支持http/https）")
    question: str = Field(default="请提取标签中所有产品信息", description="分析问题")
    platform: str = Field(default="none", description="推送平台：feishu/wechat/none")
    tenant_id: Optional[str] = Field(default=None, description="租户ID（多租户场景）")
    custom_model: Optional[str] = Field(default=None, description="自定义模型名，如gpt-4o")
    ocr_engine: Optional[str] = Field(default=None, description="OCR引擎：builtin/smart/paddleocr/tesseract")


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

def run_ocr_workflow(image_url: str, question: str = "请提取标签中所有产品信息", platform: str = "none", tenant_id: Optional[str] = None, runtime_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    运行OCR工作流的核心函数
    同步执行：图片预处理 → OCR识别 → 文本纠错 → 结构化提取 → 智能问答 → 结果输出
    """
    try:
        from graphs.graph import main_graph

        # 三级配置解析
        resolved = _config_manager.resolve(tenant_id, runtime_config)
        
        # 构建输入（携带租户配置）
        graph_input = {
            "package_image": File(url=image_url, file_type="image"),
            "question": question,
            "platform": platform,
            "tenant_id": tenant_id,
            "custom_model_config": resolved.get("llm", {}),
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
        "version": "5.5.0",
        "docs": "/docs",
        "endpoints": {
            "POST /api/auth/register": "用户注册（username+password→获取API Key）",
            "POST /api/auth/login": "用户登录（username+password→JWT/Token）",
            "POST /api/auth/api-key": "创建新API Key（需Authorization Header）",
            "GET /api/auth/api-keys": "列出我的API Keys",
            "POST /api/ocr": "OCR识别（传入图片URL，返回结构化数据）",
            "POST /api/feishu/callback": "飞书事件回调",
            "POST /api/dingtalk/callback": "钉钉事件回调",
            "POST /api/wecom/callback": "企业微信事件回调",
            "POST /api/im/send": "通用IM消息Webhook推送",
            "GET /api/admin/stats": "调用统计Dashboard",
            "GET /api/admin/usage?api_key=xxx": "查询指定API Key用量",
            "GET /metrics": "Prometheus指标端点",
            "GET /health": "健康检查"
        }
    }


# ==================== 用户与API Key管理 ====================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=32, description="用户名")
    password: str = Field(..., min_length=6, max_length=64, description="密码")
    email: str = Field(default="", description="邮箱（可选）")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class ApiKeyCreateReq(BaseModel):
    name: str = Field(default="default", description="API Key名称")


@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """用户注册，自动生成默认API Key"""
    try:
        result = _register_user(req.username, req.password, req.email)
        return {"success": True, "message": "注册成功", "data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """用户登录，返回用户信息"""
    try:
        result = _authenticate(req.username, req.password)
        return {"success": True, "message": "登录成功", "data": result}
    except HTTPException as e:
        raise e


@app.post("/api/auth/api-key")
async def create_api_key(req: ApiKeyCreateReq, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """创建新的API Key（需要Authorization: Bearer <existing_api_key>）"""
    if not credentials:
        raise HTTPException(status_code=401, detail="需要Authorization Header（使用已有API Key）")
    api_key = credentials.credentials
    user_info = _validate_api_key(api_key)
    new_key = _generate_api_key()
    conn = _get_user_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO api_keys (api_key, user_id, name, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
        (new_key, user_info["user_id"], req.name, datetime.now().isoformat(),
         (datetime.now() + timedelta(days=365)).isoformat())
    )
    conn.commit()
    conn.close()
    return {"success": True, "api_key": new_key, "name": req.name}


@app.get("/api/auth/api-keys")
async def list_api_keys(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """列出当前用户的所有API Key"""
    if not credentials:
        raise HTTPException(status_code=401, detail="需要Authorization Header")
    api_key = credentials.credentials
    user_info = _validate_api_key(api_key)
    conn = _get_user_db()
    c = conn.cursor()
    c.execute(
        "SELECT api_key, name, created_at, expires_at, is_active, rate_limit FROM api_keys WHERE user_id=? ORDER BY created_at DESC",
        (user_info["user_id"],)
    )
    keys = [{"api_key": r[0], "name": r[1], "created_at": r[2], "expires_at": r[3],
             "is_active": bool(r[4]), "rate_limit": r[5]} for r in c.fetchall()]
    conn.close()
    return {"success": True, "api_keys": keys}


@app.get("/api/admin/usage")
async def get_usage(api_key: str = "", days: int = 7):
    """查询API Key用量统计"""
    if not api_key:
        raise HTTPException(status_code=400, detail="需要指定api_key参数")
    conn = _get_user_db()
    c = conn.cursor()
    cutoff = time.time() - (days * 86400)
    c.execute(
        "SELECT COUNT(*), SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), AVG(response_time_ms) "
        "FROM usage_log WHERE api_key=? AND timestamp>?", (api_key, cutoff)
    )
    row = c.fetchone()
    conn.close()
    return {
        "api_key": api_key,
        "days": days,
        "total_requests": row[0] or 0,
        "success": row[1] or 0,
        "failed": row[2] or 0,
        "avg_response_time_ms": round(row[3] or 0, 1)
    }


# ==================== API Key中间件（可选鉴权） ====================

async def optional_api_key_auth(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
) -> Optional[Dict[str, Any]]:
    """可选API Key鉴权（免费模式可无key调用，但有速率限制）"""
    api_key = x_api_key
    if not api_key and authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")
    if api_key:
        try:
            return _validate_api_key(api_key)
        except HTTPException:
            return None
    return None


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "PackCV-OCR"}


@app.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    """上传图片进行OCR识别（Web Demo上传接口）"""
    import asyncio
    import tempfile
    suffix = os.path.splitext(file.filename or "image.png")[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
    content = await file.read()
    tmp.write(content)
    tmp_path = tmp.name
    tmp.close()
    
    try:
        # 上传到对象存储
        from coze_coding_dev_sdk.storage import StorageClient
        st = StorageClient()
        upload_result = st.upload_file(tmp_path)
        file_url = upload_result["url"] if isinstance(upload_result, dict) else str(upload_result)
        
        # 调用OCR工作流（复用核心函数）
        result = run_ocr_workflow(file_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ocr", response_model=OCRResponse)
async def ocr_recognize(request: OCRRequest, x_api_key: Optional[str] = Header(None)):
    """
    OCR识别接口
    
    上传图片URL，返回结构化提取结果。
    支持食品、日化、饮品等各类产品包装。
    
    API Key鉴权（可选）：
    - 有API Key：x-api-key 或 Authorization: Bearer <key>
    - 无API Key：免费模式，速率限制更严格（10次/小时）
    
    请求示例:
    ```bash
    curl -X POST http://localhost:8000/api/ocr \\
      -H "Content-Type: application/json" \\
      -H "x-api-key: pck-xxxx..." \\
      -d '{"image_url": "https://example.com/packaging.jpg"}'
    ```
    """
    # 验证API Key（可选）
    api_key = x_api_key
    user_info = None
    if api_key:
        try:
            user_info = _validate_api_key(api_key)
            # 检查速率限制
            rate_limit = user_info.get("rate_limit", 100)
            if not _check_rate_limit(api_key, max_requests=rate_limit):
                raise HTTPException(status_code=429, detail=f"速率限制：每小时内最多{rate_limit}次请求")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="无效的API Key")
    else:
        # 免费模式：限制更严格
        free_key = "free_mode"
        if not _check_rate_limit(free_key, max_requests=10):
            raise HTTPException(status_code=429, detail="免费模式速率限制：10次/小时，请注册获取API Key")
    
    start_time = time.time()
    logger.info(f"OCR请求: image_url={request.image_url[:60]}...")
    
    # 构建运行时配置
    runtime_cfg = {}
    if request.custom_model:
        runtime_cfg["model_override"] = request.custom_model
    result = run_ocr_workflow(
        request.image_url, request.question, request.platform,
        tenant_id=request.tenant_id,
        runtime_config=runtime_cfg
    )
    
    elapsed = int((time.time() - start_time) * 1000)
    
    # 记录用量
    used_key = api_key or free_key
    _log_usage(used_key, "/api/ocr", elapsed, success=result.get("success", False))
    
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


# ==================== V5.3 三平台回调 & 监控 ====================

class DingTalkEventRequest(BaseModel):
    """钉钉事件回调请求"""
    msgtype: Optional[str] = Field(default=None, description="消息类型")
    text: Optional[Dict[str, Any]] = Field(default=None, description="文本内容")
    senderId: Optional[str] = Field(default=None, description="发送者ID")
    conversationId: Optional[str] = Field(default=None, description="会话ID")
    conversationType: Optional[str] = Field(default=None, description="会话类型")
    pictureUrl: Optional[str] = Field(default=None, description="图片URL")
    encrypt: Optional[str] = Field(default=None, description="加密消息体")


class WeComEventRequest(BaseModel):
    """企业微信事件回调请求"""
    msgtype: Optional[str] = Field(default=None, description="消息类型")
    from_user: Optional[str] = Field(default=None, alias="from", description="发送者")
    chat_id: Optional[str] = Field(default=None, description="会话ID")
    chat_type: Optional[str] = Field(default=None, description="会话类型")
    text: Optional[Dict[str, Any]] = Field(default=None, description="文本内容")
    image: Optional[Dict[str, Any]] = Field(default=None, description="图片内容")
    Encrypt: Optional[str] = Field(default=None, description="加密消息体")
    echostr: Optional[str] = Field(default=None, description="验证串")
    event: Optional[str] = Field(default=None, description="事件类型")

    class Config:
        populate_by_name = True


def _handle_im_message(platform: str, body: str, image_url: str = "", text: str = "") -> Dict[str, Any]:
    """统一处理三平台消息事件
    1. 解析事件
    2. 命令路由
    3. 返回对应平台格式的响应
    """
    dispatcher = get_dispatcher()
    adapter = dispatcher.get_adapter(platform)
    if not adapter:
        return {"error": f"未知平台: {platform}"}

    # 1. 解析事件
    event = adapter.parse_event(body)
    if event.get("event_type") == "url_verification":
        return {"challenge": event.get("challenge", "")}

    # 2. 命令路由
    extracted_text = text or adapter.extract_text_from_event(event)
    parsed_cmd = dispatcher.parse_command(extracted_text)
    image_from_event = ""
    if event.get("message_type") == "image" and isinstance(event.get("content"), dict):
        image_from_event = event["content"].get("image_url", "") or event["content"].get("image_key", "")

    final_image_url = image_url or image_from_event

    if parsed_cmd["is_command"] and parsed_cmd["known"]:
        response = dispatcher.route_command(
            parsed_cmd["command"],
            parsed_cmd["args"],
            image_url=final_image_url
        )
    elif event.get("message_type") == "image" and final_image_url:
        # 直接发送图片，触发OCR
        response = dispatcher.route_command("/ocr", "", image_url=final_image_url)
    else:
        response = dispatcher.route_command("/help", "")

    return dispatcher.dispatch_to_platform(
        platform, response["title"], response["content"], response.get("actions", [])
    )


@app.post("/api/dingtalk/callback")
async def dingtalk_callback(request: Request):
    """钉钉事件回调接口
    配置方式：
    1. 创建钉钉群机器人或应用机器人
    2. 配置回调URL为本接口
    3. 接收文本/图片消息 → 触发OCR → 返回结果卡片
    """
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")
    headers = {k.lower(): v for k, v in request.headers.items()}

    dispatcher = get_dispatcher()
    adapter = dispatcher.get_adapter("dingtalk")
    if not adapter.verify_signature(headers, body_str):
        raise HTTPException(status_code=401, detail="签名验证失败")

    logger.info(f"钉钉事件: {body_str[:200]}")
    return _handle_im_message("dingtalk", body_str)


@app.post("/api/wecom/callback")
async def wecom_callback(request: Request):
    """企业微信事件回调接口
    配置方式：
    1. 创建企业微信智能机器人或自建应用
    2. 配置回调URL为本接口
    3. 接收消息 → 触发OCR → 返回结果
    """
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")
    headers = {k.lower(): v for k, v in request.headers.items()}

    dispatcher = get_dispatcher()
    adapter = dispatcher.get_adapter("wecom")
    if not adapter.verify_signature(headers, body_str):
        raise HTTPException(status_code=401, detail="签名验证失败")

    logger.info(f"企微事件: {body_str[:200]}")
    return _handle_im_message("wecom", body_str)


@app.post("/api/im/send")
async def im_send(platform: str, webhook_url: str, title: str, content: str, request: Request = None):
    """通用IM消息推送接口（Webhook方式）
    用法：
    POST /api/im/send?platform=feishu&webhook_url=https://...&title=识别结果&content=...
    """
    dispatcher = get_dispatcher()
    payload = dispatcher.dispatch_to_platform(platform, title, content)
    adapter = dispatcher.get_adapter(platform)
    if not adapter:
        raise HTTPException(status_code=400, detail=f"未知平台: {platform}")
    result = adapter.send_webhook(webhook_url, payload)
    return result


@app.get("/api/admin/stats")
async def get_stats():
    """调用统计Dashboard接口 - 滑动窗口统计"""
    store = AuditStore.get_instance()
    return {
        "stats": store.get_stats(),
        "recent_calls": store.get_recent(20)
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus指标端点 - 文本格式"""
    from fastapi.responses import PlainTextResponse
    store = AuditStore.get_instance()
    return PlainTextResponse(
        content=store.to_prometheus(),
        media_type="text/plain; version=0.0.4"
    )


# ==================== 配置管理API（ConfigManager） ====================

@app.get("/api/config/nodes")
async def list_config_nodes():
    """获取所有可配置节点"""
    return {"nodes": _config_manager.get_all_nodes()}

@app.get("/api/config/summary")
async def get_config_summary(tenant_id: Optional[str] = None):
    """获取配置摘要（含租户覆盖）"""
    return _config_manager.get_config_summary(tenant_id)

@app.get("/api/config/resolve")
async def resolve_config(tenant_id: Optional[str] = None):
    """三级解析完整配置"""
    return _config_manager.resolve(tenant_id)

@app.put("/api/config/tenant/{tenant_id}")
async def update_tenant_config(tenant_id: str, config: Dict[str, Any]):
    """更新租户配置"""
    success = _config_manager.set_tenant_config(tenant_id, config)
    if not success:
        raise HTTPException(status_code=500, detail="保存租户配置失败")
    return {"success": True, "tenant_id": tenant_id, "config": config}

@app.get("/api/config/tenant/{tenant_id}")
async def get_tenant_config(tenant_id: str):
    """获取租户配置"""
    config = _config_manager.get_tenant_config(tenant_id)
    if config is None:
        # 返回默认配置
        resolved = _config_manager.resolve(tenant_id)
        return {"tenant_id": tenant_id, "using_default": True, "config": resolved}
    return {"tenant_id": tenant_id, "using_default": False, "config": config}

@app.delete("/api/config/tenant/{tenant_id}")
async def delete_tenant_config(tenant_id: str):
    """删除租户配置（恢复默认）"""
    success = _config_manager.delete_tenant_config(tenant_id)
    return {"success": success, "tenant_id": tenant_id}

@app.get("/api/config/tenants")
async def list_tenants():
    """列出所有有配置的租户"""
    tenants = _config_manager.list_tenant_configs()
    return {"tenants": tenants, "total": len(tenants)}


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