"""工作流API路由 - 调用PackCV-OCR主图

提供：
- POST /api/v1/extract - 单文件信息提取
- POST /api/v1/qa - 智能问答
- POST /api/v1/batch - 批量处理
- GET  /api/v1/scenarios - 列出所有支持场景
"""

import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.file.file import File  # noqa: F401  # 确保类型注册


router = APIRouter(prefix="/api/v1", tags=["workflow"])


# ============= Pydantic Models =============

class ExtractRequest(BaseModel):
    """单文件提取请求"""
    input_file: File = Field(..., description="输入文件（图片/PDF）")
    user_question: Optional[str] = Field(default=None, description="用户问题")
    scenario_hint: Optional[str] = Field(default=None, description="场景提示")
    language: str = Field(default="zh-CN", description="语言代码")


class ExtractResponse(BaseModel):
    """单文件提取响应"""
    request_id: str
    tenant_id: str
    scenario: str
    structured_data: dict
    ocr_text: Optional[str] = None
    confidence: float
    qa_answer: Optional[str] = None
    models_used: List[str]
    cost_usd: float
    processing_time_ms: int
    timestamp: str


class QARequest(BaseModel):
    """QA问答请求"""
    input_file: File = Field(..., description="已提取的文件")
    question: str = Field(..., description="用户问题")
    context: Optional[dict] = Field(default=None, description="上下文（已提取数据）")


class QAResponse(BaseModel):
    """QA问答响应"""
    request_id: str
    question: str
    answer: str
    confidence: float
    model_used: str
    cost_usd: float
    processing_time_ms: int


class BatchRequest(BaseModel):
    """批量处理请求"""
    input_files: List[File] = Field(..., min_items=1, max_items=20)
    scenario_hint: Optional[str] = None


class BatchItemResult(BaseModel):
    """批量项结果"""
    file_url: str
    success: bool
    scenario: Optional[str] = None
    structured_data: Optional[dict] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """批量处理响应"""
    request_id: str
    total: int
    success_count: int
    failed_count: int
    results: List[BatchItemResult]
    total_cost_usd: float
    processing_time_ms: int


class ScenarioInfo(BaseModel):
    """场景信息"""
    scenario_id: str
    scenario_name: str
    description: str
    languages: List[str]
    sample_schema: dict


# ============= API端点 =============

@router.get("/scenarios", response_model=List[ScenarioInfo])
async def list_scenarios():
    """列出所有支持的提取场景"""
    from scenarios import SCENARIO_REGISTRY
    from utils.i18n import Language  # noqa: F401
    SUPPORTED_LANGUAGES = [lang.value for lang in Language]

    scenarios = []
    for sid, schema_cls in SCENARIO_REGISTRY.items():
        try:
            sample = schema_cls.model_json_schema()
        except Exception:
            sample = {}
        scenarios.append(
            ScenarioInfo(
                scenario_id=sid,
                scenario_name=sid.replace("_", " ").title(),
                description=f"{sid}场景的结构化提取",
                languages=SUPPORTED_LANGUAGES,
                sample_schema=sample,
            )
        )
    return scenarios


@router.post("/extract", response_model=ExtractResponse)
async def extract_info(req: ExtractRequest):
    """单文件信息提取

    完整工作流：输入路由 → 场景检测 → 预处理 → OCR → 信息提取 → 后处理 → 输出
    """
    import time

    from graphs.graph import main_graph
    from tenancy.context import TenantContext

    start = time.time()
    ctx = TenantContext.require()

    # 构造工作流输入
    wf_input = {
        "input_file": req.input_file.model_dump(),
        "user_question": req.user_question or "",
    }

    try:
        result = main_graph.invoke(wf_input)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "workflow_failed", "message": str(e)},
        )

    elapsed_ms = int((time.time() - start) * 1000)

    return ExtractResponse(
        request_id=ctx["request_id"],
        tenant_id=ctx["tenant_id"],
        scenario=result.get("detected_scenario", "unknown"),
        structured_data=result.get("structured_data", {}),
        ocr_text=result.get("ocr_text"),
        confidence=result.get("confidence", 0.0),
        qa_answer=result.get("qa_answer"),
        models_used=result.get("models_used", []),
        cost_usd=result.get("cost_usd", 0.0),
        processing_time_ms=elapsed_ms,
        timestamp=datetime.now().isoformat(),
    )


@router.post("/qa", response_model=QAResponse)
async def qa_answer(req: QARequest):
    """基于已提取数据的智能问答"""
    import time

    from graphs.graph import main_graph
    from tenancy.context import TenantContext

    start = time.time()
    ctx = TenantContext.require()

    wf_input = {
        "input_file": req.input_file.model_dump(),
        "user_question": req.question,
    }

    try:
        result = main_graph.invoke(wf_input)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "workflow_failed", "message": str(e)},
        )

    elapsed_ms = int((time.time() - start) * 1000)
    qa = result.get("qa_answer", "")

    return QAResponse(
        request_id=ctx["request_id"],
        question=req.question,
        answer=qa,
        confidence=result.get("confidence", 0.0),
        model_used=result.get("model_used", ""),
        cost_usd=result.get("cost_usd", 0.0),
        processing_time_ms=elapsed_ms,
    )


@router.post("/batch", response_model=BatchResponse)
async def batch_extract(req: BatchRequest):
    """批量文件处理（最多20个/请求）"""
    import time

    from graphs.graph import main_graph
    from tenancy.context import TenantContext

    start = time.time()
    ctx = TenantContext.require()

    if len(req.input_files) > 20:
        raise HTTPException(
            status_code=400,
            detail="批量请求最多20个文件",
        )

    results: List[BatchItemResult] = []
    success_count = 0
    failed_count = 0
    total_cost = 0.0

    for file in req.input_files:
        try:
            wf_input = {
                "input_file": file.model_dump(),
                "user_question": "",
            }
            result = main_graph.invoke(wf_input)
            results.append(
                BatchItemResult(
                    file_url=file.url,
                    success=True,
                    scenario=result.get("detected_scenario"),
                    structured_data=result.get("structured_data", {}),
                )
            )
            success_count += 1
            total_cost += result.get("cost_usd", 0.0)
        except Exception as e:
            results.append(
                BatchItemResult(
                    file_url=file.url,
                    success=False,
                    error=str(e),
                )
            )
            failed_count += 1

    elapsed_ms = int((time.time() - start) * 1000)
    return BatchResponse(
        request_id=ctx["request_id"],
        total=len(req.input_files),
        success_count=success_count,
        failed_count=failed_count,
        results=results,
        total_cost_usd=round(total_cost, 4),
        processing_time_ms=elapsed_ms,
    )
