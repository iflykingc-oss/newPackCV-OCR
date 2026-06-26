"""PackCV-OCR 类型定义（Pydantic）"""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl


class ExtractRequest(BaseModel):
    """信息提取请求"""
    file_url: HttpUrl = Field(..., description="文件URL（图片或PDF）")
    scenario: Optional[str] = Field(None, description="业务场景（自动检测则不传）")
    user_question: str = Field(default="", description="用户提问")
    language: str = Field(default="zh-CN", description="文档语言")
    enable_preprocess: bool = Field(default=True, description="是否预处理")


class ExtractResponse(BaseModel):
    """信息提取响应"""
    scenario: str = Field(..., description="识别出的场景")
    confidence: float = Field(..., description="整体置信度 0-1")
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="提取的结构化数据")
    raw_text: Optional[str] = Field(None, description="OCR 原文")
    model_used: str = Field(..., description="使用的模型ID")
    fallback_chain: List[str] = Field(default_factory=list, description="降级链路")
    latency_ms: int = Field(..., description="处理耗时")
    cost: float = Field(default=0.0, description="本次调用费用")


class QAResponse(BaseModel):
    """问答响应"""
    question: str
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    model_used: str
    latency_ms: int


class UsageInfo(BaseModel):
    """用量信息"""
    tenant_id: str
    period: str
    call_count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    rpm_peak: int = 0
    tpm_peak: int = 0


class TenantInfo(BaseModel):
    """租户信息"""
    tenant_id: str
    name: Optional[str] = None
    tier: Literal["FREE", "BASIC", "PRO", "ENTERPRISE", "FLAGSHIP"] = "FREE"
    active: bool = True
    created_at: Optional[int] = None
    api_key: Optional[str] = None  # 仅创建时返回


class ScenarioInfo(BaseModel):
    """场景信息"""
    scenario_id: str
    name: str
    description: str
    fields: List[str] = Field(default_factory=list)
    sample_count: int = 0
