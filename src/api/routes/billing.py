"""计费、审计和数据脱敏API路由"""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from pydantic import BaseModel, Field

from tenancy.context import TenantContext
from billing.engine import billing_engine
from billing.invoice import invoice_generator
from audit.logger import audit_logger, AuditAction
from security.data_masker import DataMasker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["billing"])


# ==================== 请求/响应模型 ====================

class RecordUsageRequest(BaseModel):
    """记录使用请求"""
    model: str = Field(..., description="模型ID")
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    scenario: Optional[str] = None
    latency_ms: float = Field(0.0, ge=0)
    success: bool = True


class RecordUsageResponse(BaseModel):
    """记录使用响应"""
    cost_usd: float
    cost_cny: float
    total_tokens: int


class InvoiceResponse(BaseModel):
    """账单响应"""
    invoice_id: str
    tenant_id: str
    tenant_name: str
    tier: str
    billing_mode: str
    year_month: str
    period_start: str
    period_end: str
    total_tokens: int
    input_tokens: int
    output_tokens: int
    call_count: int
    failed_count: int
    package_fee: float
    package_quota: int
    overage_tokens: int
    overage_cost: float
    by_token_cost: float
    by_call_cost: float
    discount: float
    total_amount: float
    total_cost_usd: float
    line_items: list
    status: str
    created_at: str
    issued_at: Optional[str] = None
    paid_at: Optional[str] = None
    due_at: Optional[str] = None
    notes: Optional[str] = None


class MaskTextRequest(BaseModel):
    """脱敏请求"""
    text: str = Field(..., description="待脱敏文本")
    mask_type: str = Field(default="partial", description="脱敏类型: full/partial/hash")


class MaskTextResponse(BaseModel):
    """脱敏响应"""
    original: str
    masked: str
    detected_types: list
    changed: bool


class ValidateSafeRequest(BaseModel):
    """验证安全请求"""
    data: dict = Field(..., description="待验证数据")


# ==================== 计费端点 ====================

@router.post("/billing/record", response_model=RecordUsageResponse, summary="记录使用量")
async def record_usage(req: RecordUsageRequest, request: Request):
    """记录一次API调用的使用量（自动计算成本）"""
    ctx = TenantContext.require()
    request_id = ctx.get("request_id", "unknown")

    cost = await billing_engine.record_usage(
        tenant_id=ctx["tenant_id"],
        request_id=request_id,
        model=req.model,
        input_tokens=req.input_tokens,
        output_tokens=req.output_tokens,
        scenario=req.scenario,
        latency_ms=req.latency_ms,
        success=req.success,
    )

    # 审计日志
    await audit_logger.log(
        action=AuditAction.BILLING_DEDUCT,
        tenant_id=ctx["tenant_id"],
        request_id=request_id,
        resource=f"model:{req.model}",
        metadata={
            "input_tokens": req.input_tokens,
            "output_tokens": req.output_tokens,
            "cost_usd": cost["cost_usd"],
        },
    )

    return RecordUsageResponse(
        cost_usd=cost["cost_usd"],
        cost_cny=cost["cost_cny"],
        total_tokens=cost["total_tokens"],
    )


@router.get("/billing/usage", summary="获取使用统计")
async def get_usage(
    request: Request,
    year_month: Optional[str] = Query(None, description="账期 YYYYMM"),
):
    """获取当前租户的使用统计"""
    ctx = TenantContext.require()

    usage = await billing_engine.get_monthly_usage(ctx["tenant_id"], year_month)
    recent = await billing_engine.get_recent_records(ctx["tenant_id"], limit=10)

    return {
        "monthly": usage,
        "recent_records": recent,
    }


@router.get("/billing/invoice", summary="生成月度账单")
async def generate_invoice(
    request: Request,
    year_month: Optional[str] = Query(None, description="账期 YYYYMM"),
    discount: float = Query(0.0, ge=0, description="折扣金额CNY"),
):
    """生成指定账期的账单"""
    ctx = TenantContext.require()

    invoice = await invoice_generator.generate(
        tenant_id=ctx["tenant_id"],
        tenant_name=ctx.get("tenant_name", ctx["tenant_id"]),
        tier=ctx.get("tier", "free"),
        billing_mode=ctx.get("billing_mode", "hybrid"),
        year_month=year_month,
        discount=discount,
    )

    # 审计日志
    await audit_logger.log(
        action=AuditAction.INVOICE_ISSUE,
        tenant_id=ctx["tenant_id"],
        request_id=ctx.get("request_id"),
        resource=f"invoice:{invoice.invoice_id}",
        metadata={
            "year_month": invoice.year_month,
            "total_amount": invoice.total_amount,
        },
    )

    return invoice.model_dump()


# ==================== 审计端点 ====================

@router.get("/audit/logs", summary="查询审计日志")
async def query_audit_logs(
    request: Request,
    action: Optional[str] = Query(None, description="动作类型"),
    start_time: Optional[float] = Query(None, description="开始时间戳"),
    end_time: Optional[float] = Query(None, description="结束时间戳"),
    limit: int = Query(100, ge=1, le=1000),
):
    """查询当前租户的审计日志"""
    ctx = TenantContext.require()

    logs = await audit_logger.query(
        tenant_id=ctx["tenant_id"],
        action=action,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )

    return {
        "total": len(logs),
        "logs": logs,
    }


# ==================== 数据脱敏端点 ====================

@router.post("/security/mask", response_model=MaskTextResponse, summary="数据脱敏")
async def mask_text(req: MaskTextRequest, request: Request):
    """对文本进行脱敏处理（身份证/手机/银行卡/邮箱等）"""
    ctx = TenantContext.require()

    detected = DataMasker.detect_sensitive(req.text)
    masked = DataMasker.mask_text(req.text, mask_type=req.mask_type)
    changed = masked != req.text

    # 审计日志（如果有敏感信息）
    if detected:
        await audit_logger.log(
            action=AuditAction.DATA_READ,
            tenant_id=ctx["tenant_id"],
            request_id=ctx.get("request_id"),
            resource="security.mask",
            metadata={
                "detected_types": detected,
                "mask_type": req.mask_type,
                "text_length": len(req.text),
            },
        )

    return MaskTextResponse(
        original=req.text,
        masked=masked,
        detected_types=detected,
        changed=changed,
    )


@router.post("/security/validate", summary="验证数据安全性")
async def validate_safe(req: ValidateSafeRequest, request: Request):
    """验证数据中是否包含未脱敏的敏感信息"""
    ctx = TenantContext.require()

    is_safe, issues = DataMasker.validate_safe(req.data)

    return {
        "is_safe": is_safe,
        "issue_count": len(issues),
        "issues": issues,
    }


# ==================== 降级策略查询 ====================

@router.get("/degradation/policy", summary="查询当前降级策略")
async def get_degradation_policy(request: Request):
    """查询当前租户的降级策略建议"""
    from tenancy.fallback_policy import FallbackPolicy

    ctx = TenantContext.require()

    quota = ctx.get("quota", {})
    tpm_used = 0  # 实际应从Redis读取
    tpm_limit = quota.get("base_tpm", 100) + quota.get("elastic_tpm", 30)

    decision = FallbackPolicy.decide(
        current_model="doubao-seed-2-0-pro-260215",
        tpm_used=tpm_used,
        tpm_limit=tpm_limit,
        error_rate=0.0,
    )

    return {
        "level": decision.level.value,
        "target_model": decision.target_model,
        "reason": decision.reason,
        "fallback_chain": FallbackPolicy.get_fallback_chain("doubao-seed-2-0-pro-260215"),
    }
