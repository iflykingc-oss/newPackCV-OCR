"""账单生成器

根据租户等级和计费模式生成月度账单
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

from billing.engine import (
    BillingEngine,
    BillingMode,
    PACKAGE_PRICING,
    USD_TO_CNY,
    billing_engine,
)

logger = logging.getLogger(__name__)


class InvoiceStatus(str, Enum):
    """账单状态"""
    DRAFT = "draft"
    ISSUED = "issued"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class LineItem(BaseModel):
    """账单明细项"""
    description: str
    quantity: float
    unit: str
    unit_price: float
    amount: float


class Invoice(BaseModel):
    """账单"""
    invoice_id: str = Field(default_factory=lambda: f"INV-{uuid.uuid4().hex[:12].upper()}")
    tenant_id: str
    tenant_name: str
    tier: str
    billing_mode: str
    year_month: str  # 格式：202601
    period_start: str  # ISO datetime
    period_end: str

    # 用量
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    failed_count: int = 0

    # 金额（人民币元）
    package_fee: float = 0.0
    package_quota: int = 0
    overage_tokens: int = 0
    overage_cost: float = 0.0
    by_token_cost: float = 0.0
    by_call_cost: float = 0.0
    discount: float = 0.0
    total_amount: float = 0.0
    total_cost_usd: float = 0.0

    # 明细
    line_items: list = Field(default_factory=list)

    # 状态
    status: str = "draft"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    issued_at: Optional[str] = None
    paid_at: Optional[str] = None
    due_at: Optional[str] = None

    notes: Optional[str] = None


class InvoiceGenerator:
    """账单生成器"""

    def __init__(self):
        self.billing_engine = billing_engine

    async def generate(
        self,
        tenant_id: str,
        tenant_name: str,
        tier: str,
        billing_mode: str = "hybrid",
        year_month: Optional[str] = None,
        discount: float = 0.0,
    ) -> Invoice:
        """生成账单

        Args:
            tenant_id: 租户ID
            tenant_name: 租户名称
            tier: 租户等级
            billing_mode: 计费模式
            year_month: 账期（YYYYMM），默认当前月
            discount: 折扣金额（CNY）
        """
        if year_month is None:
            year_month = datetime.now(timezone.utc).strftime("%Y%m")

        # 1. 获取使用量
        usage = await self.billing_engine.get_monthly_usage(tenant_id, year_month)

        # 2. 获取套餐配置
        package = PACKAGE_PRICING.get(tier, PACKAGE_PRICING["free"])

        # 3. 计算账期
        period_start = f"{year_month[:4]}-{year_month[4:]}-01T00:00:00Z"
        if year_month[4:] == "12":
            next_year = str(int(year_month[:4]) + 1)
            period_end = f"{next_year}-01-01T00:00:00Z"
        else:
            next_month = int(year_month[4:]) + 1
            period_end = f"{year_month[:4]}-{next_month:02d}-01T00:00:00Z"

        # 4. 初始化账单
        invoice = Invoice(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tier=tier,
            billing_mode=billing_mode,
            year_month=year_month,
            period_start=period_start,
            period_end=period_end,
            total_tokens=usage["total_tokens"],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            call_count=usage["call_count"],
            failed_count=usage["failed_count"],
            total_cost_usd=usage["cost_usd"],
            package_quota=package["monthly_quota"],
            discount=discount,
        )

        # 5. 根据计费模式计算金额
        if billing_mode == BillingMode.PACKAGE.value:
            invoice = self._calc_package(invoice, package, usage)
        elif billing_mode == BillingMode.BY_TOKEN.value:
            invoice = self._calc_by_token(invoice, usage)
        elif billing_mode == BillingMode.BY_CALL.value:
            invoice = self._calc_by_call(invoice, usage)
        else:  # hybrid
            invoice = self._calc_hybrid(invoice, package, usage)

        # 6. 应用折扣
        if discount > 0:
            invoice.total_amount = max(0, invoice.total_amount - discount)

        # 7. 添加明细
        invoice.line_items = self._build_line_items(invoice, package, usage)

        # 8. 设置应付日期（账期结束+15天）
        invoice.due_at = f"{period_end[:10]}T00:00:00Z"
        invoice.issued_at = datetime.now(timezone.utc).isoformat()
        invoice.status = InvoiceStatus.ISSUED.value

        return invoice

    def _calc_package(self, invoice: Invoice, package: dict, usage: dict) -> Invoice:
        """套餐制计费"""
        invoice.package_fee = package["monthly_fee"]

        if usage["total_tokens"] > package["monthly_quota"]:
            overage = usage["total_tokens"] - package["monthly_quota"]
            invoice.overage_tokens = overage
            invoice.overage_cost = round(
                overage * package["overage_rate"] / 1000, 2
            )
            invoice.total_amount = round(
                invoice.package_fee + invoice.overage_cost, 2
            )
        else:
            invoice.total_amount = invoice.package_fee

        return invoice

    def _calc_by_token(self, invoice: Invoice, usage: dict) -> Invoice:
        """按Token计费"""
        invoice.by_token_cost = round(usage["cost_cny"], 2)
        invoice.total_amount = invoice.by_token_cost
        return invoice

    def _calc_by_call(self, invoice: Invoice, usage: dict) -> Invoice:
        """按调用计费（每次¥0.1）"""
        invoice.by_call_cost = round(usage["call_count"] * 0.1, 2)
        invoice.total_amount = invoice.by_call_cost
        return invoice

    def _calc_hybrid(self, invoice: Invoice, package: dict, usage: dict) -> Invoice:
        """混合计费（套餐+超额按量，推荐）"""
        invoice.package_fee = package["monthly_fee"]

        if usage["total_tokens"] > package["monthly_quota"]:
            overage = usage["total_tokens"] - package["monthly_quota"]
            invoice.overage_tokens = overage
            invoice.overage_cost = round(
                overage * package["overage_rate"] / 1000, 2
            )
            invoice.total_amount = round(
                invoice.package_fee + invoice.overage_cost, 2
            )
        else:
            invoice.total_amount = invoice.package_fee

        return invoice

    def _build_line_items(self, invoice: Invoice, package: dict, usage: dict) -> list:
        """构建账单明细"""
        items = []

        if invoice.package_fee > 0:
            items.append(LineItem(
                description=f"{invoice.tier.upper()} 套餐月费",
                quantity=1,
                unit="月",
                unit_price=invoice.package_fee,
                amount=invoice.package_fee,
            ).model_dump())

        if invoice.overage_tokens > 0:
            items.append(LineItem(
                description="超额Token费",
                quantity=invoice.overage_tokens,
                unit="tokens",
                unit_price=package["overage_rate"] / 1000,
                amount=invoice.overage_cost,
            ).model_dump())

        if invoice.by_token_cost > 0 and invoice.billing_mode == "by_token":
            items.append(LineItem(
                description="按Token计费",
                quantity=usage["total_tokens"],
                unit="tokens",
                unit_price=usage["cost_cny"] / max(1, usage["total_tokens"]),
                amount=invoice.by_token_cost,
            ).model_dump())

        if invoice.by_call_cost > 0:
            items.append(LineItem(
                description="按调用计费",
                quantity=usage["call_count"],
                unit="次",
                unit_price=0.1,
                amount=invoice.by_call_cost,
            ).model_dump())

        if invoice.discount > 0:
            items.append(LineItem(
                description="优惠折扣",
                quantity=1,
                unit="次",
                unit_price=-invoice.discount,
                amount=-invoice.discount,
            ).model_dump())

        return items

    async def issue(self, invoice: Invoice) -> Invoice:
        """开具账单"""
        invoice.status = InvoiceStatus.ISSUED.value
        invoice.issued_at = datetime.now(timezone.utc).isoformat()
        return invoice

    async def mark_paid(self, invoice: Invoice) -> Invoice:
        """标记已支付"""
        invoice.status = InvoiceStatus.PAID.value
        invoice.paid_at = datetime.now(timezone.utc).isoformat()
        return invoice


# 全局账单生成器
invoice_generator = InvoiceGenerator()
