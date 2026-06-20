"""场景Schema注册中心 - 8大场景统一注册与检测"""
from typing import Dict, List, Optional, Any
from utils.scenario_schemas.base import BaseSchema
from utils.scenario_schemas.packaging import PACKAGING_SCHEMA
from utils.scenario_schemas.finance import FINANCE_RECEIPT_SCHEMA, FINANCE_STATEMENT_SCHEMA
from utils.scenario_schemas.pharma import PHARMA_SCHEMA
from utils.scenario_schemas.contract import CONTRACT_SCHEMA
from utils.scenario_schemas.id_card import ID_CARD_SCHEMA
from utils.scenario_schemas.logistics import LOGISTICS_SCHEMA
from utils.scenario_schemas.general import GENERAL_SCHEMA


class SchemaRegistry:
    """全局场景Schema注册中心（单例）"""

    _instance: Optional["SchemaRegistry"] = None
    _schemas: Dict[str, BaseSchema] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self):
        """注册全部8大场景"""
        for s in [
            PACKAGING_SCHEMA,
            FINANCE_RECEIPT_SCHEMA,
            FINANCE_STATEMENT_SCHEMA,
            PHARMA_SCHEMA,
            CONTRACT_SCHEMA,
            ID_CARD_SCHEMA,
            LOGISTICS_SCHEMA,
            GENERAL_SCHEMA,
        ]:
            self._schemas[s.scenario_type] = s

    def register(self, schema: BaseSchema):
        """注册新场景"""
        self._schemas[schema.scenario_type] = schema

    def get(self, scenario_type: str) -> BaseSchema:
        """按类型获取场景Schema，不存在返回通用"""
        return self._schemas.get(scenario_type, GENERAL_SCHEMA)

    def get_all(self) -> Dict[str, BaseSchema]:
        """返回全部注册的场景"""
        return dict(self._schemas)

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """场景列表（含字段摘要）"""
        result = []
        for name, s in self._schemas.items():
            result.append({
                "type": s.scenario_type,
                "name": s.scenario_name,
                "desc": s.description,
                "fields": len(s.fields),
                "required": len([f for f in s.fields if f.required]),
            })
        return result

    def detect_scenario(self, text: str) -> str:
        """基于关键词检测场景类型（文本兜底方案）"""
        text_upper = text.upper()

        # 合同/协议
        contract_kw = ["合同", "协议", "合约", "契约", "CONTRACT", "AGREEMENT",
                       "甲方", "乙方", "签约", "章程", "条款", "保密"]
        score = sum(2 for kw in contract_kw if kw in text_upper or kw in text)
        if score >= 4:
            return "contract"

        # 身份证
        id_kw = ["身份证", "居民身份证", "ID", "IDENTITY", "公民身份",
                 "民族", "出生", "住址", "签发机关"]
        score = sum(2 for kw in id_kw if kw in text_upper or kw in text)
        if score >= 4:
            return "id_card"

        # 物流单
        logistics_kw = ["运单", "快递", "物流", "包裹", "TRACKING", "EXPRESS",
                        "寄件", "收件", "始发", "目的地", "顺丰", "中通", "韵达"]
        score = sum(2 for kw in logistics_kw if kw in text_upper or kw in text)
        if score >= 4:
            return "logistics"

        # 金融票据（原有逻辑增强）
        finance_kw = ["发票", "收据", "INVOICE", "TAX", "金额", "税率",
                      "收款", "付款", "开户行", "账号"]
        score = sum(2 for kw in finance_kw if kw in text_upper or kw in text)
        if score >= 4:
            # 区分回单和票据
            statement_kw = ["银行回单", "交易流水", "账户余额", "对方户名",
                            "BANK", "BALANCE", "STATEMENT"]
            stmt_score = sum(2 for kw in statement_kw if kw in text_upper or kw in text)
            return "finance_statement" if stmt_score >= 4 else "finance_receipt"

        # 药品（原有逻辑增强）
        pharma_kw = ["国药准字", "批准文号", "生产批号", "药品", "OTC",
                     "适应症", "用法用量", "禁忌", "不良反应", "PHARM"]
        score = sum(2 for kw in pharma_kw if kw in text_upper or kw in text)
        if score >= 3:
            return "pharmaceutical"

        # 食品/日化包装（原有逻辑）
        packaging_kw = ["配料", "成分", "净含量", "保质期", "生产日期",
                        "贮存条件", "营养成分", "生产商", "INGREDIENTS"]
        score = sum(2 for kw in packaging_kw if kw in text_upper or kw in text)
        if score >= 3:
            return "packaging"

        return "general_document"


# 全局单例
registry = SchemaRegistry()