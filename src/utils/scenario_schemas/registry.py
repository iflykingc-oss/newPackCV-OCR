"""场景Schema注册中心"""
from typing import Dict, Optional
from utils.scenario_schemas.base import BaseSchema, FieldDef
from utils.scenario_schemas.packaging import PACKAGING_SCHEMA
from utils.scenario_schemas.finance import FINANCE_RECEIPT_SCHEMA, FINANCE_STATEMENT_SCHEMA
from utils.scenario_schemas.pharma import PHARMA_SCHEMA
from utils.scenario_schemas.general import GENERAL_SCHEMA


class SchemaRegistry:
    """场景Schema注册中心，支持注册/查询/动态获取"""

    def __init__(self):
        self._schemas: Dict[str, BaseSchema] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册内置场景"""
        self.register(PACKAGING_SCHEMA)
        self.register(FINANCE_RECEIPT_SCHEMA)
        self.register(FINANCE_STATEMENT_SCHEMA)
        self.register(PHARMA_SCHEMA)
        self.register(GENERAL_SCHEMA)

    def register(self, schema: BaseSchema):
        """注册一个Schema"""
        self._schemas[schema.scenario_type] = schema

    def get(self, scenario_type: str) -> Optional[BaseSchema]:
        """按场景类型获取Schema"""
        return self._schemas.get(scenario_type)

    def get_all(self) -> Dict[str, BaseSchema]:
        """获取所有已注册Schema"""
        return dict(self._schemas)

    def resolve(self, doc_type: str) -> BaseSchema:
        """智能解析：根据文档类型返回最佳匹配Schema"""
        exact = self.get(doc_type)
        if exact:
            return exact
        # 模糊匹配
        type_to_scenario = {
            "invoice": "finance_receipt",
            "receipt": "finance_receipt",
            "receipt": "finance_receipt",
            "bank_slip": "finance_statement",
            "bank_statement": "finance_statement",
            "medicine": "pharmaceutical",
            "drug": "pharmaceutical",
            "pill": "pharmaceutical",
            "food": "packaging",
            "beverage": "packaging",
            "cosmetic": "packaging",
            "chemical": "packaging",
        }
        dt_lower = doc_type.lower()
        for key, scenario in type_to_scenario.items():
            if key in dt_lower:
                return self.get(scenario)
        return self.get("general_document")

    def get_field_summary(self, scenario_type: str) -> Dict:
        """获取场景字段概要"""
        schema = self.get(scenario_type)
        if not schema:
            return {"error": f"Unknown scenario: {scenario_type}"}
        return {
            "scenario": schema.scenario_type,
            "name": schema.scenario_name,
            "total_fields": len(schema.fields),
            "required": [f.name for f in schema.fields if f.required],
            "optional": [f.name for f in schema.fields if not f.required],
            "field_types": {f.name: f.field_type for f in schema.fields if f.field_type},
        }

    def detect_scenario(self, image_description: str, ocr_text: str = "") -> str:
        """基于图像描述和OCR文本智能判断场景类型"""
        desc_lower = (image_description + " " + ocr_text).lower()

        # 金融票据关键词
        finance_receipt_keywords = ["发票", "invoice", "收据", "receipt", "小票", "tax", "payment",
                                     "金额", "amount", "seller", "buyer", "纳税人", "税号"]
        finance_statement_keywords = ["银行", "bank", "回单", "流水", "statement", "account",
                                       "交易", "transaction", "转账", "余额", "balance"]
        pharma_keywords = ["药品", "medicine", "drug", "药", "国药准字", "批准文号",
                            "manufacturer", "batch", "expiry", "有效期", "贮藏"]
        packaging_keywords = ["食品", "food", "饮料", "beverage", "包装", "package",
                               "净含量", "配料", "ingredient", "保质期", "营养"]

        # 评分制判断
        scores = {
            "finance_receipt": sum(1 for kw in finance_receipt_keywords if kw in desc_lower),
            "finance_statement": sum(1 for kw in finance_statement_keywords if kw in desc_lower),
            "pharmaceutical": sum(1 for kw in pharma_keywords if kw in desc_lower),
            "packaging": sum(1 for kw in packaging_keywords if kw in desc_lower),
        }
        best = max(scores, key=scores.get)
        if scores[best] >= 2:
            return best
        return "general_document"


# 全局单例
default_registry = SchemaRegistry()

def get_registry() -> SchemaRegistry:
    return default_registry