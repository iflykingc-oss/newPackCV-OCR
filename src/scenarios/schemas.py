#!/usr/bin/env python3
"""场景Schema定义 - V7.0统一架构

12个核心场景的Pydantic Schema定义，支持：
- 字段类型约束
- 必填/可选标识
- 描述信息（用于Prompt注入）
- 场景元数据（推荐模型、验证规则等）
"""
from typing import Dict, Any, List, Type
from pydantic import BaseModel, Field


# ==========================================
# 1. 包装场景 Schema
# ==========================================
class PackagingSchema(BaseModel):
    """产品包装/标签 Schema"""

    product_name: str = Field(default="", description="产品名称")
    brand: str = Field(default="", description="品牌")
    specification: str = Field(default="", description="规格/净含量")
    production_date: str = Field(default="", description="生产日期")
    expiry_date: str = Field(default="", description="保质期/到期日期")
    shelf_life: str = Field(default="", description="保质期时长")
    batch_number: str = Field(default="", description="批次号")
    ingredients: str = Field(default="", description="配料表")
    nutrition_facts: str = Field(default="", description="营养成分表")
    manufacturer: str = Field(default="", description="生产商")
    manufacturer_address: str = Field(default="", description="生产地址")
    storage_conditions: str = Field(default="", description="储存条件")
    allergen_info: str = Field(default="", description="过敏原信息")
    certification: str = Field(default="", description="认证标识")
    barcode: str = Field(default="", description="条形码")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 2. 金融票据 Schema
# ==========================================
class FinanceReceiptSchema(BaseModel):
    """发票/收据/小票 Schema"""

    receipt_type: str = Field(default="", description="票据类型（发票/收据/小票）")
    invoice_number: str = Field(default="", description="发票号码")
    invoice_code: str = Field(default="", description="发票代码")
    issue_date: str = Field(default="", description="开票日期")
    seller_name: str = Field(default="", description="销售方名称")
    seller_tax_id: str = Field(default="", description="销售方税号")
    buyer_name: str = Field(default="", description="购买方名称")
    buyer_tax_id: str = Field(default="", description="购买方税号")
    total_amount: str = Field(default="", description="价税合计")
    amount_in_figures: str = Field(default="", description="小写金额")
    amount_in_words: str = Field(default="", description="大写金额")
    tax_amount: str = Field(default="", description="税额")
    items: str = Field(default="", description="商品明细")
    payment_method: str = Field(default="", description="支付方式")
    remarks: str = Field(default="", description="备注")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 3. 合同/协议 Schema
# ==========================================
class ContractSchema(BaseModel):
    """合同/协议 Schema"""

    contract_title: str = Field(default="", description="合同标题")
    contract_number: str = Field(default="", description="合同编号")
    contract_type: str = Field(default="", description="合同类型")
    party_a: str = Field(default="", description="甲方")
    party_b: str = Field(default="", description="乙方")
    party_c: str = Field(default="", description="丙方（如有）")
    sign_date: str = Field(default="", description="签订日期")
    effective_date: str = Field(default="", description="生效日期")
    expiry_date: str = Field(default="", description="到期日期")
    contract_amount: str = Field(default="", description="合同金额")
    payment_terms: str = Field(default="", description="付款条款")
    delivery_date: str = Field(default="", description="交付日期")
    key_obligations: str = Field(default="", description="主要义务")
    liability_terms: str = Field(default="", description="违约责任")
    dispute_resolution: str = Field(default="", description="争议解决")
    governing_law: str = Field(default="", description="适用法律")
    signatures: str = Field(default="", description="签字盖章")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 4. 证件 Schema
# ==========================================
class IDCardSchema(BaseModel):
    """身份证/护照/驾照 Schema"""

    document_type: str = Field(default="", description="证件类型")
    name: str = Field(default="", description="姓名")
    name_pinyin: str = Field(default="", description="姓名拼音（护照）")
    gender: str = Field(default="", description="性别")
    ethnicity: str = Field(default="", description="民族")
    birth_date: str = Field(default="", description="出生日期")
    id_number: str = Field(default="", description="证件号码")
    address: str = Field(default="", description="住址")
    issuing_authority: str = Field(default="", description="签发机关")
    issue_date: str = Field(default="", description="签发日期")
    expiry_date: str = Field(default="", description="有效期至")
    nationality: str = Field(default="", description="国籍（护照）")
    photo_detected: str = Field(default="", description="是否检测到照片")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 5. 教育场景 Schema (V7.0新增)
# ==========================================
class EducationSchema(BaseModel):
    """教育-试卷/成绩单/证书 Schema"""

    document_type: str = Field(default="", description="文档类型（试卷/成绩单/证书）")
    student_name: str = Field(default="", description="学生姓名")
    student_id: str = Field(default="", description="学号")
    class_name: str = Field(default="", description="班级")
    grade: str = Field(default="", description="年级")
    school: str = Field(default="", description="学校")
    semester: str = Field(default="", description="学期")
    exam_date: str = Field(default="", description="考试日期")
    subjects: str = Field(default="", description="科目成绩列表（JSON字符串）")
    total_score: str = Field(default="", description="总分")
    average_score: str = Field(default="", description="平均分")
    class_rank: str = Field(default="", description="班级排名")
    grade_rank: str = Field(default="", description="年级排名")
    teacher_comment: str = Field(default="", description="教师评语")
    certificate_number: str = Field(default="", description="证书编号（证书类）")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 6. 医疗病历 Schema (V7.0新增)
# ==========================================
class MedicalRecordSchema(BaseModel):
    """医疗-病历/检查报告/处方 Schema"""

    document_type: str = Field(default="", description="文档类型（病历/检查报告/处方）")
    patient_name: str = Field(default="", description="患者姓名")
    patient_id: str = Field(default="", description="患者ID")
    gender: str = Field(default="", description="性别")
    age: str = Field(default="", description="年龄")
    department: str = Field(default="", description="科室")
    doctor_name: str = Field(default="", description="医生姓名")
    visit_date: str = Field(default="", description="就诊日期")
    chief_complaint: str = Field(default="", description="主诉")
    present_illness: str = Field(default="", description="现病史")
    past_history: str = Field(default="", description="既往史")
    physical_exam: str = Field(default="", description="查体")
    diagnosis: str = Field(default="", description="诊断")
    examination_items: str = Field(default="", description="检查项目")
    examination_results: str = Field(default="", description="检查结果")
    prescription: str = Field(default="", description="处方")
    doctor_advice: str = Field(default="", description="医嘱")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 7. 法律文书 Schema (V7.0新增)
# ==========================================
class LegalDocumentSchema(BaseModel):
    """法律-裁判文书/起诉状 Schema"""

    document_type: str = Field(default="", description="文书类型")
    case_number: str = Field(default="", description="案号")
    court: str = Field(default="", description="审理法院")
    case_type: str = Field(default="", description="案件类型")
    plaintiff: str = Field(default="", description="原告/上诉人")
    defendant: str = Field(default="", description="被告/被上诉人")
    judge: str = Field(default="", description="审判员")
    filing_date: str = Field(default="", description="立案日期")
    hearing_date: str = Field(default="", description="开庭日期")
    judgment_date: str = Field(default="", description="判决日期")
    claims: str = Field(default="", description="诉讼请求")
    facts: str = Field(default="", description="事实与理由")
    evidence: str = Field(default="", description="证据")
    judgment_result: str = Field(default="", description="判决结果")
    legal_basis: str = Field(default="", description="法律依据")
    appeal_right: str = Field(default="", description="上诉权利")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 8. 电商商品 Schema (V7.0新增)
# ==========================================
class EcommerceProductSchema(BaseModel):
    """电商-商品/订单 Schema"""

    document_type: str = Field(default="", description="文档类型（商品/订单）")
    product_name: str = Field(default="", description="商品名称")
    product_sku: str = Field(default="", description="商品SKU")
    brand: str = Field(default="", description="品牌")
    category: str = Field(default="", description="类目")
    price: str = Field(default="", description="价格")
    original_price: str = Field(default="", description="原价")
    specifications: str = Field(default="", description="规格")
    product_description: str = Field(default="", description="商品描述")
    product_images: str = Field(default="", description="商品图片")
    sales_count: str = Field(default="", description="销量")
    rating: str = Field(default="", description="评分")
    reviews_count: str = Field(default="", description="评价数")
    seller_name: str = Field(default="", description="卖家")
    order_number: str = Field(default="", description="订单号（订单类）")
    order_date: str = Field(default="", description="订单日期")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 9. 物流 Schema
# ==========================================
class LogisticsSchema(BaseModel):
    """物流-快递单/运单 Schema"""

    waybill_number: str = Field(default="", description="运单号")
    carrier: str = Field(default="", description="快递公司")
    sender_name: str = Field(default="", description="寄件人姓名")
    sender_phone: str = Field(default="", description="寄件人电话")
    sender_address: str = Field(default="", description="寄件人地址")
    recipient_name: str = Field(default="", description="收件人姓名")
    recipient_phone: str = Field(default="", description="收件人电话")
    recipient_address: str = Field(default="", description="收件人地址")
    item_description: str = Field(default="", description="物品描述")
    weight: str = Field(default="", description="重量")
    shipping_fee: str = Field(default="", description="运费")
    pickup_date: str = Field(default="", description="取件日期")
    expected_delivery: str = Field(default="", description="预计送达")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 10. 药品 Schema
# ==========================================
class PharmaceuticalSchema(BaseModel):
    """药品-包装/说明书 Schema"""

    drug_name: str = Field(default="", description="药品名称")
    drug_name_en: str = Field(default="", description="英文名")
    approval_number: str = Field(default="", description="批准文号")
    specification: str = Field(default="", description="规格")
    dosage_form: str = Field(default="", description="剂型")
    manufacturer: str = Field(default="", description="生产企业")
    ingredients: str = Field(default="", description="成分")
    indications: str = Field(default="", description="适应症")
    usage_dosage: str = Field(default="", description="用法用量")
    adverse_reactions: str = Field(default="", description="不良反应")
    contraindications: str = Field(default="", description="禁忌")
    precautions: str = Field(default="", description="注意事项")
    storage: str = Field(default="", description="贮藏")
    expiry_date: str = Field(default="", description="有效期")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 11. 银行流水 Schema
# ==========================================
class FinanceStatementSchema(BaseModel):
    """银行流水/对账单 Schema"""

    account_holder: str = Field(default="", description="账户持有人")
    account_number: str = Field(default="", description="账号")
    bank_name: str = Field(default="", description="银行名称")
    statement_period: str = Field(default="", description="账单周期")
    opening_balance: str = Field(default="", description="期初余额")
    closing_balance: str = Field(default="", description="期末余额")
    total_income: str = Field(default="", description="总收入")
    total_expense: str = Field(default="", description="总支出")
    transactions: str = Field(default="", description="交易明细")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 12. 通用文档 Schema
# ==========================================
class GeneralDocumentSchema(BaseModel):
    """通用文档 Schema"""

    title: str = Field(default="", description="文档标题")
    document_type: str = Field(default="", description="文档类型")
    author: str = Field(default="", description="作者")
    publish_date: str = Field(default="", description="发布日期")
    summary: str = Field(default="", description="摘要")
    key_points: str = Field(default="", description="关键点")
    keywords: str = Field(default="", description="关键字")
    main_content: str = Field(default="", description="主要内容")
    confidence: float = Field(default=0.0, description="提取置信度")


# ==========================================
# 场景注册表
# ==========================================
SCENARIO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "packaging": {
        "schema": PackagingSchema,
        "display_name": "产品包装",
        "recommended_model": "doubao-seed-2-0-lite-260215",
        "category": "goods"
    },
    "finance_receipt": {
        "schema": FinanceReceiptSchema,
        "display_name": "金融票据",
        "recommended_model": "doubao-seed-2-0-pro-260215",
        "category": "finance"
    },
    "finance_statement": {
        "schema": FinanceStatementSchema,
        "display_name": "银行流水",
        "recommended_model": "doubao-seed-2-0-pro-260215",
        "category": "finance"
    },
    "pharmaceutical": {
        "schema": PharmaceuticalSchema,
        "display_name": "药品",
        "recommended_model": "doubao-seed-2-0-pro-260215",
        "category": "medical"
    },
    "contract": {
        "schema": ContractSchema,
        "display_name": "合同协议",
        "recommended_model": "kimi-k2-5-260127",
        "category": "legal"
    },
    "id_card": {
        "schema": IDCardSchema,
        "display_name": "身份证件",
        "recommended_model": "doubao-seed-2-0-mini-260215",
        "category": "identity"
    },
    "logistics": {
        "schema": LogisticsSchema,
        "display_name": "物流面单",
        "recommended_model": "doubao-seed-2-0-lite-260215",
        "category": "logistics"
    },
    "education": {
        "schema": EducationSchema,
        "display_name": "教育文档",
        "recommended_model": "kimi-k2-5-260127",
        "category": "education"
    },
    "medical_record": {
        "schema": MedicalRecordSchema,
        "display_name": "医疗病历",
        "recommended_model": "kimi-k2-5-260127",
        "category": "medical"
    },
    "legal_document": {
        "schema": LegalDocumentSchema,
        "display_name": "法律文书",
        "recommended_model": "kimi-k2-5-260127",
        "category": "legal"
    },
    "ecommerce_product": {
        "schema": EcommerceProductSchema,
        "display_name": "电商商品",
        "recommended_model": "qwen-3-5-plus-260215",
        "category": "ecommerce"
    },
    "general_document": {
        "schema": GeneralDocumentSchema,
        "display_name": "通用文档",
        "recommended_model": "doubao-seed-2-0-lite-260215",
        "category": "general"
    },
}


def get_scenario_schema(scenario: str) -> BaseModel:
    """获取场景对应的Pydantic Schema"""
    if scenario not in SCENARIO_REGISTRY:
        scenario = "general_document"
    return SCENARIO_REGISTRY[scenario]["schema"]


def get_scenario_info(scenario: str) -> Dict[str, Any]:
    """获取场景的完整元信息"""
    if scenario not in SCENARIO_REGISTRY:
        scenario = "general_document"
    return SCENARIO_REGISTRY[scenario]


def get_schema_json_description(scenario: str) -> str:
    """获取场景Schema的JSON描述（用于Prompt注入）"""
    schema_class = get_scenario_schema(scenario)
    return schema_class.model_json_schema()


def list_all_scenarios() -> List[str]:
    """列出所有支持的场景"""
    return list(SCENARIO_REGISTRY.keys())
