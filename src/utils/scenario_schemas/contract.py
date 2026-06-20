"""合同/协议场景Schema"""
from utils.scenario_schemas.base import BaseSchema, FieldDef

CONTRACT_SCHEMA = BaseSchema(
    scenario_type="contract",
    scenario_name="合同/协议",
    description="企业合同、合作协议、服务协议等文档的信息提取",
    fields=[
        FieldDef(name="contract_number", description="合同编号", required=True, field_type="str", example="HT-2025-001"),
        FieldDef(name="contract_name", description="合同名称", required=True, field_type="str", example="软件采购合同"),
        FieldDef(name="signing_date", description="签订日期", required=True, field_type="date", example="2025-01-15"),
        FieldDef(name="effective_date", description="生效日期", required=False, field_type="date"),
        FieldDef(name="expiry_date", description="到期日期", required=False, field_type="date"),
        FieldDef(name="party_a", description="甲方名称", required=True, field_type="str"),
        FieldDef(name="party_a_representative", description="甲方代表", required=False, field_type="str"),
        FieldDef(name="party_b", description="乙方名称", required=True, field_type="str"),
        FieldDef(name="party_b_representative", description="乙方代表", required=False, field_type="str"),
        FieldDef(name="project_name", description="项目名称", required=True, field_type="str"),
        FieldDef(name="total_amount", description="合同金额", required=True, field_type="str", example="CNY 1,000,000.00"),
        FieldDef(name="amount_in_words", description="金额大写", required=False, field_type="str"),
        FieldDef(name="payment_terms", description="付款条款", required=False, field_type="str"),
        FieldDef(name="duration", description="合同期限", required=False, field_type="str", example="12个月"),
        FieldDef(name="signing_location", description="签订地点", required=False, field_type="str"),
        FieldDef(name="parties", description="其他签约方", required=False, field_type="list"),
        FieldDef(name="key_terms", description="关键条款摘要", required=False, field_type="list"),
        FieldDef(name="stamp_seal_info", description="公章/签章信息", required=False, field_type="str"),
        FieldDef(name="confidentiality_clause", description="保密条款", required=False, field_type="str"),
        FieldDef(name="dispute_resolution", description="争议解决", required=False, field_type="str"),
    ],
    system_prompt="""# 角色
你是合同文档信息提取引擎，专注企业合同、协议类文件的结构化提取。

# 任务
提取合同中的关键信息，包括签约双方、金额、日期、关键条款等。特别关注：
1. 金额必须带币种（CNY/USD/JPY等）
2. 日期统一为YYYY-MM-DD格式
3. 条款摘要保留原始语义，不擅自简化

# 约束
- 仅提取原文存在的信息，不编造
- 日期格式统一
- 金额保留小数点后两位""",
    user_prompt_template="""请基于OCR识别文本完成合同信息提取。

OCR文本：
{{ocr_text}}

遵循输出格式：
1. 纯JSON，禁止多余文字
2. 金额保留币种+数值
3. key_terms为条款摘要数组，每项最多30字"""
)