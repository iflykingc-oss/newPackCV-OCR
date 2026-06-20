"""医药包装/药品信息场景Schema"""
from utils.scenario_schemas.base import BaseSchema, FieldDef

PHARMA_SP = """\
# 角色定义
你是药品信息提取引擎，负责将药品包装、说明书上的文本提取为标准化结构化数据。
# 任务目标
提取药品的关键信息，包括药品名称、批准文号、生产批号、有效期等核心字段。
# 约束
- 药品名称提取通用名称
- 批准文号必须完整（含字母数字）
- 日期格式YYYY-MM-DD"""

PHARMA_UP = """\
请基于OCR识别文本完成药品信息提取，严格遵守系统规则。

OCR文本：
{{ocr_text}}

输出要求：
1. 输出纯JSON
2. 批准文号必须完整
3. ingredients/excipients为字符串数组"""

PHARMA_SCHEMA = BaseSchema(
    scenario_type="pharmaceutical",
    scenario_name="医药包装/药品信息提取",
    description="药品包装、说明书、标签上的结构化信息提取",
    system_prompt=PHARMA_SP,
    user_prompt_template=PHARMA_UP,
    fields=[
        FieldDef(name="drug_name", description="药品通用名称", required=True),
        FieldDef(name="brand_name", description="商品名/品牌名"),
        FieldDef(name="english_name", description="英文名称"),
        FieldDef(name="active_ingredients", description="活性成分/有效成分，字符串数组", field_type="list"),
        FieldDef(name="excipients", description="辅料成分，字符串数组", field_type="list"),
        FieldDef(name="dosage_form", description="剂型（片剂/胶囊/注射液/口服液/外用等）"),
        FieldDef(name="specification", description="规格（如'0.25g×12片'）"),
        FieldDef(name="package_unit", description="包装单位（盒/瓶/袋等）"),
        FieldDef(name="manufacturer", description="生产企业", required=True),
        FieldDef(name="approval_number", description="批准文号（国药准字xxxxx）", required=True),
        FieldDef(name="batch_number", description="生产批号", required=True),
        FieldDef(name="production_date", description="生产日期，格式YYYY-MM-DD"),
        FieldDef(name="expiry_date", description="有效期至，格式YYYY-MM-DD", required=True),
        FieldDef(name="storage_condition", description="贮藏条件（阴凉/冷藏/常温等）"),
        FieldDef(name="usage_dosage", description="用法用量"),
        FieldDef(name="indications", description="适应症/功能主治"),
        FieldDef(name="contraindications", description="禁忌"),
        FieldDef(name="adverse_reactions", description="不良反应，字符串数组", field_type="list"),
        FieldDef(name="precautions", description="注意事项，字符串数组", field_type="list"),
        FieldDef(name="drug_interactions", description="药物相互作用"),
        FieldDef(name="barcode", description="条形码/商品码"),
        FieldDef(name="qr_code_content", description="追溯码/二维码内容"),
        FieldDef(name="ext_info", description="其他有价值信息，字符串数组", field_type="list"),
    ]
)