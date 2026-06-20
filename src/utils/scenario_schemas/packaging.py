"""包装/日化/食品类场景Schema"""
from utils.scenario_schemas.base import BaseSchema, FieldDef

PACKAGING_SYSTEM_PROMPT = """\
# 角色定义
你是商品包装信息提取引擎，负责将商品包装上的OCR文本与视觉信息提取为标准化结构化数据。
# 任务目标
提取商品包装上的所有有效信息，按指定JSON结构输出。
# 约束
- 仅提取实际存在的信息，不得编造
- 输出必须可被JSON.parse()直接解析
- 品类限定为：食品/饮料/日化清洁/个人护理/药品/电子产品/其他"""

PACKAGING_USER_PROMPT = """\
请基于OCR识别文本完成商品包装信息提取，严格遵守系统规则。

OCR文本：
{{ocr_text}}

输出要求：
1. 输出纯JSON，禁止任何多余文字
2. 品类限定为：食品/饮料/日化清洁/个人护理/药品/电子产品/其他
3. 无对应信息的字段为null，warnings/ext_info为空数组"""

PACKAGING_SCHEMA = BaseSchema(
    scenario_type="packaging",
    scenario_name="商品包装信息提取",
    description="食品、饮料、日化清洁、个人护理、药品、电子产品等商品包装上的标签信息提取",
    system_prompt=PACKAGING_SYSTEM_PROMPT,
    user_prompt_template=PACKAGING_USER_PROMPT,
    fields=[
        FieldDef(name="product_type", description="品类名称（食品/饮料/日化清洁/个人护理/药品/电子产品/其他）", required=True),
        FieldDef(name="brand", description="品牌名称，含中英文", required=True),
        FieldDef(name="product_name", description="产品全称，含中英文"),
        FieldDef(name="specification", description="规格/净含量/型号，如'500ml'、'200g'"),
        FieldDef(name="manufacturer", description="生产商/出品方"),
        FieldDef(name="production_date", description="生产日期，格式YYYY-MM-DD"),
        FieldDef(name="shelf_life", description="保质期/有效期"),
        FieldDef(name="batch_number", description="生产批号/批次号"),
        FieldDef(name="ingredients", description="配料表/主要成分，字符串数组", field_type="list"),
        FieldDef(name="nutrition_info", description="营养成分信息，字符串数组", field_type="list"),
        FieldDef(name="usage_method", description="使用方法"),
        FieldDef(name="storage_condition", description="贮存条件"),
        FieldDef(name="warnings", description="注意事项/安全警示，字符串数组", field_type="list"),
        FieldDef(name="ext_info", description="其他有价值信息（宣传语、认证标识等），字符串数组", field_type="list"),
    ]
)