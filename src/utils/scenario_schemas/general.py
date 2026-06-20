"""通用文档/灵活Key-Value场景Schema"""
from utils.scenario_schemas.base import BaseSchema, FieldDef

GENERAL_SP = """\
# 角色定义
你是通用文档信息提取引擎，负责将各类文档中的关键信息提取为结构化数据。
# 任务目标
提取文档中的实体信息（日期、编号、名称、金额），以及灵活的Key-Value字段。
# 约束
- 提取有业务价值的字段
- 不确定的字段不编造
- 提取至少3个以上的key_fields"""

GENERAL_UP = """\
请基于OCR识别文本完成通用文档信息提取，严格遵守系统规则。

OCR文本：
{{ocr_text}}

输出要求：
1. 输出纯JSON
2. key_fields对象包含至少3个有业务价值的字段
3. 表格数据按行提取到table_data数组"""

GENERAL_SCHEMA = BaseSchema(
    scenario_type="general_document",
    scenario_name="通用文档信息提取",
    description="各类文档、标签、证书等通用场景下的灵活Key-Value信息提取",
    system_prompt=GENERAL_SP,
    user_prompt_template=GENERAL_UP,
    fields=[
        FieldDef(name="document_type", description="文档类型（根据内容自动判断）"),
        FieldDef(name="title", description="文档标题/名称"),
        FieldDef(name="key_fields", description="提取的键值对字段，如{'编号':'xxxx','日期':'yyyy'}，至少包含3个有业务价值的字段", field_type="dict"),
        FieldDef(name="dates", description="提取到的所有日期信息，字符串数组", field_type="list"),
        FieldDef(name="numbers", description="提取到的所有编号/号码信息，字符串数组", field_type="list"),
        FieldDef(name="names", description="提取到的所有人名/企业名，字符串数组", field_type="list"),
        FieldDef(name="amounts", description="提取到的所有金额/数量，字符串数组", field_type="list"),
        FieldDef(name="table_data", description="表格数据（如有），每个元素为一行", field_type="list"),
        FieldDef(name="seal_text", description="印章/公章上的文字"),
        FieldDef(name="ext_info", description="其他有价值信息，字符串数组", field_type="list"),
    ]
)