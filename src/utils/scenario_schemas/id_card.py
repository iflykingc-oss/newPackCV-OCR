"""证件场景Schema"""
from utils.scenario_schemas.base import BaseSchema, FieldDef

ID_CARD_SCHEMA = BaseSchema(
    scenario_type="id_card",
    scenario_name="证件/身份证",
    description="身份证、护照、驾驶证等个人证件的信息提取",
    fields=[
        FieldDef(name="document_type", description="证件类型", required=True, field_type="str", example="身份证/护照/驾驶证"),
        FieldDef(name="full_name", description="姓名", required=True, field_type="str"),
        FieldDef(name="name_en", description="英文姓名（护照）", required=False, field_type="str"),
        FieldDef(name="gender", description="性别", required=True, field_type="str"),
        FieldDef(name="ethnicity", description="民族", required=False, field_type="str", example="汉族"),
        FieldDef(name="birth_date", description="出生日期", required=True, field_type="date", example="1990-01-01"),
        FieldDef(name="id_number", description="证件号码", required=True, field_type="str"),
        FieldDef(name="nationality", description="国籍", required=False, field_type="str"),
        FieldDef(name="address", description="住址", required=True, field_type="str"),
        FieldDef(name="issuing_authority", description="签发机关", required=True, field_type="str"),
        FieldDef(name="valid_from", description="有效期起始", required=True, field_type="date"),
        FieldDef(name="valid_until", description="有效期截止", required=True, field_type="date"),
        FieldDef(name="photo_available", description="是否含照片", required=False, field_type="str", example="是/否"),
        FieldDef(name="issuing_date", description="签发日期", required=False, field_type="date"),
        FieldDef(name="vehicle_category", description="准驾车型（驾驶证）", required=False, field_type="str", example="C1"),
        FieldDef(name="passport_number", description="护照号", required=False, field_type="str"),
    ],
    system_prompt="""# 角色
你是证件信息提取引擎，专注身份证、护照、驾驶证等个人证件的信息提取。

# 任务
提取证件上的所有可见信息，注意：
1. 证件号码必须完整（身份证18位、护照号字母+数字）
2. 日期统一YYYY-MM-DD格式
3. 地址保留完整省市县区街道

# 约束
- 不编造，只提取可见信息
- 证件类型自动识别（身份证/护照/驾驶证）
- 敏感信息仅返回提取结果，不储存、不传播""",
    user_prompt_template="""请基于OCR识别文本完成证件信息提取。

OCR文本：
{{ocr_text}}

输出要求：
1. 纯JSON格式
2. 证件号码必须完整
3. 地址保留原始格式"""
)