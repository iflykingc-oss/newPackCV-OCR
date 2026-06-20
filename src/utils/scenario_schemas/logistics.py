"""物流单场景Schema"""
from utils.scenario_schemas.base import BaseSchema, FieldDef

LOGISTICS_SCHEMA = BaseSchema(
    scenario_type="logistics",
    scenario_name="物流单/快递单",
    description="快递面单、物流运单、货运单等信息提取",
    fields=[
        FieldDef(name="tracking_number", description="运单号/追踪号", required=True, field_type="str"),
        FieldDef(name="logistics_company", description="物流公司", required=True, field_type="str", example="顺丰/中通/韵达/FedEx"),
        FieldDef(name="sender_name", description="寄件人姓名", required=True, field_type="str"),
        FieldDef(name="sender_phone", description="寄件人电话", required=False, field_type="str"),
        FieldDef(name="sender_address", description="寄件人地址", required=True, field_type="str"),
        FieldDef(name="sender_company", description="寄件公司", required=False, field_type="str"),
        FieldDef(name="receiver_name", description="收件人姓名", required=True, field_type="str"),
        FieldDef(name="receiver_phone", description="收件人电话", required=False, field_type="str"),
        FieldDef(name="receiver_address", description="收件人地址", required=True, field_type="str"),
        FieldDef(name="receiver_company", description="收件公司", required=False, field_type="str"),
        FieldDef(name="origin", description="始发地", required=False, field_type="str"),
        FieldDef(name="destination", description="目的地", required=False, field_type="str"),
        FieldDef(name="weight_kg", description="重量(kg)", required=False, field_type="str"),
        FieldDef(name="dimensions", description="尺寸(长宽高cm)", required=False, field_type="str"),
        FieldDef(name="item_description", description="物品描述", required=True, field_type="str"),
        FieldDef(name="item_quantity", description="物品数量", required=False, field_type="str"),
        FieldDef(name="declared_value", description="声明价值", required=False, field_type="str"),
        FieldDef(name="cod_amount", description="代收货款金额", required=False, field_type="str"),
        FieldDef(name="shipping_fee", description="运费", required=False, field_type="str"),
        FieldDef(name="service_type", description="服务类型", required=False, field_type="str", example="标快/特惠/次日达"),
        FieldDef(name="order_number", description="订单号", required=False, field_type="str"),
        FieldDef(name="shipping_date", description="发货日期", required=False, field_type="date"),
        FieldDef(name="estimated_delivery", description="预计送达", required=False, field_type="date"),
        FieldDef(name="package_count", description="包裹件数", required=False, field_type="str"),
        FieldDef(name="remarks", description="备注", required=False, field_type="str"),
    ],
    system_prompt="""# 角色
你是物流信息提取引擎，专注快递面单、物流运单的结构化信息提取。

# 任务
提取物流单据中的关键信息，特别关注：
1. 运单号必须完整
2. 地址保留省市县区街道四级
3. 电话保留完整号码
4. 物品描述提取主要品类

# 约束
- 仅提取可见信息
- 地址格式标准化
- 个人信息注意保护""",
    user_prompt_template="""请基于OCR识别文本完成物流单据信息提取。

OCR文本：
{{ocr_text}}

输出要求：
1. 纯JSON格式
2. 运单号完整提取
3. 地址保留省市县区"""
)