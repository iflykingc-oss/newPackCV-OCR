"""金融票据/回单/流水单场景Schema"""
from utils.scenario_schemas.base import BaseSchema, FieldDef

FINANCE_RECEIPT_SP = """\
# 角色定义
你是金融票据信息提取引擎，负责将发票、收据、银行回单、流水单上的文本提取为标准化结构化数据。
# 任务目标
提取金融票据上的所有关键信息，特别关注金额、日期、号码、双方信息。
# 约束
- 金额精确到分（小数点后两位）
- 日期统一格式YYYY-MM-DD
- 不得编造任何信息"""

FINANCE_RECEIPT_UP = """\
请基于OCR识别文本完成金融票据信息提取，严格遵守系统规则。

OCR文本：
{{ocr_text}}

输出要求：
1. 输出纯JSON，禁止任何多余文字
2. 金额字段为数字类型
3. items数组每项含name/qty/price/amount"""

FINANCE_RECEIPT_SCHEMA = BaseSchema(
    scenario_type="finance_receipt",
    scenario_name="金融票据/发票信息提取",
    description="发票、收据、小票等金融票据上的结构化信息提取",
    system_prompt=FINANCE_RECEIPT_SP,
    user_prompt_template=FINANCE_RECEIPT_UP,
    fields=[
        FieldDef(name="document_type", description="票据类型（增值税发票/普通发票/收据/小票/其他）", required=True),
        FieldDef(name="invoice_number", description="发票号码/票据编号", required=True),
        FieldDef(name="invoice_code", description="发票代码（增值税发票特有）"),
        FieldDef(name="issue_date", description="开票日期，格式YYYY-MM-DD", required=True),
        FieldDef(name="seller_name", description="销售方名称", required=True),
        FieldDef(name="seller_tax_id", description="销售方纳税人识别号"),
        FieldDef(name="buyer_name", description="购买方名称", required=True),
        FieldDef(name="buyer_tax_id", description="购买方纳税人识别号"),
        FieldDef(name="items", description="商品/服务明细列表，每项含名称、数量、单价、金额", field_type="list"),
        FieldDef(name="total_amount", description="总金额（数字，不含单位）", required=True),
        FieldDef(name="total_amount_cn", description="总金额（大写中文）"),
        FieldDef(name="tax_amount", description="税额"),
        FieldDef(name="tax_rate", description="税率"),
        FieldDef(name="check_code", description="校验码（增值税发票尾部）"),
        FieldDef(name="qr_code_content", description="二维码内容（如有）"),
        FieldDef(name="seal_info", description="印章信息"),
        FieldDef(name="ext_info", description="其他有价值信息，字符串数组", field_type="list"),
    ]
)

FINANCE_STATEMENT_SP = """\
# 角色定义
你是银行回单/流水单信息提取引擎，负责将银行单据上的交易信息提取为标准化结构化数据。
# 任务目标
提取银行回单、流水单上的交易日期、金额、双方账户信息。
# 约束
- 金额保留两位小数
- 账号可掩码显示
- 交易流水号完整提取"""

FINANCE_STATEMENT_UP = """\
请基于OCR识别文本完成银行回单/流水单信息提取，严格遵守系统规则。

OCR文本：
{{ocr_text}}

输出要求：
1. 输出纯JSON
2. 交易金额含正负符号
3. 日期格式YYYY-MM-DD"""

FINANCE_STATEMENT_SCHEMA = BaseSchema(
    scenario_type="finance_statement",
    scenario_name="银行回单/流水单信息提取",
    description="银行回单、流水单、对账单等银行单据上的结构化信息提取",
    system_prompt=FINANCE_STATEMENT_SP,
    user_prompt_template=FINANCE_STATEMENT_UP,
    fields=[
        FieldDef(name="document_type", description="单据类型（银行回单/流水单/对账单/其他）", required=True),
        FieldDef(name="bank_name", description="银行名称", required=True),
        FieldDef(name="account_name", description="账户名称/户名", required=True),
        FieldDef(name="account_number", description="账号（后4位或掩码）", required=True),
        FieldDef(name="transaction_date", description="交易日期，格式YYYY-MM-DD", required=True),
        FieldDef(name="transaction_type", description="交易类型（转账/汇款/存款/取款/消费/其他）"),
        FieldDef(name="counterparty_name", description="对方户名"),
        FieldDef(name="counterparty_account", description="对方账号"),
        FieldDef(name="amount", description="交易金额（数字，含符号）", required=True),
        FieldDef(name="currency", description="币种，默认RMB"),
        FieldDef(name="balance_before", description="交易前余额"),
        FieldDef(name="balance_after", description="交易后余额"),
        FieldDef(name="transaction_id", description="交易流水号"),
        FieldDef(name="channel", description="交易渠道（柜面/网银/手机银行/ATM等）"),
        FieldDef(name="remark", description="摘要/备注/用途"),
        FieldDef(name="seal_info", description="印章/电子回单章信息"),
        FieldDef(name="ext_info", description="其他有价值信息，字符串数组", field_type="list"),
    ]
)

FINANCE_SCHEMAS = {
    "finance_receipt": FINANCE_RECEIPT_SCHEMA,
    "finance_statement": FINANCE_STATEMENT_SCHEMA,
}