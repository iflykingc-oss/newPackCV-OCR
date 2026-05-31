# -*- coding: utf-8 -*-
"""
模型结构化提取节点 - 深度优化版 V2.0
增强规则引擎：更多正则模式、包装场景专用提取、字段间关联校验
LLM降级策略保留
"""

import os
import re
import json
import time
import logging
from jinja2 import Template
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ModelExtractInput, ModelExtractOutput
from langchain_core.messages import SystemMessage, HumanMessage
from coze_coding_dev_sdk import LLMClient

logger = logging.getLogger(__name__)


def rule_based_extract(ocr_text: str, template_fields: List[str]) -> Dict[str, Any]:
    """
    增强版规则引擎提取
    1. 更丰富的正则模式（覆盖更多OCR变体）
    2. 多模式匹配优先级
    3. 字段间关联校验（如生产日期+保质期→计算到期日期）
    4. 包装场景专用模式（食品/饮料/日用品）
    """
    result: Dict[str, Any] = {}

    # ==================== 品牌提取 ====================
    brand_rules = [
        # 带标签的模式（中文）
        r"(?:品牌|商标|牌名|品牌名称)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9]+)",
        # 英文标签
        r"(?:Brand)[：:]\s*([A-Za-z0-9\u4e00-\u9fa5]+)",
        r"(?:brand)[：:]\s*([A-Za-z0-9\u4e00-\u9fa5]+)",
        # 常见品牌直接匹配（扩展列表）
        r"(金龙鱼|海天|康师傅|统一|蒙牛|伊利|娃哈哈|王老吉|加多宝|百事|可口可乐|雀巢|达利园|三只松鼠|良品铺子|农夫山泉|怡宝|百威|青岛|雪花|加酱油|老干妈|恒顺|太太乐|李锦记|双汇|金锣|雨润|正大|福临门|鲁花|多力|胡姬花|长寿花|西王|刀唛|红蜻蜓|日清|出前一丁|公仔|合味道|五谷道场|白家|阿宽|拉面说|自嗨锅|海底捞|小龙坎|大龙燚|桥头|德庄|秋霞|秦妈|周君记|秋林|百草味|来伊份|卫龙|三只松鼠|洽洽|良品铺子|百味林|天喔|溜溜梅|有友|绝味|周黑鸭|煌上煌|紫燕|廖排骨|川娃子|饭扫光|老干妈|饭遭殃|虎邦|仲景|川南|乌江|饭爷|李子柒|蜀中|桥头|黄飞红|甘源|盐津铺子|劲仔|甘竹|鹰金钱|梅林|古龙|银鹭|椰树|汇源|味全|养乐多|光明|三元|新希望|燕塘|晨光|风行|香满楼|维他|豆本豆|维维|永和|冰泉|露露|六个核桃|承德露露|椰树|银鹭|达利园|好吃点|可比克|好丽友|乐事|品客|上好佳|旺旺|徐福记|嘉士利|奥利奥|趣多多|太平梳打|优冠|达能|卡尔顿|盼盼|法丽兹|瑞可德|来伊份|百草味)",
        # 公司名提取品牌
        r"([\u4e00-\u9fa5]{2,6})(?:集团|股份|有限公司|公司|厂)",
    ]

    # ==================== 产品名称提取 ====================
    product_name_rules = [
        r"(?:产品名称|品名|名称|产品名|商品名称|商品名)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9/（()）\s]+?)(?:\n|$|规格|净含量|配料)",
        r"(?:品名)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9/]+)",
        # 英文标签
        r"(?:Product(?:\s+Name)?)[：:]\s*([A-Za-z0-9\u4e00-\u9fa5\s/]+?)(?:\n|$)",
        r"(?:Product)[：:]\s*([A-Za-z0-9\u4e00-\u9fa5\s/]+?)(?:\n|$)",
        # 食品/饮料类产品名
        r"([\u4e00-\u9fa5]{2,15}(?:菜籽油|花生油|大豆油|调和油|橄榄油|玉米油|葵花籽油|芝麻油|香油|色拉油|猪油|黄油|奶油|牛奶|酸奶|奶粉|奶酪|饮料|果汁|茶|咖啡|啤酒|白酒|红酒|黄酒|米酒|酱油|醋|酱|味精|鸡精|盐|糖|蜂蜜|巧克力|饼干|蛋糕|面包|方便面|火腿|香肠|肉松|鱼罐头|果酱|花生酱|番茄酱|芝麻酱|辣椒酱|豆腐乳|榨菜|泡菜|蜜饯|坚果|瓜子|花生|核桃|红枣|枸杞|燕麦|大米|面粉|面条|米粉|粉丝|淀粉|食用菌|紫菜|海带|虾米|干贝|腊肉|板鸭|烤鸭|咸鸭蛋|皮蛋|豆腐|豆浆|豆干|腐竹|年糕|汤圆|粽子|月饼|饺子|包子|馒头|花卷|烧麦|春卷|馄饨))",
        # 英文产品名（如 Rapeseed Oil）
        r"((?:Rapeseed|Sunflower|Olive|Soybean|Corn|Peanut|Coconut|Palm|Vegetable|Sesame)\s+(?:Oil|Water|Juice|Milk|Tea|Coffee|Wine|Beer))",
    ]

    # ==================== 规格提取 ====================
    specification_rules = [
        r"(?:净含量|规格|容量|体积|重量|净重|毛重|含量)[：:]*\s*([\d.]+\s*(?:L|l|ml|mL|ML|Ml|g|G|kg|KG|Kg|千克|公斤|克|升|毫升|斤|磅|oz|OZ))",
        r"(?:净含量|规格|容量|体积|重量)[：:]*\s*(\d+\.?\d*\s*(?:L|ml|g|kg|毫升|升|克|千克))",
        # 英文标签
        r"(?:Net\s+Content|Net\s+Weight|Volume|Capacity|Weight|Specification|Spec)[：:]*\s*([\d.]+\s*(?:L|l|ml|mL|g|G|kg|KG|千克|公斤|克|升|毫升|斤|磅|oz|OZ))",
        # 数字+单位（无标签）
        r"(\d+\.?\d*\s*(?:L|ml|g|kg|毫升|升|克|千克|公斤|斤))",
        # 特殊格式：5L/5升
        r"(?:净含量)[：:]*\s*(\d+\s*[L升毫升克千克kgg]+)",
    ]

    # ==================== 生产日期提取 ====================
    production_date_rules = [
        r"(?:生产日期|制造日期|生产|日期|DATE|PROD)[：:]*\s*(\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}\s*日?)",
        r"(?:生产日期|制造日期)[：:]*\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        r"(?:生产日期|制造日期)[：:]*\s*(\d{4}年\d{1,2}月\d{1,2}日?)",
        # 英文标签
        r"(?:Production\s+Date|Date\s+of\s+Manufacture|Mfg\s+Date|DATE|PROD)[：:]*\s*(\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2})",
        r"(?:Production\s+Date)[：:]*\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        # 紧凑格式
        r"(?:生产日期|DATE)[：:]*\s*(\d{8})",
        # 无标签但有日期格式
        r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)(?:\s|$)",
    ]

    # ==================== 保质期提取 ====================
    shelf_life_rules = [
        r"(?:保质期|有效期|有效期限|保存期|保存期限|保质)[：:]*\s*(\d{1,3}\s*(?:天|日|个月|月|年))",
        r"(?:保质期|有效期|保存期)[：:]*\s*(\d+\s*[天日月年])",
        # 英文标签
        r"(?:Shelf\s+Life|Best\s+Before|Expiration|Expiry|Use\s+By)[：:]*\s*(\d{1,3}\s*(?:days?|months?|years?|天|日|个月|月|年))",
        r"(?:Shelf\s+Life)[：:]*\s*(\d+\s*(?:days?|months?|years?))",
        # 特殊格式
        r"(?:保质期)[：:]*\s*(至\s*\d{4}[-/年]\d{1,2}[-/月]\d{1,2})",
        r"(?:保质期至|有效期至|有效期到|保存期至)[：:]*\s*(\d{4}[-/年.]\s*\d{1,2}[-/月.]?\s*\d{0,2})",
    ]

    # ==================== 生产商提取 ====================
    manufacturer_rules = [
        r"(?:制造商|生产商|生产厂家|厂家|委托方|出品|生产商地址|生产者)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9（()）\s]+?)(?:\n|$|电话|地址|邮编)",
        r"(?:委托生产|受委托方|经销|总经销|总代理)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9]+)",
        # 英文标签
        r"(?:Manufacturer|Mfg|Mfg\s+by|Produced\s+by|Producer|Made\s+by)[：:]*\s*([A-Za-z0-9\u4e00-\u9fa5\s.]+?)(?:\n|$)",
        # 公司名模式
        r"([\u4e00-\u9fa5]{2,10}(?:有限公司|股份公司|集团|公司|厂|厂址))",
        # 英文公司名
        r"([A-Z][A-Za-z0-9\s]+(?:Co\.|Ltd\.|Inc\.|Corp\.|Company|Group))",
    ]

    # ==================== 配料提取 ====================
    ingredients_rules = [
        r"(?:配料|成分|原料|原料与辅料|配料表)[：:]\s*(.+?)(?:\n|$|保质期|生产日期|生产商|贮存|储藏)",
        r"(?:配料表|成分表|Ingredients)[：:]\s*(.+?)(?:\n|$)",
    ]

    # ==================== 执行标准提取 ====================
    standard_rules = [
        r"(?:执行标准|标准号|产品标准|标准代号|产品标准号)[：:]*\s*([A-Z]{1,3}\s*/?\s*\d{4,5}[-—.]?\d{0,4})",
        r"(?:标准)[：:]*\s*(GB\s*/?\s*\d+[-—.]?\d*|Q\s*/?\s*\d+[-—.]?\d*|DB\s*/?\s*\d+[-—.]?\d*|SB\s*/?\s*\d+[-—.]?\d*)",
        # 英文标签
        r"(?:Standard|Standard\s+No|Exec\s+Standard)[：:]*\s*(GB\s*/?\s*\d+[-—.]?\d*|Q\s*/?\s*\d+[-—.]?\d*|DB\s*/?\s*\d+[-—.]?\d*)",
        r"(GB/?\d+[-—.]?\d*)",
        r"(Q/?\d+[-—.]?\d*)",
    ]

    # ==================== 批号提取 ====================
    batch_number_rules = [
        r"(?:批号|批次|生产批号|批编码|LOT|Batch)[：:]*\s*([A-Za-z0-9\-_]+)",
        r"(?:LOT\s*NO|BATCH\s*NO)[.:：]*\s*([A-Za-z0-9\-_]+)",
    ]

    # ==================== 许可证号提取 ====================
    license_number_rules = [
        r"(?:生产许可证|许可证|SC编号|食品生产许可证|许可证编号|QS|SC)[：:]*\s*(SC\d{10,14})",
        r"(?:生产许可证|许可证|QS|SC)[：:]*\s*(QS\d{10,12})",
        # 英文标签
        r"(?:License|License\s+No|Production\s+License|SC\s+No)[：:]*\s*(SC\d{10,14})",
        r"(?:License|License\s+No)[：:]*\s*(QS\d{10,12})",
        r"(SC\d{10,14})",
        r"(QS\d{10,12})",
    ]

    # ==================== 贮存条件提取 ====================
    storage_rules = [
        r"(?:贮存条件|贮存方法|保存方法|储藏方法|存放条件|储藏条件|保存条件|贮藏)[：:]*\s*(.+?)(?:\n|$|生产日期|生产商|保质期)",
        r"(?:贮存|保存|储藏|冷藏|冷冻|避光|阴凉|干燥)[：:]*\s*(.+?)(?:\n|$)",
        # 英文标签
        r"(?:Storage|Storage\s+Condition|Storage\s+Method|Keep|Store)[：:]*\s*(.+?)(?:\n|$)",
    ]

    # 汇总所有规则
    all_rules: Dict[str, List[str]] = {
        "brand": brand_rules,
        "product_name": product_name_rules,
        "specification": specification_rules,
        "production_date": production_date_rules,
        "shelf_life": shelf_life_rules,
        "manufacturer": manufacturer_rules,
        "ingredients": ingredients_rules,
        "standard": standard_rules,
        "batch_number": batch_number_rules,
        "license_number": license_number_rules,
        "storage_condition": storage_rules,
    }

    # 执行规则匹配
    for field, patterns in all_rules.items():
        for pattern in patterns:
            match = re.search(pattern, ocr_text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    if match.lastindex and match.lastindex >= 1:
                        value = match.group(1).strip()
                    else:
                        value = match.group(0).strip()
                    if value:
                        result[field] = value
                        break
                except IndexError:
                    result[field] = match.group(0).strip()
                    break

    # ==================== 关联校验 ====================
    # 如果有生产日期和保质期，计算到期日期
    if "production_date" in result and "shelf_life" in result:
        try:
            _compute_expiry_date(result)
        except Exception as e:
            pass  # 计算失败不影响主流程

    # 填充默认字段
    if template_fields:
        for field in template_fields:
            if field not in result:
                result[field] = "N/A"
    else:
        default_fields = [
            "brand", "product_name", "specification",
            "production_date", "shelf_life", "manufacturer"
        ]
        for field in default_fields:
            if field not in result:
                result[field] = "N/A"
        for field in all_rules:
            if field not in default_fields and field not in result:
                result[field] = "N/A"

    return result


def _compute_expiry_date(result: Dict[str, Any]) -> None:
    """根据生产日期和保质期计算到期日期"""
    import datetime

    prod_date_str = result.get("production_date", "")
    shelf_life_str = result.get("shelf_life", "")

    if not prod_date_str or not shelf_life_str:
        return

    # 解析生产日期
    date_match = re.search(r'(\d{4})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})', prod_date_str)
    if not date_match:
        return

    year = int(date_match.group(1))
    month = int(date_match.group(2))
    day = int(date_match.group(3))

    try:
        prod_date = datetime.date(year, month, day)
    except ValueError:
        return

    # 解析保质期
    shelf_match = re.match(r'(\d+)\s*(天|日|个月|月|年)', shelf_life_str)
    if not shelf_match:
        return

    value = int(shelf_match.group(1))
    unit = shelf_match.group(2)

    if unit in ("天", "日"):
        expiry_date = prod_date + datetime.timedelta(days=value)
    elif unit in ("个月", "月"):
        # 简化计算：按月加
        expiry_month = month + value
        expiry_year = year + (expiry_month - 1) // 12
        expiry_month = (expiry_month - 1) % 12 + 1
        try:
            expiry_date = datetime.date(expiry_year, expiry_month, min(day, 28))
        except ValueError:
            expiry_date = datetime.date(expiry_year, expiry_month, 1)
    elif unit == "年":
        try:
            expiry_date = datetime.date(year + value, month, day)
        except ValueError:
            expiry_date = datetime.date(year + value, month, min(day, 28))
    else:
        return

    result["expiry_date"] = expiry_date.strftime("%Y-%m-%d")


def model_extract_node(
    state: ModelExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ModelExtractOutput:
    """
    title: 结构化信息提取
    desc: 使用大语言模型从OCR识别文本中提取结构化信息（如品牌、规格、生产日期等），
          LLM不可用时自动降级到增强版规则引擎
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 预先获取ocr_text（避免在except块中使用未定义变量）
    ocr_text = state.ocr_text or state.raw_text or state.ocr_raw_result or ""

    try:
        # 加载模型配置
        cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH"), config['metadata']['llm_cfg'])
        with open(cfg_file, 'r', encoding='utf-8') as f:
            _cfg = json.load(f)

        llm_config = _cfg.get("config", {})
        sp = _cfg.get("sp", "")
        up_tpl = Template(_cfg.get("up", ""))

        # 使用自定义提示词或默认提示词
        user_prompt = state.custom_prompt if state.custom_prompt else up_tpl.render({
            "ocr_text": ocr_text,
            "fields": json.dumps(state.template_fields, ensure_ascii=False) if state.template_fields else ""
        })

        # 构造消息
        messages = [
            SystemMessage(content=sp),
            HumanMessage(content=user_prompt)
        ]

        # 初始化并调用模型
        client = LLMClient(ctx=ctx)

        model_params = {
            "model": llm_config.get("model", state.model_name),
            "temperature": llm_config.get("temperature", 0.1),
            "max_tokens": llm_config.get("max_completion_tokens", 2000),
        }

        response = client.invoke(
            messages=messages,
            **model_params
        )

        # 解析响应 - 安全获取文本内容
        result_text = ""
        if response and hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                result_text = content
            elif isinstance(content, list):
                if content and isinstance(content[0], str):
                    result_text = " ".join(content)
                else:
                    text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                    result_text = " ".join(text_parts).strip()
            else:
                result_text = str(content)

        # 尝试解析JSON
        structured_data: Dict[str, Any] = {}
        confidence = 0.0
        missing_fields: List[str] = []

        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group())
            else:
                structured_data = json.loads(result_text)

            if state.template_fields:
                filled_fields = [f for f in state.template_fields if f in structured_data and structured_data[f]]
                missing_fields = [f for f in state.template_fields if f not in filled_fields]
                confidence = len(filled_fields) / len(state.template_fields) if state.template_fields else 1.0
            else:
                confidence = 0.8

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败，尝试文本提取: {str(e)}")
            structured_data = {"raw_extract": result_text}
            confidence = 0.5

        logger.info(f"LLM结构化提取完成，置信度: {confidence:.2f}")

        return ModelExtractOutput(
            structured_data=structured_data,
            confidence=confidence,
            missing_fields=missing_fields
        )

    except Exception as e:
        logger.warning(f"模型结构化提取失败: {str(e)}，降级到规则引擎")
        # 降级到规则引擎 - ocr_text已在函数开头安全获取
        rule_data = rule_based_extract(ocr_text, state.template_fields or [])
        filled_count = sum(1 for v in rule_data.values() if v != "N/A")
        rule_confidence = filled_count / len(rule_data) if rule_data else 0.3

        logger.info(f"规则引擎提取完成: {json.dumps(rule_data, ensure_ascii=False)}, 置信度: {rule_confidence:.2f}")
        return ModelExtractOutput(
            structured_data=rule_data,
            confidence=rule_confidence,
            missing_fields=[f for f in rule_data if rule_data[f] == "N/A"]
        )
