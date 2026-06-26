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
try:
    from coze_coding_dev_sdk import LLMClient
except ImportError:
    LLMClient = None

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
        # 常见品牌直接匹配（扩展列表-包括米多奇等区域品牌）
        r"(金龙鱼|海天|康师傅|统一|蒙牛|伊利|娃哈哈|王老吉|加多宝|百事|可口可乐|雀巢|达利园|三只松鼠|良品铺子|农夫山泉|怡宝|百威|青岛|雪花|加酱油|老干妈|恒顺|太太乐|李锦记|双汇|金锣|雨润|正大|福临门|鲁花|多力|胡姬花|长寿花|西王|刀唛|红蜻蜓|日清|出前一丁|公仔|合味道|五谷道场|白家|阿宽|拉面说|自嗨锅|海底捞|小龙坎|大龙燚|桥头|德庄|秋霞|秦妈|周君记|秋林|百草味|来伊份|卫龙|三只松鼠|洽洽|良品铺子|百味林|天喔|溜溜梅|有友|绝味|周黑鸭|煌上煌|紫燕|廖排骨|川娃子|饭扫光|老干妈|饭遭殃|虎邦|仲景|川南|乌江|饭爷|李子柒|蜀中|桥头|黄飞红|甘源|盐津铺子|劲仔|甘竹|鹰金钱|梅林|古龙|银鹭|椰树|汇源|味全|养乐多|光明|三元|新希望|燕塘|晨光|风行|香满楼|维他|豆本豆|维维|永和|冰泉|露露|六个核桃|承德露露|椰树|银鹭|达利园|好吃点|可比克|好丽友|乐事|品客|上好佳|旺旺|徐福记|嘉士利|奥利奥|趣多多|太平梳打|优冠|达能|卡尔顿|盼盼|法丽兹|瑞可德|来伊份|百草味|米多奇|嗨吃家|食族人|莫小仙|开小灶|统一|今麦郎|白象|克明|陈克明|金沙河|想念|五得利|中裕|发达|利生|鲁王)",
        # 公司名提取品牌（仅当品牌尚未提取时，避免公司名误作品牌）
        r"(?<![\u4e00-\u9fa5])([\u4e00-\u9fa5]{2,6})(?:集团|股份|有限公司|公司)(?![\u4e00-\u9fa5])",
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
        # 优先：带标签的规格（排除"每"前缀如"每100克"）
        r"(?:净含量|规格|容量|体积|重量|净重|毛重|含量)[：:]*\s*([\d.]+\s*(?:L|l|ml|mL|ML|Ml|g|G|kg|KG|Kg|千克|公斤|克|升|毫升|斤|磅|oz|OZ))",
        r"(?:净含量|规格|容量|体积|重量)[：:]*\s*(\d+\.?\d*\s*(?:L|ml|g|kg|毫升|升|克|千克))",
        # 英文标签
        r"(?:Net\s+Content|Net\s+Weight|Volume|Capacity|Weight|Specification|Spec)[：:]*\s*([\d.]+\s*(?:L|l|ml|mL|g|G|kg|KG|千克|公斤|克|升|毫升|斤|磅|oz|OZ))",
        # 数字+单位（无标签）- 排除营养表中的营养素数值（至少2位数字）
        r"(?:^|(?<!\d)(?<![每]))((?:[1-9]\d+|[1-9])\.?\d*\s*(?:L|ml|g|kg|毫升|升|千克|公斤|斤))(?!\s*(?:千焦|蛋白质|脂肪|碳水|钠|能量|膳食纤维|反式|营养|参考值|\d|米|克|毫克|微克))",
        # 特殊格式：5L/5升
        r"(?:净含量)[：:]*\s*(\d+\s*[L升毫升克千克kgg]+)",
    ]

    # ==================== 生产日期提取 ====================
    production_date_rules = [
        # 标准中文标签
        r"(?:生产日期|制造日期|生产|日期|DATE|PROD)[：:]*\s*(\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}\s*日?)",
        r"(?:生产日期|制造日期)[：:]*\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        r"(?:生产日期|制造日期)[：:]*\s*(\d{4}年\d{1,2}月\d{1,2}日?)",
        # OCR容错：生→三、产→广、产→严 等常见OCR误识别（全4字匹配）
        r"(?:[三生主]产日期|制造日期|生产[日期])[：:]*\s*(\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}\s*日?)",
        r"(?:[三生主]产日?期)[：:]*\s*(\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}\s*日?)",
        # OCR容错+非标准日期值（如"见包装上"）- 仅匹配4字以内
        r"(?:[三生主]产日?期)[：:]*\s*([\u4e00-\u9fa5]{1,4})",
        # 英文标签
        r"(?:Production\s+Date|Date\s+of\s+Manufacture|Mfg\s+Date|DATE|PROD)[：:]*\s*(\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2})",
        r"(?:Production\s+Date)[：:]*\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        # 紧凑格式
        r"(?:生产日期|DATE)[：:]*\s*(\d{8})",
        # 无标签但有日期格式 - 排除营养表中的非日期数字
        r"(?<!\d)(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)(?!\s*(?:千焦|蛋白质|脂肪|碳水|钠|能量|克|毫升))",
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
        # 食品标签 - 多行匹配（配料可能跨多行）- 含OCR容错如"料表"→"配料表"
        r"(?:[配]?料表|配料|成分|原料|原料与辅料)[：:]\s*([\s\S]+?)(?=\n\s*(?:保质期|生产日期|生产商|贮存|储藏|贮藏|产品标准|产品标准号|执行标准|产地|制造商|生产厂家|地址|电话|网址|食用方法|使用方法|[过敏改]敏|[过敏改]敏原))",
        r"(?:配料表|成分表|Ingredients)[：:]\s*([\s\S]+?)(?=\n\s*(?:保质期|生产日期|生产商|贮存))",
        # 单行标签（无后续标签时）
        r"(?:[配]?料表|配料|成分|原料)[：:]\s*([\u4e00-\u9fa5a-zA-Z0-9，、（）()\s/]{3,120})",
        # 英文标签
        r"(?:Ingredients)[：:]*\s*([A-Za-z0-9,\s.]{5,200})",
    ]

    # ==================== 执行标准提取 ====================
    standard_rules = [
        r"(?:执行标准|标准号|产品标准|标准代号|产品标准号)[：:]*\s*([A-Z]{1,3}\s*/?\s*[A-Z]?\s*\d{4,5}[-—.]?\d{0,4})",
        r"(?:标准)[：:]*\s*(GB\s*/?\s*[A-Z]?\s*\d+[-—.]?\d*|Q\s*/?\s*\d+[-—.]?\d*|DB\s*/?\s*\d+[-—.]?\d*|SB\s*/?\s*\d+[-—.]?\d*)",
        # 英文标签
        r"(?:Standard|Standard\s+No|Exec\s+Standard)[：:]*\s*(GB\s*/?\s*[A-Z]?\s*\d+[-—.]?\d*|Q\s*/?\s*\d+[-—.]?\d*|DB\s*/?\s*\d+[-—.]?\d*)",
        r"(GB/?[A-Z]?/?\d+[-—.]?\d*)",
        r"(Q/?\d+[-—.]?\d*)",
    ]

    # ==================== 批号提取 ====================
    batch_number_rules = [
        r"(?:批号|批次|生产批号|批编码|LOT|Batch)\s*(?:No|NO|编号)?[：:]*\s*([A-Za-z0-9\-_/]{4,})",
        r"(?:LOT\s*NO|BATCH\s*NO|LOT\s+#)[.:：]*\s*([A-Za-z0-9\-_/]{4,})",
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

    # ==================== 自由格式文本提取（无标签场景） ====================
    # 分场景策略：
    #   场景A：规则引擎几乎没提取到字段 → 完全信任自由格式结果
    #   场景B：规则引擎提取到部分字段但有偏差 → 合并自由格式结果（品牌/配料/规格等优先用自由格式）
    filled_fields = [k for k, v in result.items() if v and v != "N/A"]
    missing_key_fields = [k for k in ["brand", "ingredients", "specification", "production_date"]
                         if k not in result or not result.get(k) or result.get(k) == "N/A"]

    # 场景A：几乎全空 → 完全信任自由格式
    if len(filled_fields) <= 1:
        free_form_result = _free_form_extract(ocr_text)
        for key, value in free_form_result.items():
            if value and value != "N/A":
                result[key] = value
    # 场景B：部分字段缺失或可疑 → 仅补充缺失字段
    elif missing_key_fields:
        free_form_result = _free_form_extract(ocr_text)
        for key in missing_key_fields:
            if key in free_form_result and free_form_result[key] and free_form_result[key] != "N/A":
                # 品牌覆盖：自由格式品牌不应是公司名片段
                if key == "brand":
                    old_brand = result.get("brand", "")
                    new_brand = free_form_result[key]
                    # 如果自由格式品牌比规则品牌短且不含公司后缀，优先用自由格式
                    if len(new_brand) <= len(old_brand) and not any(
                        suffix in new_brand for suffix in ["有限", "公司", "股份", "集团"]
                    ):
                        result[key] = new_brand
                else:
                    result[key] = free_form_result[key]

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

    # ==================== 置信度打分 ====================
    filled = {k: v for k, v in result.items() if v and v != "N/A"}
    total_key_fields = len([f for f in default_fields if f in result])
    filled_key = len([f for f in default_fields if f in filled])

    if total_key_fields > 0:
        key_field_ratio = filled_key / total_key_fields
    else:
        key_field_ratio = 0.0

    # 综合置信度 = 关键字段填充率
    if key_field_ratio >= 0.8:
        confidence = "high"
    elif key_field_ratio >= 0.4:
        confidence = "medium"
    elif filled_key > 0:
        confidence = "low"
    else:
        confidence = "failed"

    result["_confidence"] = confidence
    result["_filled_fields"] = filled_key
    result["_total_expected"] = total_key_fields

    # ==================== 交叉关联推理 ====================
    # 1. 品牌+成分→推断产品类型（补充product_name）
    if (result.get("product_name", "N/A") == "N/A"
            and result.get("brand", "N/A") != "N/A"
            and result.get("ingredients", "N/A") != "N/A"):
        ingredients_lower = result["ingredients"].lower()
        if any(kw in ingredients_lower for kw in ["面粉", "小麦", "大米", "玉米"]):
            result["_inferred_type"] = "食品"
        elif any(kw in ingredients_lower for kw in ["水", "甘油", "表面活性剂", "月桂醇"]):
            result["_inferred_type"] = "日化品"
        elif any(kw in ingredients_lower for kw in ["草药", "中药", "提取物"]):
            result["_inferred_type"] = "药品"

    # 2. 生产日期+保质期→自动计算到期日期（已实现）
    # 3. 营养表字段合理性验证
    if "nutrition_facts" in result:
        nf = result["nutrition_facts"]
        if isinstance(nf, dict):
            # 检查是否包含关键营养指标
            has_energy = any("能量" in str(k) or "energy" in str(k).lower() or "calor" in str(k).lower() for k in nf)
            has_protein = any("蛋白" in str(k) or "protein" in str(k).lower() for k in nf)
            if not has_energy or not has_protein:
                result["_nutrition_warning"] = "营养表可能不完整（缺少能量或蛋白质）"

    return result


def _free_form_extract(ocr_text: str) -> Dict[str, Any]:
    """
    自由格式文本提取 - 处理无标签的包装文字（如湿巾、日化品正面）
    通过文本模式匹配直接提取品牌、产品名、成分、特点等
    """
    result: Dict[str, Any] = {}

    if not ocr_text.strip():
        return result

    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]

    # ===== 产品类型检测 =====
    product_keywords = {
        "湿巾|湿纸巾|清洁巾|擦手巾|棉柔巾|洗脸巾": "湿巾",
        "纸巾|抽纸|面巾纸|餐巾纸|手帕纸": "纸巾",
        "洗发水|洗发露|洗发乳|洗发液|洗发膏|洗发精": "洗发水",
        "沐浴露|沐浴乳|沐浴液|沐浴|沐浴泡泡": "沐浴露",
        "洗面奶|洁面乳|洁面泡沫|洗面乳|洁面膏": "洗面奶",
        "护肤霜|护肤乳|面霜|乳液|润肤乳|身体乳": "护肤品",
        "洗衣液|洗衣粉|洗衣凝珠|洗衣皂": "洗衣液",
        "洗手液|洗手露|洗手泡沫": "洗手液",
        "牙膏|牙粉|漱口水|牙线": "牙膏",
        "消毒液|消毒水|除菌液|抗菌液": "消毒液",
        "矿泉水|纯净水|饮用水|苏打水": "饮用水",
        "饮料|果汁|茶饮|奶茶|乳酸菌": "饮料",
        "饼干|糕点|面包|蛋糕|点心|酥": "饼干糕点",
        "巧克力|糖果|口香糖|棒棒糖": "糖果",
        "牛奶|酸奶|纯奶|鲜奶|奶粉": "乳制品",
        "酱油|醋|料酒|蚝油|调味料|酱": "调味品",
        "食用油|花生油|菜籽油|橄榄油|调和油": "食用油",
        "大米|面粉|面条|米粉|挂面|杂粮": "粮食",
    }

    product_type = ""
    full_text_all = "".join(lines)
    for pattern, ptype in product_keywords.items():
        if re.search(pattern, full_text_all):
            product_type = ptype
            break

    # ===== 品牌提取（自由格式） =====
    # 使用逐行搜索，避免跨行拼接导致的错误匹配
    brands = []

    # 先逐行搜索：英中文混合品牌 "Dafi答菲", "NIVEA妮维雅"
    for line in lines:
        # 允许中间有空格或没有空格
        m = re.search(r'([A-Z][a-zA-Z0-9]{1,15}[\s]?[\u4e00-\u9fa5]{2,6})', line)
        if m:
            candidate = m.group(1).strip()
            # 过滤掉明显不是品牌的匹配（如"MINUS无酒精"这种跨词组合）
            # 检查前半部分是否像品牌（有大小写字母）
            eng_part = re.match(r'^[A-Z][a-zA-Z0-9]+', candidate)
            chn_part = re.search(r'[\u4e00-\u9fa5]{2,}$', candidate)
            if eng_part and chn_part:
                brands.append(candidate)

        # 也搜中文在前英文在后
        m = re.search(r'([\u4e00-\u9fa5]{2,6}[A-Z][a-zA-Z0-9]{0,10})', line)
        if m:
            candidate = m.group(1).strip()
            brands.append(candidate)

    # 仍没找到：尝试纯中文2-6字品牌
    if not brands:
        for line in lines:
            # 去掉标点和英文数字
            pure_cn = re.sub(r'[A-Za-z0-9\s+\-—()（）「」【】\[\]《》<>：:，。,.。!！?？""\'\"''/\\\\]', '', line).strip()
            if 2 <= len(pure_cn) <= 6 and pure_cn:
                # 排除常见的产品名/成分名/噪音
                if not any(kw in pure_cn for kw in '的、了、是、在、有、和、就、不、人、都、而、及、与、着、或、一个、没有、我们、可以、这个、那个、什么、怎么'):
                    # 排除明显是产品描述的关键词
                    if not any(kw in pure_cn for kw in ['雪饼', '饼干', '面包', '蛋糕', '薯片', '饮料', '食品', '用品',
                                                         '类型', '规格', '含量', '成分', '配料', '标准']):
                        brands.append(pure_cn)
                        break

    # 仍没找到：尝试行首单个英文品牌（如 "Dafi", "NIVEA"）
    if not brands:
        for line in lines[:3]:
            m = re.match(r'^([A-Z][a-z]{2,15})$', line.strip())
            if m:
                brands.append(m.group(1))
                break

    if brands:
        result["brand"] = brands[0].strip()
    else:
        # Try single English word brand (like "Dafi", "NIVEA" etc.)
        eng_brand_match = re.search(r'^([A-Z][a-z]{2,10})$', lines[0] if lines else "", re.MULTILINE)
        if eng_brand_match:
            result["brand"] = eng_brand_match.group(1)

    # ===== 产品名称提取 =====
    if product_type:
        if result.get("brand"):
            result["product_name"] = f"{result['brand']}{product_type}"
        else:
            result["product_name"] = product_type

    # Try to find product name in first few lines
    for line in lines[:4]:
        # Check if line contains product keywords
        for pattern, ptype in product_keywords.items():
            m = re.search(pattern, line)
            if m:
                match_text = m.group(0)
                # Try to get full product name (text before the keyword)
                prefix = line[:line.index(match_text)].strip()
                if prefix:
                    result["product_name"] = f"{prefix}{match_text}"
                break
        if "product_name" in result:
            break

    # ===== 成分/配料提取（自由格式） =====
    ingredient_keywords = [
        r'(?:水|纯水|纯净水|去离子水|EDI纯水|超纯水)',
        r'(?:甘油|丙二醇|透明质酸|玻尿酸|烟酰胺|维E|维生素E|维生素C|VC|神经酰胺|角鲨烷|氨基酸|水杨酸|果酸|乳酸|尿素|尿囊素|泛醇|苯氧乙醇|尼泊金酯|甲基异噻唑啉酮|MIT|CMIT|DMDM乙内酰脲|碘丙炔醇丁基氨甲酸酯|苯甲酸钠|山梨酸钾|脱氢乙酸钠|双氧水|过氧化氢|次氯酸|次氯酸钠|乙醇|异丙醇|芦荟|洋甘菊|茶树|薰衣草|金盏花)',
        r'(?:Purified[\s]+water[\s]*|purified[\s]+water[\s]*|aqua[\s]*|Glycerin[\s]*|Glycerol[\s]*|Propylene[\s]+Glycol[\s]*|Butylene[\s]+Glycol[\s]*|Hyaluronic[\s]+Acid[\s]*|Sodium[\s]+Hyaluronate[\s]*|Niacinamide[\s]*|Tocopherol[\s]*|Vitamin[\s]+[CEB][\s]*|Ceramide[\s]*|Squalane[\s]*|Amino[\s]+Acid[\s]*|Salicylic[\s]+Acid[\s]*|Lactic[\s]+Acid[\s]*|Aloe[\s]+Vera[\s]*|Chamomile[\s]*|Green[\s]+Tea[\s]*|Panthenol[\s]*|Phenoxyethanol[\s]*|Ethylhexylglycerin[\s]*|Caprylyl[\s]+Glycol[\s]*)',
    ]

    ingredients_found = []
    for line in lines:
        # 跳过"无XX"模式——表示不含该成分
        if re.match(r'^无', line.strip()):
            continue
        for pattern in ingredient_keywords:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for m in matches:
                cleaned = m.strip()
                if cleaned and cleaned not in ingredients_found:
                    ingredients_found.append(cleaned)

    if ingredients_found:
        result["ingredients"] = "，".join(ingredients_found[:8])

    # ===== 产品特点/卖点提取 =====
    # 提取所有看起来像卖点的短句
    feature_lines = []

    # 排除的噪音模式（OCR伪影）
    noise_patterns = [
        r'^[\+\-]+\s*[A-Za-z]+\s*$', r'^[A-Za-z]{1,3}\s*[\+\-]+\s*[A-Za-z]*\s*$',
        r'^r[\-\s]*j', r'^[\+\-][Jj]', r'^I\+', r'^[A-Z][a-z]?[\-\s][Jj]',
        r'^\d+$', r'^[\+\-]+$', r'^[A-Za-z\s]+$',
    ]

    for line in lines:
        # 跳过明显的噪点行
        is_noise = False
        for np in noise_patterns:
            if re.match(np, line.strip(), re.IGNORECASE):
                is_noise = True
                break
        if is_noise:
            continue

        cleaned = line.strip()
        # 清理行首的中继噪点
        cleaned = re.sub(r'^[\+\-]+\s*[A-Za-z]*\s*', '', cleaned).strip()
        cleaned = re.sub(r'^[A-Za-z]{1,3}\s*[\+\-]+\s*', '', cleaned).strip()
        cleaned = re.sub(r'^r[\-\s]*j[\s\-]*[A-Za-z]*', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'^[\+\-][Jj][A-Za-z]*', '', cleaned).strip()

        if cleaned and len(cleaned) >= 4 and cleaned not in feature_lines:
            # Not just numbers or English-only
            has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', cleaned))
            if has_chinese:
                feature_lines.append(cleaned)

    if feature_lines:
        # 去重+排除已提取为品牌/产品名的内容
        unique_features = []
        brand_name = result.get("brand", "")
        product_name = result.get("product_name", "")
        exclude_texts = [brand_name, product_name] if brand_name or product_name else []
        for f in feature_lines:
            # 排除品牌和产品名
            if any(exclude in f or f in exclude for exclude in exclude_texts):
                continue
            if not any(f in uf or uf in f for uf in unique_features):
                unique_features.append(f)
        result["features"] = "，".join(unique_features[:8])

    # ===== 规格提取（自由格式） =====
    # 优先找有明确标签的规格行
    spec_with_label = re.search(r'(?:净含量|规格|容量|净重)[：:]*\s*(\d+\.?\d*\s*(?:L|l|ml|mL|g|G|kg|KG|片|抽|层|包|袋|盒|瓶|罐|支|条|枚|张))', full_text_all)
    if spec_with_label:
        result["specification"] = spec_with_label.group(1).strip()
    else:
        # 无标签时找完整数字+单位（排除"每"前缀如"每100克"）
        spec_match = re.search(r'(?<![每])(\d+\.?\d*\s*(?:L|l|ml|mL|g|G|kg|KG|片|抽|层|包|袋|盒|瓶|罐|支|条|枚|张))', full_text_all)
        if spec_match:
            result["specification"] = spec_match.group(1).strip()

    # 检测时标注产品类型
    if product_type:
        result["product_type"] = product_type

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


def _layout_aware_extract(regions: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    布局感知的键值对提取
    利用OCR区域的bbox坐标，按水平对齐关系识别标签名→值的配对
    """
    kv_pairs: Dict[str, str] = {}

    if not regions:
        return kv_pairs

    # 按Y轴中心排序（从上到下）
    sorted_regions = sorted(regions, key=lambda r: (r.get("bbox", [0, 0, 0, 0])[1] + r.get("bbox", [0, 0, 0, 0])[3]) / 2)

    # 同行文本合并：Y坐标相差<20px的视为同一行
    lines: List[List[Dict]] = []
    current_line: List[Dict] = []
    last_y = -100

    for region in sorted_regions:
        bbox = region.get("bbox", [0, 0, 0, 0])
        center_y = (bbox[1] + bbox[3]) / 2
        if abs(center_y - last_y) > 20 and current_line:
            lines.append(current_line)
            current_line = [region]
        else:
            current_line.append(region)
        last_y = center_y

    if current_line:
        lines.append(current_line)

    # 按X轴排序每行内的元素
    for line in lines:
        sorted_line = sorted(line, key=lambda r: r.get("bbox", [0, 0, 0, 0])[0])

        # 尝试构建标签→值对
        texts = [r.get("text", "").strip() for r in sorted_line if r.get("text")]
        full_line = " ".join(texts)

        # 查找标签:值模式（同一行内）
        kv_match = re.match(r'^([^:：]+)[:：]\s*(.+)$', full_line)
        if kv_match:
            label = kv_match.group(1).strip().lower()
            value = kv_match.group(2).strip()

            # 将常见标签映射到字段名
            label_map = {
                "brand": "brand", "品牌": "brand", "商标": "brand",
                "product": "product_name", "product name": "product_name",
                "品名": "product_name", "产品名称": "product_name", "名称": "product_name",
                "规格": "specification", "净含量": "specification", "net content": "specification",
                "specification": "specification", "net weight": "specification",
                "生产日期": "production_date", "production date": "production_date",
                "date": "production_date", "生产": "production_date",
                "保质期": "shelf_life", "shelf life": "shelf_life", "保质": "shelf_life",
                "manufacturer": "manufacturer", "制造商": "manufacturer", "生产商": "manufacturer",
                "配料": "ingredients", "ingredients": "ingredients", "成分": "ingredients",
                "执行标准": "standard", "standard": "standard", "标准号": "standard",
                "批号": "batch_number", "batch": "batch_number", "lot": "batch_number",
                "许可证": "license_number", "license": "license_number",
                "贮存条件": "storage_condition", "storage": "storage_condition",
                "storage condition": "storage_condition", "存放条件": "storage_condition",
            }

            for key, field in label_map.items():
                if label.startswith(key) or key in label:
                    if field not in kv_pairs:
                        kv_pairs[field] = value
                    break

    return kv_pairs


def _llm_post_correct(
    structured_data: Dict[str, Any],
    ocr_text: str,
    ctx: Context,
    llm_config: Dict
) -> Dict[str, Any]:
    """
    LLM后验证和纠错：
    1. 识别明显的OCR识别错误
    2. 标准化日期格式
    3. 纠正品牌名称
    """
    try:
        # 使用LLM验证关键字段
        correction_prompt = f"""请验证并纠正以下包装标签的结构化提取结果。如果发现明显错误请修正。

OCR原始文本：
{ocr_text}

当前提取结果：
{json.dumps(structured_data, ensure_ascii=False, indent=2)}

修正规则：
1. 日期格式统一为YYYY-MM-DD
2. 生产商名称纠正常见OCR识别错误
3. 保质期格式统一为"X个月"或"X天"
4. 保留已经正确的值，不要随意替换为N/A
5. 对于明显错误的OCR识别（如Storege→Storage）自动纠正
6. 输出JSON格式，保持所有现有字段

请只返回JSON："""
        client = LLMClient(ctx=ctx)
        messages = [
            SystemMessage(content="你是一个专业的OCR结果校验助手。仅返回JSON，不要包含其他内容。"),
            HumanMessage(content=correction_prompt)
        ]
        response = client.invoke(
            messages=messages,
            model=llm_config.get("model", "doubao-seed-2-0-pro-260215"),
            temperature=0.1,
            max_tokens=2000,
        )

        if response and hasattr(response, 'content'):
            content = response.content if isinstance(response.content, str) else str(response.content)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                corrected = json.loads(json_match.group())
                # 只保留原始字段集
                for key in structured_data:
                    if key in corrected and corrected[key]:
                        structured_data[key] = corrected[key]
    except Exception as e:
        logger.warning(f"LLM后验证失败，保留原始结果: {str(e)}")

    return structured_data


def model_extract_node(
    state: ModelExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ModelExtractOutput:
    """
    title: 结构化信息提取
    desc: 使用大语言模型从OCR识别文本中提取结构化信息（如品牌、规格、生产日期等），
          LLM不可用时自动降级到增强版规则引擎。
          支持布局感知提取（利用OCR区域坐标）和LLM后验证纠错。
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 预先获取ocr_text（避免在except块中使用未定义变量）
    ocr_text = state.ocr_text or state.raw_text or state.ocr_raw_result or ""
    regions = state.regions or []

    # Step 1: 布局感知提取（快速KV配对）
    layout_data = _layout_aware_extract(regions)

    # Step 2: 规则引擎提取（与布局数据互补）
    rule_data = rule_based_extract(ocr_text, state.template_fields or [])

    # Step 3: 融合布局感知结果和规则结果（布局优先）
    for key, value in layout_data.items():
        rule_data[key] = value  # 布局结果覆盖规则结果（更准确）

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
            logger.warning(f"JSON解析失败，使用规则引擎结果: {str(e)}")
            # 使用融合后的规则引擎结果
            structured_data = rule_data
            confidence = 0.7

        # Step 4: LLM后验证（纠正明显OCR错误）
        if confidence < 0.9 and structured_data:
            structured_data = _llm_post_correct(structured_data, ocr_text, ctx, llm_config)

        # V5.4 商业化统一结构解析：提取嵌套字段，扁平化处理
        # 1) 提取 product_type（顶层）
        product_type = structured_data.pop("product_type", "") if isinstance(structured_data, dict) else ""

        # 2) 提取 category_info（保持原始嵌套）
        category_info = structured_data.pop("category_info", {}) if isinstance(structured_data, dict) else {}
        if not isinstance(category_info, dict):
            category_info = {}

        # 3) 提取 warnings / ext_info（新结构字段）
        warnings = structured_data.pop("warnings", []) if isinstance(structured_data, dict) else []
        if not isinstance(warnings, list):
            warnings = []
        ext_info = structured_data.pop("ext_info", []) if isinstance(structured_data, dict) else []
        if not isinstance(ext_info, list):
            ext_info = []

        # 4) 兼容旧规则引擎数据：如果structured_data里没有category_info且都是顶层字段，
        #    视为旧版数据，直接当顶层使用
        #    如果有category_info，将其字段扁平化到structured_data顶层（供下游融合使用）
        if category_info:
            # 检查是否有字段冲突（同一字段同时在顶层和category_info）
            for k, v in category_info.items():
                if k not in structured_data or structured_data.get(k) is None:
                    structured_data[k] = v
                # 顶层已有值则保留顶层值

        # 计算统一字段覆盖率（V5.4：基于9个标准字段）
        standard_fields = ["brand", "product_name", "specification", "manufacturer",
                           "production_date", "shelf_life", "batch_number"]
        filled_standard = sum(1 for f in standard_fields if structured_data.get(f))
        if product_type and product_type != "其他":
            filled_standard += 1
        if warnings:
            filled_standard += 1
        if category_info:
            filled_standard += min(len(category_info), 2)  # 品类字段最多算2个
        confidence = min(0.99, 0.5 + filled_standard * 0.05)

        logger.info(
            f"结构化提取完成，置信度: {confidence:.2f}, 产品类型: {product_type or '未知'}, "
            f"category_info字段: {list(category_info.keys())}, warnings: {len(warnings)}, ext_info: {len(ext_info)}"
        )

        return ModelExtractOutput(
            structured_data=structured_data,
            category_info=category_info,
            warnings=warnings,
            ext_info=ext_info,
            confidence=confidence,
            missing_fields=missing_fields,
            product_type=product_type
        )

    except Exception as e:
        logger.warning(f"模型结构化提取失败: {str(e)}，降级到规则引擎")
        # 降级到规则引擎
        filled_count = sum(1 for v in rule_data.values() if v != "N/A")
        rule_confidence = filled_count / len(rule_data) if rule_data else 0.3

        logger.info(f"规则引擎提取完成: {json.dumps(rule_data, ensure_ascii=False)}, 置信度: {rule_confidence:.2f}")
        # 提取产品类型
        product_type = rule_data.pop("product_type", "") if isinstance(rule_data, dict) else ""
        return ModelExtractOutput(
            structured_data=rule_data,
            category_info={},
            warnings=[],
            ext_info=[],
            confidence=rule_confidence,
            missing_fields=[f for f in rule_data if rule_data[f] == "N/A"],
            product_type=product_type
        )
