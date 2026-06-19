#!/usr/bin/env python3
"""
PackCV-OCR 多引擎对比评测脚本 V2.0
对比引擎：PackCV(我们的OCR管线) vs EasyOCR
覆盖12张不同品类的商品包装图
"""

import os, sys, json, time, datetime, re
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# ========== 导入EasyOCR ==========
import easyocr

# ========== S3存储 ==========
from coze_coding_dev_sdk.s3 import S3SyncStorage

storage = S3SyncStorage(
    endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
    access_key="",
    secret_key="",
    bucket_name=os.getenv("COZE_BUCKET_NAME"),
    region="cn-beijing",
)

# ========== 加载图片URL映射 ==========
with open("assets/mock/eval_urls.json", "r") as f:
    IMAGE_URLS = json.load(f)

# ========== EasyOCR初始化 ==========
print("[INFO] 初始化 EasyOCR (中英文模型)...")
easyocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
print("[INFO] EasyOCR 初始化完成")


def download_image(url: str) -> bytes:
    """从签名URL下载图片"""
    import requests
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def run_easyocr(image_bytes: bytes) -> Dict[str, Any]:
    """运行EasyOCR并提取文本"""
    import cv2
    import numpy as np

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"text": "", "confidence": 0.0, "error": "decode_failed"}

    t0 = time.time()
    results = easyocr_reader.readtext(img)
    elapsed = time.time() - t0

    texts = []
    confs = []
    for (bbox, text, conf) in results:
        texts.append(text)
        confs.append(conf)

    avg_conf = float(np.mean(confs)) if confs else 0.0
    full_text = "\n".join(texts)

    return {
        "text": full_text,
        "confidence": round(avg_conf, 4),
        "time_seconds": round(elapsed, 3),
        "num_texts": len(texts),
    }


def expected_fields(category: str) -> int:
    """根据品类预估应提取的字段数"""
    mapping = {
        "零食": 10,
        "饼干": 12,
        "薯片": 12,
        "糖果": 8,
        "方便面": 10,
        "茶饮料": 10,
        "酸奶": 12,
        "酱油": 10,
        "酱料": 10,
        "洗发水": 8,
        "清洁": 8,
        "药品": 10,
    }
    for k, v in mapping.items():
        if k in category:
            return v
    return 8


def extract_fields_easyocr(text: str, category: str) -> Dict[str, Any]:
    """从EasyOCR文本中尝试提取结构化字段"""
    fields = {}

    # 产品名称
    name_patterns = [
        r"(?:产品名称|品名|产品名)[：:]\s*([^\n]+)",
        r"^([^\n]{2,30}(?:面|饼|饮料|酱油|醋|酱|糖|药|膏|露|奶|茶|片))\s*$",
    ]
    for p in name_patterns:
        m = re.search(p, text, re.MULTILINE)
        if m:
            fields["product_name"] = m.group(1).strip()
            break

    # 品牌
    brand_match = re.search(r"(?:品牌|商标)[：:]\s*([^\n]+)", text)
    if brand_match:
        fields["brand"] = brand_match.group(1).strip()

    # 配料/成分
    ing_match = re.search(r"(?:配料表?|成分|成分表)[：:]\s*([^\n]+(?:\n[^\n]*)*?)(?=\n(?:保质|生产|储存|产品|净含|食用|用法|注意|贮藏|执行|生产厂))", text)
    if ing_match:
        fields["ingredients"] = ing_match.group(1).strip()[:200]

    # 净含量
    net_match = re.search(r"(?:净含量|净重|规格)[：:]\s*([^\n]+)", text)
    if net_match:
        fields["net_content"] = net_match.group(1).strip()

    # 生产日期
    prod_match = re.search(r"(?:生产日期)[：:]\s*([^\n]+)", text)
    if prod_match:
        fields["production_date"] = prod_match.group(1).strip()

    # 保质期
    shelf_match = re.search(r"(?:保质期)[：:]\s*([^\n]+)", text)
    if shelf_match:
        fields["shelf_life"] = shelf_match.group(1).strip()

    # 生产许可证
    sc_match = re.search(r"(?:生产许可证|SC)[：:]*\s*([A-Za-z0-9]+)", text)
    if sc_match:
        fields["production_license"] = sc_match.group(1).strip()

    # 储存条件
    store_match = re.search(r"(?:储存条件|贮藏条件|贮存条件)[：:]\s*([^\n]+)", text)
    if store_match:
        fields["storage_condition"] = store_match.group(1).strip()

    # 营养成分 - 检测是否有营养表关键词
    if re.search(r"(?:营养|能量|蛋白质|脂肪|碳水)", text):
        fields["has_nutrition_info"] = True

    # 生产厂家
    mfr_match = re.search(r"(?:生产[厂方]|制造商|厂家)[：:]\s*([^\n]+)", text)
    if mfr_match:
        fields["manufacturer"] = mfr_match.group(1).strip()

    # 地址
    addr_match = re.search(r"(?:地址)[：:]\s*([^\n]+)", text)
    if addr_match:
        fields["address"] = addr_match.group(1).strip()

    # 条形码
    barcode_match = re.search(r"(\d{6,13})", text)
    if barcode_match:
        fields["barcode"] = barcode_match.group(1)

    # 批准文号 (药品)
    approve_match = re.search(r"(?:国药准字|批准文号|备案号)[：:]*\s*([A-Za-z0-9]+)", text)
    if approve_match:
        fields["approval_number"] = approve_match.group(1).strip()

    return fields


def evaluate_all():
    """运行完整评测"""
    results = []

    print(f"\n{'='*80}")
    print(f"PackCV-OCR vs EasyOCR 对比评测")
    print(f"共 {len(IMAGE_URLS)} 张图片")
    print(f"{'='*80}\n")

    for i, (img_name, info) in enumerate(IMAGE_URLS.items(), 1):
        category = info["category"]
        url = info["url"]

        print(f"[{i}/{len(IMAGE_URLS)}] {img_name}")
        print(f"  品类: {category}")

        try:
            # 下载图片
            img_bytes = download_image(url)
            print(f"  图片大小: {len(img_bytes)/1024:.0f}KB")

            # 运行EasyOCR
            easyocr_result = run_easyocr(img_bytes)
            print(f"  EasyOCR: 文本段数={easyocr_result['num_texts']}, "
                  f"平均置信度={easyocr_result['confidence']:.2%}, "
                  f"耗时={easyocr_result['time_seconds']:.2f}s")

            # EasyOCR字段提取
            easyocr_fields = extract_fields_easyocr(easyocr_result["text"], category)
            exp_fields = expected_fields(category)
            easyocr_field_count = len(easyocr_fields)
            easyocr_field_rate = round(easyocr_field_count / exp_fields * 100, 1) if exp_fields > 0 else 0

            print(f"  EasyOCR 字段提取: {easyocr_field_count}/{exp_fields} "
                  f"({easyocr_field_rate}%)")
            if easyocr_fields:
                print(f"    提取到: {', '.join(list(easyocr_fields.keys())[:8])}")

        except Exception as e:
            print(f"  ERROR: {e}")
            easyocr_result = {"text": "", "confidence": 0, "time_seconds": 0, "num_texts": 0}
            easyocr_fields = {}
            easyocr_field_count = 0
            easyocr_field_rate = 0

        results.append({
            "image_name": img_name,
            "category": category,
            "url": url,
            "easyocr": {
                "text_length": len(easyocr_result.get("text", "")),
                "confidence": easyocr_result.get("confidence", 0),
                "time_seconds": easyocr_result.get("time_seconds", 0),
                "num_texts": easyocr_result.get("num_texts", 0),
                "fields_extracted": easyocr_field_count,
                "field_rate": easyocr_field_rate,
                "extracted_fields": {k: str(v)[:50] for k, v in easyocr_fields.items()},
            },
            "packcv": None,  # 将由test_run填充
        })

        print()

    # 保存结果
    report = {
        "report_time": datetime.datetime.now().isoformat(),
        "total_images": len(results),
        "easyocr_only": True,
        "packcv_status": "pending_test_run",
        "results": results,
    }

    with open("assets/mock/eval_comparison_v2.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print(f"\n{'='*80}")
    print(f"EasyOCR 评测汇总")
    print(f"{'='*80}")
    total_fields = sum(r["easyocr"]["fields_extracted"] for r in results)
    total_rate = sum(r["easyocr"]["field_rate"] for r in results)
    avg_rate = round(total_rate / len(results), 1) if results else 0
    avg_conf = np.mean([r["easyocr"]["confidence"] for r in results])
    avg_time = np.mean([r["easyocr"]["time_seconds"] for r in results])

    print(f"平均字段提取率: {avg_rate}%")
    print(f"平均文本置信度: {avg_conf:.2%}")
    print(f"平均处理耗时: {avg_time:.2f}s/张")

    print(f"\n各品类详情:")
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        cat_key = r["category"].split("/")[0]
        by_cat[cat_key].append(r)
    for cat, items in sorted(by_cat.items()):
        avg_cat_rate = np.mean([i["easyocr"]["field_rate"] for i in items])
        avg_cat_conf = np.mean([i["easyocr"]["confidence"] for i in items])
        print(f"  {cat}: {len(items)}张, 字段率={avg_cat_rate:.1f}%, 置信度={avg_cat_conf:.2%}")

    print(f"\n结果已保存至: assets/mock/eval_comparison_v2.json")


if __name__ == "__main__":
    evaluate_all()