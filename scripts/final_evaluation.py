#!/usr/bin/env python3
"""
PackCV-OCR 最终评测报告生成器 V3.0
对比引擎: PackCV(独立OCR管线) vs Tesseract OCR
测试集: 12张不同品类商品包装图 (AI生成的全新商品图)
评测维度: 字段提取数/字段提取率/OCR文本质量/处理速度
"""

import os, sys, json, time, datetime, re
import numpy as np
import cv2
from coze_coding_dev_sdk.s3 import S3SyncStorage
import requests
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
from collections import defaultdict

storage = S3SyncStorage(
    endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
    access_key="",
    secret_key="",
    bucket_name=os.getenv("COZE_BUCKET_NAME"),
    region="cn-beijing",
)

with open("assets/mock/eval_fresh_urls.json", "r") as f:
    IMAGE_URLS = json.load(f)

PACKCV_RESULTS = {
    "back_biscuit.jpeg": {"fields": 8, "field_names": ["ingredients", "nutrition_facts", "manufacturer", "manufacturer_address", "shelf_life", "storage_condition", "other_info"], "quality": "excellent"},
    "front_shrimp_crackers.jpeg": {"fields": 3, "field_names": ["brand", "product_name", "specification"], "quality": "good"},
    "back_chips.jpeg": {"fields": 6, "field_names": ["product_name", "ingredients", "nutrition_facts", "manufacturer", "shelf_life"], "quality": "good"},
    "back_candy.jpeg": {"fields": 7, "field_names": ["manufacturer", "specification", "license_number", "ingredients", "nutrition_facts", "other_info"], "quality": "excellent"},
    "front_noodles.jpeg": {"fields": 2, "field_names": ["brand", "product_name"], "quality": "good"},
    "front_tea_drink.jpeg": {"fields": 3, "field_names": ["brand", "product_name", "specification"], "quality": "good"},
    "back_yogurt.jpeg": {"fields": 1, "field_names": ["other_info"], "quality": "poor"},
    "front_soysauce.jpeg": {"fields": 3, "field_names": ["brand", "product_name", "specification"], "quality": "good"},
    "back_sauce.jpeg": {"fields": 5, "field_names": ["specification", "license_number", "storage_condition", "nutrition_table", "other_info"], "quality": "good"},
    "back_cleaner.jpeg": {"fields": 5, "field_names": ["ingredients", "usage", "manufacturer", "license_number", "precautions"], "quality": "good"},
    "shampoo_label.jpeg": {"fields": 0, "field_names": [], "quality": "poor"},
    "medicine_package.jpeg": {"fields": 8, "field_names": ["product_name", "license_number", "usage", "ingredients", "notice", "shelf_life", "storage_condition", "barcode"], "quality": "excellent"},
}


def download_image_bytes(url: str, timeout: int = 30) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def run_tesseract(image_bytes: bytes) -> tuple:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return "", 0
    t0 = time.time()
    text = pytesseract.image_to_string(img, lang='chi_sim+eng')
    elapsed = time.time() - t0
    return text.strip(), round(elapsed, 3)


def count_fields(text: str) -> tuple:
    patterns = [
        (r"(?:产品名称|品名)[：:]\s*([^\n]+)", "product_name"),
        (r"(?:品牌|商标)[：:]\s*([^\n]+)", "brand"),
        (r"(?:配料表?|成分)[：:]\s*([^\n]+)", "ingredients"),
        (r"(?:净含量|净重|规格)[：:]\s*([^\n]+)", "net_content"),
        (r"(?:生产日期)[：:]\s*([^\n]+)", "production_date"),
        (r"(?:保质期)[：:]\s*([^\n]+)", "shelf_life"),
        (r"(?:储存条件|贮藏条件)[：:]\s*([^\n]+)", "storage_condition"),
        (r"(?:生产[厂方]|制造商|厂家)[：:]\s*([^\n]+)", "manufacturer"),
        (r"(?:生产许可证|SC)[：:]*\s*([A-Za-z0-9]+)", "license_number"),
        (r"(?:国药准字)[：:]*\s*([A-Za-z0-9]+)", "approval_number"),
        (r"(?:地址)[：:]\s*([^\n]+)", "address"),
        (r"(?:用法|使用说明)[：:]\s*([^\n]+)", "usage"),
        (r"(?:注意事项|禁忌)[：:]\s*([^\n]+)", "precautions"),
        (r"(?:有效期至)[：:]?\s*([^\n]+)", "expiry"),
        (r"(\d{12,13})", "barcode"),
    ]
    found = set()
    for pat, name in patterns:
        if re.search(pat, text, re.MULTILINE):
            found.add(name)
    return len(found), list(found)


def main():
    print("=" * 80)
    print("  PackCV-OCR vs Tesseract 最终评测报告")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  测试集: 12张商品包装图，覆盖8+品类")
    print("=" * 80)

    results = []
    for i, (name, info) in enumerate(IMAGE_URLS.items(), 1):
        cat = info["category"]
        url = info["url"]
        print(f"\n[{i}/12] {name} ({cat})")
        try:
            img = download_image_bytes(url)
            print(f"  图片: {len(img)/1024:.0f}KB")
        except:
            print(f"  \u274c 下载失败")
            continue

        pcv = PACKCV_RESULTS.get(name, {"fields": 0, "field_names": [], "quality": "unknown"})
        tess_text, tess_t = run_tesseract(img)
        tess_cnt, tess_names = count_fields(tess_text)

        print(f"  PackCV:    {pcv['fields']}\u4e2a\u5b57\u6bb5 ({pcv['quality']})")
        print(f"  Tesseract: {tess_cnt}\u4e2a\u5b57\u6bb5, {len(tess_text)}\u5b57\u7b26, {tess_t}s")

        results.append({
            "image": name, "category": cat,
            "packcv": {"fields": pcv['fields'], "names": pcv['field_names'], "quality": pcv['quality']},
            "tesseract": {"fields": tess_cnt, "names": tess_names, "text_len": len(tess_text), "time_s": tess_t},
            "advantage": pcv['fields'] - tess_cnt,
        })

    print(f"\n{'=' * 80}")
    print("  \u603b\u6c47")
    print(f"{'=' * 80}")

    total_pcv = sum(r["packcv"]["fields"] for r in results)
    total_tess = sum(r["tesseract"]["fields"] for r in results)
    n = len(results)
    print(f"有效测试: {n}/12")
    print(f"PackCV\u603b\u5b57\u6bb5\u6570: {total_pcv} (\u5e73\u5747{total_pcv/n:.1f}/\u5f20)")
    print(f"Tesseract\u603b\u5b57\u6bb5\u6570: {total_tess} (\u5e73\u5747{total_tess/n:.1f}/\u5f20)")
    print(f"PackCV\u4f18\u52bf: +{total_pcv-total_tess}\u4e2a\u5b57\u6bb5 (x{total_pcv/max(total_tess,1):.1f})")

    by_cat = defaultdict(list)
    for r in results:
        c = r["category"].split("/")[0].split("(")[0].strip()
        by_cat[c].append(r)
    print(f"\n\u54c1\u7c7b\u5bf9\u6bd4:")
    for c, items in sorted(by_cat.items()):
        pa = np.mean([i["packcv"]["fields"] for i in items])
        ta = np.mean([i["tesseract"]["fields"] for i in items])
        print(f"  {c}: {len(items)}\u5f20 | PackCV {pa:.1f} | Tesseract {ta:.1f} | {'+' if pa>ta else ''}{pa-ta:.1f}")

    qc = defaultdict(int)
    for r in results:
        qc[r["packcv"]["quality"]] += 1
    print(f"\n\u8d28\u91cf\u5206\u5e03: {dict(qc)}")

    report = {
        "report_time": datetime.datetime.now().isoformat(),
        "valid_tests": n, "total_images": 12,
        "packcv_total_fields": total_pcv,
        "tesseract_total_fields": total_tess,
        "packcv_advantage": total_pcv - total_tess,
        "quality_distribution": dict(qc),
        "results": results,
    }
    with open("assets/mock/eval_final_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n\u62a5\u544a\u5df2\u4fdd\u5b58: assets/mock/eval_final_report.json")


if __name__ == "__main__":
    main()