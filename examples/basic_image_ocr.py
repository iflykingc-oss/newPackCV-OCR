"""
VibeCoding-OCR 基础示例：图片 OCR 识别

最简单的调用方式 - 包装场景
"""
import requests

# API 地址
API_URL = "http://localhost:9000/api/ocr"

# 准备输入
payload = {
    "package_image": {
        "url": "https://example.com/package.jpg",
        "file_type": "image"
    }
}

# 调用 OCR
response = requests.post(API_URL, json=payload, timeout=60)
response.raise_for_status()

result = response.json()
print("品牌:", result["structured_data"]["brand"])
print("产品名:", result["structured_data"]["product_name"])
print("生产日期:", result["structured_data"]["production_date"])
print("保质期:", result["structured_data"]["shelf_life"])
