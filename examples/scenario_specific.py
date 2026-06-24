"""
VibeCoding-OCR 场景化提取示例

演示不同业务场景的提取
"""
import requests

API_URL = "http://localhost:9000/api/ocr"

def ocr_extract(input_data):
    """统一的 OCR 调用入口"""
    response = requests.post(API_URL, json=input_data, timeout=120)
    response.raise_for_status()
    return response.json()


# ===== 场景 1: 产品包装 =====
print("=" * 50)
print("场景 1: 产品包装")
print("=" * 50)
result = ocr_extract({
    "package_image": {"url": "https://example.com/package.jpg", "file_type": "image"}
})
print(f"  检测场景: {result.get('detected_category')}")
print(f"  置信度: {result.get('scenario_confidence')}")
print(f"  品牌: {result['structured_data'].get('brand')}")
print()


# ===== 场景 2: 金融票据 =====
print("=" * 50)
print("场景 2: 金融票据")
print("=" * 50)
result = ocr_extract({
    "package_image": {"url": "https://example.com/receipt.jpg", "file_type": "image"}
})
print(f"  金额: {result['structured_data'].get('amount')}")
print(f"  开票方: {result['structured_data'].get('issuer')}")
print()


# ===== 场景 3: 合同 =====
print("=" * 50)
print("场景 3: 合同(PDF)")
print("=" * 50)
result = ocr_extract({
    "document_file": {"url": "https://example.com/contract.pdf", "file_type": "document"}
})
print(f"  合同号: {result['structured_data'].get('contract_number')}")
print(f"  金额: {result['structured_data'].get('amount')}")
