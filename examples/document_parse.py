"""
VibeCoding-OCR 文档解析示例

演示 PDF 解析 (走 MinerU 引擎)
"""
import requests

API_URL = "http://localhost:9000/api/ocr"

# 准备输入 - PDF 文档
payload = {
    "document_file": {
        "url": "https://example.com/contract.pdf",
        "file_type": "document"
    }
}

response = requests.post(API_URL, json=payload, timeout=120)
response.raise_for_status()

result = response.json()
print("合同编号:", result["structured_data"].get("contract_number"))
print("甲方:", result["structured_data"].get("party_a"))
print("乙方:", result["structured_data"].get("party_b"))
print("金额:", result["structured_data"].get("amount"))
print("表格行数:", len(result.get("tables", [])))
