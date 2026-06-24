"""
VibeCoding-OCR 租户配置示例

演示 SaaS 多租户场景下的模型配置
"""
import requests

CONFIG_URL = "http://localhost:9000/api/config/tenant"

# ===== 1. 查询租户配置 =====
tenant_id = "company-a"
response = requests.get(f"{CONFIG_URL}/{tenant_id}")
current = response.json()
print(f"租户 {tenant_id} 当前配置:")
for k, v in current.items():
    print(f"  {k}: {v}")
print()

# ===== 2. 更新租户配置 =====
new_config = {
    "llm_config": {
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 2000
    },
    "ocr_engine": "paddleocr",
    "vl_engine": "minicpm-o"
}
response = requests.put(f"{CONFIG_URL}/{tenant_id}", json=new_config)
print("更新结果:", response.json())
print()

# ===== 3. 删除租户配置（恢复默认） =====
response = requests.delete(f"{CONFIG_URL}/{tenant_id}")
print("删除结果:", response.json())
