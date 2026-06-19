#!/usr/bin/env python3
"""上传所有评测图片到对象存储，生成访问URL"""
import os, json
from coze_coding_dev_sdk.s3 import S3SyncStorage

storage = S3SyncStorage(
    endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
    access_key="",
    secret_key="",
    bucket_name=os.getenv("COZE_BUCKET_NAME"),
    region="cn-beijing",
)

# 评测图片清单
images = {
    # 品类：零食
    "front_shrimp_crackers.jpeg": ("零食虾片/薯片", "assets/mock/front_shrimp_crackers.jpeg"),
    "back_biscuit.jpeg": ("饼干背面标签", "assets/mock/back_biscuit.jpeg"),
    "back_chips.jpeg": ("薯片背面标签", "assets/mock/back_chips.jpeg"),
    "back_candy.jpeg": ("糖果/巧克力包装", "assets/mock/back_candy.jpeg"),
    # 品类：方便食品
    "front_noodles.jpeg": ("方便面包装正面", "assets/mock/front_noodles.jpeg"),
    # 品类：饮料
    "front_tea_drink.jpeg": ("茶饮料瓶标", "assets/mock/front_tea_drink.jpeg"),
    "back_yogurt.jpeg": ("酸奶饮品背面标签", "assets/mock/back_yogurt.jpeg"),
    # 品类：调味品
    "front_soysauce.jpeg": ("酱油瓶标签正面", "assets/mock/front_soysauce.jpeg"),
    "back_sauce.jpeg": ("酱料瓶背面标签", "assets/mock/back_sauce.jpeg"),
    # 品类：日化清洁
    "back_cleaner.jpeg": ("清洁用品瓶标", "assets/mock/back_cleaner.jpeg"),
}

urls = {}
for img_name, (category, path) in images.items():
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        continue
    with open(path, "rb") as f:
        content = f.read()
    key = storage.upload_file(
        file_content=content,
        file_name=f"eval/{img_name}",
        content_type="image/jpeg",
    )
    url = storage.generate_presigned_url(key=key, expire_time=86400)
    urls[img_name] = {"category": category, "url": url, "key": key}
    print(f"  {img_name}: {category} -> uploaded")

# 保存URL映射
with open("assets/mock/eval_urls.json", "w") as f:
    json.dump(urls, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(urls)} image URLs to assets/mock/eval_urls.json")