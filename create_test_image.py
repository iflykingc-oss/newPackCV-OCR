#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速测试完整流程"""

import sys
import os
sys.path.insert(0, '/workspace/projects')
sys.path.insert(0, '/workspace/projects/src')

from PIL import Image, ImageDraw, ImageFont
from coze_coding_dev_sdk.s3 import S3SyncStorage
import base64

# 创建测试图片
img = Image.new('RGB', (800, 200), color='white')
draw = ImageDraw.Draw(img)
text = "Product Name: 金龙鱼菜籽油\nBatch: 20240429\nExpire: 20250429\nPrice: 59.9元"

# 使用系统字体
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
except:
    font = ImageFont.load_default()

# 绘制多行文本
y_offset = 30
for line in text.split('\n'):
    draw.text((50, y_offset), line, fill='black', font=font)
    y_offset += 40

# 保存并上传到S3
import tempfile
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    img.save(f.name)
    temp_path = f.name

# 读取图片并上传
with open(temp_path, 'rb') as f:
    img_bytes = f.read()

storage = S3SyncStorage(
    endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
    access_key="",
    secret_key="",
    bucket_name=os.getenv("COZE_BUCKET_NAME"),
    region="cn-beijing",
)
key = storage.upload_file(
    file_content=img_bytes,
    file_name="test_ocr_image.png",
    content_type='image/png'
)
url = storage.generate_presigned_url(key=key, expire_time=86400)

print(f"测试图片已上传: {url}")

# 清理临时文件
os.unlink(temp_path)
