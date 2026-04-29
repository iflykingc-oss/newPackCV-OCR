#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""创建测试图片并测试OCR"""

from PIL import Image, ImageDraw, ImageFont
import pytesseract

# 创建图片
img = Image.new('RGB', (800, 200), color='white')
draw = ImageDraw.Draw(img)

# 绘制文本
text = "Hello World 你好世界 123456"
try:
    # 尝试使用系统字体
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
except:
    font = ImageFont.load_default()

draw.text((100, 80), text, fill='black', font=font)

# 保存图片
img.save('/tmp/test_ocr.png')
print(f"测试图片已创建: /tmp/test_ocr.png")

# 测试Tesseract
print("\n测试Tesseract OCR...")
try:
    text_result = pytesseract.image_to_string(img, lang='chi_sim+eng')
    print(f"识别结果: {text_result}")

    # 获取详细信息
    data = pytesseract.image_to_data(img, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
    print(f"\n识别到{len([t for t in data['text'] if t.strip()])}个文本块")
    for i, (t, c) in enumerate(zip(data['text'], data['conf'])):
        if t.strip():
            print(f"  {i+1}. '{t}' (置信度: {c})")

except Exception as e:
    print(f"错误: {e}")
