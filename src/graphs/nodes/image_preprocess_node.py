# -*- coding: utf-8 -*-
"""
图片预处理节点
针对瓶子等包装的图像增强、去噪、校正等预处理操作
"""

import os
import time
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ImagePreprocessInput, ImagePreprocessOutput
from utils.file.file import File, FileOps


def image_preprocess_node(
    state: ImagePreprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ImagePreprocessOutput:
    """
    title: 图片预处理
    desc: 对包装图片进行增强、去噪、校正等预处理，提升OCR识别准确率
    integrations: 图像处理
    """
    ctx = runtime.context
    
    # 获取图片路径
    image_path = state.package_image.url
    if not os.path.exists(image_path):
        # 如果是URL，先下载到临时目录
        import requests
        temp_path = f"/tmp/preprocess_{int(time.time())}.jpg"
        try:
            resp = requests.get(image_path, timeout=30)
            resp.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(resp.content)
            image_path = temp_path
        except Exception as e:
            # 下载失败，直接使用原路径
            image_path = state.package_image.url
    
    # 使用OpenCV进行图像预处理
    try:
        import cv2
        import numpy as np
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"无法读取图片: {image_path}")
        
        processing_info = {}
        is_rotated = False
        is_enhanced = False
        
        # 1. 图像增强（提升对比度、亮度）
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        is_enhanced = True
        processing_info["enhanced"] = True
        
        # 2. 去噪（针对瓶子包装的噪点）
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        processing_info["denoised"] = True
        
        # 3. 边缘增强（针对小字体喷码）
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        
        # 4. 灰度化
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        
        # 5. 自适应阈值处理（Sauvola算法思想，适合弱信号识别）
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # 6. 形态学处理（消除小噪点）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 7. 保存预处理后的图片
        output_path = f"/tmp/preprocessed_{int(time.time())}.jpg"
        cv2.imwrite(output_path, morph)
        
        # 创建File对象
        processed_file = File(url=output_path, file_type="image")
        
        processing_info["original_size"] = f"{img.shape[1]}x{img.shape[0]}"
        processing_info["output_size"] = f"{morph.shape[1]}x{morph.shape[0]}"
        processing_info["processing_steps"] = ["enhance", "denoise", "sharpen", "threshold", "morphology"]
        
        print(f"图片预处理完成: {processing_info}")
        
        return ImagePreprocessOutput(
            preprocessed_image=processed_file,
            is_rotated=is_rotated,
            is_enhanced=is_enhanced,
            processing_info=processing_info
        )
        
    except Exception as e:
        # 如果OpenCV不可用或处理失败，返回原图
        print(f"图像预处理失败，使用原图: {str(e)}")
        return ImagePreprocessOutput(
            preprocessed_image=state.package_image,
            is_rotated=False,
            is_enhanced=False,
            processing_info={"error": str(e), "fallback": "use_original"}
        )
