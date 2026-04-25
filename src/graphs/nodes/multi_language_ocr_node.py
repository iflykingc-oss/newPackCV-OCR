# -*- coding: utf-8 -*-
"""
多语言OCR识别节点
基于PaddleOCR 3.1.0，支持80+种语言识别
应用场景：进口商品识别、多语言包装说明、跨境电商文档
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    MultiLanguageOCRInput,
    MultiLanguageOCROutput
)


def multi_language_ocr_node(state: MultiLanguageOCRInput, config: RunnableConfig, runtime: Runtime[Context]) -> MultiLanguageOCROutput:
    """
    title: 多语言OCR识别
    desc: 支持80+种语言OCR识别，自动语言检测，适用于进口商品、跨境电商场景
    integrations: PaddleOCR 3.1.0
    """
    ctx = runtime.context
    
    print(f"[多语言OCR] 开始处理图片...")
    
    try:
        # 导入依赖
        import cv2
        import numpy as np
        import requests
        
        # 下载图片
        print(f"[多语言OCR] 下载图片: {state.image.url}")
        img_data = download_image(state.image.url)
        if img_data is None:
            raise Exception("图片下载失败")
        
        # 解码图片
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise Exception("图片解码失败")
        
        # 执行多语言OCR识别
        start_time = datetime.now()
        result = perform_multi_language_ocr(
            image,
            state.target_language,
            state.auto_detect_language
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"[多语言OCR] 识别完成，耗时 {processing_time:.2f} 秒")
        print(f"[多语言OCR] 检测语言: {result['detected_language']}, 置信度: {result['confidence']:.2%}")
        
        return MultiLanguageOCROutput(
            recognized_text=result['recognized_text'],
            detected_language=result['detected_language'],
            confidence=result['confidence'],
            regions=result['regions'],
            processing_time=processing_time
        )
        
    except Exception as e:
        error_msg = f"多语言OCR节点发生错误: {str(e)}\n{traceback.format_exc()}"
        print(f"[多语言OCR] 错误: {error_msg}")
        
        return MultiLanguageOCROutput(
            recognized_text="",
            detected_language="",
            confidence=0.0,
            regions=[],
            processing_time=0.0
        )


def download_image(image_url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"[多语言OCR] 下载图片失败: {str(e)}")
        return None


def perform_multi_language_ocr(
    image: np.ndarray,
    target_language: str,
    auto_detect_language: bool
) -> Dict[str, Any]:
    """
    执行多语言OCR识别
    
    Returns:
        {
            'recognized_text': str,
            'detected_language': str,
            'confidence': float,
            'regions': List[Dict]
        }
    """
    try:
        from paddleocr import PaddleOCR
        
        # 确定语言代码
        if auto_detect_language or target_language == "auto":
            # 自动检测语言：使用多语言模型
            lang_code = "ch"  # PP-OCRv5单模型支持多语言
            print(f"[多语言OCR] 自动检测语言模式")
        else:
            # 使用目标语言
            lang_code = map_language_to_paddle_code(target_language)
            print(f"[多语言OCR] 目标语言: {target_language} ({lang_code})")
        
        # 初始化PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang_code,
            use_gpu=False,
            show_log=False,
            # 优化参数
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.5,
        )
        
        # 执行识别
        result = ocr.ocr(image, cls=True)
        
        # 解析结果
        return parse_multi_language_ocr_result(result, lang_code)
    
    except ImportError as e:
        print(f"[多语言OCR] 无法导入PaddleOCR: {str(e)}")
        print(f"[多语言OCR] 尝试降级方案...")
        return perform_ocr_fallback(image)
    
    except Exception as e:
        print(f"[多语言OCR] 多语言OCR失败: {str(e)}")
        print(f"[多语言OCR] 尝试降级方案...")
        return perform_ocr_fallback(image)


def parse_multi_language_ocr_result(
    result: List,
    lang_code: str
) -> Dict[str, Any]:
    """
    解析多语言OCR结果
    """
    regions = []
    recognized_text = ""
    
    try:
        if not result or not result[0]:
            return {
                'recognized_text': "",
                'detected_language': "",
                'confidence': 0.0,
                'regions': []
            }
        
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            
            text_region = line[0]
            text_info = line[1]
            
            if text_info and len(text_info) >= 2:
                text = text_info[0]
                confidence = text_info[1]
                
                regions.append({
                    "text": text,
                    "text_region": text_region,
                    "confidence": float(confidence),
                    "language": lang_code
                })
    
    except Exception as e:
        print(f"[多语言OCR] 解析结果失败: {str(e)}")
    
    # 合并文本
    recognized_text = "\n".join([r["text"] for r in regions])
    
    # 计算整体置信度
    confidence = sum(r["confidence"] for r in regions) / len(regions) if regions else 0.0
    
    # 检测实际语言（基于文本内容）
    detected_language = lang_code
    if recognized_text:
        detected_language = detect_language_from_text(recognized_text)
    
    return {
        'recognized_text': recognized_text,
        'detected_language': detected_language,
        'confidence': confidence,
        'regions': regions
    }


def map_language_to_paddle_code(lang: str) -> str:
    """
    将语言代码映射到PaddleOCR支持的语言
    PaddleOCR 3.1.0支持80+种语言
    """
    lang_map = {
        # 核心语言（PP-OCRv5单模型支持）
        "ch": "ch",  # 简体中文
        "zh": "ch",
        "en": "en",  # 英文
        "japan": "japan",  # 日文
        "korean": "korean",  # 韩文
        "french": "french",  # 法文
        "german": "german",  # 德文
        "spanish": "spanish",  # 西班牙文
        "chinese_cht": "chinese_cht",  # 繁体中文
        
        # 其他常见语言（PaddleOCR 3.1.0支持）
        "italian": "it",  # 意大利文
        "portuguese": "pt",  # 葡萄牙文
        "russian": "ru",  # 俄文
        "arabic": "ar",  # 阿拉伯文
        "thai": "th",  # 泰文
        "vietnamese": "vi",  # 越南文
        "dutch": "nl",  # 荷兰文
        "swedish": "sv",  # 瑞典文
        "finnish": "fi",  # 芬兰文
        "danish": "da",  # 丹麦文
        "norwegian": "no",  # 挪威文
        "polish": "pl",  # 波兰文
        "czech": "cs",  # 捷克文
        "hungarian": "hu",  # 匈牙利文
        "turkish": "tr",  # 土耳其文
        "hebrew": "he",  # 希伯来文
        "hindi": "hi",  # 印地文
        "bengali": "bn",  # 孟加拉文
        "javanese": "jv",  # 爪哇文
        "korean": "ko",  # 韩文（另一种表示）
        "malay": "ms",  # 马来文
        "thai": "th",  # 泰文
        "vietnamese": "vi",  # 越南文
    }
    
    return lang_map.get(lang.lower(), "ch")


def detect_language_from_text(text: str) -> str:
    """
    从文本内容检测语言类型
    """
    if not text:
        return "ch"
    
    # 统计字符类型
    chinese_count = 0
    english_count = 0
    japanese_count = 0
    korean_count = 0
    other_count = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符
            chinese_count += 1
        elif '\u3040' <= char <= '\u309f':  # 日文平假名
            japanese_count += 1
        elif '\u30a0' <= char <= '\u30ff':  # 日文片假名
            japanese_count += 1
        elif '\uac00' <= char <= '\ud7af':  # 韩文
            korean_count += 1
        elif char.isalpha() or char.isdigit():
            english_count += 1
        else:
            other_count += 1
    
    # 判断主导语言
    max_count = max(chinese_count, english_count, japanese_count, korean_count, other_count)
    
    if korean_count == max_count and korean_count > 0:
        return "korean"
    elif japanese_count == max_count and japanese_count > 0:
        return "japan"
    elif english_count == max_count and english_count > 0:
        return "en"
    else:
        return "ch"


def perform_ocr_fallback(image: np.ndarray) -> Dict[str, Any]:
    """
    降级方案：使用Tesseract
    """
    try:
        import pytesseract
        
        print(f"[多语言OCR] 使用Tesseract降级方案")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='chi_sim+eng+jpn')
        
        return {
            'recognized_text': text.strip(),
            'detected_language': detect_language_from_text(text),
            'confidence': 0.5,
            'regions': [
                {
                    "text": text.strip(),
                    "text_region": [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]],
                    "confidence": 0.5,
                    "language": detect_language_from_text(text)
                }
            ]
        }
    
    except ImportError as e:
        print(f"[多语言OCR] 无法导入Tesseract: {str(e)}")
        return {
            'recognized_text': "",
            'detected_language': "",
            'confidence': 0.0,
            'regions': []
        }
    except Exception as e:
        print(f"[多语言OCR] Tesseract降级方案失败: {str(e)}")
        return {
            'recognized_text': "",
            'detected_language': "",
            'confidence': 0.0,
            'regions': []
        }


# 支持的语言列表（用于用户界面展示）
SUPPORTED_LANGUAGES = [
    {"code": "ch", "name": "简体中文", "native_name": "中文"},
    {"code": "en", "name": "英文", "native_name": "English"},
    {"code": "japan", "name": "日文", "native_name": "日本語"},
    {"code": "korean", "name": "韩文", "native_name": "한국어"},
    {"code": "french", "name": "法文", "native_name": "Français"},
    {"code": "german", "name": "德文", "native_name": "Deutsch"},
    {"code": "spanish", "name": "西班牙文", "native_name": "Español"},
    {"code": "chinese_cht", "name": "繁体中文", "native_name": "繁體中文"},
    {"code": "italian", "name": "意大利文", "native_name": "Italiano"},
    {"code": "portuguese", "name": "葡萄牙文", "native_name": "Português"},
    {"code": "russian", "name": "俄文", "native_name": "Русский"},
    {"code": "arabic", "name": "阿拉伯文", "native_name": "العربية"},
    {"code": "thai", "name": "泰文", "native_name": "ไทย"},
    {"code": "vietnamese", "name": "越南文", "native_name": "Tiếng Việt"},
    {"code": "dutch", "name": "荷兰文", "native_name": "Nederlands"},
    {"code": "swedish", "name": "瑞典文", "native_name": "Svenska"},
    {"code": "finnish", "name": "芬兰文", "native_name": "Suomi"},
    {"code": "danish", "name": "丹麦文", "native_name": "Dansk"},
    {"code": "norwegian", "name": "挪威文", "native_name": "Norsk"},
    {"code": "polish", "name": "波兰文", "native_name": "Polski"},
    {"code": "czech", "name": "捷克文", "native_name": "Čeština"},
    {"code": "hungarian", "name": "匈牙利文", "native_name": "Magyar"},
    {"code": "turkish", "name": "土耳其文", "native_name": "Türkçe"},
    {"code": "hebrew", "name": "希伯来文", "native_name": "עברית"},
    {"code": "hindi", "name": "印地文", "native_name": "हिन्दी"},
    {"code": "bengali", "name": "孟加拉文", "native_name": "বাংলা"},
    {"code": "javanese", "name": "爪哇文", "native_name": "Basa Jawa"},
    {"code": "malay", "name": "马来文", "native_name": "Bahasa Melayu"},
]


def get_supported_languages() -> List[Dict[str, str]]:
    """
    获取支持的语言列表
    
    Returns:
        语言列表，每个元素包含：
        {
            "code": "语言代码",
            "name": "中文名称",
            "native_name": "本地名称"
        }
    """
    return SUPPORTED_LANGUAGES


def is_language_supported(lang_code: str) -> bool:
    """
    检查语言是否支持
    
    Args:
        lang_code: 语言代码
    
    Returns:
        是否支持
    """
    return lang_code in [lang["code"] for lang in SUPPORTED_LANGUAGES]
