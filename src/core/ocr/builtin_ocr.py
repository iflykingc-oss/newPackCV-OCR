"""
内嵌轻量级OCR引擎 - 不依赖外部模型下载
基于OpenCV图像处理和模板匹配
"""

import os
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TextRegion:
    """文本区域"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class OCRResult:
    """OCR识别结果"""
    raw_text: str
    confidence: float
    regions: List[TextRegion]
    metadata: Dict[str, Any]


class BuiltinOCR:
    """
    内嵌OCR引擎 - 基于OpenCV图像处理
    不需要下载任何模型，零依赖
    """
    
    # 数字模板 (0-9)
    DIGIT_TEMPLATES = {
        '0': np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]], dtype=np.uint8),
        '1': np.array([[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,1,0,0],[1,1,1,0]], dtype=np.uint8),
        '2': np.array([[0,1,1,0],[1,0,0,1],[0,0,1,0],[0,1,0,0],[1,1,1,1]], dtype=np.uint8),
        '3': np.array([[1,1,1,0],[0,0,1,0],[0,1,1,0],[0,0,0,1],[1,1,1,0]], dtype=np.uint8),
        '4': np.array([[1,0,1,0],[1,0,1,0],[1,1,1,1],[0,0,1,0],[0,0,1,0]], dtype=np.uint8),
        '5': np.array([[1,1,1,1],[1,0,0,0],[1,1,1,0],[0,0,0,1],[1,1,1,0]], dtype=np.uint8),
        '6': np.array([[0,1,1,0],[1,0,0,0],[1,1,1,0],[1,0,0,1],[0,1,1,0]], dtype=np.uint8),
        '7': np.array([[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,0,0]], dtype=np.uint8),
        '8': np.array([[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,0,0,1],[0,1,1,0]], dtype=np.uint8),
        '9': np.array([[0,1,1,0],[1,0,0,1],[0,1,1,1],[0,0,0,1],[0,1,1,0]], dtype=np.uint8),
    }
    
    # 常见字母模板
    LETTER_TEMPLATES = {
        'A': np.array([[0,1,0],[1,0,1],[1,1,1],[1,0,1],[1,0,1]], dtype=np.uint8),
        'B': np.array([[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,1,0]], dtype=np.uint8),
        'C': np.array([[0,1,1],[1,0,0],[1,0,0],[1,0,0],[0,1,1]], dtype=np.uint8),
        'D': np.array([[1,1,0],[1,0,1],[1,0,1],[1,0,1],[1,1,0]], dtype=np.uint8),
        'E': np.array([[1,1,1],[1,0,0],[1,1,0],[1,0,0],[1,1,1]], dtype=np.uint8),
        'F': np.array([[1,1,1],[1,0,0],[1,1,0],[1,0,0],[1,0,0]], dtype=np.uint8),
        'G': np.array([[0,1,1],[1,0,0],[1,0,1],[1,0,1],[0,1,1]], dtype=np.uint8),
        'H': np.array([[1,0,1],[1,0,1],[1,1,1],[1,0,1],[1,0,1]], dtype=np.uint8),
        'I': np.array([[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1]], dtype=np.uint8),
        'J': np.array([[0,0,1],[0,0,1],[0,0,1],[1,0,1],[0,1,0]], dtype=np.uint8),
        'K': np.array([[1,0,1],[1,0,1],[1,1,0],[1,0,1],[1,0,1]], dtype=np.uint8),
        'L': np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,1,1]], dtype=np.uint8),
        'M': np.array([[1,0,1],[1,1,1],[1,0,1],[1,0,1],[1,0,1]], dtype=np.uint8),
        'N': np.array([[1,0,1],[1,1,1],[1,1,1],[1,0,1],[1,0,1]], dtype=np.uint8),
        'O': np.array([[0,1,0],[1,0,1],[1,0,1],[1,0,1],[0,1,0]], dtype=np.uint8),
        'P': np.array([[1,1,0],[1,0,1],[1,1,0],[1,0,0],[1,0,0]], dtype=np.uint8),
        'Q': np.array([[0,1,0],[1,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.uint8),
        'R': np.array([[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,0,1]], dtype=np.uint8),
        'S': np.array([[0,1,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[1,1,0,0]], dtype=np.uint8),
        'T': np.array([[1,1,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0]], dtype=np.uint8),
        'U': np.array([[1,0,1],[1,0,1],[1,0,1],[1,0,1],[0,1,0]], dtype=np.uint8),
        'V': np.array([[1,0,1],[1,0,1],[1,0,1],[0,1,0],[0,1,0]], dtype=np.uint8),
        'W': np.array([[1,0,1],[1,0,1],[1,0,1],[1,1,1],[1,0,1]], dtype=np.uint8),
        'X': np.array([[1,0,1],[1,0,1],[0,1,0],[1,0,1],[1,0,1]], dtype=np.uint8),
        'Y': np.array([[1,0,1],[1,0,1],[0,1,0],[0,1,0],[0,1,0]], dtype=np.uint8),
        'Z': np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,0],[1,1,1]], dtype=np.uint8),
    }
    
    def __init__(self):
        """初始化内嵌OCR"""
        self.templates = {**self.DIGIT_TEMPLATES, **self.LETTER_TEMPLATES}
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 转为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 去噪
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def find_text_regions(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """查找文本区域"""
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        min_area = 20  # 最小区域面积
        max_area = binary.shape[0] * binary.shape[1] * 0.5  # 最大区域面积
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 过滤太小的区域
            if area < min_area:
                continue
            
            # 过滤太大的区域
            if area > max_area:
                continue
            
            # 宽高比过滤（排除极端形状）
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            regions.append((x, y, x + w, y + h))
        
        # 按x坐标排序（从左到右）
        regions.sort(key=lambda r: r[0])
        
        return regions
    
    def segment_characters(self, char_img: np.ndarray, num_chars: int = None) -> List[np.ndarray]:
        """分割字符"""
        if len(char_img.shape) == 3:
            gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = char_img
        
        # 垂直投影
        vertical_proj = np.sum(gray, axis=0)
        
        # 找到字符间隔
        threshold = np.mean(vertical_proj) * 0.3
        is_char = vertical_proj > threshold
        
        # 找到字符边界
        char_boundaries = []
        in_char = False
        start = 0
        
        for i, val in enumerate(is_char):
            if val and not in_char:
                start = i
                in_char = True
            elif not val and in_char:
                char_boundaries.append((start, i))
                in_char = False
        
        if in_char:
            char_boundaries.append((start, len(is_char)))
        
        # 提取字符图像
        chars = []
        for start, end in char_boundaries:
            if end - start > 3:  # 最小字符宽度
                char_crop = gray[:, start:end]
                chars.append(char_crop)
        
        return chars
    
    def recognize_character(self, char_img: np.ndarray) -> Tuple[str, float]:
        """识别单个字符"""
        # 调整大小
        char_resized = cv2.resize(char_img, (5, 7))
        
        # 二值化
        _, char_binary = cv2.threshold(char_resized, 127, 1, cv2.THRESH_BINARY)
        
        best_match = '?'
        best_score = 0.0
        
        for char, template in self.templates.items():
            # 调整模板大小
            template_resized = cv2.resize(template, (5, 7))
            
            # 计算相似度
            score = np.sum(char_binary == template_resized) / 35.0
            
            if score > best_score:
                best_score = score
                best_match = char
        
        return best_match, best_score
    
    def ocr(self, image_path: str) -> OCRResult:
        """
        执行OCR识别
        
        Args:
            image_path: 图片路径或URL
            
        Returns:
            OCRResult: 识别结果
        """
        # 读取图片
        if image_path.startswith(('http://', 'https://')):
            import requests
            response = requests.get(image_path, timeout=10)
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_path)
        
        if img is None:
            return OCRResult(raw_text="", confidence=0.0, regions=[], metadata={"error": "无法读取图片"})
        
        # 预处理
        binary = self.preprocess_image(img)
        
        # 查找文本区域
        regions = self.find_text_regions(binary)
        
        # 识别文字
        all_texts = []
        all_confidences = []
        text_regions = []
        
        for i, (x1, y1, x2, y2) in enumerate(regions):
            # 裁剪文本区域
            roi = binary[y1:y2, x1:x2]
            
            # 分割字符
            chars = self.segment_characters(roi)
            
            # 识别每个字符
            text = ""
            confidence = 0.0
            for char_img in chars:
                if char_img.size > 0:
                    char, score = self.recognize_character(char_img)
                    text += char
                    confidence += score
            
            if chars:
                confidence /= len(chars)
            
            if text.strip():
                all_texts.append(text)
                all_confidences.append(confidence)
                text_regions.append(TextRegion(
                    text=text,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2)
                ))
        
        # 合并结果
        raw_text = " ".join(all_texts)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        return OCRResult(
            raw_text=raw_text,
            confidence=avg_confidence,
            regions=text_regions,
            metadata={"engine": "builtin", "regions_count": len(regions)}
        )


class PatternOCR:
    """
    基于模式的OCR - 用于识别常见的日期、批号等格式
    不需要机器学习，纯规则匹配
    """
    
    # 日期模式
    DATE_PATTERNS = [
        r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',  # 2024/01/01, 2024年1月1日
        r'(\d{4}\.\d{1,2}\.\d{1,2})',  # 2024.01.01
        r'(\d{8})',  # 20240101
        r'(\d{2}[-/]\d{2}[-/]\d{2,4})',  # 24/01/01
    ]
    
    # 批号模式
    BATCH_PATTERNS = [
        r'([A-Z]\d{6,})',  # A123456
        r'(\d{10,})',  # 长数字
        r'([A-Z]{2,}\d+)',  # AB123
    ]
    
    def __init__(self):
        self.date_regex = [re.compile(p) for p in self.DATE_PATTERNS]
        self.batch_regex = [re.compile(p) for p in self.BATCH_PATTERNS]
    
    def extract_dates(self, text: str) -> List[str]:
        """提取日期"""
        dates = []
        for regex in self.date_regex:
            matches = regex.findall(text)
            dates.extend(matches)
        return dates
    
    def extract_batch_numbers(self, text: str) -> List[str]:
        """提取批号"""
        batches = []
        for regex in self.batch_regex:
            matches = regex.findall(text)
            batches.extend(matches)
        return batches
    
    def parse_expiration_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        # 移除分隔符
        date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
        date_str = date_str.replace('/', '-').replace('.', '-')
        
        # 尝试不同格式
        formats = [
            '%Y-%m-%d',
            '%Y%m%d',
            '%y-%m-%d',
            '%y%m%d',
            '%Y-%m',
            '%Y%m',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None


# 全局实例
builtin_ocr = BuiltinOCR()
pattern_ocr = PatternOCR()


def recognize_text(image_path: str) -> OCRResult:
    """
    识别图片中的文字
    
    Args:
        image_path: 图片路径或URL
        
    Returns:
        OCRResult: 识别结果
    """
    return builtin_ocr.ocr(image_path)


def extract_structured_info(text: str) -> Dict[str, Any]:
    """
    从识别文本中提取结构化信息
    
    Args:
        text: OCR识别的原始文本
        
    Returns:
        Dict: 结构化信息
    """
    result = {
        'dates': pattern_ocr.extract_dates(text),
        'batch_numbers': pattern_ocr.extract_batch_numbers(text),
    }
    
    # 尝试解析日期
    for date_str in result['dates']:
        parsed = pattern_ocr.parse_expiration_date(date_str)
        if parsed:
            result['parsed_date'] = parsed.isoformat()
            break
    
    return result
