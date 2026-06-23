"""多引擎 OCR 融合 + LLM 智能纠错

核心思路：
  不依赖单一 OCR 引擎的精度，而是：
  1. 并行运行多个轻量级 OCR 引擎（RapidOCR + Tesseract）
  2. 用 LLM 从多个引擎的输出中"取其精华"
  3. LLM 同时做语义纠错（比纯规则纠错强得多）

无需 GPU，完全基于现有 LLM API + CPU OCR 引擎。
"""

import os, time, logging, re
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# ==================== 多引擎 OCR ====================

def run_rapidocr(image_path_or_array) -> Tuple[str, float, List]:
    """RapidOCR 识别，返回 (text, confidence, regions)"""
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        result, _ = engine(image_path_or_array)
        if result:
            lines = [line for _, line, conf in result]
            confs = [conf for _, _, conf in result]
            regions = [(line, conf, bbox) for (bbox, line, conf) in result]
            return "\n".join(lines), sum(confs)/len(confs), regions
        return "", 0.0, []
    except Exception as e:
        logger.warning(f"RapidOCR failed: {e}")
        return "", 0.0, []


def run_tesseract(image_path_or_array) -> Tuple[str, float, List]:
    """Tesseract OCR 识别 — 仅用于英文/数字场景，中文场景置信度打折"""
    try:
        import cv2
        import pytesseract

        import platform
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        if hasattr(image_path_or_array, 'shape'):
            img = image_path_or_array
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(image_path_or_array)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            return "", 0.0, []

        # 先用英文模式检测
        text_eng = pytesseract.image_to_string(img, lang='eng')
        # 再用中英文模式
        text_mixed = pytesseract.image_to_string(img, lang='chi_sim+eng')

        # 选择更好的结果
        text = text_mixed if len(text_mixed.strip()) > len(text_eng.strip()) else text_eng

        data = pytesseract.image_to_data(img, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data['conf'] if int(c) > 0]
        avg_conf = sum(confs) / len(confs) / 100.0 if confs else 0.0

        # 检测中文占比：如果中文多，Tesseract 结果不可靠，置信度打折
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.strip())
        if total_chars > 10 and chinese_chars / total_chars > 0.3:
            avg_conf *= 0.5  # 中文场景置信度减半

        return text.strip(), avg_conf, []
    except ImportError:
        return "", 0.0, []
    except Exception as e:
        logger.warning(f"Tesseract failed: {e}")
        return "", 0.0, []


def multi_engine_ocr(image_path_or_array) -> Dict[str, Any]:
    """并行运行多个 OCR 引擎，收集各自结果"""
    results = {}

    # RapidOCR（主力）
    r_text, r_conf, r_regions = run_rapidocr(image_path_or_array)
    if r_text:
        results["rapidocr"] = {"text": r_text, "confidence": r_conf, "regions": r_regions}

    # Tesseract（辅助）
    t_text, t_conf, _ = run_tesseract(image_path_or_array)
    if t_text:
        results["tesseract"] = {"text": t_text, "confidence": t_conf}

    return results


# ==================== LLM 智能融合 ====================

def llm_fusion_correction(ocr_results: Dict[str, Any], image_context: str = "") -> Dict[str, Any]:
    """用 LLM 从多个 OCR 引擎的输出中融合最佳结果

    策略：
    1. 如果只有一个引擎有结果，直接用 LLM 纠错
    2. 如果多个引擎有结果，用 LLM 做融合+纠错
    3. LLM 的 prompt 针对包装标签场景优化
    """
    if not ocr_results:
        return {"text": "", "method": "none", "confidence": 0}

    # 只有一个引擎结果
    if len(ocr_results) == 1:
        engine_name = list(ocr_results.keys())[0]
        text = ocr_results[engine_name]["text"]
        conf = ocr_results[engine_name]["confidence"]
        corrected = llm_correct_text(text, engine_name=engine_name)
        return {"text": corrected, "method": f"single_{engine_name}", "confidence": conf}

    # 多个引擎结果 → LLM 融合
    return llm_fuse_ocr_results(ocr_results)


def llm_correct_text(text: str, engine_name: str = "unknown") -> str:
    """用 LLM 纠错单个 OCR 引擎的结果"""
    if not text or len(text.strip()) < 3:
        return text

    prompt = f"""你是OCR文本纠错专家。以下是从产品包装图片中OCR识别的文本（引擎: {engine_name}）。

请完成以下任务：
1. 修正明显的OCR错别字（特别是中文形近字错误）
2. 修正中英混合时的乱码
3. 保持原文的排版格式和换行
4. 如果某段文本明显是乱码无法修复，用[...]标记

原始文本：
{text}

直接输出纠正后的文本，不要加任何解释。"""

    return _call_llm(prompt, text)


def llm_fuse_ocr_results(ocr_results: Dict[str, Any]) -> Dict[str, Any]:
    """用 LLM 从多个 OCR 结果中融合最佳文本"""
    if not ocr_results:
        return {"text": "", "method": "none", "confidence": 0}

    # 构建融合 prompt
    engine_texts = []
    for name, result in ocr_results.items():
        engine_texts.append(f"=== {name} (置信度: {result['confidence']:.2f}) ===\n{result['text']}")

    all_texts = "\n\n".join(engine_texts)

    prompt = f"""你是OCR文本融合专家。以下是从同一张产品包装图片中，用不同OCR引擎识别的结果。

请完成以下任务：
1. 对比多个引擎的输出，识别每个引擎的优势部分
2. 从每个引擎中提取最准确的文本片段
3. 融合成一份完整、准确的文本
4. 修正明显的OCR错误（中文形近字、中英乱码等）
5. 保持原文的排版格式

多个引擎的结果：
{all_texts}

直接输出融合后的最佳文本，不要加任何解释。"""

    fused = _call_llm(prompt, all_texts)

    # 计算融合后的置信度（取最高引擎置信度 + 融合增益）
    max_conf = max(r["confidence"] for r in ocr_results.values())
    fusion_bonus = 0.05 if len(ocr_results) > 1 else 0
    final_conf = min(0.99, max_conf + fusion_bonus)

    return {"text": fused, "method": "llm_fusion", "confidence": final_conf}


def _call_llm(prompt: str, fallback_text: str) -> str:
    """调用 LLM API — 支持 coze SDK / 通用 OpenAI 兼容 API"""
    try:
        # 模式1: 尝试 coze SDK（如果在 coze 环境中）
        try:
            from coze_coding_dev_sdk import LLMClient
            from langchain_core.messages import HumanMessage
            client = LLMClient()
            response = client.invoke(
                messages=[HumanMessage(content=prompt)],
                model=os.getenv("LLM_MODEL", "doubao-seed-2-0-pro-260215"),
                temperature=0.0,
                max_tokens=4096,
            )
            if response and hasattr(response, 'content'):
                result = response.content if isinstance(response.content, str) else str(response.content)
                if len(result) > len(fallback_text) * 0.3:
                    return result.strip()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"coze SDK failed: {e}")

        # 模式2: 通用 OpenAI 兼容 API
        import requests
        endpoint = os.getenv("LLM_ENDPOINT", "")
        api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        model = os.getenv("LLM_MODEL", "doubao-seed-2-0-pro-260215")

        if not endpoint:
            return fallback_text

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.0,
        }

        resp = requests.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            json=payload, headers=headers, timeout=30
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()

        if len(result) < len(fallback_text) * 0.3:
            return fallback_text

        return result

    except Exception as e:
        logger.debug(f"LLM call failed: {e}")
        return fallback_text


# ==================== 图像预处理增强 ====================

def enhance_image_for_ocr(image_array):
    """增强图像以提升 OCR 效果（纯算法，无需GPU）"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image_array

    if image_array is None:
        return None

    img = image_array.copy()
    h, w = img.shape[:2]

    # 1. 自适应去噪
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 双边滤波（保留边缘的同时去噪）
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. 自适应直方图均衡化（CLAHE）提升对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3. 自适应二值化（对光照不均匀的图片效果好）
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )

    # 4. 形态学操作去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 转回三通道
    if len(image_array.shape) == 3:
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    else:
        result = cleaned

    return result


def deskew_image(image_array):
    """倾斜校正（针对倾斜的包装标签）"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image_array, 0

    if image_array is None:
        return image_array, 0

    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array

    # 用 Hough 变换检测文本行角度
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return image_array, 0

    # 计算平均倾斜角度
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:  # 只取合理角度
            angles.append(angle)

    if not angles:
        return image_array, 0

    avg_angle = np.median(angles)

    # 如果倾斜角度小于0.5度，不校正
    if abs(avg_angle) < 0.5:
        return image_array, 0

    # 旋转校正
    h, w = image_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated = cv2.warpAffine(image_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    logger.info(f"  Deskewed by {avg_angle:.1f} degrees")
    return rotated, avg_angle


# ==================== 完整流程 ====================

def enhanced_ocr_pipeline(image_path_or_array, use_enhancement: bool = True,
                           use_llm_fusion: bool = True) -> Dict[str, Any]:
    """增强版 OCR 流程：预处理 → 多引擎 → 自动择优 → LLM 融合

    策略：
    1. 原图 + 增强图各跑一遍 RapidOCR
    2. 自动选择置信度更高的结果（增强不一定总有效）
    3. 如果有 Tesseract，也跑一遍
    4. 多引擎结果用 LLM 融合

    Args:
        image_path_or_array: 图片路径或 numpy 数组
        use_enhancement: 是否启用图像增强（自动择优）
        use_llm_fusion: 是否启用 LLM 融合（需要 LLM 端点）

    Returns:
        {"text": str, "confidence": float, "method": str, "engines_used": list}
    """
    import cv2

    # 获取图片数组
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    elif hasattr(image_path_or_array, 'shape'):
        img = image_path_or_array
    else:
        return {"text": "", "confidence": 0, "method": "error", "engines_used": []}

    if img is None:
        return {"text": "", "confidence": 0, "method": "error", "engines_used": []}

    # 1. 倾斜校正
    deskewed, angle = deskew_image(img)

    # 2. 多引擎 OCR（原图 + 增强图自动择优）
    results = {}

    # 原图 RapidOCR
    r_text, r_conf, _ = run_rapidocr(deskewed)
    if r_text:
        results["rapidocr"] = {"text": r_text, "confidence": r_conf}

    # 增强图 RapidOCR（自动择优：只有增强后更好才采用）
    if use_enhancement:
        enhanced = enhance_image_for_ocr(deskewed)
        e_text, e_conf, _ = run_rapidocr(enhanced)
        if e_text and e_conf > r_conf:
            results["rapidocr_enhanced"] = {"text": e_text, "confidence": e_conf}
            logger.info(f"  Enhanced image better: {r_conf:.3f} → {e_conf:.3f}")

    # Tesseract（如果可用）
    t_text, t_conf, _ = run_tesseract(deskewed)
    if t_text:
        results["tesseract"] = {"text": t_text, "confidence": t_conf}

    if not results:
        return {"text": "", "confidence": 0, "method": "no_results", "engines_used": []}

    # 3. LLM 融合
    if use_llm_fusion and len(results) > 0:
        fused = llm_fusion_correction(results)
        return {
            "text": fused["text"],
            "confidence": fused["confidence"],
            "method": fused["method"],
            "engines_used": list(results.keys()),
        }

    # 无 LLM 时，取置信度最高的结果
    best = max(results.values(), key=lambda x: x["confidence"])
    return {
        "text": best["text"],
        "confidence": best["confidence"],
        "method": "best_engine",
        "engines_used": list(results.keys()),
    }
