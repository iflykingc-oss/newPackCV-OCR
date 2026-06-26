# -*- coding: utf-8 -*-
"""
批量处理节点
对多张图片进行批量OCR识别处理，使用Tesseract引擎
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import BatchProcessInput, BatchProcessOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage

logger = logging.getLogger(__name__)


def _get_s3_storage() -> S3SyncStorage:
    """获取S3对象存储客户端"""
    return S3SyncStorage(
        endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
        access_key="",
        secret_key="",
        bucket_name=os.getenv("COZE_BUCKET_NAME"),
        region="cn-beijing",
    )


def download_image(url: str) -> Optional[np.ndarray]:
    """下载图片并转换为OpenCV格式"""
    try:
        if url.startswith('http://') or url.startswith('https://'):
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            pil_img = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif os.path.exists(url):
            return cv2.imread(url)
        return None
    except Exception as e:
        logger.error(f"下载图片失败: {e}")
        return None


def recognize_single_image(url: str) -> Dict[str, Any]:
    """对单张图片进行OCR识别（用于多进程）"""
    try:
        img = download_image(url)
        if img is None:
            return {"url": url, "text": "", "error": "图片下载失败"}

        # 灰度化 + CLAHE增强
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        pil_img = Image.fromarray(enhanced)
        config = '--psm 6 -l chi_sim+eng --oem 3'
        text = pytesseract.image_to_string(pil_img, config=config)

        return {"url": url, "text": text.strip(), "error": None}
    except Exception as e:
        return {"url": url, "text": "", "error": str(e)}


def batch_process_node(
    state: BatchProcessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> BatchProcessOutput:
    """
    title: 批量OCR处理
    desc: 对多张图片进行批量OCR识别，使用Tesseract引擎并行处理
    integrations: Tesseract OCR, 对象存储
    """
    ctx = runtime.context
    start_time = time.time()

    try:
        image_list = state.images or []
        if not image_list:
            logger.warning("无图片列表")
            return BatchProcessOutput(
                results=[],
                all_text="",
                success_count=0,
                failed_count=0,
                errors=["无图片列表"],
                export_file_url=""
            )

        logger.info(f"开始批量处理 {len(image_list)} 张图片")

        # 并行OCR识别（IO密集型用线程池）
        results: List[Dict[str, Any]] = []
        max_workers = min(len(image_list), 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(recognize_single_image, img.url): img.url
                for img in image_list if img and img.url
            }

            for future in as_completed(future_to_url):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    url = future_to_url[future]
                    logger.error(f"处理超时: {url}, 错误: {e}")
                    results.append({"url": url, "text": "", "error": str(e)})

        # 统计结果
        success_count = sum(1 for r in results if r.get("text"))
        total_count = len(results)

        logger.info(f"批量处理完成: {success_count}/{total_count} 成功")

        # 导出结果
        export_file_url = ""
        try:
            storage = _get_s3_storage()
            export_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_count": total_count,
                "success_count": success_count,
                "results": results
            }
            json_bytes = json.dumps(export_data, ensure_ascii=False, indent=2).encode('utf-8')

            file_name = f"batch_output/batch_result_{int(time.time())}.json"
            key = storage.upload_file(
                file_content=json_bytes,
                file_name=file_name,
                content_type='application/json'
            )
            export_file_url = storage.generate_presigned_url(key=key, expire_time=3600)
        except Exception as e:
            logger.warning(f"批量结果上传S3失败: {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"批量处理总耗时: {elapsed_time:.2f}秒")

        # 合并所有识别文本
        all_text = "\n".join([r.get("text", "") for r in results if r.get("text")])
        failed_count = total_count - success_count
        errors = [r.get("error", "") for r in results if r.get("error")]

        return BatchProcessOutput(
            results=results,
            all_text=all_text,
            success_count=success_count,
            failed_count=failed_count,
            errors=errors,
            export_file_url=export_file_url
        )

    except Exception as e:
        logger.error(f"批量处理异常: {e}")
        return BatchProcessOutput(
            results=[],
            all_text="",
            success_count=0,
            failed_count=0,
            errors=[str(e)],
            export_file_url=""
        )
