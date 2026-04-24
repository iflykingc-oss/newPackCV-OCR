# -*- coding: utf-8 -*-
"""
ROI分层裁切节点
自动分割货架图片中的每个商品区域
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    ROISegmentationInput,
    ROISegmentationOutput
)


def roi_segmentation_node(state: ROISegmentationInput, config: RunnableConfig, runtime: Runtime[Context]) -> ROISegmentationOutput:
    """
    title: ROI分层裁切
    desc: 自动分割货架图片中的每个商品区域，生成独立的裁切图片
    """
    ctx = runtime.context
    
    print(f"[ROI裁切] 开始分割 {len(state.detected_objects)} 个商品区域...")
    
    try:
        # 导入依赖
        import cv2
        import numpy as np
        import requests
        import tempfile
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        start_time = datetime.now()
        
        # 下载原始图片
        print(f"[ROI裁切] 下载原始图片...")
        img_data = download_image(state.shelf_image.url)
        if img_data is None:
            raise Exception("图片下载失败")
        
        # 解码图片
        img_array = np.frombuffer(img_data, np.uint8)
        original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise Exception("图片解码失败")
        
        # 限制最大裁切区域数
        objects_to_process = state.detected_objects[:state.max_regions]
        
        # 对每个检测到的商品进行ROI裁切
        roi_regions = []
        roi_images = []
        
        for idx, obj in enumerate(objects_to_process):
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            
            # 添加边距
            h, w = original_image.shape[:2]
            padding = state.padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 裁切区域
            roi = original_image[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # 保存裁切图片
            roi_image_url = save_roi_image(roi, idx, obj.get('id', idx))
            
            # 记录ROI信息
            roi_region = {
                'id': obj.get('id', idx),
                'index': idx,
                'bbox': [x1, y1, x2, y2],
                'size': [x2 - x1, y2 - y1],
                'confidence': obj.get('confidence', 0.0),
                'class_name': obj.get('class_name', 'unknown'),
                'roi_image_url': roi_image_url,
                'original_object': obj
            }
            
            roi_regions.append(roi_region)
            roi_images.append({'url': roi_image_url, 'region_id': idx})
            
            print(f"[ROI裁切] 处理区域 {idx + 1}/{len(objects_to_process)}: {x2-x1}x{y2-y1}")
        
        segmentation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"[ROI裁切] 完成，共生成 {len(roi_regions)} 个ROI区域，耗时 {segmentation_time:.2f} 秒")
        
        return ROISegmentationOutput(
            roi_regions=roi_regions,
            roi_images=roi_images,
            region_count=len(roi_regions),
            segmentation_time=segmentation_time
        )
        
    except Exception as e:
        error_msg = f"ROI裁切节点发生错误: {str(e)}\n{traceback.format_exc()}"
        print(f"[ROI裁切] 错误: {error_msg}")
        
        return ROISegmentationOutput(
            roi_regions=[],
            roi_images=[],
            region_count=0,
            segmentation_time=0.0
        )


def download_image(image_url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        import requests
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        print(f"下载图片失败: {str(e)}")
        return None


def save_roi_image(roi: np.ndarray, idx: int, obj_id: int) -> str:
    """保存ROI裁切图片"""
    try:
        import tempfile
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, roi)
        
        # 上传到对象存储
        storage = S3SyncStorage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"cv/roi_{timestamp}_id{obj_id}_{idx}.jpg"
        url = storage.upload_file(temp_path, object_name)
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return url
        
    except Exception as e:
        print(f"保存ROI图片失败: {str(e)}")
        return None
