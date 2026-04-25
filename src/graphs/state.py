# -*- coding: utf-8 -*-
"""
多平台OCR包装识别系统 - 状态定义
包含全局状态、图出入参、节点出入参定义
"""

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from utils.file.file import File


# ==================== 全局状态 ====================

class GlobalState(BaseModel):
    """全局状态定义 - 包含工作流流转的所有数据"""
    # 输入数据（从GraphInput合并）
    package_image: File = Field(..., description="包装图片（瓶子、包装盒、包装袋等）")
    shelf_image: Optional[File] = Field(default=None, description="货架图片（PackCV-OCR场景）")
    images: Optional[List[File]] = Field(default=None, description="多张图片列表（批量处理）")
    processing_mode: Optional[str] = Field(default="single", description="处理模式：single（单图）或 batch（批量）")
    cv_model: Optional[str] = Field(default="yolov8", description="CV模型类型（PackCV-OCR）")
    ocr_engine_type: Literal["builtin", "api", "tesseract"] = Field(default="builtin", description="OCR引擎类型")
    ocr_api_config: Optional[Dict[str, Any]] = Field(default=None, description="OCR API配置")
    model_type: Literal["extract", "correct", "qa"] = Field(default="extract", description="模型调用类型")
    model_name: str = Field(default="doubao-seed-2-0-pro-260215", description="大模型名称")
    model_prompt: Optional[str] = Field(default="", description="自定义模型提示词")
    platform: Literal["wechat", "feishu", "none"] = Field(default="none", description="目标平台")
    export_format: Literal["json", "excel", "pdf"] = Field(default="json", description="导出格式")
    user_question: Optional[str] = Field(default="", description="用户提问")
    ocr_engine_config: Dict[str, Any] = Field(default_factory=dict, description="OCR引擎配置")
    llm_model_config: Dict[str, Any] = Field(default_factory=dict, description="大模型配置")
    platform_config: Dict[str, Any] = Field(default_factory=dict, description="多平台配置")
    
    # PackCV-OCR 特有字段
    enable_expiry_detection: bool = Field(default=True, description="是否启用效期检测")
    enable_inventory_analysis: bool = Field(default=True, description="是否启用库存分析")
    enable_alerts: bool = Field(default=True, description="是否启用告警")
    alert_rules: Dict[str, Any] = Field(default_factory=dict, description="告警规则配置")
    near_expiry_days: int = Field(default=30, description="临期预警天数")
    low_stock_threshold: int = Field(default=10, description="低库存阈值")
    max_workers: int = Field(default=10, description="最大并行处理数")
    detection_threshold: float = Field(default=0.5, description="CV检测置信度阈值")
    
    # 中间状态（单图处理）
    preprocessed_image: Optional[File] = Field(default=None, description="预处理后的图片")
    ocr_raw_result: Optional[str] = Field(default="", description="OCR识别的原始文本")
    ocr_confidence: Optional[float] = Field(default=0.0, description="OCR识别置信度")
    ocr_regions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="OCR识别区域列表（坐标+文本）")
    structured_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="模型结构化提取的数据")
    corrected_result: Optional[str] = Field(default="", description="智能纠错后的文本")
    qa_answer: Optional[str] = Field(default="", description="语义问答的答案")
    
    # 中间状态（PackCV-OCR）
    detected_objects: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="CV检测到的商品列表")
    detection_confidence: Optional[float] = Field(default=0.0, description="CV检测整体置信度")
    roi_regions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="裁切后的ROI区域列表")
    roi_images: Optional[List[File]] = Field(default_factory=list, description="裁切后的ROI图片列表")
    processing_results: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="并行处理结果列表")
    expiry_data: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="效期数据列表")
    quantity_stats: Optional[Dict[str, Any]] = Field(default_factory=dict, description="数量统计信息")
    alerts: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="告警列表")
    reports: Optional[Dict[str, Any]] = Field(default_factory=dict, description="生成的报表")
    
    # 输出数据
    final_result: Dict[str, Any] = Field(default_factory=dict, description="最终输出结果")
    export_file_url: Optional[str] = Field(default="", description="导出文件URL（TXT/PDF/Excel）")
    success: bool = Field(default=True, description="是否成功")
    error_message: Optional[str] = Field(default=None, description="错误信息")


# ==================== 图出入参 ====================

class GraphInput(BaseModel):
    """工作流输入"""
    package_image: File = Field(..., description="上传的包装图片（单图处理）")
    images: Optional[List[File]] = Field(default=None, description="上传的图片列表（批量处理，优先级高于package_image）")
    ocr_engine_type: Literal["builtin", "api"] = Field(default="builtin", description="OCR引擎类型：builtin（内置算法）或 api（外部API）")
    ocr_api_config: Optional[Dict[str, Any]] = Field(default=None, description="OCR API配置（当engine_type=api时使用）")
    model_type: Literal["extract", "correct", "qa"] = Field(default="extract", description="模型调用类型：extract（结构化提取）、correct（智能纠错）、qa（语义问答）")
    model_name: Optional[str] = Field(default="doubao-seed-2-0-pro-260215", description="使用的大模型名称")
    model_prompt: Optional[str] = Field(default="", description="自定义模型提示词")
    platform: Literal["wechat", "feishu", "none"] = Field(default="none", description="目标平台：wechat（微信）、feishu（飞书）、none（不推送）")
    export_format: Literal["json", "excel", "pdf"] = Field(default="json", description="导出格式")
    user_question: Optional[str] = Field(default="", description="用户提问（仅用于qa模式）")


class GraphOutput(BaseModel):
    """工作流输出"""
    success: bool = Field(default=True, description="是否成功")
    ocr_result: str = Field(default="", description="OCR识别结果")
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="结构化提取数据（如有）")
    corrected_text: Optional[str] = Field(default=None, description="纠错后的文本（如有）")
    qa_answer: Optional[str] = Field(default=None, description="问答答案（如有）")
    export_file_url: Optional[str] = Field(default=None, description="导出文件URL")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    
    # 兼容性字段
    raw_text: Optional[str] = Field(default=None, description="OCR识别结果（兼容字段）")
    answer: Optional[str] = Field(default=None, description="问答答案（兼容字段）")


# ==================== 图片预处理节点 ====================

class ImagePreprocessInput(BaseModel):
    """图片预处理节点输入"""
    package_image: File = Field(..., description="原始包装图片")


class ImagePreprocessOutput(BaseModel):
    """图片预处理节点输出"""
    preprocessed_image: File = Field(..., description="预处理后的图片")
    is_rotated: bool = Field(default=False, description="是否进行了旋转校正")
    is_enhanced: bool = Field(default=False, description="是否进行了图像增强")
    processing_info: Dict[str, Any] = Field(default_factory=dict, description="处理信息（角度、亮度等）")


# ==================== OCR识别节点 ====================

class OCRRecognizeInput(BaseModel):
    """OCR识别节点输入"""
    image: Optional[File] = Field(default=None, description="待识别图片（可能是原始图或预处理图）")
    package_image: Optional[File] = Field(default=None, description="原始包装图片")
    preprocessed_image: Optional[File] = Field(default=None, description="预处理后的图片")
    ocr_engine_type: Literal["builtin", "api"] = Field(default="builtin", description="OCR引擎类型")
    ocr_api_config: Optional[Dict[str, Any]] = Field(default=None, description="OCR API配置")


class OCRRecognizeOutput(BaseModel):
    """OCR识别节点输出"""
    ocr_raw_result: str = Field(default="", description="识别的原始文本")
    raw_text: str = Field(default="", description="识别的原始文本（兼容字段）")
    ocr_confidence: float = Field(default=0.0, description="整体置信度")
    confidence: float = Field(default=0.0, description="整体置信度（兼容字段）")
    ocr_regions: List[Dict[str, Any]] = Field(default_factory=list, description="识别区域列表")
    regions: List[Dict[str, Any]] = Field(default_factory=list, description="识别区域列表（兼容字段）")
    engine_used: str = Field(default="", description="使用的引擎名称")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# ==================== 模型结构化提取节点 ====================

class ModelExtractInput(BaseModel):
    """模型结构化提取节点输入"""
    ocr_text: Optional[str] = Field(default="", description="OCR识别的文本")
    raw_text: Optional[str] = Field(default="", description="OCR识别的文本（兼容字段）")
    ocr_raw_result: Optional[str] = Field(default="", description="OCR识别的文本（兼容字段）")
    model_name: str = Field(default="doubao-seed-2-0-pro-260215", description="使用的模型名称")
    custom_prompt: Optional[str] = Field(default="", description="自定义提示词")
    template_fields: Optional[List[str]] = Field(default_factory=list, description="需要提取的字段列表")


class ModelExtractOutput(BaseModel):
    """模型结构化提取节点输出"""
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="提取的结构化数据")
    confidence: float = Field(default=0.0, description="提取置信度")
    missing_fields: List[str] = Field(default_factory=list, description="缺失字段列表")


# ==================== 智能纠错节点 ====================

class CorrectTextInput(BaseModel):
    """智能纠错节点输入"""
    ocr_text: Optional[str] = Field(default="", description="需要纠错的文本")
    raw_text: Optional[str] = Field(default="", description="需要纠错的文本（兼容字段）")
    ocr_raw_result: Optional[str] = Field(default="", description="需要纠错的文本（兼容字段）")
    model_name: str = Field(default="doubao-seed-2-0-pro-260215", description="使用的模型名称")
    correction_rules: Optional[List[str]] = Field(default_factory=list, description="纠错规则列表")


class CorrectTextOutput(BaseModel):
    """智能纠错节点输出"""
    corrected_text: str = Field(..., description="纠错后的文本")
    changes: List[Dict[str, Any]] = Field(default_factory=list, description="修改记录（原文->修正）")
    correction_count: int = Field(default=0, description="修正数量")


# ==================== 语义问答节点 ====================

class QaAnswerInput(BaseModel):
    """语义问答节点输入"""
    ocr_text: Optional[str] = Field(default="", description="OCR识别的文本（可作为上下文）")
    raw_text: Optional[str] = Field(default="", description="OCR识别的文本（可作为上下文，兼容字段）")
    ocr_raw_result: Optional[str] = Field(default="", description="OCR识别的文本（可作为上下文，兼容字段）")
    structured_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="结构化数据（可作为上下文）")
    user_question: str = Field(default="", description="用户提问")
    model_name: str = Field(default="doubao-seed-2-0-pro-260215", description="使用的模型名称")


class QaAnswerOutput(BaseModel):
    """语义问答节点输出"""
    answer: str = Field(..., description="答案")
    confidence: float = Field(default=0.0, description="答案置信度")
    references: List[str] = Field(default_factory=list, description="参考来源")


# ==================== 结果输出节点 ====================

class ResultOutputInput(BaseModel):
    """结果输出节点输入"""
    ocr_result: Optional[str] = Field(default="", description="OCR识别结果")
    raw_text: Optional[str] = Field(default="", description="OCR识别结果（兼容字段）")
    ocr_raw_result: Optional[str] = Field(default="", description="OCR识别结果（兼容字段）")
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="结构化数据")
    corrected_text: Optional[str] = Field(default=None, description="纠错文本")
    corrected_result: Optional[str] = Field(default=None, description="纠错文本（兼容字段）")
    qa_answer: Optional[str] = Field(default=None, description="问答答案")
    answer: Optional[str] = Field(default=None, description="问答答案（兼容字段）")
    export_format: Literal["json", "excel", "pdf"] = Field(default="json", description="导出格式")
    platform: Literal["wechat", "feishu", "none"] = Field(default="none", description="目标平台")
    package_image: Optional[File] = Field(default=None, description="原始图片（用于生成报告）")
    preprocessed_image: Optional[File] = Field(default=None, description="预处理图片（用于生成报告）")


class ResultOutputOutput(BaseModel):
    """结果输出节点输出"""
    final_result: Dict[str, Any] = Field(default_factory=dict, description="最终结果（包含所有数据）")
    export_file_url: Optional[str] = Field(default=None, description="导出文件URL")
    platform_push_result: Dict[str, Any] = Field(default_factory=dict, description="平台推送结果")


# ==================== 路由处理节点 ====================

class RouteProcessingInput(BaseModel):
    """路由处理节点输入"""
    images: Optional[List[File]] = Field(default=None, description="多张图片列表（批量处理）")
    package_image: File = Field(default=..., description="包装图片（单图处理）")


class RouteProcessingOutput(BaseModel):
    """路由处理节点输出"""
    processing_mode: str = Field(default="single", description="处理模式：single（单图）或 batch（批量）")


# ==================== 批量处理节点 ====================

class BatchProcessInput(BaseModel):
    """批量处理节点输入"""
    images: List[File] = Field(..., description="待处理的图片列表")
    ocr_engine_type: Literal["builtin", "api", "tesseract"] = Field(default="builtin", description="OCR引擎类型")
    ocr_api_config: Optional[Dict[str, Any]] = Field(default=None, description="OCR API配置")
    export_format: Literal["json", "excel", "pdf"] = Field(default="json", description="导出格式")
    model_type: Optional[Literal["extract", "correct", "qa"]] = Field(default=None, description="模型调用类型（可选）")
    model_name: Optional[str] = Field(default="doubao-seed-2-0-pro-260215", description="使用的模型名称")


class BatchProcessOutput(BaseModel):
    """批量处理节点输出"""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="每个图片的处理结果列表")
    all_text: str = Field(default="", description="所有识别文本合并")
    success_count: int = Field(default=0, description="成功数量")
    failed_count: int = Field(default=0, description="失败数量")
    errors: List[str] = Field(default_factory=list, description="错误信息列表")
    export_file_url: Optional[str] = Field(default=None, description="导出文件URL")


# ==================== PackCV-OCR 融合算法状态定义 ====================

# --- CV目标检测节点 ---

class CVDetectionInput(BaseModel):
    """CV目标检测节点输入"""
    shelf_image: File = Field(..., description="货架/多包装图片")
    cv_model: str = Field(default="yolov8", description="CV模型类型：yolov8/mmdetection")
    detection_threshold: float = Field(default=0.5, description="检测置信度阈值")
    enable_image_enhancement: bool = Field(default=True, description="是否启用图像增强")


class CVDetectionOutput(BaseModel):
    """CV目标检测节点输出"""
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list, description="检测到的商品列表")
    total_count: int = Field(default=0, description="商品总数")
    detection_confidence: float = Field(default=0.0, description="整体检测置信度")
    processed_image: Optional[File] = Field(default=None, description="标注后的图片（带检测框）")
    detection_time: float = Field(default=0.0, description="检测耗时（秒）")


# --- ROI分层裁切节点 ---

class ROISegmentationInput(BaseModel):
    """ROI分层裁切节点输入"""
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list, description="检测到的商品列表")
    shelf_image: File = Field(..., description="原始货架图片")
    padding: int = Field(default=10, description="裁切边距（像素）")
    max_regions: int = Field(default=100, description="最大裁切区域数")


class ROISegmentationOutput(BaseModel):
    """ROI分层裁切节点输出"""
    roi_regions: List[Dict[str, Any]] = Field(default_factory=list, description="裁切后的区域列表")
    roi_images: List[File] = Field(default_factory=list, description="裁切后的图片列表")
    region_count: int = Field(default=0, description="裁切区域总数")
    segmentation_time: float = Field(default=0.0, description="分割耗时（秒）")


# --- 并行处理引擎节点 ---

class ParallelProcessingInput(BaseModel):
    """并行处理引擎节点输入"""
    roi_images: List[File] = Field(..., description="裁切后的图片列表")
    roi_regions: List[Dict[str, Any]] = Field(default_factory=list, description="裁切区域信息")
    ocr_engine_type: Literal["builtin", "api", "tesseract"] = Field(default="builtin", description="OCR引擎类型")
    ocr_api_config: Optional[Dict[str, Any]] = Field(default=None, description="OCR API配置")
    max_workers: int = Field(default=10, description="最大并行处理数")
    enable_expiry_detection: bool = Field(default=True, description="是否启用效期检测")


class ParallelProcessingOutput(BaseModel):
    """并行处理引擎节点输出"""
    processing_results: List[Dict[str, Any]] = Field(default_factory=list, description="每个区域的处理结果")
    total_processed: int = Field(default=0, description="处理总数")
    success_count: int = Field(default=0, description="成功数量")
    failed_count: int = Field(default=0, description="失败数量")
    processing_time: float = Field(default=0.0, description="总处理耗时（秒）")


# --- 效期识别节点 ---

class ExpiryDateInput(BaseModel):
    """效期识别节点输入"""
    processing_results: List[Dict[str, Any]] = Field(default_factory=list, description="并行处理结果")
    date_format_preference: Optional[List[str]] = Field(default_factory=list, description="日期格式偏好（如['YYYY-MM-DD', 'YYYYMMDD']）")
    enable_correction: bool = Field(default=True, description="是否启用日期格式纠正")


class ExpiryDateOutput(BaseModel):
    """效期识别节点输出"""
    expiry_data: List[Dict[str, Any]] = Field(default_factory=list, description="效期数据列表")
    total_products: int = Field(default=0, description="商品总数")
    valid_count: int = Field(default=0, description="有效期内数量")
    near_expiry_count: int = Field(default=0, description="临期数量（30天内）")
    expired_count: int = Field(default=0, description="过期数量")
    recognition_accuracy: float = Field(default=0.0, description="识别准确率")


# --- 数量统计节点 ---

class QuantityCountInput(BaseModel):
    """数量统计节点输入"""
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list, description="CV检测结果")
    processing_results: List[Dict[str, Any]] = Field(default_factory=list, description="OCR处理结果")
    enable_inventory_analysis: bool = Field(default=True, description="是否启用库存分析")


class QuantityCountOutput(BaseModel):
    """数量统计节点输出"""
    quantity_stats: Dict[str, Any] = Field(default_factory=dict, description="数量统计信息")
    category_breakdown: List[Dict[str, Any]] = Field(default_factory=list, description="分类统计")
    inventory_status: Dict[str, Any] = Field(default_factory=dict, description="库存状态")
    total_items: int = Field(default=0, description="总物品数")


# --- 智能告警引擎节点 ---

class AlertEngineInput(BaseModel):
    """智能告警引擎节点输入"""
    expiry_data: List[Dict[str, Any]] = Field(default_factory=list, description="效期数据")
    quantity_stats: Dict[str, Any] = Field(default_factory=dict, description="数量统计")
    inventory_status: Dict[str, Any] = Field(default_factory=dict, description="库存状态")
    alert_rules: Dict[str, Any] = Field(default_factory=dict, description="告警规则配置")
    near_expiry_days: int = Field(default=30, description="临期预警天数")
    low_stock_threshold: int = Field(default=10, description="低库存阈值")


class AlertEngineOutput(BaseModel):
    """智能告警引擎节点输出"""
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="告警列表")
    alert_summary: Dict[str, Any] = Field(default_factory=dict, description="告警摘要")
    critical_count: int = Field(default=0, description="严重告警数")
    warning_count: int = Field(default=0, description="警告告警数")
    info_count: int = Field(default=0, description="信息告警数")


# --- 自动报表生成节点 ---

class ReportGenerationInput(BaseModel):
    """自动报表生成节点输入"""
    expiry_data: List[Dict[str, Any]] = Field(default_factory=list, description="效期数据")
    quantity_stats: Dict[str, Any] = Field(default_factory=dict, description="数量统计")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="告警列表")
    shelf_image: Optional[File] = Field(default=None, description="原始货架图片")
    report_type: Literal["expiry", "inventory", "compliance", "all"] = Field(default="all", description="报表类型")
    export_format: Literal["json", "excel", "pdf"] = Field(default="excel", description="导出格式")
    include_charts: bool = Field(default=True, description="是否包含图表")


class ReportGenerationOutput(BaseModel):
    """自动报表生成节点输出"""
    reports: Dict[str, Any] = Field(default_factory=dict, description="生成的报表")
    expiry_report_url: Optional[str] = Field(default=None, description="效期报表URL")
    inventory_report_url: Optional[str] = Field(default=None, description="库存报表URL")
    compliance_report_url: Optional[str] = Field(default=None, description="合规台账URL")
    combined_report_url: Optional[str] = Field(default=None, description="合并报表URL")
    generation_time: float = Field(default=0.0, description="生成耗时（秒）")


# --- PackCV-OCR 工作流输入输出 ---

class PackCVGraphInput(BaseModel):
    """PackCV-OCR工作流输入"""
    shelf_image: File = Field(..., description="货架/多包装图片")
    cv_model: str = Field(default="yolov8", description="CV模型类型")
    ocr_engine_type: Literal["builtin", "api", "tesseract"] = Field(default="builtin", description="OCR引擎类型")
    enable_expiry_detection: bool = Field(default=True, description="是否启用效期检测")
    enable_inventory_analysis: bool = Field(default=True, description="是否启用库存分析")
    enable_alerts: bool = Field(default=True, description="是否启用告警")
    export_format: Literal["json", "excel", "pdf"] = Field(default="excel", description="导出格式")
    platform: Literal["wechat", "feishu", "none"] = Field(default="none", description="目标平台")
    alert_rules: Dict[str, Any] = Field(default_factory=dict, description="告警规则")


class PackCVGraphOutput(BaseModel):
    """PackCV-OCR工作流输出"""
    success: bool = Field(default=True, description="是否成功")
    detection_result: Dict[str, Any] = Field(default_factory=dict, description="CV检测结果")
    expiry_data: List[Dict[str, Any]] = Field(default_factory=list, description="效期数据")
    quantity_stats: Dict[str, Any] = Field(default_factory=dict, description="数量统计")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="告警列表")
    reports: Dict[str, Any] = Field(default_factory=dict, description="报表URLs")
    total_products: int = Field(default=0, description="总商品数")


# ==================== V1.1 新增节点状态定义 ====================

# --- 图像预处理增强节点（V1.1新增）---

class ImagePreprocessEnhanceInput(BaseModel):
    """图像预处理增强节点输入"""
    image: File = Field(..., description="原始图片")
    enable_orientation_classify: bool = Field(default=True, description="是否启用方向分类")
    enable_dewarp: bool = Field(default=True, description="是否启用去畸变")
    enable_denoise: bool = Field(default=True, description="是否启用去噪")
    enable_enhance: bool = Field(default=True, description="是否启用增强")
    enhance_denoise_kernel: int = Field(default=3, description="去噪核大小")
    enhance_contrast: float = Field(default=1.5, description="对比度增强系数")


class ImagePreprocessEnhanceOutput(BaseModel):
    """图像预处理增强节点输出"""
    preprocessed_image: File = Field(..., description="预处理后的图片")
    orientation_angle: int = Field(default=0, description="检测到的方向角度（0/90/180/270）")
    is_corrected: bool = Field(default=False, description="是否进行了矫正")
    enhancement_steps: List[str] = Field(default_factory=list, description="执行的增强步骤")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 文本方向矫正节点（V1.1新增）---

class TextDirectionCorrectInput(BaseModel):
    """文本方向矫正节点输入"""
    image: File = Field(..., description="输入图片")
    use_edge_projection: bool = Field(default=True, description="是否使用边缘投影法")
    use_cls_model: bool = Field(default=True, description="是否使用分类模型")
    angle_range: int = Field(default=45, description="角度搜索范围（±angle）")


class TextDirectionCorrectOutput(BaseModel):
    """文本方向矫正节点输出"""
    corrected_image: File = Field(..., description="矫正后的图片")
    detected_angle: float = Field(default=0.0, description="检测到的旋转角度")
    correction_method: str = Field(default="", description="使用的矫正方法")
    confidence: float = Field(default=0.0, description="置信度")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 智能排版解析节点（V1.1新增）---

class LayoutParseInput(BaseModel):
    """智能排版解析节点输入"""
    ocr_regions: List[Dict[str, Any]] = Field(default_factory=list, description="OCR识别区域列表")
    parse_mode: Literal["auto", "multi_column", "single_column", "preserve_indent"] = Field(default="auto", description="解析模式")
    enable_paragraph_break: bool = Field(default=True, description="是否启用自然段换行")
    enable_vertical_text: bool = Field(default=True, description="是否支持竖排文本")


class LayoutParseOutput(BaseModel):
    """智能排版解析节点输出"""
    parsed_text: str = Field(default="", description="解析后的文本")
    layout_type: str = Field(default="", description="检测到的布局类型（single_column/multi_column）")
    paragraph_count: int = Field(default=0, description="段落数量")
    column_count: int = Field(default=1, description="栏数")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 忽略区域配置节点（V1.1新增）---

class IgnoreRegionInput(BaseModel):
    """忽略区域配置节点输入"""
    image: File = Field(..., description="输入图片")
    ocr_regions: List[Dict[str, Any]] = Field(default_factory=list, description="OCR识别区域列表")
    ignore_regions: List[Dict[str, int]] = Field(default_factory=list, description="忽略区域列表（x1,y1,x2,y2）")
    auto_detect_watermark: bool = Field(default=True, description="是否自动检测水印")
    watermark_threshold: float = Field(default=0.9, description="水印检测阈值")


class IgnoreRegionOutput(BaseModel):
    """忽略区域配置节点输出"""
    filtered_regions: List[Dict[str, Any]] = Field(default_factory=list, description="过滤后的OCR区域")
    filtered_text: str = Field(default="", description="过滤后的文本")
    ignored_count: int = Field(default=0, description="忽略的区域数量")
    detected_watermarks: List[Dict[str, Any]] = Field(default_factory=list, description="检测到的水印位置")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 文本后处理节点（V1.1新增）---

class TextPostProcessInput(BaseModel):
    """文本后处理节点输入"""
    text: str = Field(default="", description="输入文本")
    enable_full_half_convert: bool = Field(default=True, description="是否启用半全角转换")
    enable_spell_correct: bool = Field(default=True, description="是否启用拼写纠错")
    enable_format_normalize: bool = Field(default=True, description="是否启用格式规范化")
    enable_cleanup_whitespace: bool = Field(default=True, description="是否启用清理多余空格")


class TextPostProcessOutput(BaseModel):
    """文本后处理节点输出"""
    processed_text: str = Field(default="", description="处理后的文本")
    corrections: List[Dict[str, str]] = Field(default_factory=list, description="进行的纠错列表（原文->修正）")
    correction_count: int = Field(default=0, description="纠错数量")
    processing_steps: List[str] = Field(default_factory=list, description="执行的处理步骤")


# --- 多语言OCR支持节点（V1.1新增）---

class MultiLangOCRInput(BaseModel):
    """多语言OCR支持节点输入"""
    image: File = Field(..., description="输入图片")
    languages: List[Literal["ch", "en", "japan", "korean", "fr", "es", "ru", "de"]] = Field(default=["ch"], description="支持的语言列表")
    auto_detect: bool = Field(default=True, description="是否自动检测语言")
    use_angle_cls: bool = Field(default=True, description="是否使用角度分类")
    use_structure_analysis: bool = Field(default=False, description="是否使用结构分析")


class MultiLangOCROutput(BaseModel):
    """多语言OCR支持节点输出"""
    ocr_text: str = Field(default="", description="识别的文本")
    detected_languages: List[str] = Field(default_factory=list, description="检测到的语言")
    regions: List[Dict[str, Any]] = Field(default_factory=list, description="识别区域列表")
    confidence: float = Field(default=0.0, description="平均置信度")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 弯曲文本矫正节点（V1.1新增）---

class CurvedTextCorrectInput(BaseModel):
    """弯曲文本矫正节点输入"""
    image: File = Field(..., description="弯曲文本图片")
    correction_method: Literal["lctp", "thin_plate_spline", "polynomial"] = Field(default="lctp", description="矫正方法")
    detect_curvature: bool = Field(default=True, description="是否检测弯曲程度")


class CurvedTextCorrectOutput(BaseModel):
    """弯曲文本矫正节点输出"""
    corrected_image: File = Field(..., description="矫正后的图片")
    curvature_degree: float = Field(default=0.0, description="弯曲程度（0=平直，1=严重弯曲）")
    keypoints: List[Dict[str, float]] = Field(default_factory=list, description="检测到的关键拐点")
    is_curved: bool = Field(default=False, description="是否为弯曲文本")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# ==================== V1.2 新增节点状态定义 ====================

# --- 图像超分辨率增强节点（V1.2新增）---

class SuperResolutionEnhanceInput(BaseModel):
    """图像超分辨率增强节点输入"""
    image: File = Field(..., description="输入图片")
    model_name: Literal["EDSR", "ESPCN", "FSRCNN"] = Field(default="EDSR", description="超分辨率模型")
    scale_factor: int = Field(default=3, description="放大倍数（2/3/4）")
    target_dpi: int = Field(default=300, description="目标DPI")
    enable_sharpen: bool = Field(default=True, description="是否启用锐化")


class SuperResolutionEnhanceOutput(BaseModel):
    """图像超分辨率增强节点输出"""
    enhanced_image: File = Field(..., description="增强后的图片")
    original_size: tuple = Field(default=(0, 0), description="原始尺寸（width, height）")
    enhanced_size: tuple = Field(default=(0, 0), description="增强后尺寸（width, height）")
    scale_factor: int = Field(default=1, description="实际放大倍数")
    enhancement_score: float = Field(default=0.0, description="增强评分（0-1）")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 智能ROI切割与增强节点（V1.2新增）---

class SmartROIExtractInput(BaseModel):
    """智能ROI切割与增强节点输入"""
    image: File = Field(..., description="输入图片")
    target_fields: List[Literal["brand", "production_date", "expiry_date", "batch_number", "barcode", "specification"]] = Field(
        default_factory=list,
        description="目标字段列表"
    )
    enable_sr_enhance: bool = Field(default=True, description="是否启用超分辨率增强")
    sr_scale_factor: int = Field(default=3, description="超分辨率放大倍数")
    roi_padding: float = Field(default=0.1, description="ROI边界扩展比例（0-1）")


class SmartROIExtractOutput(BaseModel):
    """智能ROI切割与增强节点输出"""
    roi_regions: List[Dict[str, Any]] = Field(default_factory=list, description="ROI区域列表")
    enhanced_rois: List[File] = Field(default_factory=list, description="增强后的ROI图片URL列表")
    extracted_texts: List[Dict[str, str]] = Field(default_factory=list, description="提取的文本列表")
    field_classification: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="字段分类结果")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 多模态验证节点（V1.2新增）---

class MultiModalValidationInput(BaseModel):
    """多模态验证节点输入"""
    image: File = Field(..., description="输入图片")
    extracted_info: Dict[str, Any] = Field(default_factory=dict, description="已提取的信息")
    enable_logic_validation: bool = Field(default=True, description="是否启用逻辑验证")
    enable_consistency_check: bool = Field(default=True, description="是否启用一致性检查")
    validation_rules: List[str] = Field(default_factory=list, description="验证规则列表")


class MultiModalValidationOutput(BaseModel):
    """多模态验证节点输出"""
    is_valid: bool = Field(default=True, description="是否验证通过")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="验证结果详情")
    corrected_info: Dict[str, Any] = Field(default_factory=dict, description="修正后的信息")
    validation_errors: List[Dict[str, str]] = Field(default_factory=list, description="验证错误列表")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 细粒度商品识别节点（V1.2新增）---

class FineGrainedRecognitionInput(BaseModel):
    """细粒度商品识别节点输入"""
    image: File = Field(..., description="输入图片")
    detection_boxes: List[Dict[str, Any]] = Field(default_factory=list, description="检测框列表")
    ocr_results: List[Dict[str, Any]] = Field(default_factory=list, description="OCR结果列表")
    enable_multimodal_fusion: bool = Field(default=True, description="是否启用多模态融合")
    enable_barcode_recognition: bool = Field(default=True, description="是否启用条形码识别")


class FineGrainedRecognitionOutput(BaseModel):
    """细粒度商品识别节点输出"""
    recognized_products: List[Dict[str, Any]] = Field(default_factory=list, description="识别的商品列表")
    product_attributes: Dict[str, List[str]] = Field(default_factory=dict, description="商品属性（规格、年份、批次等）")
    recognition_confidence: float = Field(default=0.0, description="整体识别置信度")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# ==================== V1.3 新增节点状态定义 ====================

# --- YOLO11-OBB旋转框检测节点（V1.3新增）---

class CVOBBDetectionInput(BaseModel):
    """YOLO11-OBB旋转框检测节点输入"""
    image: File = Field(..., description="待检测图片")
    detection_threshold: float = Field(default=0.5, description="检测置信度阈值")
    use_gpu: bool = Field(default=True, description="是否使用GPU加速")
    enable_image_enhancement: bool = Field(default=True, description="是否启用图像增强")


class CVOBBDetectionOutput(BaseModel):
    """YOLO11-OBB旋转框检测节点输出"""
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list, description="检测到的对象列表")
    """对象结构：
    {
        "bbox": [cx, cy, w, h, angle],  # 旋转框（中心点坐标、宽高、角度）
        "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 四个角点坐标
        "confidence": 0.95,
        "class_id": 0,
        "class_name": "bottle",
        "is_rotated": True,  # 是否为倾斜对象
        "rotation_angle": 45.5  # 旋转角度
    }
    """
    detection_confidence: float = Field(default=0.0, description="整体检测置信度")
    rotated_count: int = Field(default=0, description="倾斜对象数量")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- PP-OCRv5多语言OCR识别节点（V1.3新增）---

class OCRRecognizeInputV2(BaseModel):
    """OCR识别节点输入（V2升级版）"""
    image: Optional[File] = Field(default=None, description="待识别图片")
    preprocessed_image: Optional[File] = Field(default=None, description="预处理后的图片（兼容字段）")
    package_image: Optional[File] = Field(default=None, description="包装图片（兼容字段）")
    # 新增参数
    auto_language_detect: bool = Field(default=True, description="自动检测语言类型")
    supported_languages: List[str] = Field(default_factory=list, description="支持的语言列表，如['ch', 'en', 'japan']")
    enable_handwriting: bool = Field(default=True, description="是否启用手写识别")
    enable_vertical_text: bool = Field(default=True, description="是否支持竖排文本")
    use_paddle_ocr_v5: bool = Field(default=True, description="是否使用PP-OCRv5模型")
    # 原有参数
    ocr_engine_type: Literal["builtin", "api", "tesseract"] = Field(default="builtin", description="OCR引擎类型")
    ocr_api_config: Optional[Dict[str, Any]] = Field(default=None, description="OCR API配置")


class OCRRecognizeOutputV2(BaseModel):
    """OCR识别节点输出（V2升级版）"""
    ocr_raw_result: str = Field(default="", description="识别的原始文本")
    raw_text: str = Field(default="", description="识别的原始文本（兼容字段）")
    ocr_confidence: float = Field(default=0.0, description="整体置信度")
    confidence: float = Field(default=0.0, description="整体置信度（兼容字段）")
    ocr_regions: List[Dict[str, Any]] = Field(default_factory=list, description="识别区域列表")
    regions: List[Dict[str, Any]] = Field(default_factory=list, description="识别区域列表（兼容字段）")
    """区域结构：
    {
        "text": "识别文本",
        "text_region": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 多边形坐标
        "confidence": 0.95,
        "language": "ch",  # 检测到的语言类型
        "is_handwriting": False,  # 是否为手写体
        "is_vertical": False  # 是否为竖排文本
    }
    """
    # 新增字段
    detected_languages: List[str] = Field(default_factory=list, description="检测到的语言类型列表")
    handwriting_ratio: float = Field(default=0.0, description="手写体占比")
    engine_used: str = Field(default="", description="使用的引擎名称和版本")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- 多语言OCR识别节点（V1.3新增）---

class MultiLanguageOCRInput(BaseModel):
    """多语言OCR输入"""
    image: File = Field(..., description="待识别图片")
    target_language: Literal["ch", "en", "japan", "korean", "french", "german", "spanish", "italian", "portuguese", "auto"] = Field(
        default="auto", description="目标语言，auto表示自动检测"
    )
    auto_detect_language: bool = Field(default=True, description="是否自动检测语言")


class MultiLanguageOCROutput(BaseModel):
    """多语言OCR输出"""
    recognized_text: str = Field(default="", description="识别的文本")
    detected_language: str = Field(default="", description="检测到的语言类型")
    confidence: float = Field(default=0.0, description="识别置信度")
    regions: List[Dict[str, Any]] = Field(default_factory=list, description="识别区域列表")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")


# --- PP-StructureV3文档解析节点（V1.3新增）---

class StructureParseInput(BaseModel):
    """文档解析节点输入"""
    image: File = Field(..., description="待解析文档图片")
    parse_mode: Literal["layout", "table", "seal", "formula", "chart"] = Field(default="layout", description="解析模式")
    export_format: Literal["markdown", "json", "html"] = Field(default="markdown", description="输出格式")
    enable_table_recognition: bool = Field(default=True, description="是否启用表格识别")
    enable_seal_recognition: bool = Field(default=True, description="是否启用印章识别")


class StructureParseOutput(BaseModel):
    """文档解析节点输出"""
    layout_blocks: List[Dict[str, Any]] = Field(default_factory=list, description="版面块列表")
    """版面块结构：
    {
        "type": "text" | "table" | "seal" | "formula" | "chart" | "image",
        "bbox": [x1, y1, x2, y2],
        "content": "文本内容或表格HTML",
        "confidence": 0.95,
        "category": "title" | "paragraph" | "table" | "figure"  # 23种版面类别之一
    }
    """
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="提取的表格列表")
    """表格结构：
    {
        "bbox": [x1, y1, x2, y2],
        "html": "<table>...</table>",  # 表格HTML
        "cells": [...],  # 单元格列表
        "confidence": 0.92
    }
    """
    seals: List[Dict[str, Any]] = Field(default_factory=list, description="提取的印章列表")
    """印章结构：
    {
        "bbox": [x1, y1, x2, y2],
        "text": "印章文字",
        "type": "official" | "private",
        "confidence": 0.88
    }
    """
    charts: List[Dict[str, Any]] = Field(default_factory=list, description="提取的图表列表")
    formulas: List[Dict[str, Any]] = Field(default_factory=list, description="提取的公式列表")
    markdown: str = Field(default="", description="Markdown格式输出")
    json_output: Dict[str, Any] = Field(default_factory=dict, description="JSON格式输出")
    processing_time: float = Field(default=0.0, description="处理耗时（秒）")

    error_message: Optional[str] = Field(default=None, description="错误信息")
