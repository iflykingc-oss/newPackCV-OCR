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
    processing_time: float = Field(default=0.0, description="总处理耗时")
    error_message: Optional[str] = Field(default=None, description="错误信息")
