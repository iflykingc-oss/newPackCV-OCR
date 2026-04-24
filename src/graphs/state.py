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
    ocr_engine_type: Literal["builtin", "api"] = Field(default="builtin", description="OCR引擎类型")
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
    
    # 中间状态
    preprocessed_image: Optional[File] = Field(default=None, description="预处理后的图片")
    ocr_raw_result: Optional[str] = Field(default="", description="OCR识别的原始文本")
    ocr_confidence: Optional[float] = Field(default=0.0, description="OCR识别置信度")
    ocr_regions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="OCR识别区域列表（坐标+文本）")
    structured_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="模型结构化提取的数据")
    corrected_result: Optional[str] = Field(default="", description="智能纠错后的文本")
    qa_answer: Optional[str] = Field(default="", description="语义问答的答案")
    
    # 输出数据
    final_result: Dict[str, Any] = Field(default_factory=dict, description="最终输出结果")
    export_file_url: Optional[str] = Field(default="", description="导出文件URL（TXT/PDF/Excel）")
    success: bool = Field(default=True, description="是否成功")
    error_message: Optional[str] = Field(default=None, description="错误信息")


# ==================== 图出入参 ====================

class GraphInput(BaseModel):
    """工作流输入"""
    package_image: File = Field(..., description="上传的包装图片")
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


# ==================== 批量处理节点 ====================

class BatchProcessInput(BaseModel):
    """批量处理节点输入"""
    image_list: List[File] = Field(..., description="待处理的图片列表")
    ocr_engine_type: Literal["builtin", "api"] = Field(default="builtin", description="OCR引擎类型")
    ocr_api_config: Optional[Dict[str, Any]] = Field(default=None, description="OCR API配置")
    llm_config: Optional[Dict[str, Any]] = Field(default=None, description="模型配置")


class BatchProcessOutput(BaseModel):
    """批量处理节点输出"""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="每个图片的处理结果列表")
    summary: Dict[str, Any] = Field(default_factory=dict, description="批量处理摘要（总数、成功数、失败数等）")
    merged_export_url: Optional[str] = Field(default=None, description="合并导出文件URL")
