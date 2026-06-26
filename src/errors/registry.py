#!/usr/bin/env python3
"""全局错误码模块
功能:
- 标准化错误响应格式
- 多语言错误消息 (中/英/日)
- HTTP 状态码映射
- 错误恢复建议
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from functools import lru_cache

logger = logging.getLogger(__name__)

# 从 api_docs 导入错误码定义
try:
    from api_docs.generator import ERROR_CODES, _get_http_status
except ImportError:
    ERROR_CODES = {}
    def _get_http_status(code: int) -> int:
        return 500


class ErrorResponse(BaseModel):
    """标准化错误响应"""
    success: bool = Field(default=False, description="成功标识")
    error_code: int = Field(..., description="错误码")
    error_type: str = Field(..., description="错误类型码")
    message: str = Field(..., description="错误消息")
    locale: str = Field(default="zh", description="语言")
    details: Optional[Dict[str, Any]] = Field(default=None, description="详细信息")
    recovery_hint: Optional[str] = Field(default=None, description="恢复建议")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")


# 错误恢复建议
RECOVERY_HINTS: Dict[int, str] = {
    100001: "请检查请求参数格式，确保 JSON 有效",
    100002: "请检查必填参数是否完整",
    100003: "支持文件类型: jpg/png/pdf/docx/txt",
    
    101001: "请在请求头添加 X-API-Key",
    101002: "请联系管理员续期或更换密钥",
    101003: "请联系客服解冻租户",
    
    102001: "请降低请求频率，或升级套餐",
    102002: "请升级套餐或等待下月配额刷新",
    102003: "请等待当前请求完成后再发起新请求",
    
    103001: "请确保图片清晰度 ≥ 300dpi",
    103002: "系统将自动降级尝试其他模型，稍后重试",
    103003: "支持场景: license/invoice/contract/id_card/general",
    
    104001: "请稍后重试，若持续失败请联系客服",
    104002: "服务正在维护，预计恢复时间 10 分钟",
    104003: "服务正在熔断保护，预计恢复时间 30 秒",
}


class ErrorRegistry:
    """错误码注册表"""
    
    _registry: Dict[int, Dict[str, Any]] = ERROR_CODES
    _locale: str = "zh"
    
    @classmethod
    def set_locale(cls, locale: str):
        """设置默认语言"""
        cls._locale = locale
    
    @classmethod
    def get_message(cls, error_code: int, locale: Optional[str] = None) -> str:
        """获取错误消息"""
        loc = locale or cls._locale
        info = cls._registry.get(error_code, {})
        return info.get(loc, info.get("zh", f"未知错误 ({error_code})"))
    
    @classmethod
    def get_error_type(cls, error_code: int) -> str:
        """获取错误类型码"""
        info = cls._registry.get(error_code, {})
        return info.get("code", "UNKNOWN_ERROR")
    
    @classmethod
    def get_http_status(cls, error_code: int) -> int:
        """获取 HTTP 状态码"""
        return _get_http_status(error_code)
    
    @classmethod
    def get_recovery_hint(cls, error_code: int) -> Optional[str]:
        """获取恢复建议"""
        return RECOVERY_HINTS.get(error_code)
    
    @classmethod
    def create_response(
        cls,
        error_code: int,
        locale: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> ErrorResponse:
        """创建标准化错误响应"""
        return ErrorResponse(
            error_code=error_code,
            error_type=cls.get_error_type(error_code),
            message=cls.get_message(error_code, locale),
            locale=locale or cls._locale,
            details=details,
            recovery_hint=cls.get_recovery_hint(error_code),
            request_id=request_id
        )
    
    @classmethod
    def register_error(
        cls,
        error_code: int,
        code: str,
        messages: Dict[str, str],
        recovery_hint: Optional[str] = None
    ):
        """注册新错误码"""
        cls._registry[error_code] = {
            "code": code,
            **messages
        }
        if recovery_hint:
            RECOVERY_HINTS[error_code] = recovery_hint
    
    @classmethod
    def list_all(cls, locale: str = "zh") -> List[Dict[str, Any]]:
        """列出所有错误码"""
        result = []
        for code_int, info in cls._registry.items():
            result.append({
                "error_code": code_int,
                "error_type": info.get("code", ""),
                "message": info.get(locale, info.get("zh", "")),
                "http_status": cls.get_http_status(code_int),
                "recovery_hint": RECOVERY_HINTS.get(code_int)
            })
        return result


# 快捷函数
def make_error(error_code: int, locale: str = "zh", **kwargs) -> ErrorResponse:
    """创建错误响应"""
    return ErrorRegistry.create_response(error_code, locale=locale, **kwargs)


def error_response_to_dict(error: ErrorResponse) -> Dict[str, Any]:
    """转换为字典"""
    return error.model_dump()