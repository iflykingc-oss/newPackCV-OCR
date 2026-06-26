#!/usr/bin/env python3
"""全局错误码模块"""
from errors.registry import (
    ErrorRegistry,
    ErrorResponse,
    make_error,
    error_response_to_dict,
    RECOVERY_HINTS
)

__all__ = [
    "ErrorRegistry",
    "ErrorResponse",
    "make_error",
    "error_response_to_dict",
    "RECOVERY_HINTS"
]