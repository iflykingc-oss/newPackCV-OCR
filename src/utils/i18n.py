# -*- coding: utf-8 -*-
"""
PackCV-OCR Internationalization (i18n) Module
多语言错误信息、API响应本地化、多时区/多币种支持
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("i18n")

# ==================== 支持的语言 ====================

SUPPORTED_LOCALES = ["zh-CN", "zh-TW", "en", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar", "th", "vi"]

# ==================== 错误消息本地化 ====================

ERROR_MESSAGES: Dict[str, Dict[str, str]] = {
    # ── 认证相关 ──
    "auth_invalid_credentials": {
        "zh-CN": "用户名或密码错误",
        "zh-TW": "使用者名稱或密碼錯誤",
        "en": "Invalid username or password",
        "ja": "ユーザー名またはパスワードが正しくありません",
        "ko": "사용자 이름 또는 비밀번호가 올바르지 않습니다",
        "fr": "Nom d'utilisateur ou mot de passe incorrect",
        "de": "Ungültiger Benutzername oder Passwort",
        "es": "Nombre de usuario o contraseña incorrectos",
        "pt": "Nome de usuário ou senha incorretos",
        "ru": "Неверное имя пользователя или пароль",
        "ar": "اسم المستخدم أو كلمة المرور غير صحيحة",
        "th": "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง",
        "vi": "Tên người dùng hoặc mật khẩu không chính xác",
    },
    "auth_account_disabled": {
        "zh-CN": "账户已停用",
        "en": "Account has been disabled",
        "ja": "アカウントが無効になっています",
        "ko": "계정이 비활성화되었습니다",
        "fr": "Le compte a été désactivé",
        "de": "Das Konto wurde deaktiviert",
        "es": "La cuenta ha sido desactivada",
    },
    "auth_invalid_api_key": {
        "zh-CN": "无效的API Key",
        "en": "Invalid API Key",
        "ja": "無効なAPIキー",
        "ko": "유효하지 않은 API 키",
        "fr": "Clé API invalide",
        "de": "Ungültiger API-Schlüssel",
    },
    "auth_api_key_expired": {
        "zh-CN": "API Key已过期",
        "en": "API Key has expired",
        "ja": "APIキーの有効期限が切れています",
        "ko": "API 키가 만료되었습니다",
    },
    "auth_api_key_disabled": {
        "zh-CN": "API Key已停用",
        "en": "API Key has been disabled",
        "ja": "APIキーが無効になっています",
    },
    # ── 速率限制 ──
    "rate_limit_exceeded": {
        "zh-CN": "速率限制：每小时内最多{limit}次请求",
        "en": "Rate limit exceeded: maximum {limit} requests per hour",
        "ja": "レート制限：1時間あたり最大{limit}リクエスト",
        "ko": "속도 제한: 시간당 최대 {limit}건 요청",
        "fr": "Limite de débit dépassée : maximum {limit} requêtes par heure",
    },
    "rate_limit_free_mode": {
        "zh-CN": "免费模式速率限制：10次/小时，请注册获取API Key",
        "en": "Free mode rate limit: 10 requests/hour. Please register for an API Key",
        "ja": "無料モード制限：10リクエスト/時間。APIキーを登録してください",
        "ko": "무료 모드 제한: 시간당 10건. API 키를 등록해 주세요",
    },
    # ── OCR处理 ──
    "ocr_processing_failed": {
        "zh-CN": "OCR识别失败: {detail}",
        "en": "OCR processing failed: {detail}",
        "ja": "OCR処理に失敗しました: {detail}",
        "ko": "OCR 처리 실패: {detail}",
    },
    "ocr_upload_failed": {
        "zh-CN": "OCR识别失败: {detail}",
        "en": "OCR upload processing failed: {detail}",
        "ja": "OCRアップロード処理に失敗しました: {detail}",
    },
    "workflow_execution_failed": {
        "zh-CN": "工作流执行失败: {detail}",
        "en": "Workflow execution failed: {detail}",
        "ja": "ワークフロー実行に失敗しました: {detail}",
    },
    # ── 配置管理 ──
    "config_save_failed": {
        "zh-CN": "保存租户配置失败",
        "en": "Failed to save tenant configuration",
        "ja": "テナント設定の保存に失敗しました",
    },
    "config_unknown_platform": {
        "zh-CN": "未知平台: {platform}",
        "en": "Unknown platform: {platform}",
        "ja": "不明なプラットフォーム: {platform}",
    },
    # ── 签名验证 ──
    "signature_verification_failed": {
        "zh-CN": "签名验证失败",
        "en": "Signature verification failed",
        "ja": "署名検証に失敗しました",
    },
}


# ==================== 场景名称本地化 ====================

SCENARIO_NAMES: Dict[str, Dict[str, str]] = {
    "packaging": {
        "zh-CN": "商品包装", "en": "Product Packaging",
        "ja": "商品包装", "ko": "제품 포장",
        "fr": "Emballage de produit", "de": "Produktverpackung",
    },
    "finance_receipt": {
        "zh-CN": "金融票据", "en": "Financial Receipt",
        "ja": "金融領収書", "ko": "금융 영수증",
        "fr": "Reçu financier", "de": "Finanzbeleg",
    },
    "finance_statement": {
        "zh-CN": "银行流水单", "en": "Bank Statement",
        "ja": "銀行取引明細書", "ko": "은행 거래 명세서",
        "fr": "Relevé bancaire", "de": "Kontoauszug",
    },
    "pharmaceutical": {
        "zh-CN": "药品包装", "en": "Pharmaceutical Packaging",
        "ja": "医薬品包装", "ko": "의약품 포장",
        "fr": "Emballage pharmaceutique", "de": "Pharma-Verpackung",
    },
    "contract": {
        "zh-CN": "合同/协议", "en": "Contract/Agreement",
        "ja": "契約書/合意書", "ko": "계약서/합의서",
        "fr": "Contrat/Accord", "de": "Vertrag/Vereinbarung",
    },
    "id_card": {
        "zh-CN": "身份证件", "en": "ID Card/Passport",
        "ja": "身分証明書", "ko": "신분증",
        "fr": "Pièce d'identité", "de": "Personalausweis",
    },
    "logistics": {
        "zh-CN": "物流单/快递单", "en": "Logistics/Shipping Label",
        "ja": "物流伝票/配送ラベル", "ko": "물류/배송 라벨",
        "fr": "Étiquette logistique", "de": "Logistik-Etikett",
    },
    "general_document": {
        "zh-CN": "通用文档", "en": "General Document",
        "ja": "一般文書", "ko": "일반 문서",
        "fr": "Document général", "de": "Allgemeines Dokument",
    },
}


# ==================== 字段名称本地化 ====================

FIELD_NAMES: Dict[str, Dict[str, str]] = {
    "brand": {"zh-CN": "品牌", "en": "Brand", "ja": "ブランド", "ko": "브랜드", "fr": "Marque"},
    "product_name": {"zh-CN": "品名", "en": "Product Name", "ja": "製品名", "ko": "제품명", "fr": "Nom du produit"},
    "specification": {"zh-CN": "规格", "en": "Specification", "ja": "仕様", "ko": "사양", "fr": "Spécification"},
    "production_date": {"zh-CN": "生产日期", "en": "Production Date", "ja": "製造日", "ko": "제조일", "fr": "Date de production"},
    "shelf_life": {"zh-CN": "保质期", "en": "Shelf Life", "ja": "賞味期限", "ko": "유통기한", "fr": "Durée de conservation"},
    "ingredients": {"zh-CN": "配料/成分", "en": "Ingredients", "ja": "原材料", "ko": "성분", "fr": "Ingrédients"},
    "manufacturer": {"zh-CN": "生产商", "en": "Manufacturer", "ja": "製造元", "ko": "제조사", "fr": "Fabricant"},
    "license_number": {"zh-CN": "许可证号", "en": "License Number", "ja": "許可番号", "ko": "허가 번호", "fr": "Numéro de licence"},
    "batch_number": {"zh-CN": "批号", "en": "Batch Number", "ja": "ロット番号", "ko": "배치 번호", "fr": "Numéro de lot"},
    "standard": {"zh-CN": "执行标准", "en": "Standard", "ja": "規格", "ko": "표준", "fr": "Norme"},
    "storage_condition": {"zh-CN": "贮存条件", "en": "Storage Condition", "ja": "保存条件", "ko": "보관 조건", "fr": "Condition de stockage"},
    "usage": {"zh-CN": "使用方法", "en": "Usage", "ja": "使用方法", "ko": "사용법", "fr": "Mode d'emploi"},
    "warnings": {"zh-CN": "注意事项", "en": "Warnings", "ja": "注意事項", "ko": "주의사항", "fr": "Avertissements"},
    "nutrition_facts": {"zh-CN": "营养成分", "en": "Nutrition Facts", "ja": "栄養成分", "ko": "영양 성분", "fr": "Informations nutritionnelles"},
    "allergen": {"zh-CN": "过敏原", "en": "Allergen", "ja": "アレルゲン", "ko": "알레르겐", "fr": "Allergène"},
    "expiry_date": {"zh-CN": "到期日", "en": "Expiry Date", "ja": "消費期限", "ko": "유통기한", "fr": "Date d'expiration"},
    "barcode": {"zh-CN": "条形码", "en": "Barcode", "ja": "バーコード", "ko": "바코드", "fr": "Code-barres"},
    "origin": {"zh-CN": "原产地", "en": "Origin", "ja": "原産地", "ko": "원산지", "fr": "Origine"},
    "amount": {"zh-CN": "金额", "en": "Amount", "ja": "金額", "ko": "금액", "fr": "Montant"},
    "currency": {"zh-CN": "币种", "en": "Currency", "ja": "通貨", "ko": "통화", "fr": "Devise"},
    "contract_number": {"zh-CN": "合同编号", "en": "Contract Number", "ja": "契約番号", "ko": "계약 번호", "fr": "Numéro de contrat"},
    "tracking_number": {"zh-CN": "运单号", "en": "Tracking Number", "ja": "追跡番号", "ko": "운송장 번호", "fr": "Numéro de suivi"},
    "id_number": {"zh-CN": "证件号", "en": "ID Number", "ja": "ID番号", "ko": "ID 번호", "fr": "Numéro d'identification"},
}


# ==================== 时区支持 ====================

COMMON_TIMEZONES: Dict[str, timedelta] = {
    "UTC": timedelta(0),
    "Asia/Shanghai": timedelta(hours=8),
    "Asia/Tokyo": timedelta(hours=9),
    "Asia/Seoul": timedelta(hours=9),
    "Asia/Singapore": timedelta(hours=8),
    "Asia/Bangkok": timedelta(hours=7),
    "Asia/Kolkata": timedelta(hours=5, minutes=30),
    "America/New_York": timedelta(hours=-5),
    "America/Los_Angeles": timedelta(hours=-8),
    "America/Chicago": timedelta(hours=-6),
    "Europe/London": timedelta(0),
    "Europe/Paris": timedelta(hours=1),
    "Europe/Berlin": timedelta(hours=1),
    "Europe/Moscow": timedelta(hours=3),
    "Australia/Sydney": timedelta(hours=11),
    "Pacific/Auckland": timedelta(hours=13),
}

# 常用货币符号
CURRENCY_SYMBOLS: Dict[str, str] = {
    "CNY": "¥", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
    "KRW": "₩", "THB": "฿", "VND": "₫", "INR": "₹", "RUB": "₽",
    "BRL": "R$", "AUD": "A$", "CAD": "C$", "SGD": "S$", "HKD": "HK$",
    "TWD": "NT$", "MYR": "RM", "IDR": "Rp", "PHP": "₱", "AED": "د.إ",
}


# ==================== 核心函数 ====================

def get_error_message(error_key: str, locale: str = "zh-CN", **kwargs) -> str:
    """获取本地化错误消息"""
    messages = ERROR_MESSAGES.get(error_key, {})
    msg = messages.get(locale) or messages.get("en") or messages.get("zh-CN") or error_key
    if kwargs:
        try:
            msg = msg.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return msg


def get_scenario_name(scenario_type: str, locale: str = "zh-CN") -> str:
    """获取本地化场景名称"""
    names = SCENARIO_NAMES.get(scenario_type, {})
    return names.get(locale) or names.get("en") or scenario_type


def get_field_name(field_key: str, locale: str = "zh-CN") -> str:
    """获取本地化字段名称"""
    names = FIELD_NAMES.get(field_key, {})
    return names.get(locale) or names.get("en") or field_key


def format_datetime(dt: datetime, timezone_name: str = "UTC", fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """将datetime格式化为指定时区"""
    tz_offset = COMMON_TIMEZONES.get(timezone_name, timedelta(0))
    tz = timezone(tz_offset)
    localized = dt.astimezone(tz)
    return localized.strftime(fmt)


def format_currency(amount: float, currency: str = "CNY") -> str:
    """格式化货币金额"""
    symbol = CURRENCY_SYMBOLS.get(currency, currency)
    if currency == "JPY" or currency == "KRW":
        return f"{symbol}{int(amount):,}"
    return f"{symbol}{amount:,.2f}"


def resolve_locale(accept_language: Optional[str] = None, default: str = "zh-CN") -> str:
    """从请求头解析最佳匹配locale"""
    if not accept_language:
        return default
    # 简单解析 Accept-Language 头
    languages = []
    for part in accept_language.split(","):
        part = part.strip().split(";")[0].strip()
        if part:
            languages.append(part)

    for lang in languages:
        lang_lower = lang.lower()
        for supported in SUPPORTED_LOCALES:
            if supported.lower().startswith(lang_lower) or lang_lower.startswith(supported.lower().split("-")[0]):
                return supported

    return default
