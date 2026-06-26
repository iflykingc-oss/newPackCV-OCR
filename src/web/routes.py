"""
Web 管理后台路由 - 基于 Jinja2 模板
提供可视化 Dashboard、租户管理、用量分析、计费账单、系统设置页面
支持 i18n 中/英/日三语言切换
"""
import os
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 模板与静态资源路径
WEB_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")
STATIC_DIR = os.path.join(WEB_DIR, "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(tags=["web"], include_in_schema=False)

# i18n 后端词典（用于 Jinja2 模板渲染）
_I18N_DICT = {
    "zh-CN": {
        "app_title": "PackCV 管理后台",
        "page_dashboard": "系统概览",
        "page_tenants": "租户管理",
        "page_usage": "用量统计",
        "page_billing": "账单管理",
        "page_settings": "系统设置",
    },
    "en": {
        "app_title": "PackCV Admin",
        "page_dashboard": "System Overview",
        "page_tenants": "Tenant Management",
        "page_usage": "Usage Statistics",
        "page_billing": "Billing",
        "page_settings": "Settings",
    },
    "ja": {
        "app_title": "PackCV 管理画面",
        "page_dashboard": "システム概要",
        "page_tenants": "テナント管理",
        "page_usage": "使用統計",
        "page_billing": "請求管理",
        "page_settings": "設定",
    },
}

_DEFAULT_LOCALE = "zh-CN"


def _get_locale(request: Request) -> str:
    """从 cookie 或 Accept-Language 推断 locale"""
    locale = request.cookies.get("packcv_locale", "")
    if locale in _I18N_DICT:
        return locale
    accept = request.headers.get("accept-language", "")
    for loc in _I18N_DICT:
        if loc[:2].lower() in accept.lower():
            return loc
    return _DEFAULT_LOCALE


def _ctx(request: Request, active: str) -> dict:
    """构建模板上下文（含 i18n 字典）"""
    locale = _get_locale(request)
    i18n = _I18N_DICT.get(locale, _I18N_DICT[_DEFAULT_LOCALE])
    return {
        "request": request,
        "active": active,
        "locale": locale,
        "i18n": i18n,
    }


@router.get("/", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    """Dashboard 主页"""
    return templates.TemplateResponse("dashboard.html", _ctx(request, "dashboard"))


@router.get("/tenants", response_class=HTMLResponse)
async def page_tenants(request: Request):
    """租户管理页"""
    return templates.TemplateResponse("tenants.html", _ctx(request, "tenants"))


@router.get("/usage", response_class=HTMLResponse)
async def page_usage(request: Request):
    """用量分析页"""
    return templates.TemplateResponse("usage.html", _ctx(request, "usage"))


@router.get("/billing", response_class=HTMLResponse)
async def page_billing(request: Request):
    """计费账单页"""
    return templates.TemplateResponse("billing.html", _ctx(request, "billing"))


@router.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request):
    """系统设置页"""
    return templates.TemplateResponse("settings.html", _ctx(request, "settings"))


@router.post("/api/set-locale")
async def set_locale(request: Request, locale: str = Query(...)):
    """设置语言偏好（写入 cookie）"""
    if locale not in _I18N_DICT:
        locale = _DEFAULT_LOCALE
    from fastapi.responses import JSONResponse
    resp = JSONResponse({"locale": locale, "ok": True})
    resp.set_cookie("packcv_locale", locale, max_age=365 * 86400, httponly=False)
    return resp


def mount_static(app):
    """挂载静态资源（在主 app 中调用）"""
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
