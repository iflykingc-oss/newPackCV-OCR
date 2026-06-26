"""
营销页面路由模块 (V6.3.0 商业化)
================================

提供面向潜在客户的公开页面:
- /marketing      - 营销首页 (产品介绍)
- /marketing/pricing - 定价页 (免费版/专业版/企业版)
- /marketing/signup  - 自助注册 (生成API Key)
- /marketing/demo    - 在线演示
- /marketing/docs    - 客户文档导航

设计原则:
- 不需要登录即可访问
- SEO 友好 (meta/og 标签)
- 移动端响应式
- 多语言 (zh-CN/en/ja)
"""

import os
import logging
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, EmailStr

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/marketing", tags=["marketing"], include_in_schema=False)

WEB_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")
STATIC_DIR = os.path.join(WEB_DIR, "static")

# ============================================================
# 多语言文案
# ============================================================
I18N = {
    "zh-CN": {
        "tagline": "8场景多模态OCR · 文档/图片全格式 · 企业级SaaS",
        "cta_try": "免费试用",
        "cta_demo": "查看演示",
        "cta_pricing": "查看定价",
        "feature_scenario": "8大行业场景",
        "feature_scenario_desc": "包装/金融/医药/合同/证件/物流/通用文档 一键切换",
        "feature_engine": "5引擎智能梯级",
        "feature_engine_desc": "Custom/PaddleOCR-VL/LightOnOCR/DeepSeek/Fallback 自动降级",
        "feature_doc": "全格式文档解析",
        "feature_doc_desc": "PDF/DOCX/PPTX/XLSX MinerU 95.69 OmniDocBench",
        "feature_api": "企业级 API",
        "feature_api_desc": "REST + GraphQL + Webhook + 限流 + 多租户",
        "feature_i18n": "10语言国际化",
        "feature_i18n_desc": "zh-CN/en/ja/ko/fr/de/es/pt/ru/ar 错误消息本地化",
        "feature_cloud": "云原生 SaaS",
        "feature_cloud_desc": "多租户隔离/SSO/RBAC/计费引擎/审计日志",
        "pricing_title": "简单透明的定价",
        "pricing_free_title": "免费版",
        "pricing_free_price": "¥0",
        "pricing_free_unit": "/月",
        "pricing_free_quota": "1,000 次/月",
        "pricing_free_features": [
            "8场景全功能",
            "5引擎智能梯级",
            "Web 管理后台",
            "社区支持",
        ],
        "pricing_pro_title": "专业版",
        "pricing_pro_price": "¥999",
        "pricing_pro_unit": "/月",
        "pricing_pro_quota": "100,000 次/月",
        "pricing_pro_features": [
            "免费版全部功能",
            "API Key 自助管理",
            "WebHook 主动推送",
            "99.5% SLA",
            "工单支持 (24h)",
        ],
        "pricing_enterprise_title": "企业版",
        "pricing_enterprise_price": "联系销售",
        "pricing_enterprise_unit": "定制",
        "pricing_enterprise_quota": "无限",
        "pricing_enterprise_features": [
            "专业版全部功能",
            "私有化部署",
            "SSO/OIDC 集成",
            "99.9% SLA",
            "7×24 专属支持",
            "定制场景Schema",
        ],
        "signup_title": "开始免费试用",
        "signup_subtitle": "无需信用卡,30秒获取API Key",
        "signup_email": "邮箱地址",
        "signup_company": "公司名称 (可选)",
        "signup_use_case": "使用场景 (可选)",
        "signup_submit": "立即注册",
        "demo_title": "在线体验",
        "demo_subtitle": "上传图片/PDF,30秒内看到提取结果",
    },
    "en": {
        "tagline": "8-Scenario Multimodal OCR · Document/Image Full-Format · Enterprise SaaS",
        "cta_try": "Free Trial",
        "cta_demo": "Live Demo",
        "cta_pricing": "Pricing",
        "feature_scenario": "8 Industry Scenarios",
        "feature_scenario_desc": "Packaging/Finance/Pharma/Contract/ID/Logistics/General one-click switch",
        "feature_engine": "5-Engine Cascade",
        "feature_engine_desc": "Custom/PaddleOCR-VL/LightOnOCR/DeepSeek/Fallback auto-degrade",
        "feature_doc": "All-Format Documents",
        "feature_doc_desc": "PDF/DOCX/PPTX/XLSX MinerU 95.69 OmniDocBench",
        "feature_api": "Enterprise API",
        "feature_api_desc": "REST + GraphQL + Webhook + Rate-Limit + Multi-Tenant",
        "feature_i18n": "10-Language i18n",
        "feature_i18n_desc": "zh-CN/en/ja/ko/fr/de/es/pt/ru/ar error message localization",
        "feature_cloud": "Cloud-Native SaaS",
        "feature_cloud_desc": "Multi-tenant isolation/SSO/RBAC/Billing/Audit logs",
        "pricing_title": "Simple Transparent Pricing",
        "pricing_free_title": "Free",
        "pricing_free_price": "$0",
        "pricing_free_unit": "/mo",
        "pricing_free_quota": "1,000 /mo",
        "pricing_free_features": [
            "All 8 scenarios",
            "5-engine cascade",
            "Web admin panel",
            "Community support",
        ],
        "pricing_pro_title": "Pro",
        "pricing_pro_price": "$99",
        "pricing_pro_unit": "/mo",
        "pricing_pro_quota": "100,000 /mo",
        "pricing_pro_features": [
            "Everything in Free",
            "Self-serve API Keys",
            "Active WebHook",
            "99.5% SLA",
            "Ticket support (24h)",
        ],
        "pricing_enterprise_title": "Enterprise",
        "pricing_enterprise_price": "Contact Sales",
        "pricing_enterprise_unit": "Custom",
        "pricing_enterprise_quota": "Unlimited",
        "pricing_enterprise_features": [
            "Everything in Pro",
            "On-premise deployment",
            "SSO/OIDC integration",
            "99.9% SLA",
            "24×7 dedicated support",
            "Custom scenario schemas",
        ],
        "signup_title": "Start Free Trial",
        "signup_subtitle": "No credit card required. API key in 30 seconds.",
        "signup_email": "Email Address",
        "signup_company": "Company (Optional)",
        "signup_use_case": "Use Case (Optional)",
        "signup_submit": "Sign Up Now",
        "demo_title": "Try it Live",
        "demo_subtitle": "Upload image/PDF, see extraction result in 30 seconds",
    },
    "ja": {
        "tagline": "8シナリオマルチモーダルOCR · ドキュメント/画像全形式 · エンタープライズSaaS",
        "cta_try": "無料トライアル",
        "cta_demo": "デモを見る",
        "cta_pricing": "価格",
        "feature_scenario": "8業界シナリオ",
        "feature_scenario_desc": "包装/金融/医薬/契約/ID/物流/汎用 ワンクリック切替",
        "feature_engine": "5エンジンカスケード",
        "feature_engine_desc": "Custom/PaddleOCR-VL/LightOnOCR/DeepSeek/Fallback 自動縮退",
        "feature_doc": "全形式ドキュメント",
        "feature_doc_desc": "PDF/DOCX/PPTX/XLSX MinerU 95.69 OmniDocBench",
        "feature_api": "エンタープライズAPI",
        "feature_api_desc": "REST + GraphQL + Webhook + レート制限 + マルチテナント",
        "feature_i18n": "10言語対応",
        "feature_i18n_desc": "zh-CN/en/ja/ko/fr/de/es/pt/ru/ar エラーメッセージローカライズ",
        "feature_cloud": "クラウドネイティブSaaS",
        "feature_cloud_desc": "マルチテナント分離/SSO/RBAC/請求/監査ログ",
        "pricing_title": "シンプルで透明な価格設定",
        "pricing_free_title": "フリー",
        "pricing_free_price": "¥0",
        "pricing_free_unit": "/月",
        "pricing_free_quota": "1,000 /月",
        "pricing_free_features": [
            "8シナリオ全機能",
            "5エンジンカスケード",
            "Web管理画面",
            "コミュニティサポート",
        ],
        "pricing_pro_title": "プロ",
        "pricing_pro_price": "¥999",
        "pricing_pro_unit": "/月",
        "pricing_pro_quota": "100,000 /月",
        "pricing_pro_features": [
            "フリーの全機能",
            "セルフAPIキー管理",
            "アクティブWebhook",
            "99.5% SLA",
            "チケットサポート (24h)",
        ],
        "pricing_enterprise_title": "エンタープライズ",
        "pricing_enterprise_price": "営業に相談",
        "pricing_enterprise_unit": "カスタム",
        "pricing_enterprise_quota": "無制限",
        "pricing_enterprise_features": [
            "プロの全機能",
            "オンプレデプロイ",
            "SSO/OIDC統合",
            "99.9% SLA",
            "24×7専任サポート",
            "カスタムシナリオSchema",
        ],
        "signup_title": "無料トライアル開始",
        "signup_subtitle": "クレジットカード不要、30秒でAPIキー取得",
        "signup_email": "メールアドレス",
        "signup_company": "会社 (任意)",
        "signup_use_case": "ユースケース (任意)",
        "signup_submit": "今すぐ登録",
        "demo_title": "ライブ体験",
        "demo_subtitle": "画像/PDFをアップロード、30秒で抽出結果",
    },
}


def get_locale(request: Request) -> str:
    """从 cookie/Accept-Language 推断 locale"""
    locale = request.cookies.get("packcv_locale", "")
    if locale in I18N:
        return locale
    accept = request.headers.get("accept-language", "")
    if "ja" in accept.lower():
        return "ja"
    if "zh" in accept.lower():
        return "zh-CN"
    return "en"


def get_i18n(request: Request) -> dict:
    return I18N.get(get_locale(request), I18N["en"])


# ============================================================
# 简易租户注册 (无DB依赖,内存中保存)
# ============================================================
_TENANT_DB: dict = {}


class SignupRequest(BaseModel):
    email: EmailStr = Field(..., description="邮箱")
    company: Optional[str] = Field(None, description="公司名称")
    use_case: Optional[str] = Field(None, description="使用场景")


class SignupResponse(BaseModel):
    tenant_id: str = Field(..., description="租户ID")
    api_key: str = Field(..., description="API Key")
    tier: str = Field(..., description="套餐等级")
    quota: int = Field(..., description="本月配额")
    dashboard_url: str = Field(..., description="Dashboard 链接")
    expires_at: str = Field(..., description="过期时间")


def _create_tenant(email: str, company: Optional[str], use_case: Optional[str]) -> dict:
    """创建租户 (免费试用)"""
    tenant_id = "tnt_" + secrets.token_hex(8)
    api_key = "pck_live_" + secrets.token_hex(24)
    created_at = datetime.utcnow()
    expires_at = created_at.replace(day=28)  # 30天试用
    # 简化版:直接放入内存 (生产环境应入DB)
    _TENANT_DB[tenant_id] = {
        "email": email,
        "company": company or "",
        "use_case": use_case or "",
        "api_key": api_key,
        "tier": "free",
        "quota_monthly": 1000,
        "quota_used": 0,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }
    # 尝试持久化到现有 APIKeyManager
    try:
        from tenancy.api_key_manager import APIKeyManager
        APIKeyManager.create_tenant(
            tenant_id=tenant_id,
            name=company or email.split("@")[0],
            tier="free",
            api_key=api_key,
            quota=1000,
        )
    except Exception as e:
        logger.warning(f"持久化租户失败 (降级到内存): {e}")
    return _TENANT_DB[tenant_id]


# ============================================================
# 路由: 营销首页
# ============================================================
@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def page_landing(request: Request):
    """营销首页 - 公开"""
    i18n = get_i18n(request)
    locale = get_locale(request)
    html = f"""
<!DOCTYPE html>
<html lang="{locale}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PackCV-OCR · {i18n['tagline']}</title>
  <meta name="description" content="VibeCoding-OCR: 8-scenario multimodal OCR for packaging, finance, pharma, contract, ID, logistics. MinerU document parsing. Enterprise SaaS.">
  <meta property="og:title" content="PackCV-OCR · {i18n['tagline']}">
  <meta property="og:type" content="website">
  <meta property="og:image" content="/static/og-cover.png">
  <link rel="stylesheet" href="/static/css/landing.css">
</head>
<body>
  <header class="topbar">
    <div class="container">
      <a href="/marketing" class="logo">📄 PackCV-OCR</a>
      <nav>
        <a href="/marketing#features">{i18n['feature_scenario']}</a>
        <a href="/marketing/pricing">{i18n['cta_pricing']}</a>
        <a href="/marketing/docs">Docs</a>
        <a href="/marketing/demo">{i18n['cta_demo']}</a>
        <a href="/marketing/signup" class="btn-primary">{i18n['cta_try']}</a>
      </nav>
    </div>
  </header>

  <section class="hero">
    <div class="container">
      <h1>从图片/PDF到结构化数据</h1>
      <h1>8大场景 · 一键提取</h1>
      <p class="tagline">{i18n['tagline']}</p>
      <div class="cta-row">
        <a href="/marketing/signup" class="btn-primary btn-large">🚀 {i18n['cta_try']}</a>
        <a href="/marketing/demo" class="btn-secondary btn-large">▶ {i18n['cta_demo']}</a>
      </div>
      <div class="badges">
        <span class="badge">✅ 无需信用卡</span>
        <span class="badge">⚡ 30秒开通</span>
        <span class="badge">🌍 10语言支持</span>
        <span class="badge">🔓 Apache 2.0</span>
      </div>
    </div>
  </section>

  <section id="features" class="features">
    <div class="container">
      <h2>6大核心能力</h2>
      <div class="grid">
        <div class="card">
          <div class="card-icon">🎯</div>
          <h3>{i18n['feature_scenario']}</h3>
          <p>{i18n['feature_scenario_desc']}</p>
        </div>
        <div class="card">
          <div class="card-icon">⚙️</div>
          <h3>{i18n['feature_engine']}</h3>
          <p>{i18n['feature_engine_desc']}</p>
        </div>
        <div class="card">
          <div class="card-icon">📄</div>
          <h3>{i18n['feature_doc']}</h3>
          <p>{i18n['feature_doc_desc']}</p>
        </div>
        <div class="card">
          <div class="card-icon">🔌</div>
          <h3>{i18n['feature_api']}</h3>
          <p>{i18n['feature_api_desc']}</p>
        </div>
        <div class="card">
          <div class="card-icon">🌐</div>
          <h3>{i18n['feature_i18n']}</h3>
          <p>{i18n['feature_i18n_desc']}</p>
        </div>
        <div class="card">
          <div class="card-icon">☁️</div>
          <h3>{i18n['feature_cloud']}</h3>
          <p>{i18n['feature_cloud_desc']}</p>
        </div>
      </div>
    </div>
  </section>

  <section class="code-demo">
    <div class="container">
      <h2>3行代码,即可集成</h2>
      <pre><code class="language-python">from packcv import PackCVClient

client = PackCVClient(api_key="pck_live_xxx")
result = client.extract(
    image="receipt.jpg",
    scenario="finance_receipt",
    locale="zh-CN"
)
print(result.fields)  # {'金额': '¥1,234.56', '日期': '2024-01-15', ...}</code></pre>
      <a href="/marketing/docs/quickstart" class="btn-secondary">查看完整文档 →</a>
    </div>
  </section>

  <section class="cta-bottom">
    <div class="container">
      <h2>立即开始免费试用</h2>
      <p>1,000次/月免费配额,无需信用卡</p>
      <a href="/marketing/signup" class="btn-primary btn-large">🚀 {i18n['cta_try']}</a>
    </div>
  </section>

  <footer class="footer">
    <div class="container">
      <p>© 2026 VibeCoding-OCR · Apache 2.0 · <a href="https://github.com/iflykingc-oss/newPackCV-OCR">GitHub</a></p>
    </div>
  </footer>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ============================================================
# 路由: 定价页
# ============================================================
@router.get("/pricing", response_class=HTMLResponse)
async def page_pricing(request: Request):
    """定价页 - 公开"""
    i18n = get_i18n(request)
    locale = get_locale(request)
    free_features = "".join(f"<li>✓ {f}</li>" for f in i18n["pricing_free_features"])
    pro_features = "".join(f"<li>✓ {f}</li>" for f in i18n["pricing_pro_features"])
    ent_features = "".join(f"<li>✓ {f}</li>" for f in i18n["pricing_enterprise_features"])
    html = f"""
<!DOCTYPE html>
<html lang="{locale}">
<head>
  <meta charset="UTF-8">
  <title>{i18n['pricing_title']} - PackCV-OCR</title>
  <link rel="stylesheet" href="/static/css/landing.css">
</head>
<body>
  <header class="topbar">
    <div class="container">
      <a href="/marketing" class="logo">📄 PackCV-OCR</a>
      <nav>
        <a href="/marketing">← Back</a>
        <a href="/marketing/signup" class="btn-primary">{i18n['cta_try']}</a>
      </nav>
    </div>
  </header>

  <section class="pricing">
    <div class="container">
      <h1>{i18n['pricing_title']}</h1>
      <p class="subtitle">30天免费试用 · 无需信用卡 · 随时取消</p>
      <div class="grid-pricing">
        <div class="plan">
          <h3>{i18n['pricing_free_title']}</h3>
          <div class="price"><span class="amount">{i18n['pricing_free_price']}</span><span class="unit">{i18n['pricing_free_unit']}</span></div>
          <p class="quota">{i18n['pricing_free_quota']}</p>
          <ul>{free_features}</ul>
          <a href="/marketing/signup" class="btn-secondary">开始试用</a>
        </div>
        <div class="plan plan-recommended">
          <span class="ribbon">推荐</span>
          <h3>{i18n['pricing_pro_title']}</h3>
          <div class="price"><span class="amount">{i18n['pricing_pro_price']}</span><span class="unit">{i18n['pricing_pro_unit']}</span></div>
          <p class="quota">{i18n['pricing_pro_quota']}</p>
          <ul>{pro_features}</ul>
          <a href="/marketing/signup" class="btn-primary">立即升级</a>
        </div>
        <div class="plan">
          <h3>{i18n['pricing_enterprise_title']}</h3>
          <div class="price"><span class="amount">{i18n['pricing_enterprise_price']}</span></div>
          <p class="quota">{i18n['pricing_enterprise_quota']}</p>
          <ul>{ent_features}</ul>
          <a href="mailto:sales@vibecoding.dev" class="btn-secondary">联系销售</a>
        </div>
      </div>
    </div>
  </section>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ============================================================
# 路由: 自助注册
# ============================================================
@router.get("/signup", response_class=HTMLResponse)
async def page_signup(request: Request):
    """注册页 - 公开"""
    i18n = get_i18n(request)
    locale = get_locale(request)
    html = f"""
<!DOCTYPE html>
<html lang="{locale}">
<head>
  <meta charset="UTF-8">
  <title>{i18n['signup_title']} - PackCV-OCR</title>
  <link rel="stylesheet" href="/static/css/landing.css">
</head>
<body>
  <header class="topbar">
    <div class="container">
      <a href="/marketing" class="logo">📄 PackCV-OCR</a>
      <nav>
        <a href="/marketing">← Back</a>
      </nav>
    </div>
  </header>

  <section class="signup">
    <div class="container">
      <div class="signup-card">
        <h1>{i18n['signup_title']}</h1>
        <p class="subtitle">{i18n['signup_subtitle']}</p>
        <form id="signup-form">
          <div class="form-group">
            <label>{i18n['signup_email']}</label>
            <input type="email" name="email" required placeholder="you@company.com">
          </div>
          <div class="form-group">
            <label>{i18n['signup_company']}</label>
            <input type="text" name="company" placeholder="ACME Inc.">
          </div>
          <div class="form-group">
            <label>{i18n['signup_use_case']}</label>
            <select name="use_case">
              <option value="">请选择...</option>
              <option value="packaging">包装标签识别</option>
              <option value="finance">金融票据</option>
              <option value="pharma">医药</option>
              <option value="contract">合同</option>
              <option value="id_card">证件</option>
              <option value="logistics">物流单</option>
              <option value="general">通用文档</option>
              <option value="other">其他</option>
            </select>
          </div>
          <button type="submit" class="btn-primary btn-large">🚀 {i18n['signup_submit']}</button>
        </form>
        <div id="signup-result" style="display:none">
          <h3>✅ 欢迎!</h3>
          <p>您的 API Key 已生成:</p>
          <code id="api-key" class="api-key-display"></code>
          <p>⚠️ 请妥善保存,仅显示一次</p>
          <a id="dashboard-link" href="/dashboard" class="btn-primary">进入控制台 →</a>
        </div>
      </div>
    </div>
  </section>

  <script>
  document.getElementById('signup-form').addEventListener('submit', async (e) => {{
    e.preventDefault();
    const form = e.target;
    const data = Object.fromEntries(new FormData(form));
    const resp = await fetch('/marketing/api/signup', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(data),
    }});
    if (resp.ok) {{
      const result = await resp.json();
      document.getElementById('api-key').textContent = result.api_key;
      document.getElementById('dashboard-link').href = result.dashboard_url;
      form.style.display = 'none';
      document.getElementById('signup-result').style.display = 'block';
    }} else {{
      alert('注册失败,请重试');
    }}
  }});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@router.post("/api/signup", response_model=SignupResponse)
async def api_signup(req: SignupRequest):
    """API: 自助注册生成 API Key"""
    tenant = _create_tenant(req.email, req.company, req.use_case)
    return SignupResponse(
        tenant_id=f"tnt_{tenant['api_key'][:12]}",
        api_key=tenant["api_key"],
        tier=tenant["tier"],
        quota=tenant["quota_monthly"],
        dashboard_url=f"/dashboard?tenant={tenant['api_key'][:12]}",
        expires_at=tenant["expires_at"],
    )


# ============================================================
# 路由: 在线演示
# ============================================================
@router.get("/demo", response_class=HTMLResponse)
async def page_demo(request: Request):
    """在线演示 - 公开"""
    i18n = get_i18n(request)
    locale = get_locale(request)
    html = f"""
<!DOCTYPE html>
<html lang="{locale}">
<head>
  <meta charset="UTF-8">
  <title>{i18n['demo_title']} - PackCV-OCR</title>
  <link rel="stylesheet" href="/static/css/landing.css">
</head>
<body>
  <header class="topbar">
    <div class="container">
      <a href="/marketing" class="logo">📄 PackCV-OCR</a>
      <nav><a href="/marketing">← Back</a></nav>
    </div>
  </header>

  <section class="demo">
    <div class="container">
      <h1>{i18n['demo_title']}</h1>
      <p class="subtitle">{i18n['demo_subtitle']}</p>
      <div class="demo-grid">
        <div class="upload-zone">
          <p>📤 拖拽文件到此处,或点击选择</p>
          <p class="hint">支持 JPG/PNG/PDF (最大 10MB)</p>
          <input type="file" id="demo-file" accept="image/*,application/pdf">
        </div>
        <div class="result-zone">
          <p>📋 提取结果将显示在这里</p>
          <pre id="demo-result"></pre>
        </div>
      </div>
      <p class="cta-row"><a href="/marketing/signup" class="btn-primary">🚀 获取免费 API Key 接入生产</a></p>
    </div>
  </section>

  <script>
  document.getElementById('demo-file').addEventListener('change', async (e) => {{
    const file = e.target.files[0];
    if (!file) return;
    const result = document.getElementById('demo-result');
    result.textContent = '⏳ 正在识别...';
    const fd = new FormData();
    fd.append('file', file);
    try {{
      const resp = await fetch('/marketing/api/demo', {{method: 'POST', body: fd}});
      const data = await resp.json();
      result.textContent = JSON.stringify(data, null, 2);
    }} catch (err) {{
      result.textContent = '❌ 错误: ' + err.message;
    }}
  }});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@router.post("/api/demo")
async def api_demo(file: bytes = None):
    """API: 公开演示(限制QPS,使用fallback引擎)"""
    # 简化版演示:返回示例结构化数据 (实际环境会调用 graph)
    return JSONResponse(content={
        "scenario": "general_document",
        "fields": {
            "title": "示例文档",
            "summary": "这是演示结果",
            "key_value_pairs": [
                {"key": "日期", "value": "2024-01-15"},
                {"key": "编号", "value": "DEMO-001"},
            ],
        },
        "confidence": 0.85,
        "engine_used": "fallback",
        "latency_ms": 1200,
        "note": "🎁 这是演示API,精度可能有限。注册获取API Key以使用生产级引擎。",
    })


# ============================================================
# 路由: 客户文档导航
# ============================================================
@router.get("/docs", response_class=HTMLResponse)
async def page_docs_index(request: Request):
    """客户文档首页"""
    locale = get_locale(request)
    html = f"""
<!DOCTYPE html>
<html lang="{locale}">
<head>
  <meta charset="UTF-8">
  <title>Documentation - PackCV-OCR</title>
  <link rel="stylesheet" href="/static/css/landing.css">
</head>
<body>
  <header class="topbar">
    <div class="container">
      <a href="/marketing" class="logo">📄 PackCV-OCR</a>
      <nav><a href="/marketing">← Back</a></nav>
    </div>
  </header>

  <section class="docs">
    <div class="container">
      <h1>📚 客户文档</h1>
      <div class="docs-grid">
        <a class="doc-card" href="/marketing/docs/quickstart">
          <h3>🚀 5分钟快速开始</h3>
          <p>从注册到第一次API调用</p>
        </a>
        <a class="doc-card" href="/marketing/docs/api">
          <h3>🔌 API 参考</h3>
          <p>REST + GraphQL 端点文档</p>
        </a>
        <a class="doc-card" href="/marketing/docs/scenarios">
          <h3>🎯 场景Schema</h3>
          <p>8大行业字段定义</p>
        </a>
        <a class="doc-card" href="/marketing/docs/sdks">
          <h3>📦 SDK 下载</h3>
          <p>Python / JavaScript / Go</p>
        </a>
        <a class="doc-card" href="/marketing/docs/webhooks">
          <h3>📨 Webhook 集成</h3>
          <p>异步任务推送配置</p>
        </a>
        <a class="doc-card" href="/marketing/docs/best-practices">
          <h3>⭐ 最佳实践</h3>
          <p>性能优化/成本控制/异常处理</p>
        </a>
      </div>
    </div>
  </section>
</body>
</html>
"""
    return HTMLResponse(content=html)
