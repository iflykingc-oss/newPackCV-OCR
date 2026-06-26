// PackCV-OCR 管理后台 - API 客户端
const API = (() => {
  const baseURL = window.location.origin;
  const DEFAULT_HEADERS = { 'Content-Type': 'application/json' };

  async function request(path, options = {}) {
    const url = path.startsWith('http') ? path : `${baseURL}${path}`;
    const apiKey = localStorage.getItem('packcv_api_key') || '';
    const headers = {
      ...DEFAULT_HEADERS,
      ...(options.headers || {}),
    };
    if (apiKey && !headers['X-API-Key']) headers['X-API-Key'] = apiKey;

    try {
      const resp = await fetch(url, { ...options, headers });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || err.message || `HTTP ${resp.status}`);
      }
      return await resp.json();
    } catch (err) {
      console.error('API Error:', err);
      throw err;
    }
  }

  return {
    get: (path) => request(path, { method: 'GET' }),
    post: (path, body) => request(path, { method: 'POST', body: JSON.stringify(body) }),
    put: (path, body) => request(path, { method: 'PUT', body: JSON.stringify(body) }),
    del: (path) => request(path, { method: 'DELETE' }),

    // 业务端点
    health: () => request('/api/v1/health'),
    dashboard: () => request('/admin/dashboard'),
    tenants: () => request('/api/v1/admin/tenants'),
    createTenant: (data) => request('/api/v1/admin/tenants', { method: 'POST', body: JSON.stringify(data) }),
    deleteTenant: (apiKey) => request(`/api/v1/admin/tenants/${apiKey}`, { method: 'DELETE' }),
    usage: (tenantId, yearMonth) => request(`/api/v1/billing/usage/${tenantId}${yearMonth ? '?year_month=' + yearMonth : ''}`),
    invoice: (tenantId, yearMonth) => request(`/api/v1/billing/invoice/${tenantId}${yearMonth ? '?year_month=' + yearMonth : ''}`),
    providers: () => request('/providers'),
    providersRouting: () => request('/providers/routing'),
    providersHealth: () => request('/providers/health'),
    scenarios: () => request('/api/v1/scenarios'),
    metrics: () => request('/metrics'),
    openapiSpec: () => request('/openapi-spec'),
    webhooksList: (tenantId) => request(`/webhooks/list/${tenantId}`),
    webhookSubscribe: (data) => request('/webhooks/subscribe', { method: 'POST', body: JSON.stringify(data) }),
  };
})();

// 工具函数
const Utils = {
  formatNumber: (n) => {
    if (n == null) return '-';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(2) + 'K';
    return n.toString();
  },
  formatDate: (d) => {
    if (!d) return '-';
    const date = new Date(d);
    return date.toLocaleString('zh-CN', { hour12: false });
  },
  formatPercent: (v) => v == null ? '-' : (v * 100).toFixed(2) + '%',
  copyToClipboard: (text) => {
    navigator.clipboard.writeText(text).then(() => UI.toast('已复制', 'success'));
  },
  getTierBadge: (tier) => {
    const map = {
      'FLAGSHIP': 'badge-danger',
      'ENTERPRISE': 'badge-info',
      'PRO': 'badge-success',
      'BASIC': 'badge-warning',
      'FREE': 'badge-neutral',
    };
    return `<span class="badge ${map[tier] || 'badge-neutral'}">${tier}</span>`;
  },
};

// UI 反馈
const UI = {
  toast: (msg, type = 'info', duration = 3000) => {
    const colors = { success: '#10b981', danger: '#ef4444', warning: '#f59e0b', info: '#3b82f6' };
    const el = document.createElement('div');
    el.style.cssText = `position:fixed;top:80px;right:20px;background:${colors[type]};color:#fff;padding:0.75rem 1.25rem;border-radius:6px;box-shadow:0 4px 6px rgba(0,0,0,.1);z-index:9999;font-size:0.875rem;`;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), duration);
  },
  modal: (title, content, actions = '') => {
    const html = `<div class="modal-backdrop" onclick="if(event.target===this) UI.closeModal()">
      <div class="modal">
        <h3 style="margin-bottom:1rem;">${title}</h3>
        <div>${content}</div>
        <div style="margin-top:1.5rem;display:flex;gap:0.5rem;justify-content:flex-end;">${actions}</div>
      </div>
    </div>`;
    const div = document.createElement('div');
    div.id = 'modal-container';
    div.innerHTML = html;
    document.body.appendChild(div);
  },
  closeModal: () => {
    const m = document.getElementById('modal-container');
    if (m) m.remove();
  },
  loading: (el) => { if (el) el.innerHTML = '<div class="loading"></div>'; },
};
