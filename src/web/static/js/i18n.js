// PackCV Web 后台 - 国际化模块 (中/英/日)
const I18n = (() => {
  const SUPPORTED_LOCALES = ['zh-CN', 'en', 'ja'];
  const STORAGE_KEY = 'packcv_locale';
  const DEFAULT_LOCALE = 'zh-CN';

  // 翻译词典
  const DICT = {
    'zh-CN': {
      // 通用
      'app.title': 'PackCV 管理后台',
      'nav.dashboard': '仪表盘',
      'nav.tenants': '租户管理',
      'nav.usage': '用量统计',
      'nav.billing': '账单管理',
      'nav.settings': '系统设置',
      'common.save': '保存',
      'common.cancel': '取消',
      'common.delete': '删除',
      'common.edit': '编辑',
      'common.create': '创建',
      'common.search': '搜索',
      'common.loading': '加载中...',
      'common.confirm': '确认',
      'common.success': '操作成功',
      'common.error': '操作失败',
      'common.yes': '是',
      'common.no': '否',
      'common.status': '状态',
      'common.actions': '操作',
      'common.name': '名称',
      'common.type': '类型',
      'common.time': '时间',
      'common.total': '合计',
      'common.detail': '详情',
      'common.refresh': '刷新',
      'common.export': '导出',

      // 仪表盘
      'dashboard.title': '系统概览',
      'dashboard.total_tenants': '租户总数',
      'dashboard.active_tenants': '活跃租户',
      'dashboard.total_calls': '调用总量',
      'dashboard.success_rate': '成功率',
      'dashboard.avg_latency': '平均延迟',
      'dashboard.provider_status': 'Provider 状态',
      'dashboard.recent_events': '最近事件',
      'dashboard.system_health': '系统健康',

      // 租户
      'tenant.title': '租户管理',
      'tenant.create': '新建租户',
      'tenant.api_key': 'API Key',
      'tenant.tier': '套餐等级',
      'tenant.quota': '配额',
      'tenant.rpm': 'RPM 限制',
      'tenant.tpm': 'TPM 限制',
      'tenant.isolation': '隔离级别',
      'tenant.models': '可用模型',
      'tier.FREE': '免费版',
      'tier.BASIC': '基础版',
      'tier.PRO': '专业版',
      'tier.ENTERPRISE': '企业版',
      'tier.FLAGSHIP': '旗舰版',

      // 用量
      'usage.title': '用量统计',
      'usage.period': '统计周期',
      'usage.api_calls': 'API 调用数',
      'usage.tokens': 'Token 消耗',
      'usage.by_scenario': '按场景统计',
      'usage.by_model': '按模型统计',
      'usage.trend': '趋势图',

      // 账单
      'billing.title': '账单管理',
      'billing.current': '当月账单',
      'billing.history': '历史账单',
      'billing.amount': '金额',
      'billing.tax': '税费',
      'billing.total_due': '应付总额',
      'billing.payment_status': '支付状态',
      'billing.paid': '已支付',
      'billing.unpaid': '未支付',
      'billing.overdue': '逾期',

      // 设置
      'settings.title': '系统设置',
      'settings.api_key': 'API Key 配置',
      'settings.api_key_placeholder': '输入您的 API Key',
      'settings.language': '语言设置',
      'settings.theme': '主题设置',
      'settings.notification': '通知设置',
      'settings.security': '安全设置',

      // 语言名
      'lang.zh-CN': '中文',
      'lang.en': 'English',
      'lang.ja': '日本語',
    },
    'en': {
      'app.title': 'PackCV Admin',
      'nav.dashboard': 'Dashboard',
      'nav.tenants': 'Tenants',
      'nav.usage': 'Usage',
      'nav.billing': 'Billing',
      'nav.settings': 'Settings',
      'common.save': 'Save',
      'common.cancel': 'Cancel',
      'common.delete': 'Delete',
      'common.edit': 'Edit',
      'common.create': 'Create',
      'common.search': 'Search',
      'common.loading': 'Loading...',
      'common.confirm': 'Confirm',
      'common.success': 'Success',
      'common.error': 'Error',
      'common.yes': 'Yes',
      'common.no': 'No',
      'common.status': 'Status',
      'common.actions': 'Actions',
      'common.name': 'Name',
      'common.type': 'Type',
      'common.time': 'Time',
      'common.total': 'Total',
      'common.detail': 'Detail',
      'common.refresh': 'Refresh',
      'common.export': 'Export',

      'dashboard.title': 'System Overview',
      'dashboard.total_tenants': 'Total Tenants',
      'dashboard.active_tenants': 'Active Tenants',
      'dashboard.total_calls': 'Total Calls',
      'dashboard.success_rate': 'Success Rate',
      'dashboard.avg_latency': 'Avg Latency',
      'dashboard.provider_status': 'Provider Status',
      'dashboard.recent_events': 'Recent Events',
      'dashboard.system_health': 'System Health',

      'tenant.title': 'Tenant Management',
      'tenant.create': 'New Tenant',
      'tenant.api_key': 'API Key',
      'tenant.tier': 'Tier',
      'tenant.quota': 'Quota',
      'tenant.rpm': 'RPM Limit',
      'tenant.tpm': 'TPM Limit',
      'tenant.isolation': 'Isolation Level',
      'tenant.models': 'Available Models',
      'tier.FREE': 'Free',
      'tier.BASIC': 'Basic',
      'tier.PRO': 'Pro',
      'tier.ENTERPRISE': 'Enterprise',
      'tier.FLAGSHIP': 'Flagship',

      'usage.title': 'Usage Statistics',
      'usage.period': 'Period',
      'usage.api_calls': 'API Calls',
      'usage.tokens': 'Token Usage',
      'usage.by_scenario': 'By Scenario',
      'usage.by_model': 'By Model',
      'usage.trend': 'Trends',

      'billing.title': 'Billing',
      'billing.current': 'Current Bill',
      'billing.history': 'History',
      'billing.amount': 'Amount',
      'billing.tax': 'Tax',
      'billing.total_due': 'Total Due',
      'billing.payment_status': 'Payment Status',
      'billing.paid': 'Paid',
      'billing.unpaid': 'Unpaid',
      'billing.overdue': 'Overdue',

      'settings.title': 'Settings',
      'settings.api_key': 'API Key',
      'settings.api_key_placeholder': 'Enter your API Key',
      'settings.language': 'Language',
      'settings.theme': 'Theme',
      'settings.notification': 'Notifications',
      'settings.security': 'Security',

      'lang.zh-CN': '中文',
      'lang.en': 'English',
      'lang.ja': '日本語',
    },
    'ja': {
      'app.title': 'PackCV 管理画面',
      'nav.dashboard': 'ダッシュボード',
      'nav.tenants': 'テナント管理',
      'nav.usage': '使用統計',
      'nav.billing': '請求管理',
      'nav.settings': '設定',
      'common.save': '保存',
      'common.cancel': 'キャンセル',
      'common.delete': '削除',
      'common.edit': '編集',
      'common.create': '作成',
      'common.search': '検索',
      'common.loading': '読み込み中...',
      'common.confirm': '確認',
      'common.success': '成功',
      'common.error': 'エラー',
      'common.yes': 'はい',
      'common.no': 'いいえ',
      'common.status': 'ステータス',
      'common.actions': '操作',
      'common.name': '名前',
      'common.type': 'タイプ',
      'common.time': '時間',
      'common.total': '合計',
      'common.detail': '詳細',
      'common.refresh': '更新',
      'common.export': 'エクスポート',

      'dashboard.title': 'システム概要',
      'dashboard.total_tenants': 'テナント総数',
      'dashboard.active_tenants': 'アクティブテナント',
      'dashboard.total_calls': '総呼出数',
      'dashboard.success_rate': '成功率',
      'dashboard.avg_latency': '平均レイテンシ',
      'dashboard.provider_status': 'プロバイダー状況',
      'dashboard.recent_events': '最近のイベント',
      'dashboard.system_health': 'システムヘルス',

      'tenant.title': 'テナント管理',
      'tenant.create': '新規テナント',
      'tenant.api_key': 'APIキー',
      'tenant.tier': 'プラン',
      'tenant.quota': 'クォータ',
      'tenant.rpm': 'RPM制限',
      'tenant.tpm': 'TPM制限',
      'tenant.isolation': '隔離レベル',
      'tenant.models': '利用可能モデル',
      'tier.FREE': '無料',
      'tier.BASIC': 'ベーシック',
      'tier.PRO': 'プロ',
      'tier.ENTERPRISE': 'エンタープライズ',
      'tier.FLAGSHIP': 'フラッグシップ',

      'usage.title': '使用統計',
      'usage.period': '期間',
      'usage.api_calls': 'API呼出数',
      'usage.tokens': 'トークン消費',
      'usage.by_scenario': 'シナリオ別',
      'usage.by_model': 'モデル別',
      'usage.trend': 'トレンド',

      'billing.title': '請求管理',
      'billing.current': '当月請求',
      'billing.history': '履歴',
      'billing.amount': '金額',
      'billing.tax': '税額',
      'billing.total_due': '合計請求額',
      'billing.payment_status': '支払状況',
      'billing.paid': '支払済',
      'billing.unpaid': '未払',
      'billing.overdue': '延滞',

      'settings.title': '設定',
      'settings.api_key': 'APIキー設定',
      'settings.api_key_placeholder': 'APIキーを入力',
      'settings.language': '言語設定',
      'settings.theme': 'テーマ設定',
      'settings.notification': '通知設定',
      'settings.security': 'セキュリティ設定',

      'lang.zh-CN': '中文',
      'lang.en': 'English',
      'lang.ja': '日本語',
    },
  };

  let _currentLocale = DEFAULT_LOCALE;

  function _init() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && SUPPORTED_LOCALES.includes(stored)) {
      _currentLocale = stored;
    }
    _applyLocale();
  }

  function _applyLocale() {
    // 更新 HTML lang 属性
    document.documentElement.lang = _currentLocale;
    // 更新所有带 data-i18n 属性的元素
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const text = t(key);
      if (text !== key) {
        el.textContent = text;
      }
    });
    // 更新所有带 data-i18n-placeholder 的元素
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const key = el.getAttribute('data-i18n-placeholder');
      const text = t(key);
      if (text !== key) {
        el.setAttribute('placeholder', text);
      }
    });
    // 更新语言选择器
    const selector = document.getElementById('locale-selector');
    if (selector) selector.value = _currentLocale;
  }

  function t(key, params) {
    let text = (DICT[_currentLocale] && DICT[_currentLocale][key]) || key;
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        text = text.replace(new RegExp(`\\{\\{${k}\\}\\}`, 'g'), String(v));
      });
    }
    return text;
  }

  function setLocale(locale) {
    if (!SUPPORTED_LOCALES.includes(locale)) return;
    _currentLocale = locale;
    localStorage.setItem(STORAGE_KEY, locale);
    _applyLocale();
  }

  function getLocale() { return _currentLocale; }
  function getSupportedLocales() { return SUPPORTED_LOCALES; }

  // 页面加载时初始化
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _init);
  } else {
    _init();
  }

  return { t, setLocale, getLocale, getSupportedLocales };
})();
