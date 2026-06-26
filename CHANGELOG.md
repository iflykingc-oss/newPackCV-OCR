# Changelog

All notable changes to PackCV-OCR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.3.0] - 2026-01-25

### 🏗 集成 (Integration) — Phase 4-7 企业级模块
- **Phase 4: 稳定性**: 断路器(`resilience/`) + 健康探针 + 优雅关停 + CLI 工具 + 性能基准 + 备份恢复
- **Phase 5: 智能化 + 企业级**: LLM缓存/小样本学习/AB测试(`intelligence/`) + SSO/RBAC(`auth_sso/`) + 多租户(`tenancy/`) + 计费引擎(`billing/`) + 数据脱敏(`security/`) + 国际化(`i18n/`) + Webhook
- **Phase 6: API 增强**: API版本管理 + GraphQL (`gql_api/`) + 限流仪表盘 + 灰度发布(`gradual_rollout/`) + Web i18n
- **Phase 7: 可观测性**: 分布式追踪(`tracing/`) + 错误码注册中心(`errors/`) + 配置热更新(`config_hotreload/`) + 数据血缘(`data_lineage/`) + 审计日志(`audit/`) + MCP Server + 流式响应

### 🔧 修复 (Fixes)
- **远程仓库合并冲突**: 解决 `origin/main` 残留的 52 处合并冲突标记 (17 个文件)。采用 V6.2.0 干净基线 + 增量引入 Phase 4-7 新模块的方式重组。
- **API 双入口问题**: `web_server.py` (V6.2.0) 与 `api/main.py` (Phase 5) 共存问题已规划整合

### 💼 商业化 (Commercialization)
- **营销页面** (`src/web/marketing_routes.py`): 公开访问的 landing / pricing / signup / demo 页面 (无需登录)
- **客户文档** (`docs/customer/`): 5 分钟快速开始 + 完整 API 参考 + 场景详解 + Webhook 集成 + 最佳实践
- **定价方案** (`docs/customer/pricing.md`): Free / Starter / Pro / Enterprise 4 档套餐 + 引擎单价 + 优惠策略
- **安全合规** (`docs/customer/security.md`): 加密 / 隔离 / 脱敏 / 合规认证 / 私有化部署
- **行业案例** (`docs/case-studies/industry-roi.md`): 5 个真实客户案例 + ROI 详细数据 (¥40-500 万/年节省)
- **Python SDK** (`sdk/python/`): 官方 Python SDK,17 个测试全部通过

### 🛠 工程化 (Engineering)
- README 顶部新增「为什么选择 VibeCoding-OCR」+ ROI 数据对比
- 19 个商业化新文件提交

---

## [6.2.0] - 2026-01-24

### 🔧 修复 (Fixes)
- **恢复误归档节点**: `category_template_node` 和 `knowledge_inference_node` 被错误归档，导致 `graph.py` 导入失败。已恢复到 `nodes/` 目录。
- **删除僵尸子图**: `packcv_graph.py` 引用了所有已归档节点，LSP 检查失败。已删除（无业务引用）。
- **清理孤立归档**: 删除 `_archived/barcode_detect_node.py` 和 `_archived/stamp_detect_node.py`（pyzbar 不可用，已合并到 `multi_channel_fusion` 内部）。

### 📚 文档 (Documentation) - 开源项目规范化
- **CONTRIBUTING.md**: 完整贡献流程（Conventional Commits、PR 流程、代码规范、测试要求）
- **CODE_OF_CONDUCT.md**: 贡献者公约（基于 Contributor Covenant 2.1）
- **SECURITY.md**: 安全政策（漏洞报告流程、响应时间、版本支持矩阵）
- **docs/architecture.md**: 完整架构详解（数据流、引擎梯级、配置链、场景检测）
- **docs/development.md**: 开发者指南（添加节点/场景/引擎的标准流程）
- **docs/api/README.md**: API 端点参考
- **docs/scenarios/README.md**: 8 场景 Schema 索引
- **examples/**: 4 个实战示例（基础图片OCR、PDF解析、多场景、租户配置）

### 🛠 工程化 (Engineering)
- **pyproject.toml 开源元数据**: 添加 `authors`/`maintainers`/`keywords`/`classifiers`/`[project.urls]`/`optional-dependencies`（dev、gpu）
- **scripts/quality_check.sh**: 提交前质量检查脚本（测试/import规范/签名/配置完整性5项）
- **.github/ISSUE_TEMPLATE/**: Bug 报告、功能请求模板 + config.yml
- **.github/PULL_REQUEST_TEMPLATE.md**: 标准 PR 模板

### ✅ 验证
- 单元测试: 42/42 通过
- 集成测试: 23/23 通过
- 质量检查: 5/5 通过
- test_run: 端到端验证通过


## [6.1.1] - 2025-07

### Added
- 整合远程V6.1代码(ocr_fusion/ocr_postprocess/table_detector/unlimited_ocr工具层)
- V6.0+V6.1全节点整合到graph.py(23节点完整拓扑)
- 图片路径4路并行: OCR+VL+条码+印章→multi_channel_fusion汇聚
- 文档路径: MinerU→场景LLM提取→直出结果
- 条件QA触发(仅user_question非空时调用qa_answer)
- input_router自动判断图片/文档→不同处理管线

### Fixed
- _SCENARIO_LLM_MAP定义恢复(远程V6.1覆盖丢失)
- tests/__init__.py旧import清理
- barcode_detect/stamp_detect独立节点归档(已合并到fusion内部)
- package_image类型改为非Optional(消除LSP类型不匹配)


## [6.0.0] - 2025-07

### Added
- **MinerU 文档解析引擎** (`src/utils/document_engines/`): 支持PDF/DOCX/PPTX/XLSX全格式输入，VLM+OCR双引擎，109语言，表格→HTML/LaTeX，阅读顺序恢复
- **输入类型路由节点** (`input_router_node`): 自动判断输入类型（Image→原有管线，Document→MinerU），有条件分支
- **文档提取节点** (`document_parse_node`): MinerU解析→场景Schema映射→LLM结构化提取
- **内嵌条码/二维码检测** (`multi_channel_fusion_node` 内置): pyzbar解码，支持QR/Code128/EAN-13/Code39等主流码制，空URL兜底
- **内嵌印章/公章检测** (`multi_channel_fusion_node` 内置): VLM多模态识别，支持圆形/椭圆/方形印章，有/无人名比对
- **PaddleOCR-VL-1.6 引擎适配器**: 0.9B参数，96.3% OmniDocBench精度，支持表格/公式/印章/图表理解
- **GraphInput 支持 document_file**: 单图+单文档+多图三种输入模式

### Changed
- 条码+印章检测从独立节点→合并到 `multi_channel_fusion` 内部（快速检测，减少图拓扑复杂度）
- `GraphInput.package_image` 改为必填 `File` 类型（带默认Default），消除类型不匹配风险
- `MultiChannelFusionInput` 新增 `package_image` + `scenario_type` 字段，支持内嵌检测
- `MultiChannelFusionOutput` 新增 `barcode_results` + `stamp_results` 字段
- 测试用例数：58→65，新增7个V6.0用例

### Dependencies
- `mineru` (文档解析引擎)
- `pyzbar` (条码/二维码解码)

### Added
- **i18n 国际化模块** (`src/utils/i18n.py`): 支持zh-CN/en/ja/ko 4种locale，覆盖错误消息/场景名/字段名/货币/日期时区
- **健康检查端点**: `/health` (liveness), `/ready` (readiness), `/metrics` (Prometheus)
- **ConfigManager IM Bot路由闭环**: 钉钉/企微回调→tenant_id解析→三级配置链→OCR执行全链路贯通
- **pytest 测试框架**: 58个测试用例（42单元+16集成），覆盖场景注册表/ConfigManager/i18n/图拓扑/Schema约束
- **CI/CD GitHub Actions**: pytest自动测试 + JSON配置校验 + Python语法检查 + src.前缀禁检
- **智能后处理节点** (`smart_postprocess_node`): 合并原知识推理+品类模板为单节点，减少1次LLM调用
- **QA条件分支**: QA节点仅在用户提问时触发，非提问场景跳过

### Changed
- 管线末端3个LLM节点合并优化：`knowledge_inference+category_template → smart_postprocess`，提速3~5s
- IM Dispatcher `route_command()` 集成真实OCR执行（原仅返回"处理中"确认）
- `OCRRequest` 新增 `locale` 字段，支持per-request语言偏好
- `web_server` 所有错误消息改用i18n，硬编码中文字符串→`get_error_message()`

### Removed
- 18个死节点文件移入 `_archived/`（占49%，减少维护认知税）

### Fixed
- ConfigManager `_SCENARIO_LLM_MAP` 未定义导致 `resolve_scenario_config()` NameError
- `tests/__init__.py` 过时import（CVPreprocessor/RuleEngine等）导致测试收集失败

---

## [5.8.1] - 2025-01

### Added
- **3个新场景Schema**: 合同(contract)/证件(id_card)/物流单(logistics)
- **场景级三级配置链**: ConfigManager `resolve_scenario_config()` 方法
- **3个新LLM配置**: `contract_extract_llm_cfg.json`, `id_card_extract_llm_cfg.json`, `logistics_extract_llm_cfg.json`

---

## [5.8.0] - 2025-01

### Added
- **5大行业场景引擎**: 包装(packaging)/金融票据(finance_receipt)/银行流水(finance_statement)/医药(pharmaceutical)/通用文档(general_document)
- **场景自动检测节点** (`scenario_detector_node`): VL多模态分类(A-H) + 关键词正则匹配融合决策
- **ScenarioPipelineFactory**: 场景级管线工厂
- **SchemaRegistry**: 8场景统一注册与关键词检测

---

## [5.7.2] - 2025-01

### Added
- **ConfigManager 统一配置管理中心**: 三级配置链（文件默认→租户DB→运行时注入）
- **配置CRUD API端点**: `/api/config/tenant/{tenant_id}`, `/api/config/summary`, `/api/config/resolve`
- **web_server** 支持tenant_id和runtime_config参数

---

## [5.7.1] - 2025-01

### Added
- **自定义模型支持**: OpenAI兼容OCR/VL适配器
- **3级配置入口**: Config文件(静态) → GraphInput运行时 → Admin API热更新
- **SmartRouter custom引擎优先级**: priority 0（最高）

---

## [5.7.0] - 2025-01

### Added
- **3大模型融合**: DeepSeek-OCR(3B) + LightOnOCR-2-1B(1B) + MiniCPM-o 8B
- **OCREngine抽象层**: base.py + smart_router.py + fallback_ocr.py + lighton_ocr.py + deepseek_ocr.py
- **VLEngine抽象层**: base.py + smart_router.py + fallback_vl.py + minicpm_vl.py
- **智能引擎梯级**: Custom(priority 0) → LightOnOCR(1) → DeepSeekOCR(2) → FallbackOCR(∞)
- **引擎配置** (`src/config/engine_adapter_cfg.json`)

---

## [5.6.0] - 2025-01

### Added
- **图像质量增强节点**: CLAHE对比度 + 维纳去模糊 + 伽马低光校正 + 透视四点校正 + 形态学阴影去除
- **弯曲文本TPS校正节点**: MSER检测 + cv2 TPS薄板样条映射 + 双路识别
- **多语言OCR增强节点**: PaddleOCR 80+语言 + CJK笔画扩展 + 阿拉伯连接 + 竖排旋转检测
- **VLM-First架构升级**: 视觉优先(0.95) → 视觉推理(0.8) → 文本推理(0.6) 三级置信度
