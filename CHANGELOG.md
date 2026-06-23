# Changelog

All notable changes to PackCV-OCR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
