# PackCV-OCR 项目结构索引

## 项目概述
- **名称**: PackCV-OCR V6.0 (Multi-Engine OCR + Document Parsing)
- **功能**: 多行业场景OCR信息提取+SaaS平台 — 包装/金融/医药/合同/证件/物流/通用文档，支持图片+PDF+DOCX+PPTX+XLSX全格式输入
- **当前版本**: V6.0 — MinerU文档引擎/输入路由/条码印章检测/PaddleOCR-VL/表格提取

## 处理流程 (V6.0)
```
GraphInput(package_image|document_file|images)
  → route_processing → [input_router]
  ├─ [图片路径] scenario_detector → image_preprocess → image_quality_enhance
  │   → text_curvature_correct → image_quality_router
  │   ├─ [OCR通路] ocr_recognize → correct_text → model_extract
  │   ├─ [VL通路]  vl_packaging_understanding
  │   ├─ [条码]    (multi_channel_fusion内嵌pyzbar)
  │   └─ [印章]    (multi_channel_fusion内嵌VLM)
  │   → multi_channel_fusion {融合+条码+印章} → smart_postprocess
  │   ├─ [有用户提问] qa_answer → result_output → call_audit → feishu_notify → END
  │   └─ [无用户提问] result_output → call_audit → feishu_notify → END
  └─ [文档路径] document_parse(MinerU) → document_extract(LLM) → result_output → END

并行执行:
  img_quality_router → ocr + vl + barcode + stamp (4路并行)
  → [model_extract, vl_packaging_understanding] → multi_channel_fusion (等待全部完成)
```

## 节点清单

| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| route_processing | `graphs/graph.py` | condition | 路由处理模式 | single→input_router, batch→batch_process | - |
| input_router | `nodes/input_router_node.py` | condition | **V6.0新增** 输入类型路由 | image→scenario_detector, document→MinerU解析路径 | - |
| scenario_detector | `nodes/scenario_detector_node.py` | agent | 8场景自动检测（VL A-H分类+关键词融合决策） | - | `config/scenario_detect_llm_cfg.json` |
| image_preprocess | `nodes/image_preprocess_node.py` | task | 自适应增强：透视校正→倾斜→格式转换→滤波→二值化→上采样 | - | - |
| image_quality_enhance | `nodes/image_quality_enhance_node.py` | task | CLAHE+维纳去模糊+伽马低光+透视四点+阴影去除 | - | - |
| text_curvature_correct | `nodes/text_curvature_correct_node.py` | task | MSER+TPS弯曲文本校正→双路输出 | - | - |
| image_quality_router | `nodes/image_quality_router_node.py` | condition | 质量评分→管线路由+语种检测 | score→4分支 | - |
| ocr_recognize | `nodes/ocr_recognize_node.py` | task | 多引擎OCR融合(RapidOCR+PaddleOCR+Tesseract) | - | `config/rapidocr_optimized.yaml` |
| correct_text | `nodes/correct_text_node.py` | agent | LLM智能纠错 | - | `config/correct_text_llm_cfg.json` |
| model_extract | `nodes/model_extract_node.py` | agent | 场景感知结构化提取（SP动态注入场景Schema） | - | `config/model_extract_llm_cfg.json` |
| vl_packaging_understanding | `nodes/vl_packaging_understanding_node.py` | agent | VLM-First端到端理解(视觉为主+OCR为辅+三级置信度) | - | `config/vl_packaging_llm_cfg.json` |
| multi_channel_fusion | `nodes/multi_channel_fusion_node.py` | task | **V6.0增强** OCR+VL+条码+印章4路融合 | - | - |
| document_parse | `nodes/document_parse_node.py` | task | **V6.0新增** MinerU文档解析(PDF/DOCX/PPTX/XLSX→Markdown/JSON) | - | - |
| document_extract | `nodes/document_extract_node.py` | agent | **V6.0新增** 文档提取(LLM结构化提取+场景Schema映射) | - | `config/document_extract_llm_cfg.json` |
| smart_postprocess | `nodes/smart_postprocess_node.py` | agent | **V5.9合并** 知识推理+品类模板→单次LLM调用 | - | `config/smart_postprocess_llm_cfg.json` |
| qa_answer | `nodes/qa_answer_node.py` | agent | 语义问答(仅user_question非空时触发) | 有提问→qa, 无→直接输出 | `config/qa_answer_llm_cfg.json` |
| result_output | `nodes/result_output_node.py` | task | 结果输出(JSON/Excel/PDF+对象存储) | - | - |
| call_audit | `nodes/call_audit_node.py` | task | 调用审计(LRU+JSONL+Prometheus) | - | - |
| feishu_notify | `nodes/feishu_notify_node.py` | task | 飞书通知(交互式卡片推送) | - | - |
| batch_process | `nodes/batch_process_node.py` | task | 批量图片处理 | - | - |

**类型说明**: task(任务节点) / agent(大模型) / condition(条件分支) / looparray(列表循环) / loopcond(条件循环)

## 归档节点
18个已归档节点移至 `src/graphs/nodes/_archived/`，包括：cv_detection, cv_obb_detection, fine_grained_recognition, ignore_region, image_preprocess_enhance(old), layout_parse, multi_modal_validation, ocr_recognize_v5(old), parallel_processing, report_generation, roi_segmentation, smart_roi_extract, structure_parse, super_resolution_enhance, text_direction_correct, text_post_process, alert_engine, knowledge_inference(已合并入smart_postprocess), category_template(已合并入smart_postprocess)

## 8场景Schema体系

| 场景 | Schema文件 | 必填字段数 | LLM配置 |
|------|-----------|-----------|---------|
| packaging(包装) | `scenario_schemas/packaging.py` | 9 | `config/vl_packaging_llm_cfg.json` |
| finance_receipt(金融票据) | `scenario_schemas/finance.py` | 7 | `config/finance_extract_llm_cfg.json` |
| finance_statement(银行流水) | `scenario_schemas/finance.py` | 8 | `config/finance_statement_llm_cfg.json` |
| pharmaceutical(医药) | `scenario_schemas/pharma.py` | 10 | `config/pharma_extract_llm_cfg.json` |
| contract(合同) | `scenario_schemas/contract.py` | 8 | `config/contract_extract_llm_cfg.json` |
| id_card(证件) | `scenario_schemas/id_card.py` | 7 | `config/id_card_extract_llm_cfg.json` |
| logistics(物流单) | `scenario_schemas/logistics.py` | 9 | `config/logistics_extract_llm_cfg.json` |
| general_document(通用文档) | `scenario_schemas/general.py` | 3+ | `config/general_extract_llm_cfg.json` |

**Schema注册表**: `src/utils/scenario_schemas/registry.py` — `SchemaRegistry.detect_scenario()` + `SchemaRegistry.get_schema()`

## 智能引擎梯级

### OCR引擎 (`src/utils/ocr_engines/`)
| 优先级 | 引擎 | 适配器 | 说明 |
|--------|------|--------|------|
| 0 | CustomOCR | `custom_ocr.py` | 用户自定义(OpenAI兼容,最高优先) |
| 5 | Unlimited-OCR | `unlimited_ocr.py` | **V6.1新增** 百度SOTA长文档端到端解析, gundam/base双模式 |
| 10 | LightOnOCR-2-1B | `lighton_ocr.py` | 1B参数,5.71页/秒,Apache 2.0 |
| 20 | DeepSeek-OCR | `deepseek_ocr.py` | 3B参数,高精度 |
| 999 | FallbackOCR | `fallback_ocr.py` | RapidOCR/PaddleOCR/Tesseract根基保底 |
| - | SmartOCREngine | `smart_router.py` | 链式路由+失败自动降级 |

### 文档解析引擎 (`src/utils/document_engines/`) — **V6.0新增**
| 引擎 | 适配器 | 输入格式 | 输出 | 精度 |
|------|--------|---------|------|------|
| MinerU | `mineru_engine.py` | PDF/DOCX/PPTX/XLSX/Images | Markdown/JSON(含表格+布局+公式) | 95.69 OmniDocBench |

### VL引擎 (`src/utils/vl_engines/`)
| 优先级 | 引擎 | 适配器 | 说明 |
|--------|------|--------|------|
| 0 | CustomVL | `custom_vl.py` | 用户自定义(OpenAI兼容) |
| 1 | PaddleOCR-VL-1.6 | `minicpm_vl.py` | **V6.0新增** 0.9B, 109语言 |
| 2 | MiniCPM-o | `minicpm_vl.py` | 8B参数,30+语言,1.8MP图像 |
| ∞ | FallbackVL | `fallback_vl.py` | 现有VL模型根基保底 |
| - | SmartVLEngine | `smart_router.py` | 链式路由+失败自动降级 |

**引擎配置**: `src/config/engine_adapter_cfg.json`

## 三级配置链 (ConfigManager)

| 级别 | 来源 | 作用 | 实现位置 |
|------|------|------|---------|
| ① | 文件默认 | 全局默认配置 | `src/config/engine_adapter_cfg.json` |
| ② | 租户DB | 租户级覆盖 | ConfigManager SQLite `tenant_configs` |
| ③ | 运行时注入 | 单次请求覆盖 | GraphInput.custom_model_config |

**ConfigManager**: `src/utils/config_manager.py`
- `resolve(tenant_id)` — 三级合并解析
- `resolve_scenario_config(scenario, tenant_id)` — 场景级配置
- `set_tenant_config()` / `get_tenant_config()` / `delete_tenant_config()` — CRUD
- IM Bot回调自动路由: `tenant_id → resolve_tenant_config → resolve_scenario_config → 执行`

## i18n海外支持 (`src/utils/i18n.py`)

| 能力 | 说明 |
|------|------|
| 多语言错误消息 | zh/en/ja/ko/fr/de/es/ar/ru/pt 10语种 |
| 多币种 | CNY/USD/EUR/JPY/GBP/KRW/AED 自动格式化 |
| 多时区 | pytz全时区支持，默认UTC+8 |
| Unicode全覆盖 | NFC标准化 + 宽窄字符映射 |
| 检测与格式化 | `detect_language()` / `format_currency()` / `format_datetime()` |

## API端点

| 端点 | 用途 |
|------|------|
| `POST /ocr/upload` | 图片上传OCR识别 |
| `POST /bot/feishu/events` | 飞书事件回调 |
| `POST /bot/dingtalk/callback` | 钉钉回调 |
| `POST /bot/wecom/callback` | 企微回调 |
| `GET /health` | 健康检查 |
| `GET /ready` | 就绪检查 |
| `GET /metrics` | Prometheus指标 |
| `GET /api/config/tenant/{tenant_id}` | 租户配置查询 |
| `PUT /api/config/tenant/{tenant_id}` | 租户配置更新 |
| `DELETE /api/config/tenant/{tenant_id}` | 租户配置删除 |
| `GET /api/config/summary` | 配置总览 |

## 测试框架

| 类别 | 文件 | 用例数 | 覆盖 |
|------|------|--------|------|
| 单元测试 | `tests/unit/test_scenario_registry.py` | 17 | SchemaRegistry + 8场景Schema |
| 单元测试 | `tests/unit/test_i18n.py` | 12 | i18n多语言/币种/时区/Unicode |
| 单元测试 | `tests/unit/test_config_manager.py` | 9 | 三级配置链CRUD+场景解析 |
| 集成测试 | `tests/integration/test_e2e_pipeline.py` | 20 | 图编译/拓扑/状态/条件分支 |
| **合计** | | **58** | |

## CI/CD
- **GitHub Actions**: `.github/workflows/ci.yml`
- 流水线: pytest + pyright + flake8
- 触发: push/PR to main

## 版本管理
- **CHANGELOG**: `CHANGELOG.md` — V5.6~V5.9全部变更记录
- **版本标签**: V5.9 (当前)

## 关键配置文件索引

| 配置文件 | 用途 |
|---------|------|
| `config/model_extract_llm_cfg.json` | 结构化提取Agent |
| `config/vl_packaging_llm_cfg.json` | VLM-First VL理解Agent |
| `config/correct_text_llm_cfg.json` | 文本纠错Agent |
| `config/qa_answer_llm_cfg.json` | 语义问答Agent |
| `config/smart_postprocess_llm_cfg.json` | 智能后处理Agent(知识推理+品类模板合并) |
| `config/scenario_detect_llm_cfg.json` | 场景检测Agent |
| `config/finance_extract_llm_cfg.json` | 金融票据提取 |
| `config/finance_statement_llm_cfg.json` | 银行流水提取 |
| `config/pharma_extract_llm_cfg.json` | 医药提取 |
| `config/contract_extract_llm_cfg.json` | 合同提取 |
| `config/id_card_extract_llm_cfg.json` | 证件提取 |
| `config/logistics_extract_llm_cfg.json` | 物流提取 |
| `config/general_extract_llm_cfg.json` | 通用文档提取 |
| `config/rapidocr_optimized.yaml` | RapidOCR优化配置 |
| `src/config/engine_adapter_cfg.json` | 引擎适配器配置 |

## 技能使用
- 场景检测节点: 大语言模型(VL多模态分类)
- model_extract/vl_packaging/correct_text/qa_answer/smart_postprocess: 大语言模型
- 对象存储: 结果文件上传
- 飞书/钉钉/企微: IM平台消息推送
