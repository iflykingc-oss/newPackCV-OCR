## 项目概述
- **名称**: PackCV-OCR
- **功能**: 全格式文档/图片智能信息提取引擎，覆盖8行业场景

### 节点清单
| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| route_processing | `nodes/route_processing_node.py` | task | 入口路由：判断处理管线 | "full"→scenario_detector, "quick"→ocr_recognize | - |
| input_router | `nodes/input_router_node.py` | condition | 输入格式路由：图片→增强管线，文档→MinerU | "图片"→image_preprocess, "文档"→document_parse | - |
| scenario_detector | `nodes/scenario_detector_node.py` | agent | 8场景自动检测(VL分类+关键词) | A→packaging, B→finance_receipt, ... H→general | - |
| image_preprocess | `nodes/image_preprocess_node.py` | task | 图片预处理(降噪/归一化) | - | - |
| image_quality_enhance | `nodes/image_quality_enhance_node.py` | task | CLAHE+维纳去模糊+伽马+透视+阴影去除 | - | - |
| text_curvature_correct | `nodes/text_curvature_correct_node.py` | task | MSER+TPS弯曲文本校正 | - | - |
| image_quality_router | `nodes/image_quality_router_node.py` | condition | 图像质量评估路由 | "enhance"→enhance, "pass"→4路并行 | - |
| ocr_recognize | `nodes/ocr_recognize_node.py` | task | OCR文本识别(SmartRouter梯级) | - | - |
| multi_language_ocr | `nodes/multi_language_ocr_node.py` | task | 多语言OCR(80+语言) | - | - |
| multi_language_ocr_enhanced | `nodes/multi_language_ocr_enhanced_node.py` | task | 增强多语言OCR(CJK/阿拉伯/竖排) | - | - |
| correct_text | `nodes/correct_text_node.py` | task | 文本纠错 | - | - |
| vl_packaging_understanding | `nodes/vl_packaging_understanding_node.py` | agent | VLM-First多模态理解 | - | `config/vl_packaging_llm_cfg.json` |
| model_extract | `nodes/model_extract_node.py` | agent | 场景LLM提取(8场景Schema) | - | `config/model_extract_llm_cfg.json` 等 |
| document_parse | `nodes/document_parse_node.py` | agent | MinerU文档解析(PDF/DOCX/PPTX/XLSX) | - | `config/document_extract_llm_cfg.json` |
| multi_channel_fusion | `nodes/multi_channel_fusion_node.py` | task | 4路融合(OCR+VL+条码+印章)+置信度加权 | - | - |
| smart_postprocess | `nodes/smart_postprocess_node.py` | agent | 知识推理+品类模板合并(单次LLM) | - | `config/knowledge_inference_llm_cfg.json` |
| qa_answer | `nodes/qa_answer_node.py` | agent | 条件QA(仅user_question时触发) | - | `config/qa_answer_llm_cfg.json` |
| result_output | `nodes/result_output_node.py` | task | 结果输出+文件导出 | - | - |
| feishu_notify | `nodes/feishu_notify_node.py` | task | 飞书消息推送 | - | - |
| call_audit | `nodes/call_audit_node.py` | task | 调用审计记录 | - | - |
| batch_process | `nodes/batch_process_node.py` | task | 批量处理入口 | - | - |

**类型说明**: task(任务节点) / agent(大模型) / condition(条件分支) / looparray(列表循环) / loopcond(条件循环)

## 子图清单
无活跃子图

## 引擎适配器
| 类别 | 文件位置 | 引擎 |
|------|---------|------|
| OCR | `utils/ocr_engines/` | LightOnOCR / DeepSeek-OCR / PaddleOCR-VL / Custom / Fallback |
| VL | `utils/vl_engines/` | MiniCPM-o / PaddleOCR-VL / Custom / Fallback |
| 文档 | `utils/document_engines/` | MinerU / SmartDocumentRouter |

## 工具层
| 文件 | 功能 |
|------|------|
| `utils/ocr_fusion.py` | 多引擎OCR融合策略 |
| `utils/ocr_postprocess.py` | OCR后处理纠错 |
| `utils/table_detector.py` | 表格检测与结构化 |
| `utils/scenario_pipeline.py` | 场景管线工厂 |
| `utils/config_manager.py` | 三级配置链管理 |
| `utils/i18n.py` | 10语种海外支持 |

## 场景Schema
| 场景 | 文件 | 必填字段数 |
|------|------|-----------|
| packaging | `scenario_schemas/packaging.py` | 9 |
| finance_receipt | `scenario_schemas/finance.py` | 7 |
| finance_statement | `scenario_schemas/finance.py` | 8 |
| pharmaceutical | `scenario_schemas/pharma.py` | 10 |
| contract | `scenario_schemas/contract.py` | 8 |
| id_card | `scenario_schemas/id_card.py` | 7 |
| logistics | `scenario_schemas/logistics.py` | 9 |
| general_document | `scenario_schemas/general.py` | 3+ |

## 技能使用
- 节点`vl_packaging_understanding`使用大语言模型
- 节点`model_extract`使用大语言模型(8场景)
- 节点`smart_postprocess`使用大语言模型
- 节点`qa_answer`使用大语言模型(条件触发)
- 节点`document_parse`使用MinerU文档引擎+大语言模型
- 节点`scenario_detector`使用大语言模型(VL分类)

## 测试
- `src/tests/unit/` — 42个单元测试
- `src/tests/integration/` — 23个集成测试
- 总计: 65个用例
