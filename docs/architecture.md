# VibeCoding-OCR 架构详解

## 设计原则

1. **能力增强，非替换** — 新引擎作为智能梯级，不放弃任何旧引擎
2. **场景化优先** — 8 场景独立 Schema，自动检测
3. **VLM-First** — 多模态理解为主，OCR 为辅
4. **三级配置** — File → Tenant DB → Runtime 注入
5. **可观测性优先** — 完整日志 + 健康检查 + 指标

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       Input Layer                           │
│  Image (URL/Local)  |  PDF  |  DOCX  |  PPTX  |  XLSX     │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  input_router (V5.9)                        │
│  Auto-detect: image  vs  document → 选路                    │
└──────┬───────────────────────────────────┬──────────────────┘
       ↓                                   ↓
┌────────────── Image Path ─────────────┐  ┌──── Document Path (V6.0) ───┐
│ scenario_detector → image_preprocess   │  │ document_parse (MinerU)     │
│   → image_quality_enhance             │  │  → document_extract         │
│   → text_curvature_correct             │  │  (LLM + 场景Schema)         │
│   → image_quality_router               │  │  → result_output            │
│       ├→ barcode+stamp (内嵌fusion)    │  └────────────────────────────┘
│       ├→ ocr_recognize (并行)          │
│       ├→ vl_packaging (并行)           │
│       └→ multi_language_ocr (并行)     │
│   multi_channel_fusion (4路汇聚)       │
│   → smart_postprocess                  │
│   → result_output                      │
│       └→ qa_answer (条件触发)          │
└───────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Output: Structured JSON / Markdown            │
│  + 飞书/钉钉/企微 IM 消息推送                              │
└─────────────────────────────────────────────────────────────┘
```

## 智能引擎梯级（Smart Engine Tiers）

```
Priority 0: Custom Engine (用户配置) — 最高优先
Priority 1: PaddleOCR-VL-1.6 (SOTA, 0.9B) — V6.0
Priority 2: LightOnOCR-2-1B (1B, fast) — V5.7
Priority 3: DeepSeek-OCR (3B, 强大) — V5.7
Priority N: FallbackOCR (Tesseract/PaddleOCR/RapidOCR)
```

**降级策略**：高优先失败 → 自动降级到下一梯级，保证可用性

## 数据流（State Flow）

```
GlobalState (98字段)
  ↓
每个节点: 独立 NodeInput → 处理 → 独立 NodeOutput
  ↓
LangGraph 自动合并 NodeOutput 到 GlobalState
  ↓
下游节点从 GlobalState 读取自己需要字段
```

## 场景检测流程

```
VL 多模态分类 (A~H 8类)
  + 关键词正则匹配 (双通道)
  ↓
融合决策:
  if VL_confidence > 0.7: 采信 VL
  elif 关键词命中 > 2:     采信关键词
  else:                   default = "general_document"
  ↓
加载场景 Schema + LLM Config
  ↓
场景化 LLM 提取
```

## 三级配置链

```
① ConfigManager 加载 config/llm_cfg.json (全局默认)
  ↓
② tenant_configs 表查询 (per-tenant 覆盖)
  ↓
③ GraphInput.custom_model_config 字段 (per-request 注入)
  ↓
最终配置 = ① ⊕ ② ⊕ ③ (后者覆盖前者)
```

详细API参考 [docs/api/](./api/)。
