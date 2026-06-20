# PackCV — 多行业文档智能提取引擎 🚀

> **从包装到票据、从合同到药品——8大行业文档的结构化信息提取引擎**

[![LangGraph](https://img.shields.io/badge/Workflow-LangGraph-blue)](https://langchain-ai.github.io/langgraph/)
[![VLM-First](https://img.shields.io/badge/Architecture-VLM--First-green)](https://github.com/OpenBMB/MiniCPM-o)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

---

## 🌟 核心亮点

| 能力 | 说明 |
|------|------|
| **8行业场景** | 包装/金融票据/银行流水/医药/合同/证件/物流/通用 |
| **VLM-First架构** | 多模态视觉理解为主，OCR为辅，三级置信度融合 |
| **智能引擎梯级** | LightOnOCR-2-1B 🥇 → DeepSeek-OCR 🥈 → 现有引擎 🏁 自动降级 |
| **三级配置链** | 文件默认 → 租户DB → 运行时注入，零重启热切换 |
| **开放模型生态** | 任意OpenAI兼容端点（vLLM/Ollama/OpenAI/Claude/通义千问等） |
| **IM平台原生** | 飞书/钉钉/企微一键发布，多租户SaaS就绪 |

---

## 📋 支持场景

| 场景 | 场景ID | 典型用例 | 核心字段 |
|------|--------|---------|---------|
| 🏪 **商品包装** | `packaging` | 食品/日化/饮料包装 | brand, ingredients, shelf_life, nutrition_info |
| 🧾 **金融票据** | `finance_receipt` | 发票/收据/购物小票 | merchant, date, total_amount, items[] |
| 🏦 **银行流水** | `finance_statement` | 回单/对账单 | account, transactions[], balance |
| 💊 **医药包装** | `pharmaceutical` | 药品说明/包装盒 | drug_name, approval_number, batch, expiry |
| 📝 **合同协议** | `contract` | 商务/租赁/采购合同 | contract_number, parties, amount, terms[] |
| 🆔 **证件识别** | `id_card` | 身份证/护照/驾照 | id_number, name, issuing_authority |
| 📦 **物流单据** | `logistics` | 快递单/运单/装箱单 | tracking_number, sender, destination, items[] |
| 📄 **通用文档** | `general_document` | 任意文档 | key_fields{}, table_data[], parties[] |

> **自动场景检测**：上传图片 → VL模型自动识别场景类型 → 调用对应Schema → 定向提取

---

## 🏗 架构全景

```
                    ┌──────────────────────────────────────────┐
  Upload Image ──▶ │        SmartOCREngine / SmartVLEngine      │
                    │  (LightOnOCR▶DeepSeekOCR▶FallbackOCR)     │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │         Scenario Detector (8场景)          │
                    │  VL多模态分类(主) + 关键词正则(备)          │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │       Scenario Schema Registry            │
                    │  场景▶字段定义▶验证规则▶LLM模板            │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │     ConfigManager 三级配置链               │
                    │  文件(默认) → DB(租户) → Runtime(注入)     │
                    └────────────────┬─────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LangGraph 22节点工作流                                          │
│  route→detect→preprocess→enhance→curvature→quality_router       │
│  → [ocr_recognize→correct] ∥ [vl_understanding]                 │
│  → fusion→inference→template→qa→output→audit→push               │
└─────────────────────────────────────────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │         结构化 JSON 输出                    │
                    │  经过Schema验证，字段级置信度标注            │
                    └──────────────────────────────────────────┘

          ┌────────────┬─────────────┬─────────────┐
          ▼            ▼             ▼             ▼
       飞书机器人    钉钉机器人    企微机器人    REST API
```

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/your-org/packcv.git
cd packcv

# 使用 uv 安装（推荐）
uv sync

# 安装中文OCR引擎
apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-jpn tesseract-ocr-kor

# 安装PaddleOCR（可选，多语言增强）
uv add paddleocr
```

### 2. 启动服务

```bash
# 启动API服务器（端口 9000）
python src/main.py

# 或使用 Docker
docker-compose up -d
```

### 3. 调用API

```bash
# 基础识别（自动场景检测）
curl -X POST http://localhost:9000/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/product.jpg"}'

# 指定场景 + 自定义模型
curl -X POST http://localhost:9000/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/invoice.jpg",
    "custom_model": "gpt-4o",
    "ocr_engine": "smart"
  }'

# 多租户（带上tenant_id，配置自动继承）
curl -X POST http://localhost:9000/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/medicine.jpg",
    "tenant_id": "pharma_corp_a"
  }'
```

### 4. Web Demo

浏览器打开 `http://localhost:9000/demo`——拖拽图片即得结构化JSON。

---

## ⚙️ 配置体系

### 三级配置链（优先级递增）

```
Level 1: 文件配置（静态全局默认）
         config/model_extract_llm_cfg.json
         config/finance_extract_llm_cfg.json
         config/pharma_extract_llm_cfg.json
         src/config/engine_adapter_cfg.json

Level 2: 租户DB（多租户覆盖）
         POST /api/config/tenant/{tenant_id}
         {"scenario_overrides": {"finance_receipt": {"model": "gpt-4o"}}}

Level 3: 运行时注入（单次请求覆盖）
         {"custom_model_config": {"model": "claude-3-5-sonnet"}}
```

### 自定义模型接入

配置文件 `src/config/engine_adapter_cfg.json`：

```json
{
  "ocr_engines": {
    "lighton_ocr": { "priority": 1 },
    "deepseek_ocr": { "priority": 2 },
    "fallback": { "engine": "builtin" },
    "custom_engines": [
      {
        "name": "my-ocr",
        "endpoint": "http://my-vllm-server:8000/v1",
        "model": "my-ocr-model",
        "api_key": "sk-xxx",
        "priority": 0
      }
    ]
  }
}
```

支持任意 **OpenAI兼容API** 的端点：vLLM / Ollama / OpenAI / Claude / 通义千问 / DeepSeek

---

## 🧩 智能引擎梯级

| 层级 | 引擎 | 条件 | 速度 | 语言支持 |
|------|------|------|------|---------|
| 🥇 | LightOnOCR-2-1B | 有GPU或API Key | 5.71页/秒/H100 | 50+语言 |
| 🥈 | DeepSeek-OCR | 有GPU或API Key | 100 tokens超压缩 | 100+语言 |
| 🥉 | 自定义模型 | 配置了endpoint | 取决于服务 | 自定义 |
| 🏁 | Fallback引擎 | 必可用（不依赖GPU） | 即时 | Tesseract/PaddleOCR |

> 无GPU、无API Key？自动降级到🏁，零中断。

---

## 🤖 IM平台发布（SaaS多租户）

### 配置管理API

```http
# 设置租户场景配置
PUT /api/config/tenant/{tenant_id}
{
  "scenario_overrides": {
    "finance_receipt": {
      "model": "gpt-4o",
      "temperature": 0.0
    },
    "pharmaceutical": {
      "model": "claude-3-5-sonnet",
      "temperature": 0.1
    }
  }
}

# 查看配置摘要
GET /api/config/summary

# 删除租户配置（恢复默认）
DELETE /api/config/tenant/{tenant_id}
```

### 飞书/钉钉/企微机器人

1. 创建飞书自定义机器人 → 复制Webhook URL
2. 配置 `src/config/im_platform.json`
3. 用户在群聊中发送图片→机器人自动识别并返回结构化结果

---

## 📊 技术栈

| 组件 | 技术 |
|------|------|
| 工作流引擎 | LangGraph (22节点DAG) |
| 多模态VL | MiniCPM-o / VLM-First |
| OCR引擎 | LightOnOCR / DeepSeek-OCR / PaddleOCR / Tesseract |
| 后端 | Python FastAPI |
| 配置存储 | SQLite (支持迁移到PostgreSQL) |
| 容器化 | Docker / Docker Compose |
| 部署 | 支持Kubernetes |

---

## 📁 目录结构

```
packcv/
├── config/                      # LLM配置文件（8场景）
│   ├── model_extract_llm_cfg.json
│   ├── finance_extract_llm_cfg.json
│   ├── pharma_extract_llm_cfg.json
│   ├── contract_extract_llm_cfg.json
│   └── ...
├── src/
│   ├── graphs/
│   │   ├── graph.py            # 22节点主图编排
│   │   ├── state.py            # 状态定义
│   │   └── nodes/              # 节点实现
│   │       ├── scenario_detector_node.py
│   │       ├── image_quality_enhance_node.py
│   │       ├── text_curvature_correct_node.py
│   │       └── ...
│   ├── utils/
│   │   ├── ocr_engines/        # OCR引擎适配器
│   │   ├── vl_engines/         # VL引擎适配器
│   │   ├── scenario_schemas/   # 8场景Schema注册中心
│   │   ├── config_manager.py   # 三级配置链
│   │   ├── scenario_pipeline.py
│   │   └── im_platform/        # 飞书/钉钉/企微
│   ├── web_server.py           # API服务 + Admin
│   ├── main.py                 # 启动入口
│   └── static/demo.html        # Web Demo
├── docker-compose.yml
├── Dockerfile
└── AGENTS.md                   # 完整节点清单
```

---

## 🧪 验证

```bash
# 完整端到端测试
python -c "
from src.graphs.graph import main_graph
result = main_graph.invoke({
    'package_image': File(url='https://...', file_type='image'),
    'ocr_engine_type': 'smart'
})
print(result['structured_data'])
"

# 运行test_run
test_run params='{"package_image": {"url": "...", "file_type": "image"}}'
```

---

## 📄 许可证

Apache License 2.0

---

## 🤝 贡献

欢迎Issue和PR！提出新场景、新引擎接入、或改进现有Schema。