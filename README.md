# PackCV-OCR — 智能文档/图片信息提取引擎

[![CI](https://github.com/your-org/packcv/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/packcv/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> 🚀 8行业场景 × 全格式输入(图片/PDF/DOCX/PPTX/XLSX) × 多引擎融合 × VLM-First架构
>
> 从图片和文档中提取结构化信息，覆盖包装、金融票据、银行流水、医药、合同、证件、物流、通用文档 8 大行业场景。

---

## ✨ 核心能力

### 📸 全格式输入
| 输入类型 | 处理引擎 | 能力 |
|---------|---------|------|
| 图片 (jpg/png/webp) | CLAHE增强+TPS校正+4路并行(OCR/VL/条码/印章) | 包装/证件/物流等视觉场景 |
| PDF | MinerU (95.69 OmniDocBench) | 表格提取+版面分析+阅读顺序恢复 |
| DOCX | MinerU | 原生文档解析，零幻觉 |
| PPTX/XLSX | MinerU | 演示文稿+电子表格结构化 |

### 🏭 8行业场景引擎
| 场景 | 必填字段 | 典型文档 |
|------|---------|---------|
| 📦 包装 (packaging) | 品牌/品名/规格/生产日期/保质期/厂家 | 产品包装、标签 |
| 💰 金融票据 (finance_receipt) | 票据类型/金额/日期/收付款方 | 发票、收据、回单 |
| 🏦 银行流水 (finance_statement) | 账户/币种/交易明细/余额 | 银行对账单 |
| 💊 医药 (pharmaceutical) | 通用名/规格/批准文号/生产企业 | 药品包装、说明书 |
| 📋 合同 (contract) | 合同编号/甲乙方/金额/有效期 | 劳动合同、商业合同 |
| 🪪 证件 (id_card) | 证号/姓名/性别/出生/住址 | 身份证、护照 |
| 🚚 物流 (logistics) | 运单号/寄件人/收件人/品名 | 快递单、运单 |
| 📄 通用 (general_document) | 标题/摘要/关键字 | 任意文档 |

### 🔍 增强检测能力
- **条码/二维码解码** — 包装SKU、物流追踪码、资产标签
- **印章/公章检测** — 合同真伪、金融单据有效性
- **表格结构化提取** — HTML/Markdown输出+场景字段映射
- **手写体识别** — MinerU内置109语言OCR含手写支持

---

## 🏗️ 技术架构

### VLM-First 双通道融合

```
               ┌─ input_router ─┐
               │  (格式路由)     │
               └───────┬────────┘
                       │
           ┌───────────┼───────────┐
           │                       │
      [图片路径]               [文档路径]
           │                       │
    ┌──────┴──────┐         ┌─────┴─────┐
    │ CLAHE增强   │         │  MinerU   │
    │ TPS弯曲校正 │         │  文档解析  │
    └──────┬──────┘         └─────┬─────┘
           │                       │
    ┌──────┴──────────┐           │
    │ 4路并行识别      │           │
    │ ├─ OCR文本      │           │
    │ ├─ VL多模态     │           │
    │ ├─ 条码/二维码  │           │
    │ └─ 印章检测     │           │
    └──────┬──────────┘           │
           │                       │
    ┌──────┴──────────┐           │
    │ 多通道融合       │◄──────────┘
    │ (置信度加权)     │
    └──────┬──────────┘
           │
    ┌──────┴──────────┐
    │ 智能后处理       │
    │ (知识推理+品类)  │
    └──────┬──────────┘
           │
    ┌──────┴──────────┐
    │ 场景提取(LLM)   │
    │ (8场景Schema)   │
    └──────┬──────────┘
           │
    ┌──────┴──────────┐
    │ 条件QA(可选)    │
    └──────┬──────────┘
           │
       结果输出
```

### 智能引擎梯级

| 层级 | 引擎 | 触发条件 | 语言支持 |
|------|------|---------|---------|
| 🥇 自定义模型 | 用户OpenAI兼容端点 | 配置了endpoint | 自定义 |
| 🥈 PaddleOCR-VL-1.6 | 0.9B SOTA VLM | GPU可用 | 109语言 |
| 🥉 LightOnOCR-2-1B | 1B轻量OCR | 有API Key | 50+语言 |
| 4️⃣ DeepSeek-OCR | 3B高精度OCR | 有API Key | 100+语言 |
| 🏁 Fallback | Tesseract/PaddleOCR/RapidOCR | 必可用 | 80+语言 |

> 无GPU、无API Key？自动降级到🏁，零中断。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/your-org/packcv.git
cd packcv

# 使用 uv 安装（推荐）
uv sync

# 安装OCR引擎
apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-jpn tesseract-ocr-kor
pip install paddleocr

# 安装MinerU（文档解析，可选）
uv add mineru

# 安装pyzbar（条码解码，可选）
uv add pyzbar
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
# 图片识别（自动场景检测）
curl -X POST http://localhost:9000/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/product.jpg"}'

# PDF文档解析
curl -X POST http://localhost:9000/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/invoice.pdf"}'

# 指定场景 + 自定义模型
curl -X POST http://localhost:9000/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/invoice.jpg",
    "custom_model": "gpt-4o",
    "ocr_engine": "smart"
  }'

# 多租户（配置自动继承）
curl -X POST http://localhost:9000/ocr/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/medicine.jpg",
    "tenant_id": "pharma_corp_a"
  }'
```

### 4. Web Demo

浏览器打开 `http://localhost:9000/demo`——拖拽图片即得结构化JSON。

### 5. 健康检查

```bash
curl http://localhost:9000/health       # 存活探针
curl http://localhost:9000/ready        # 就绪探针
curl http://localhost:9000/metrics      # 运行指标
```

---

## ⚙️ 配置体系

### 三级配置链（优先级递增）

```
Level 1: 文件配置（静态全局默认）
         config/model_extract_llm_cfg.json
         config/finance_extract_llm_cfg.json
         config/pharma_extract_llm_cfg.json
         ...

Level 2: 租户DB（多租户覆盖）
         PUT /api/config/tenant/{tenant_id}
         {"scenario_overrides": {"finance_receipt": {"model": "gpt-4o"}}}

Level 3: 运行时注入（单次请求覆盖）
         {"custom_model_config": {"model": "claude-3-5-sonnet"}}
```

### 自定义模型接入

配置文件 `src/config/engine_adapter_cfg.json`：

```json
{
  "ocr_engines": {
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

支持任意 **OpenAI兼容API** 端点：vLLM / Ollama / OpenAI / Claude / 通义千问 / DeepSeek

---

## 🤖 IM平台发布（SaaS多租户）

### 配置管理API

```http
# 设置租户场景配置
PUT /api/config/tenant/{tenant_id}
{
  "scenario_overrides": {
    "finance_receipt": {"model": "gpt-4o", "temperature": 0.0},
    "pharmaceutical": {"model": "claude-3-5-sonnet", "temperature": 0.1}
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

## 🌍 海外支持

| 能力 | 覆盖范围 |
|------|---------|
| 错误消息 | 10语种 (zh/en/ja/ko/fr/de/es/ar/ru/pt) |
| 币种识别 | 7种 (CNY/USD/EUR/JPY/GBP/KRW/INR) |
| 时区支持 | 全部 IANA 时区 |
| Unicode | NFC标准化 + 全字符集覆盖 |
| OCR语言 | 109语言 (MinerU) + 80+语言 (PaddleOCR) |

---

## 📊 技术栈

| 组件 | 技术 |
|------|------|
| 工作流引擎 | LangGraph (23节点DAG) |
| 文档解析 | MinerU 3.1+ (pipeline/VLM/hybrid三引擎) |
| 多模态VL | PaddleOCR-VL-1.6 / MiniCPM-o / VLM-First |
| OCR引擎 | LightOnOCR / DeepSeek-OCR / PaddleOCR / Tesseract |
| 条码解码 | pyzbar (1D/2D全格式) |
| 增强处理 | CLAHE + TPS弯曲校正 + 维纳去模糊 |
| 后端 | Python FastAPI |
| 配置存储 | SQLite (支持迁移到PostgreSQL) |
| 测试 | pytest (65用例) |
| CI/CD | GitHub Actions |
| 容器化 | Docker / Docker Compose |

---

## 📁 目录结构

```
packcv/
├── config/                      # LLM配置文件（8场景+检测）
│   ├── model_extract_llm_cfg.json
│   ├── finance_extract_llm_cfg.json
│   ├── pharma_extract_llm_cfg.json
│   ├── contract_extract_llm_cfg.json
│   ├── document_extract_llm_cfg.json
│   └── ...
├── src/
│   ├── graphs/
│   │   ├── graph.py            # 23节点主图编排
│   │   ├── state.py            # 全局+节点状态定义
│   │   └── nodes/              # 活跃节点
│   │       ├── input_router_node.py
│   │       ├── scenario_detector_node.py
│   │       ├── image_quality_enhance_node.py
│   │       ├── text_curvature_correct_node.py
│   │       ├── multi_channel_fusion_node.py  # 含条码+印章
│   │       ├── document_parse_node.py        # MinerU
│   │       ├── smart_postprocess_node.py
│   │       └── ...
│   ├── utils/
│   │   ├── document_engines/   # MinerU文档引擎适配器
│   │   ├── ocr_engines/        # OCR引擎适配器+SmartRouter
│   │   ├── vl_engines/         # VL引擎适配器+SmartRouter
│   │   ├── scenario_schemas/   # 8场景Schema注册中心
│   │   ├── config_manager.py   # 三级配置链
│   │   ├── i18n.py             # 海外多语言支持
│   │   ├── ocr_fusion.py       # OCR多引擎融合
│   │   ├── ocr_postprocess.py  # OCR后处理纠错
│   │   ├── table_detector.py   # 表格检测
│   │   └── im_platform/        # 飞书/钉钉/企微
│   ├── tests/                  # pytest (65用例)
│   ├── web_server.py           # API服务 + Admin
│   └── main.py                 # 启动入口
├── .github/workflows/ci.yml   # CI/CD
├── CHANGELOG.md                # 版本变更记录
└── AGENTS.md                   # 完整节点清单
```

---

## 🧪 测试

```bash
# 运行全部测试（65用例）
PYTHONPATH=. python -m pytest src/tests/ -v

# 仅单元测试
PYTHONPATH=. python -m pytest src/tests/unit/ -v

# 仅集成测试
PYTHONPATH=. python -m pytest src/tests/integration/ -v
```

---

## 📄 许可证

Apache License 2.0

---

## 🤝 贡献

欢迎Issue和PR！提出新场景、新引擎接入、或改进现有Schema。
