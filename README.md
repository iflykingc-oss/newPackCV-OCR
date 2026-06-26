# VibeCoding-OCR — 智能文档/图片信息提取引擎

[![CI](https://github.com/iflykingc-oss/newPackCV-OCR/actions/workflows/ci.yml/badge.svg)](https://github.com/iflykingc-oss/newPackCV-OCR/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-22%20passed-brightgreen.svg)](./src/tests/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

> 🚀 8行业场景 × 全格式输入(图片/PDF/DOCX/PPTX/XLSX) × 多引擎融合 × VLM-First架构 × Phase 7功能优化
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

### 本地运行（开发模式）

```bash
# 运行完整流程
bash scripts/local_run.sh -m flow

# 运行单个节点
bash scripts/local_run.sh -m node -n node_name

# 启动HTTP服务
bash scripts/http_run.sh -m http -p 9001
```

### 1. 安装

```bash
git clone https://github.com/iflykingc-oss/newPackCV-OCR.git
cd newPackCV-OCR

# 使用 uv 安装（推荐）
uv sync

# 安装OCR引擎
apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-jpn tesseract-ocr-kor
pip install paddleocr
```

### 2. 启动服务

```bash
# 启动API服务器（端口 9001）
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 9001

# 或使用 Docker
docker-compose up -d
```

### 3. 调用API

```bash
# 图片识别（自动场景检测）
curl -X POST http://localhost:9001/api/v1/extract \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"input_file":{"url":"https://example.com/product.jpg"}}'

# GraphQL 查询（Phase 7 新增）
curl -X POST http://localhost:9001/graphql/ \
  -H "Content-Type: application/json" \
  -d '{"query":"{ health { live ready } providers { id name } }"}'
```

### 4. Web Demo

浏览器打开 `http://localhost:9001/` — 拖拽图片即得结构化JSON。

### 5. 健康检查

```bash
curl http://localhost:9001/health/live     # 存活探针
curl http://localhost:9001/health/ready    # 就绪探针
curl http://localhost:9001/health/startup  # 启动探针
curl http://localhost:9001/tracing/status  # 追踪状态（Phase 7）
```

---

## 🆕 Phase 7 功能优化

| 模块 | 端点 | 功能 |
|------|------|------|
| 分布式追踪 | `/tracing/status` | OpenTelemetry 集成 |
| API 文档增强 | `/docs-enhanced/*` | 错误码表 + SDK 指南 |
| 错误码标准化 | `/errors/list` | 15 个预定义错误码 |
| 配置热更新 | `/config/*` | 无重启切换 Provider |
| 数据血缘 | `/lineage/stats` | 每条记录可溯源 |
| GraphQL API | `/graphql/` | 统一查询接口 |
| 限流仪表盘 | `/monitoring/*` | 实时延迟 + TopN |

---

## 🌍 海外支持

| 能力 | 覆盖范围 |
|------|---------|
| 错误消息 | 10语种 (zh/en/ja/ko/fr/de/es/ar/ru/pt) |
| Web 后台 | 中/英/日三语言切换 |
| 币种识别 | 7种 (CNY/USD/EUR/JPY/GBP/KRW/INR) |
| OCR语言 | 109语言 (MinerU) + 80+语言 (PaddleOCR) |

---

## 📊 技术栈

| 组件 | 技术 |
|------|------|
| 工作流引擎 | LangGraph 1.2 (9节点DAG) |
| 后端 | FastAPI + uvicorn |
| 多租户 | 6层隔离 + 5级套餐 |
| 断路器 | CLOSED/OPEN/HALF_OPEN 三态机 |
| GraphQL | Ariadne |
| 追踪 | OpenTelemetry |
| 缓存 | Redis + LRU TTL |
| 测试 | pytest (22用例) |
| CI/CD | GitHub Actions |
| 容器化 | Docker / K8s Helm |

---

## 📁 目录结构

```
newPackCV-OCR/
├── config/                      # LLM配置文件
├── data/                        # Provider配置
├── src/
│   ├── api/                     # FastAPI 路由
│   │   ├── main.py              # 入口
│   │   ├── middleware/          # 鉴权
│   │   └── routes/              # 端点
│   ├── graphs/                  # LangGraph 工作流
│   │   ├── graph.py             # 主图编排
│   │   ├── state.py             # 状态定义
│   │   └── nodes/               # 节点实现
│   ├── tenancy/                 # 多租户
│   ├── intelligence/            # LLM缓存/A-B/Few-shot
│   ├── resilience/              # 断路器/优雅关停
│   ├── tracing/                 # OpenTelemetry
│   ├── errors/                  # 错误码标准化
│   ├── config_hotreload/        # 配置热更新
│   ├── data_lineage/            # 数据血缘
│   ├── gql_api/                 # GraphQL
│   ├── i18n/                    # 国际化
│   ├── web/                     # Web 后台
│   ├── tests/                   # pytest
│   └── utils/                   # 工具类
├── k8s/helm/                    # K8s Helm Chart
├── sdk/                         # Python/JS SDK
├── docs/                        # 文档
├── monitoring/                  # Prometheus 告警
├── .github/workflows/           # CI/CD
├── AGENTS.md                    # 项目索引
└── README.md                    # 本文件
```

---

## 🧪 测试

```bash
# 运行单元测试
PYTHONPATH=src pytest src/tests/unit/ -v

# 运行覆盖率
PYTHONPATH=src pytest src/tests/unit/ -v --cov=src --cov-report=term-missing
```

---

## 📚 进阶阅读

- 📐 [AGENTS.md](./AGENTS.md) — 项目结构索引 + 模块清单
- 🔌 GraphQL Schema — `/graphql/` 端点查询

---

## 🤝 贡献

欢迎贡献！提交规范遵循 [Conventional Commits](https://www.conventionalcommits.org/)。

## 📄 许可证

Apache License 2.0 — 详见 [LICENSE](./LICENSE)