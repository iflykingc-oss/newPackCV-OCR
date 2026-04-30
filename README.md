# PackCV-OCR 完整文档

## 项目概述
PackCV-OCR 是一款面向货架/包装场景的高精度OCR识别解决方案，支持多引擎调度、智能纠错、效期检测和告警功能。

## 核心功能
- 📷 多引擎OCR识别（Tesseract/EasyOCR/PaddleOCR）
- 🔍 YOLO目标检测与ROI裁切
- ✨ 图像预处理增强
- 🤖 LLM智能纠错与结构化提取
- ⚠️ 效期检测与告警
- 📊 报表生成与导出

## 目录结构
```
newPackCV-OCR/
├── config/                  # 配置文件
├── assets/                  # 静态资源与测试图片
├── scripts/                 # 运维脚本
├── deploy/                  # 部署配置
├── docs/                    # 文档
├── src/
│   ├── core/               # 核心能力模块
│   │   ├── cv/            # CV视觉层
│   │   ├── ocr/           # OCR识别层
│   │   ├── llm/           # 融合决策层
│   │   └── rule_engine/   # 规则引擎
│   ├── graphs/            # LangGraph工作流
│   ├── storage/           # 存储层
│   ├── api/               # HTTP API
│   ├── cli/               # 命令行工具
│   ├── utils/             # 工具函数
│   └── tests/             # 测试套件
├── pyproject.toml
└── uv.lock
```

## 快速开始

### 安装
```bash
# 克隆项目
git clone https://github.com/your-org/PackCV-OCR.git
cd PackCV-OCR

# 安装依赖
uv sync

# 安装OCR引擎
apt-get install tesseract-ocr tesseract-ocr-chi-sim
```

### 使用CLI
```bash
# 单图识别
packcv recognize image.jpg

# 批量识别
packcv batch ./images/ -o ./output/

# 生成报表
packcv report ./output/ -t expiry
```

### 使用Python API
```python
from src.core.ocr.ocr_scheduler import OCRScheduler
from utils.file.file import File

scheduler = OCRScheduler()
result = scheduler.recognize(image_bytes)

print(result.raw_text)
print(result.structured_data)
```

### Docker部署
```bash
cd deploy
docker-compose up -d
```

## API文档

### 识别接口
```http
POST /api/v1/recognize
Content-Type: application/json

{
    "image_base64": "...",
    "enhance": true,
    "detect": false
}
```

### 批量识别
```http
POST /api/v1/recognize/batch
Content-Type: application/json

{
    "images": ["...", "..."],
    "enhance": true
}
```

### 获取告警
```http
GET /api/v1/alerts?status=pending&limit=100
```

## 架构设计

### 三层融合架构
1. **CV视觉层**：图像预处理、目标检测、ROI裁切
2. **OCR识别层**：多引擎调度、文本识别、置信度评分
3. **融合决策层**：结果合并、LLM条件触发、规则引擎

### OCR引擎优先级
1. Tesseract（最快，无需下载模型）
2. EasyOCR（多语言支持）
3. PaddleOCR（中文印刷体最优）

### LLM触发条件
- OCR置信度 < 0.85
- 日期格式无法解析
- 多引擎结果冲突
- 核心字段缺失

## 配置说明

### 环境变量
| 变量 | 说明 | 默认值 |
|------|------|--------|
| DB_HOST | 数据库主机 | localhost |
| DB_PORT | 数据库端口 | 5432 |
| REDIS_HOST | Redis主机 | localhost |
| OSS_ENDPOINT | OSS端点 | - |
| LOG_LEVEL | 日志级别 | INFO |

### 引擎配置
在 `config/engines.yaml` 中配置各引擎的优先级和参数。

## 测试

### 运行测试
```bash
pytest src/tests/ -v

# 性能基准测试
pytest src/tests/benchmark.py -v
```

### 准确率测试
使用标准测试集验证识别准确率：
```bash
python -m src.tests.accuracy_test --dataset ./test_dataset/
```

## 部署

### Docker Compose
```bash
docker-compose -f deploy/docker-compose.yml up -d
```

### Kubernetes
```bash
kubectl apply -f deploy/k8s-deployment.yaml
```

## 性能指标
- 单图OCR识别：< 2秒
- 批量处理吞吐量：> 5张/秒
- OCR识别准确率：> 90%
- 服务可用性：> 99.9%

## 许可证
MIT License

## 贡献指南
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request
