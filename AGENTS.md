# PackCV-OCR 项目结构索引

## 项目概述
- **名称**: PackCV-OCR
- **功能**: 面向货架/包装场景的高精度OCR识别解决方案

### 核心特性
- 多引擎OCR识别（Tesseract/EasyOCR/PaddleOCR）
- YOLO目标检测与ROI裁切
- 图像预处理增强
- LLM智能纠错与结构化提取
- 效期检测与告警
- 报表生成与导出

## 目录结构

```
PackCV-OCR/
├── config/                  # 配置文件
├── assets/                  # 静态资源与测试图片
├── scripts/                 # 运维脚本
├── deploy/                  # 部署配置
│   ├── Dockerfile           # Docker镜像
│   ├── docker-compose.yml   # Docker Compose
│   ├── k8s-deployment.yaml  # K8s部署
│   └── nginx.conf           # Nginx配置
├── docs/                    # 文档
├── .github/workflows/       # CI/CD配置
├── src/
│   ├── core/               # 核心能力模块 ⭐
│   │   ├── __init__.py     # 核心接口定义
│   │   ├── cv/             # CV视觉层
│   │   │   ├── detector.py     # YOLO检测器
│   │   │   ├── preprocessor.py # 预处理算子
│   │   │   └── cropper.py      # ROI裁切
│   │   ├── ocr/            # OCR识别层
│   │   │   └── ocr_scheduler.py # 多引擎调度器
│   │   ├── llm/            # 融合决策层
│   │   │   ├── decision_maker.py # LLM触发器
│   │   │   └── prompts.py       # 提示词模板
│   │   └── rule_engine/    # 规则引擎
│   │       ├── validator.py    # 效期校验
│   │       └── alert.py        # 告警管理
│   ├── graphs/            # LangGraph工作流
│   │   ├── state.py       # 状态定义
│   │   ├── graph.py       # 主图编排
│   │   ├── base/          # 基础组件
│   │   │   └── base_graph.py
│   │   └── nodes/         # 节点实现
│   ├── storage/           # 存储层
│   │   ├── oss.py         # 对象存储
│   │   ├── db.py          # 数据库
│   │   └── cache.py       # 缓存
│   ├── api/               # HTTP API
│   ├── cli/               # 命令行工具
│   ├── utils/             # 工具函数
│   │   └── performance.py # 性能优化
│   └── tests/             # 测试套件
└── README.md
```

## 节点清单

| 节点名 | 文件位置 | 类型 | 功能描述 | 配置文件 |
|-------|---------|------|---------|---------|
| image_preprocess | `graphs/nodes/image_preprocess_node.py` | task | 图像预处理 | - |
| ocr_recognize | `graphs/nodes/ocr_recognize_node.py` | task | OCR识别 | - |
| intelligent_correction | `graphs/nodes/intelligent_correction_node.py` | agent | 智能纠错 | `config/correction_llm_cfg.json` |
| structure_parse | `graphs/nodes/structure_parse_node.py` | task | 结构化提取 | - |
| semantic_qa | `graphs/nodes/semantic_qa_node.py` | agent | 语义问答 | `config/qa_llm_cfg.json` |
| cv_detection | `graphs/nodes/cv_detection_node.py` | task | 目标检测 | - |
| alert_engine | `graphs/nodes/alert_engine_node.py` | task | 告警引擎 | - |
| report_generation | `graphs/nodes/report_generation_node.py` | task | 报表生成 | - |

**类型说明**: task(任务节点) / agent(大模型) / condition(条件分支) / looparray(列表循环) / loopcond(条件循环)

## 核心模块说明

### 三层融合架构

1. **CV视觉层** (`src/core/cv/`)
   - `detector.py`: YOLO目标检测，支持OBB有向边界框
   - `preprocessor.py`: 标准化预处理（去噪/CLAHE/锐化/反光去除）
   - `cropper.py`: ROI裁切（NMS去重/边缘补全）

2. **OCR识别层** (`src/core/ocr/`)
   - `ocr_scheduler.py`: 多引擎调度器
     - 优先级: Tesseract → EasyOCR → PaddleOCR
     - 健康检测与自动降级
     - 多引擎结果融合

3. **融合决策层** (`src/core/llm/`)
   - `decision_maker.py`: LLM条件触发器
     - 置信度阈值判断
     - 格式/逻辑校验
   - `prompts.py`: 标准化提示词

4. **规则引擎** (`src/core/rule_engine/`)
   - `validator.py`: 效期校验（日期格式/逻辑/临期告警）
   - `alert.py`: 告警管理（分级/去重/确认）

## 存储层

| 模块 | 文件 | 说明 |
|------|------|------|
| 对象存储 | `storage/oss.py` | S3兼容存储封装 |
| 数据库 | `storage/db.py` | PostgreSQL ORM模型 |
| 缓存 | `storage/cache.py` | Redis/内存缓存 |

## 部署方式

### Docker Compose (推荐)
```bash
cd deploy
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f deploy/k8s-deployment.yaml
```

## CLI命令

```bash
# 单图识别
packcv recognize image.jpg

# 批量识别
packcv batch ./images/ -o ./output/

# 生成报表
packcv report ./output/ -t expiry
```

## API接口

- `POST /api/v1/recognize` - 单图识别
- `POST /api/v1/recognize/batch` - 批量识别
- `GET /api/v1/alerts` - 获取告警
- `POST /api/v1/alerts/{id}/acknowledge` - 确认告警
- `POST /api/v1/reports/generate` - 生成报表

## 性能指标

- 单图OCR识别：< 2秒
- 批量处理吞吐量：> 5张/秒
- OCR识别准确率：> 90%
- 服务可用性：> 99.9%

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| DB_HOST | 数据库主机 | localhost |
| REDIS_HOST | Redis主机 | localhost |
| OSS_ENDPOINT | OSS端点 | - |
| LOG_LEVEL | 日志级别 | INFO |
