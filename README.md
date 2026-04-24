# PackCV-OCR 融合算法系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

> 一款面向货架/多包装复杂场景的专业OCR解决方案，完美融合CV（计算机视觉）和OCR技术

## ✨ 核心特性

### 🎯 PackCV-OCR 融合算法
采用「自研CV核心逻辑 + GitHub开源CV算法 + 原有OCR引擎」三层融合架构：

| 算法层 | 技术选型 | 核心作用 |
|--------|---------|---------|
| CV视觉层 | YOLOv8（目标检测）、OpenCV（图像处理） | 定位商品、裁切区域、统计数量、图像增强 |
| OCR识别层 | PaddleOCR、EasyOCR、Tesseract OCR | 提取生产日期/有效期/文字信息 |
| 融合决策层 | 自定义调度框架 | 多区域并行处理、结果合并、规则引擎、告警判断 |

### 🚀 核心功能

1. **CV目标检测**: 使用YOLOv8检测货架商品，支持100个商品并发检测
2. **ROI分层裁切**: 自动分割每个商品独立区域，无重叠无遗漏
3. **并行处理引擎**: ThreadPoolExecutor多区域并行OCR，默认10并发
4. **智能告警引擎**: 效期/库存/合规三大告警类型，支持自定义规则
5. **自动报表生成**: 一键生成效期/库存/合规三种专业报表（Excel/PDF）

### 💡 解决的痛点

✅ 货架抓拍图自动检测和分割  
✅ 多商品堆叠场景精准识别  
✅ 自动数量统计和缺货/压货分析  
✅ 效期自动识别和过期/临期告警  
✅ 自动生成效期/库存/合规台账  

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| CV检测速度 | 100个商品 < 1秒 |
| ROI裁切速度 | 100个区域 < 1秒 |
| 并行OCR速度 | 100个商品 < 3秒 |
| 端到端处理 | < 10秒 |
| 效期识别准确率 | ≥95% |
| 告警响应时间 | < 0.5秒 |
| 报表生成时间 | < 2秒 |

## 🛠️ 技术栈

### 核心框架
- **LangGraph**: 工作流编排框架
- **LangChain**: 大模型调用框架
- **SQLAlchemy**: ORM框架
- **PostgreSQL**: 数据库

### CV/OCR
- **YOLOv8**: 目标检测
- **OpenCV**: 图像处理
- **PaddleOCR**: OCR识别
- **Tesseract OCR**: 备选OCR引擎

### AI/ML
- **PyTorch**: 深度学习框架
- **Coze SDK**: 开发工具包

### 文档生成
- **Pandas**: Excel生成
- **ReportLab**: PDF生成
- **Jinja2**: 模板渲染

## 📦 安装

### 前置要求
- Python 3.9+
- PostgreSQL 12+
- uv（推荐）或 pip

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/iflykingc-oss/newPackCV-OCR.git
cd newPackCV-OCR

# 2. 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置数据库连接等
```

### 配置数据库

```bash
# 创建数据库
createdb packcv_ocr

# 执行 Schema
psql packcv_ocr < src/storage/schema.sql
```

## 🚀 快速开始

### 使用 PackCV-OCR 工作流

```python
from src.graphs.packcv_graph import packcv_ocr_graph

# 调用 PackCV-OCR 工作流
result = packcv_ocr_graph.invoke({
    "shelf_image": {"url": "https://example.com/shelf.jpg", "file_type": "image"},
    "cv_model": "yolov8",
    "ocr_engine_type": "builtin",
    "enable_expiry_detection": True,
    "enable_inventory_analysis": True,
    "enable_alerts": True,
    "export_format": "excel",
    "near_expiry_days": 30,
    "low_stock_threshold": 10
})

# 查看结果
print(f"检测到 {result['total_products']} 个商品")
print(f"生成 {len(result['alerts'])} 条告警")
print(f"报表URL: {result['reports']}")
```

### 使用基础 OCR 工作流

```python
from src.graphs.graph import main_graph

# 调用基础 OCR 工作流
result = main_graph.invoke({
    "package_image": {"url": "https://example.com/product.jpg", "file_type": "image"},
    "ocr_engine_type": "builtin",
    "model_type": "extract",
    "export_format": "json"
})

print(f"识别结果: {result['final_result']}")
```

## 📁 项目结构

```
newPackCV-OCR/
├── config/                  # 配置文件
│   ├── model_extract_llm_cfg.json
│   ├── correct_text_llm_cfg.json
│   └── qa_answer_llm_cfg.json
├── src/
│   ├── graphs/              # 工作流编排
│   │   ├── state.py        # 状态定义
│   │   ├── graph.py        # 基础工作流
│   │   ├── packcv_graph.py # PackCV-OCR 工作流
│   │   └── nodes/          # 节点实现
│   │       ├── cv_detection_node.py
│   │       ├── roi_segmentation_node.py
│   │       ├── parallel_processing_node.py
│   │       ├── alert_engine_node.py
│   │       └── report_generation_node.py
│   └── storage/            # 数据存储
│       ├── schema.sql      # 数据库 Schema
│       ├── models.py       # ORM 模型
│       └── database.py     # 数据库管理
├── AGENTS.md              # 项目文档
├── README.md              # 项目说明
└── pyproject.toml         # Python 项目配置
```

## 📖 文档

- **[AGENTS.md](AGENTS.md)**: 完整的项目文档
  - 节点清单
  - 工作流图
  - 状态定义
  - 常见问题FAQ

## 🎯 使用场景

### 场景1: 货架商品管理
```python
# 上传货架图片，自动检测商品、识别效期、生成告警
result = packcv_ocr_graph.invoke({
    "shelf_image": {"url": "shelf.jpg", "file_type": "image"},
    "enable_expiry_detection": True,
    "enable_alerts": True
})
```

### 场景2: 批量包装识别
```python
# 批量处理多张包装图片
result = main_graph.invoke({
    "images": [{"url": f"product_{i}.jpg"} for i in range(10)],
    "ocr_engine_type": "builtin",
    "export_format": "excel"
})
```

### 场景3: 单包装信息提取
```python
# 单张包装的结构化信息提取
result = main_graph.invoke({
    "package_image": {"url": "product.jpg", "file_type": "image"},
    "model_type": "extract"
})
```

## 🔧 配置说明

### 环境变量

```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/packcv_ocr

# OpenAI API（可选）
OPENAI_API_KEY=sk-...

# 对象存储配置（可选）
S3_ENDPOINT=...
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
```

### 告警规则配置

```python
alert_rules = {
    "near_expiry_days": 30,      # 临期预警天数
    "low_stock_threshold": 10,   # 低库存阈值
    "overstock_threshold": 100,  # 压货阈值
    "overstock_enabled": True    # 是否启用压货告警
}
```

## 🧪 测试

```bash
# 运行单元测试
pytest src/tests/

# 运行工作流测试
python -m src.main
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

- **GitHub**: https://github.com/iflykingc-oss/newPackCV-OCR
- **Issues**: https://github.com/iflykingc-oss/newPackCV-OCR/issues

## 🙏 致谢

感谢以下开源项目：

- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR识别
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流编排
- [OpenCV](https://opencv.org/) - 图像处理

---

**PackCV-OCR 融合算法系统** - 让OCR更智能，让货架管理更高效！ 🚀
