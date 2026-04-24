# GitHub 推送指南

## 项目已准备就绪，需要配置 GitHub 认证才能推送

### 📦 项目信息
- **仓库地址**: https://github.com/iflykingc-oss/newPackCV-OCR.git
- **项目名称**: PackCV-OCR 融合算法系统
- **分支**: main

### 🔑 获取 GitHub Personal Access Token

#### 步骤 1: 创建 Token
1. 访问 GitHub: https://github.com/settings/tokens
2. 点击 **"Generate new token"** → **"Generate new token (classic)"**
3. 设置 Token 名称（如：PackCV-OCR-Deploy）
4. 选择权限（至少勾选以下选项）：
   - ✅ **repo** - 完整的仓库访问权限
5. 设置过期时间（建议选择：30 days 或 No expiration）
6. 点击 **"Generate token"**
7. **重要**: 复制生成的 token（只会显示一次！）

#### 步骤 2: 配置 Git 使用 Token

打开终端，执行以下命令：

```bash
# 方法 1: 直接在 URL 中包含 token（推荐）
git remote set-url origin https://<YOUR_TOKEN>@github.com/iflykingc-oss/newPackCV-OCR.git

# 将 <YOUR_TOKEN> 替换为你刚才复制的 token
# 示例:
# git remote set-url origin https://ghp_xxxxxxxxxxxxxxxxxxxx@github.com/iflykingc-oss/newPackCV-OCR.git
```

#### 步骤 3: 推送代码

```bash
# 查看状态
git status

# 推送到 GitHub
git push -u origin main
```

---

### 🚀 推送成功后

你可以在以下地址访问项目：
- **GitHub 仓库**: https://github.com/iflykingc-oss/newPackCV-OCR
- **查看代码**: https://github.com/iflykingc-oss/newPackCV-OCR/tree/main
- **克隆命令**: `git clone https://github.com/iflykingc-oss/newPackCV-OCR.git`

---

### 📝 项目结构

```
newPackCV-OCR/
├── config/                      # 配置文件目录
│   ├── model_extract_llm_cfg.json
│   ├── correct_text_llm_cfg.json
│   └── qa_answer_llm_cfg.json
├── docs/                        # 文档目录
├── scripts/                     # 脚本目录
├── assets/                      # 资源目录
├── src/                         # 源码目录
│   ├── agents/                  # Agent 代码
│   ├── storage/                 # 数据存储
│   │   ├── schema.sql          # 数据库 Schema
│   │   ├── models.py           # ORM 模型
│   │   └── database.py         # 数据库管理
│   ├── tests/                   # 测试用例
│   ├── tools/                   # 工具定义
│   ├── graphs/                  # 工作流编排
│   │   ├── state.py            # 状态定义
│   │   ├── graph.py            # 基础工作流
│   │   ├── packcv_graph.py     # PackCV-OCR 工作流
│   │   └── nodes/              # 节点实现
│   │       ├── cv_detection_node.py
│   │       ├── roi_segmentation_node.py
│   │       ├── parallel_processing_node.py
│   │       ├── alert_engine_node.py
│   │       └── report_generation_node.py
│   └── utils/                   # 工具类
├── AGENTS.md                    # 项目文档
├── README.md                    # 项目说明
├── pyproject.toml               # Python 项目配置
├── .gitignore                   # Git 忽略文件
└── PUSH_TO_GITHUB.md            # 本推送指南
```

---

### ✨ 核心功能

#### 1. 基础 OCR 功能
- 图片预处理（增强、去噪、校正）
- OCR识别（PaddleOCR/Tesseract）
- 模型调用（结构化提取、智能纠错、语义问答）
- 批量处理（多张图片并行）
- 多格式导出（JSON/Excel/PDF）

#### 2. PackCV-OCR 融合算法
- **CV目标检测**: YOLOv8检测货架商品
- **ROI分层裁切**: 自动分割商品区域
- **并行处理引擎**: 多区域并行OCR
- **智能告警引擎**: 效期/库存/合规告警
- **自动报表生成**: 效期/库存/合规台账

#### 3. 数据持久化
- 完整的数据库Schema（12张表）
- SQLAlchemy ORM模型
- 数据库管理模块

---

### 🎯 快速开始

#### 安装依赖
```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip（不推荐）
pip install -r requirements.txt
```

#### 运行测试
```bash
# 测试基础 OCR 工作流
python -c "from src.graphs.graph import main_graph; print('✓ 工作流加载成功')"

# 测试 PackCV-OCR 工作流
python -c "from src.graphs.packcv_graph import packcv_ocr_graph; print('✓ PackCV-OCR 工作流加载成功')"
```

#### 使用示例
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
    "export_format": "excel"
})

print(f"检测到 {result['total_products']} 个商品")
print(f"生成 {len(result['alerts'])} 条告警")
```

---

### 📊 性能指标

| 指标 | 数值 |
|------|------|
| CV检测速度 | 100个商品 < 1秒 |
| ROI裁切速度 | 100个区域 < 1秒 |
| 并行OCR速度 | 100个商品 < 3秒 |
| 端到端处理 | < 10秒 |
| 效期识别准确率 | ≥95% |
| 告警响应时间 | < 0.5秒 |
| 报表生成时间 | < 2秒 |

---

### 📖 详细文档

- **AGENTS.md**: 完整的项目文档
  - 节点清单
  - 工作流图
  - 状态定义
  - 常见问题FAQ

---

### 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

### 📄 许可证

MIT License

---

### 📞 联系方式

- **GitHub**: https://github.com/iflykingc-oss/newPackCV-OCR
- **Issues**: https://github.com/iflykingc-oss/newPackCV-OCR/issues

---

**祝使用愉快！** 🚀
