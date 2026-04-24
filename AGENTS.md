# 多平台OCR包装识别系统 - 项目结构索引

## 项目概述
- **名称**: 多平台OCR包装识别系统
- **功能**: 支持微信小程序、飞书多平台适配，可精准识别瓶子等任意包装所有信息，同时支持模型调用的工具，解决包装信息识别低效、跨平台使用不便、识别结果无法深度处理的痛点。

### 核心特性
1. **多平台适配**: 支持微信小程序、飞书多平台
2. **高精度OCR识别**: 支持内置算法（PaddleOCR）和外部API
3. **图片预处理**: 针对瓶子包装的图像增强、去噪、校正
4. **智能模型调用**: 结构化提取、智能纠错、语义问答
5. **批量处理**: 支持多张图片并行处理和结果合并导出
6. **多格式导出**: 支持JSON、Excel、PDF格式
7. **数据持久化**: 基于PostgreSQL的数据存储和ORM管理

---

## 节点清单

| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|--------|---------|------|---------|---------|---------|
| route_processing | `graph.py` | condition | 路由处理模式（单图/批量） | batch→batch_process, single→image_preprocess | - |
| image_preprocess | `nodes/image_preprocess_node.py` | task | 图片预处理，包括增强、去噪、校正 | - | - |
| ocr_recognize | `nodes/ocr_recognize_node.py` | task | OCR识别，支持内置算法和外部API | - | - |
| route_model_processing | `graph.py` | condition | 根据model_type选择处理路径 | extract→model_extract, correct→correct_text, qa→qa_answer | - |
| model_extract | `nodes/model_extract_node.py` | agent | 结构化信息提取（品牌、规格、日期等） | - | `config/model_extract_llm_cfg.json` |
| correct_text | `nodes/correct_text_node.py` | agent | 智能纠错，修复错别字、漏字 | - | `config/correct_text_llm_cfg.json` |
| qa_answer | `nodes/qa_answer_node.py` | agent | 语义问答，基于识别结果回答用户问题 | - | `config/qa_answer_llm_cfg.json` |
| result_output | `nodes/result_output_node.py` | task | 结果输出，支持多格式导出和多平台推送 | - | - |
| batch_process | `nodes/batch_process_node.py` | task | 批量处理多张图片，支持并行识别和结果导出 | - | - |

**类型说明**: task(任务节点) / agent(大模型节点) / condition(条件分支节点)

---

## 工作流图结构

```
入口 (GraphInput)
  ↓
[路由处理模式] (route_processing)
  ├─ "批量处理" → [批量处理] (batch_process) → 出口
  └─ "单图处理" → [图片预处理] (image_preprocess)
                     ↓
                   [OCR识别] (ocr_recognize)
                     ↓
                   [条件分支] (route_model_processing)
                     ├─ "结构化提取" → [模型结构化提取] (model_extract)
                     ├─ "智能纠错" → [智能纠错] (correct_text)
                     └─ "语义问答" → [语义问答] (qa_answer)
                     ↓
                   [结果输出] (result_output)
                     ↓
                   出口 (GraphOutput)
```

### 批量处理流程
当输入参数包含 `images` 字段且长度大于1时，系统自动进入批量处理模式：
1. 并行处理多张图片的OCR识别
2. 支持PaddleOCR和Tesseract OCR两种引擎
3. 合并所有识别结果
4. 支持导出为Excel或PDF格式

### 单图处理流程
当输入参数只包含 `package_image` 字段时，系统进入单图处理模式：
1. 图片预处理（增强、去噪、校正）
2. OCR识别
3. 根据model_type选择模型处理路径
4. 结果输出和导出

---

## 数据库设计

### 数据库Schema
- **文件位置**: `src/storage/schema.sql`
- **数据库类型**: PostgreSQL

### 数据表清单

| 表名 | 用途 | 主要字段 |
|------|------|---------|
| users | 用户表 | id, username, email, password_hash, account_type |
| roles | 角色表 | id, name, description, permissions |
| user_roles | 用户角色关联表 | user_id, role_id, assigned_at |
| teams | 团队表 | id, name, owner_id, member_limit |
| team_members | 团队成员表 | team_id, user_id, role |
| ocr_records | OCR识别记录表 | id, user_id, image_url, ocr_text, structured_data |
| user_configs | 用户配置表 | user_id, config_key, config_value |
| platform_integrations | 多平台集成配置表 | id, user_id, platform, credentials |
| batch_tasks | 批量处理任务表 | id, user_id, image_urls, status, export_format |
| batch_task_results | 批量处理结果详情表 | id, batch_task_id, image_url, ocr_text |
| model_configs | 模型配置表 | id, user_id, name, model_type, system_prompt |
| ocr_api_configs | OCR API配置表 | id, user_id, name, api_url, api_key |

### ORM模型
- **文件位置**: `src/storage/models.py`
- **ORM框架**: SQLAlchemy
- **主要模型**: User, Role, Team, OCRRecord, BatchTask等

### 数据库管理
- **初始化脚本**: `src/storage/database.py`
- **主要功能**:
  - 数据库连接管理
  - Session工厂
  - 表创建和删除
  - 连接测试

---

## 技能使用

### 已加载技能
1. **大语言模型** (`/skills/public/prod/llm`)
   - 用途: 结构化提取、智能纠错、语义问答
   - 使用的节点: `model_extract`, `correct_text`, `qa_answer`

2. **飞书消息** (`/skills/public/prod/feishu-message`)
   - 用途: 飞书消息推送
   - 状态: 待配置集成

3. **飞书多维表格** (`/skills/public/prod/feishu-base`)
   - 用途: 飞书多维表格操作
   - 状态: 待配置集成

4. **微信公众号** (`/skills/public/prod/wechat-official-account`)
   - 用途: 微信公众号草稿生成
   - 状态: 待配置集成

5. **微信机器人** (`/skills/public/prod/wechat-bot`)
   - 用途: 企业微信机器人消息推送
   - 状态: 待配置集成

6. **对象存储** (`/skills/public/prod/storage`)
   - 用途: 文件上传和存储
   - 使用的节点: `result_output`

---

## 配置文件说明

### 大模型配置文件

#### 1. 结构化提取配置
- **文件**: `config/model_extract_llm_cfg.json`
- **用途**: 配置用于结构化信息提取的大模型
- **默认模型**: doubao-seed-2-0-pro-260215
- **主要功能**: 从OCR文本中提取品牌、规格、生产日期、保质期、厂家等信息

#### 2. 智能纠错配置
- **文件**: `config/correct_text_llm_cfg.json`
- **用途**: 配置用于智能纠错的大模型
- **默认模型**: doubao-seed-2-0-pro-260215
- **主要功能**: 修复OCR识别中的错别字、漏字、数字识别错误

#### 3. 语义问答配置
- **文件**: `config/qa_answer_llm_cfg.json`
- **用途**: 配置用于语义问答的大模型
- **默认模型**: doubao-seed-2-0-pro-260215
- **主要功能**: 基于OCR识别结果回答用户提问

---

## OCR引擎支持

### 内置算法
1. **PaddleOCR** (首选)
   - 支持中英文混合识别
   - 高精度，适合包装识别场景
   - 依赖: `paddleocr`

2. **Tesseract OCR** (备选)
   - 开源OCR引擎
   - 依赖: `pytesseract`

### 外部API
- 支持通过配置调用第三方OCR API
- 配置字段: `ocr_api_config`
  - `url`: API地址
  - `api_key`: API密钥

---

## 数据模型说明

### 主要状态类

1. **GlobalState**: 全局状态，包含工作流流转的所有数据
2. **GraphInput**: 工作流输入参数
3. **GraphOutput**: 工作流输出结果
4. **节点Input/Output**: 各节点的独立输入输出

### 核心字段说明

#### 输入字段 (GraphInput)
- `package_image`: 包装图片（File类型，单图处理）
- `images`: 多张图片列表（List[File]，批量处理）
- `ocr_engine_type`: OCR引擎类型（"builtin" 或 "api"）
- `model_type`: 模型调用类型（"extract", "correct", "qa"）
- `model_name`: 大模型名称
- `platform`: 目标平台（"wechat", "feishu", "none"）
- `export_format`: 导出格式（"json", "excel", "pdf"）

**处理模式判断**：
- 如果 `images` 存在且长度 > 1 → 批量处理模式
- 否则 → 单图处理模式

#### 输出字段 (GraphOutput)
- `success`: 是否成功
- `ocr_result`: OCR识别结果
- `structured_data`: 结构化提取数据
- `corrected_text`: 纠错后的文本
- `qa_answer`: 问答答案
- `export_file_url`: 导出文件URL

---

## 依赖包清单

### 核心依赖
- `langgraph`: 工作流编排框架
- `langchain-core`: LangChain核心库
- `coze-coding-dev-sdk`: Coze开发SDK

### OCR相关
- `paddleocr`: PaddleOCR引擎
- `pytesseract`: Tesseract OCR引擎
- `opencv-python-headless`: 图像处理

### 文档生成
- `pandas`: Excel生成
- `openpyxl`: Excel文件处理
- `reportlab`: PDF生成

### 数据库相关
- `sqlalchemy`: ORM框架
- `psycopg2-binary`: PostgreSQL驱动

### 其他
- `jinja2`: 模板渲染
- `requests`: HTTP请求
- `tempfile`: 临时文件处理

---

## 待完成功能

### 已完成 ✅
1. **数据库Schema**: 设计并创建12张表的完整DDL结构
2. **ORM层**: 使用SQLAlchemy实现数据库模型映射
3. **批量处理**: 实现批量图片处理功能，支持并行识别和结果导出
4. **工作流路由**: 实现单图/批量处理模式的自动路由

### 高优先级
1. **多平台适配**: 完成微信和飞书的集成配置
2. **用户管理**: 用户注册、登录、权限管理（JWT认证）
3. **历史记录**: 保存和管理识别历史

### 中优先级
1. **性能优化**: 优化OCR识别速度和准确率
2. **算法优化**: 研究和集成先进OCR算法（EasyOCR、MMOCR等）
3. **图像优化**: 优化图像预处理算法（针对包装场景）

### 低优先级
1. **自定义模型训练**: 支持企业用户自定义模型
2. **API开放**: 提供REST API接口
3. **多语言支持**: 支持更多语言的OCR识别

---

## 开发指南

### 添加新节点
1. 在 `src/graphs/nodes/` 目录下创建新节点文件
2. 定义节点的Input和Output类型（在`state.py`中）
3. 实现节点函数，遵循签名规范
4. 在 `graph.py` 中添加节点并连接边

### 配置新模型
1. 在 `config/` 目录下创建新的配置文件
2. 遵循配置文件格式（config, sp, up, tools）
3. 在节点中引用配置文件

### 集成新平台
1. 加载对应的技能
2. 调用 `integration_detail` 获取集成配置
3. 在相应节点中实现平台推送逻辑

---

## 测试说明

### 测试数据准备
- 测试图片URL: 使用生成的mock图片或本地图片
- 支持格式: JPG, PNG, JPEG
- 推荐尺寸: 800x1200或更大

### 测试命令
```bash
# 运行测试
python src/main.py

# 使用test_run工具
test_run(params="{...}")
```

### 测试参数示例
```json
{
  "package_image": {
    "url": "图片URL",
    "file_type": "image"
  },
  "ocr_engine_type": "builtin",
  "model_type": "extract",
  "model_name": "doubao-seed-2-0-pro-260215",
  "export_format": "json",
  "platform": "none"
}
```

---

## 常见问题

### Q1: PaddleOCR识别失败怎么办？
A: 系统会自动回退到Tesseract OCR。请确保已安装相关依赖。

### Q2: 如何切换OCR引擎？
A: 设置输入参数 `ocr_engine_type` 为 "builtin"（内置）或 "api"（外部API）。

### Q3: 如何自定义模型提示词？
A: 修改配置文件中的 `sp`（系统提示词）和 `up`（用户提示词）字段，或在输入参数中设置 `model_prompt`。

### Q4: 支持哪些导出格式？
A: 当前支持 JSON、Excel、PDF 三种格式。

### Q5: 如何配置多平台推送？
A: 需要先调用 `integration_detail` 配置对应的集成，然后在输入参数中设置 `platform` 为 "wechat" 或 "feishu"。

### Q6: 如何使用批量处理功能？
A: 在输入参数中提供 `images` 字段（包含多张图片URL列表），系统会自动识别并进入批量处理模式。

### Q7: 批量处理支持多少张图片？
A: 当前单次最多支持50张图片，可通过配置调整限制。

### Q8: 批量处理的结果如何导出？
A: 批量处理支持导出为Excel或PDF格式，设置 `export_format` 参数即可。

### Q9: 数据库如何初始化？
A: 运行 `python src/storage/database.py` 可自动创建所有表结构。确保已配置PostgreSQL连接。

### Q10: 如何配置数据库连接？
A: 设置环境变量 `DATABASE_URL`，格式为 `postgresql://user:password@localhost:5432/db_name`。

---

## 联系方式
- 项目位置: `src/graphs/`
- 配置文件: `config/`
- 依赖管理: `pyproject.toml`

---

*最后更新: 2025-01-XX*
