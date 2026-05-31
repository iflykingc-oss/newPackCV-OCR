# PackCV-OCR 项目结构索引

## 项目概述
- **名称**: PackCV-OCR V2.3
- **功能**: 面向货架/包装场景的高精度OCR识别解决方案
- **核心升级**: RapidOCR(ONNX)为主引擎 + Tesseract备选 + 增强中英文规则提取

### 核心特性
- RapidOCR(ONNX)主引擎 + Tesseract备选（PaddleOCR已禁用-OneDNN兼容性问题）
- 自动引擎降级与原始图像回退
- 轻量级图像预处理（CLAHE增强+锐化，RapidOCR自带文本检测）
- LLM智能纠错与结构化提取
- 增强版规则引擎（11个字段、中英文双语支持、关联校验、到期日计算）
- 效期检测与告警
- 报表生成与导出

## 节点清单

| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| route_processing | `graphs/graph.py` | task | 路由处理模式（单图/批量） | single→image_preprocess, batch→batch_process | - |
| batch_process | `graphs/nodes/batch_process_node.py` | task | 批量图片处理 | - | - |
| image_preprocess | `graphs/nodes/image_preprocess_node.py` | task | 图像预处理（CLAHE+锐化+S3上传） | - | - |
| ocr_recognize | `graphs/nodes/ocr_recognize_node.py` | task | 多引擎OCR（RapidOCR→Tesseract） | - | - |
| correct_text | `graphs/nodes/correct_text_node.py` | agent | LLM智能纠错 | - | `config/correct_text_llm_cfg.json` |
| model_extract | `graphs/nodes/model_extract_node.py` | agent | LLM结构化提取 + 增强规则引擎降级 | - | `config/model_extract_llm_cfg.json` |
| qa_answer | `graphs/nodes/qa_answer_node.py` | agent | 语义问答 | - | `config/qa_answer_llm_cfg.json` |
| result_output | `graphs/nodes/result_output_node.py` | task | 结果输出（JSON/Excel/PDF+平台推送） | - | - |

**类型说明**: task(任务节点) / agent(大模型) / condition(条件分支) / looparray(列表循环) / loopcond(条件循环)

## OCR引擎架构

### 多引擎融合策略
1. **RapidOCR (ONNX)** - 主引擎，基于PaddleOCR模型的ONNX推理
   - 优点：速度快、无系统依赖、中文识别率高、pip install即可用
   - 安装：`pip install rapidocr_onnxruntime`
2. **Tesseract** - 备选引擎，传统OCR
   - 优点：稳定可靠、支持多语言
   - 缺点：需要chi_sim中文包、中文识别率较低
3. **PaddleOCR** - 已禁用（OneDNN兼容性问题，环境变量ENABLE_PADDLEOCR=1可启用）

### 自动降级流程
RapidOCR(置信度≥0.3) → Tesseract(中英→纯英) → 原始图像回退

## 预处理管线
1. 大图智能缩放（最大边≤2000px）
2. CLAHE对比度增强（Lab空间亮度通道处理）
3. Unsharp Masking锐化
4. 上传S3供OCR节点下载

## 规则引擎增强
- 11个提取字段：brand, product_name, specification, production_date, shelf_life, manufacturer, ingredients, standard, batch_number, license_number, storage_condition
- 中英文双语正则支持（Brand/Product/Manufacturer/Shelf Life等英文标签）
- 关联校验：生产日期+保质期→自动计算到期日期
- 100+品牌直接匹配模式
- 多格式日期/规格/标准号正则
- 许可证号支持10-14位SC编号

## 依赖包
- rapidocr-onnxruntime>=1.4.4 (OCR主引擎)
- paddleocr (已禁用，保留安装)
- pytesseract (OCR备选引擎)
- opencv-python (图像处理)
- coze-coding-dev-sdk (LLM/S3)
- jinja2 (提示词模板)

## 技能使用
- OCR节点使用 RapidOCR(ONNX)、Tesseract
- 预处理/输出节点使用 对象存储(S3SyncStorage)
- 纠错/提取/问答节点使用 大语言模型(LLMClient)

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| COZE_WORKSPACE_PATH | 项目工作目录 | /workspace/projects |
| COZE_BUCKET_ENDPOINT_URL | S3端点URL | - |
| COZE_BUCKET_NAME | S3桶名称 | - |
| ENABLE_PADDLEOCR | 启用PaddleOCR | 0 |
