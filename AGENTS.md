# PackCV-OCR 项目结构索引

## 项目概述
- **名称**: PackCV-OCR V4.0
- **功能**: 面向货架/包装场景的高精度OCR识别解决方案 (全品类通用)
- **核心升级**: RapidOCR(ONNX)主引擎 + 自适应5策略预处理 + 多尺度检测 + 布局感知KV提取 + LLM后验证 + 全品类自由格式提取 + 飞书机器人 + HTTP API服务

### 核心特性
- RapidOCR(ONNX)主引擎 + Tesseract多PSM扫描（PaddleOCR已禁用-OneDNN兼容性问题）
- 自定义RapidOCR优化配置（config/rapidocr_optimized.yaml）
- **自适应5策略预处理（正常/暗图/模糊/低对比/强文本）+拉普拉斯质量评估**
- **极低质量图上采样（1.5x~3x INTER_CUBIC）+双边滤波去噪+形态学文本增强**
- **多尺度OCR检测（1x原始+0.5x小图双通道融合）**
- **竞品级后处理管线：IoU去重→Y轴阅读顺序排序→置信度过滤→词汇校正**
- **布局感知KV提取：基于bbox坐标的同行标签-值配对**
- **LLM后验证纠错：低置信度时自动LLM修正**
- **OCR重试机制：二值化（OTSU/自适应/CLAHE-OTSU）+ 2x上采样**
- **中文词汇校正字典（50+模式：日/月、0/O、l/1等常见混淆）**
- **智能倾斜校正（霍夫变换直线检测+自动旋转）**
- **增强型规则引擎（12个字段、中英文双语正则、关联校验）**
- **效期检测与到期日自动计算**
- **报表生成与导出（JSON/Excel/PDF）**

## 节点清单

| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| route_processing | `graphs/graph.py` | task | 路由处理模式（单图/批量） | single→image_preprocess, batch→batch_process | - |
| batch_process | `graphs/nodes/batch_process_node.py` | task | 批量图片处理 | - | - |
| image_preprocess | `graphs/nodes/image_preprocess_node.py` | task | **自适应增强**：质量评估→5策略选择→双边滤波去噪→形态学梯度→伽马校正→CLAHE→自适应二值化→上采样→锐化→倾斜校正+S3上传 | - | - |
| ocr_recognize | `graphs/nodes/ocr_recognize_node.py` | task | **增强OCR**：质量感知缩放+多尺度(1x/0.5x)RapidOCR+Tesseract备选+二值化重试(3种)+2x上采样重试+IoU去重+Y轴阅读排序+中文词汇校正 | - | `config/rapidocr_optimized.yaml` |
| correct_text | `graphs/nodes/correct_text_node.py` | agent | LLM智能纠错 | - | `config/correct_text_llm_cfg.json` |
| model_extract | `graphs/nodes/model_extract_node.py` | agent | **全品类自适应提取**：产品类型自动分类(食品/日化/饮料)+布局感知KV提取+12字段食品规则+自由格式文本提取(无标签场景)+LLM后验证 | - | `config/model_extract_llm_cfg.json` |
| qa_answer | `graphs/nodes/qa_answer_node.py` | agent | 语义问答（5段结构化分析+参考来源标注） | - | `config/qa_answer_llm_cfg.json` |
| result_output | `graphs/nodes/result_output_node.py` | task | 结果输出（JSON/Excel/PDF+平台推送） | - | - |
| feishu_notify | `graphs/nodes/feishu_notify_node.py` | task | **飞书通知**：将结构化数据通过飞书机器人卡片消息推送至群聊 | - | - |

**类型说明**: task(任务节点) / agent(大模型) / condition(条件分支) / looparray(列表循环) / loopcond(条件循环)

## OCR引擎架构

### 多引擎融合策略
1. **RapidOCR (ONNX)** - 主引擎，基于PaddleOCR模型的ONNX推理（单例复用）
   - 自定义优化配置：box_thresh=0.25, thresh=0.2, unclip_ratio=2.0, text_score=0.4
   - 多尺度检测：1x原始 + 0.5x小图（IoU合并）
2. **Tesseract** - 备选引擎，多PSM扫描（PSM 6→4→3）
3. **PaddleOCR** - 已禁用（OneDNN兼容性问题，环境变量ENABLE_PADDLEOCR=1可启用）

### 自动降级流程
RapidOCR → 二值化重试(OTSU) → 二值化重试(自适应高斯) → 二值化重试(CLAHE+OTSU) → 2x上采样重试 → Tesseract(中英→纯英) → 原始图像回退

## 预处理管线（V3.5增强版）
1. **极低质量图上采样**：min_dim<800→1.5x-3x / lap_var<40→1.5x / contrast<15→1.5x
2. **大图智能缩放**（最大边≤2000px）
3. **去噪**：双边滤波(保边) + fastNlMeans去噪(极端图)
4. **形态学文本增强**：梯度算子(MORPH_GRADIENT)增强文字边缘
5. **倾斜校正**（霍夫变换直线检测，|角度| 0.5-45度）
6. **5策略自适应增强**：
   - 正常→CLAHE 2.0+锐化
   - 暗图→Gamma 0.6-0.7校正+CLAHE 3.5+锐化
   - 模糊→拉普拉斯锐化9倍+Unsharp Masking+CLAHE 3.5
   - 低对比度→高CLAHE 4.0+对比度拉伸(5%-95%分位)
   - 强文本→CLAHE 2.0均衡化
7. **自适应阈值二值化**（极端图：原图60%+二值化40%混合）
8. **最终CLAHE+锐化**（自适应clipLimit）
9. 上传S3供OCR节点下载

## 后处理管线（竞品级）
1. IoU去重（交并比>0.5的相邻文本框合并）
2. 阅读顺序排序（Y轴同行分组→X轴排序）
3. 置信度过滤（阈值0.15，保留低置信度有效文本）
4. 中文词汇校正（50+常见OCR错误纠正）
5. 段落合并（Y坐标邻近的文本合并为段落）

## 规则引擎增强
- **11个提取字段 + 自由格式自适应**：brand, product_name, specification, production_date, shelf_life, manufacturer, ingredients, standard, batch_number, license_number, storage_condition, expiry_date + features(卖点), other_info(其他)
- **自由格式文本提取**：针对无标签文本（湿巾、日化品等），自动识别品牌/品名/成分/卖点
- **产品类型自动分类**：基于OCR文本自动判断食品/日化/饮料/其他，动态适配提取字段
- 中英文双语正则支持（Brand/Product/Manufacturer/Shelf Life等英文标签）
- 关联校验：生产日期+保质期→自动计算到期日期
- 200+品牌直接匹配模式（含金龙鱼/海天/蒙牛/伊利等）
- 多格式日期/规格/标准号正则（YYYY/MM/DD、YYYY-MM-DD、YYYY年MM月DD日）
- **标准号修复**：支持GB/T 18186格式（含推荐性标准T字母）
- 许可证号支持10-14位SC编号
- 食品生产资质SC编号专用模式
- 中文OCR词汇校正（日→月、0→O、l→1等常见混淆）
- 厂商名规范化（YiHaijiaLi→YiHaiJiaLi等）
- **布局感知KV提取**：基于bbox坐标的同行标签-值配对
- **LLM后验证**：提取置信度<0.3时自动LLM二次验证

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
- 飞书通知节点使用 飞书消息集成(integration-feishu-message)
- **API服务**：`src/web_server.py` - FastAPI HTTP API (端口8000)，支持POST /api/ocr直接调用工作流

## 工作流主线
```
route_processing → image_preprocess → ocr_recognize → correct_text → model_extract → qa_answer → result_output → feishu_notify → END
```

## 配置文件

| 文件 | 用途 |
|------|------|
| `config/rapidocr_optimized.yaml` | RapidOCR自定义参数（box_thresh=0.25, unclip_ratio=2.0, text_score=0.4） |
| `config/correct_text_llm_cfg.json` | 纠错LLM配置（doubao-seed-2-0-pro-260215） |
| `config/model_extract_llm_cfg.json` | 提取LLM配置+结构化提取SP |
| `config/qa_answer_llm_cfg.json` | 问答LLM配置+5段分析SP |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| COZE_WORKSPACE_PATH | 项目工作目录 | /workspace/projects |
| COZE_BUCKET_ENDPOINT_URL | S3端点URL | - |
| COZE_BUCKET_NAME | S3桶名称 | - |
| ENABLE_PADDLEOCR | 启用PaddleOCR | 0 |
