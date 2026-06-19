# PackCV-OCR 项目结构索引

## 项目概述
- **名称**: PackCV-OCR V5.0
- **功能**: 面向货架/包装场景的高精度OCR识别解决方案 (全品类通用)
- **核心升级**: RapidOCR(ONNX)主引擎 + 自适应5策略预处理 + 多尺度检测 + 布局感知KV提取 + LLM后验证 + 全品类自由格式提取 + 专线营养表提取 + 投影法布局分析 + 小字超分增强 + 飞书机器人 + HTTP API服务

### 核心特性
- RapidOCR(ONNX)主引擎 + Tesseract多PSM扫描（PaddleOCR已禁用-OneDNN兼容性问题）
- 自定义RapidOCR优化配置（config/rapidocr_optimized.yaml）
- **自适应5策略预处理（正常/暗图/模糊/低对比/强文本）+拉普拉斯质量评估**
- **质量分级处理：高质量图（score>70）跳过预处理增强+单次OCR快速通过**
- **OCR结果缓存（LRU 200条，相同URL秒级返回）**
- **Direction ② 小字超分辨率增强：形态学梯度检测小字区域→自动2x-3x上采样+自适应二值化增强**
- **Direction ① 投影法布局分析：垂直投影检测列间隙→水平投影检测行间隙→多区域独立OCR→结果合并**
- **极低质量图上采样（1.5x~3x INTER_CUBIC）+双边滤波去噪+形态学文本增强**
- **多尺度OCR检测（1x原始+0.5x小图双通道融合）**
- **竞品级后处理管线：IoU去重→Y轴阅读顺序排序→置信度过滤→词汇校正**
- **布局感知KV提取：基于bbox坐标的同行标签-值配对**
- **LLM后验证纠错：低置信度时自动LLM修正**
- **OCR重试机制：二值化（OTSU/自适应/CLAHE-OTSU）+ 2x上采样**
- **中文词汇校正字典（80+模式：日/月、0/O、l/1、254%/25.4%等常见混淆）**
- **Direction ③ 营养表专用提取器：行列检测→行识别→结构化营养键值对输出**
- **智能倾斜校正（霍夫变换直线检测+自动旋转）**
- **增强型规则引擎（12个字段、中英文双语正则、关联校验）**
- **效期检测与到期日自动计算**
- **报表生成与导出（JSON/Excel/PDF）**
- **Direction ⑤ 图片格式扩展：HEIC/WebP/BMP/TIFF自动转换**

## 节点清单

| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| route_processing | `graphs/graph.py` | task | 路由处理模式（单图/批量） | single→image_preprocess, batch→batch_process | - |
| batch_process | `graphs/nodes/batch_process_node.py` | task | 批量图片处理 | - | - |
| image_preprocess | `graphs/nodes/image_preprocess_node.py` | task | **自适应增强**：质量评估→5策略→**D②小字检测放大**→**D⑤格式转换**→双边滤波去噪→形态学梯度→伽马校正→CLAHE→自适应二值化→上采样→锐化→倾斜校正+S3上传 | - | - |
| ocr_recognize | `graphs/nodes/ocr_recognize_node.py` | task | **增强OCR**：**D①投影布局分析(列检测→多区域OCR)**→质量感知缩放→多尺度(1x/0.5x)RapidOCR+Tesseract备选→**D④缓存200条优化**→二值化重试(3种)→2x上采样重试→IoU去重→Y轴阅读排序→中文词汇校正(80+模式) | - | `config/rapidocr_optimized.yaml` |
| correct_text | `graphs/nodes/correct_text_node.py` | agent | LLM智能纠错 | - | `config/correct_text_llm_cfg.json` |
| model_extract | `graphs/nodes/model_extract_node.py` | agent | **全品类自适应提取**：产品类型自动分类(食品/日化/饮料)+布局感知KV提取+12字段食品规则+OCR容错标签匹配(三产→生产)+品牌直接匹配+多行配料跨行捕获+规格排除+**D③营养表专用提取器(行列解析→结构化键值对)**+自由格式提取+LLM后验证 | - | `config/model_extract_llm_cfg.json` |
| qa_answer | `graphs/nodes/qa_answer_node.py` | agent | 语义问答（5段结构化分析+参考来源标注） | - | `config/qa_answer_llm_cfg.json` |
| result_output | `graphs/nodes/result_output_node.py` | task | 结果输出（JSON/Excel/PDF+平台推送） | - | - |
| feishu_notify | `graphs/nodes/feishu_notify_node.py` | task | **飞书通知**：结构化数据→飞书机器人交互式卡片推送（含字段统计+导出链接+智能问答） | - | - |

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
4. 中文词汇校正（80+常见OCR错误纠正，包括254%→25.4%等百分比修复）
5. 段落合并（Y坐标邻近的文本合并为段落）

## 规则引擎增强
- **11个提取字段 + 自由格式自适应**：brand, product_name, specification, production_date, shelf_life, manufacturer, ingredients, standard, batch_number, license_number, storage_condition + features(卖点), other_info(其他), product_type(产品类型)
- **自由格式文本提取**：针对无标签文本（湿巾、日化品等），自动识别品牌/品名/成分/卖点
- **产品类型自动分类**：基于OCR文本自动判断食品/日化/饮料/其他，动态适配提取字段
- **品牌列表直接匹配**：200+品牌直接匹配，正确提取"米多奇"而非截取公司名
- **OCR容错标签匹配**：三产→生产、改敏→过敏、三产日期→生产日期
- **多行配料跨行捕获**：使用`[\s\S]+?`跨行匹配，OCR容错终止符（改敏/过敏原）
- **规格营养表排除**：使用`(?<!每)`负向后顾排除"每100克"营养表数据
- **生产日期"见包装"兜底**：非日期格式值时保留标签原文（如"见包装上"）
- **merge_rule_and_freeform**：规则引擎+自由格式双通道结果智能合并
- 中英文双语正则支持（Brand/Product/Manufacturer/Shelf Life等英文标签）
- 多格式日期/规格/标准号正则（YYYY/MM/DD、YYYY-MM-DD、YYYY年MM月DD日）

## 评测脚本
| 脚本 | 功能 | 说明 |
|-----|------|------|
| `scripts/final_evaluation.py` | 批量评测脚本 | 对比PackCV vs Tesseract，12张图全品类覆盖 |
| `scripts/upload_images.py` | 图片上传 | 上传包装图到对象存储 |

## 评测结果（2026-06-19）
### 整体对比
| 指标 | PackCV-OCR | Tesseract | 优势 |
|-----|-----------|-----------|------|
| 总字段提取数 | **51** | 1 | **x51** |
| 平均字段/张 | **4.2** | 0.1 | +4.1 |
| 处理速度 | ~8s/张 | ~5s/张 | Tesseract略快但无结构化 |

### 品类表现
| 品类 | 图片数 | PackCV平均 | 标杆表现 |
|-----|-------|-----------|---------|
| 🏆 **OTC药品包装** | 1 | **8.0字段** | 急支糖浆9字段：药准字+成分+用法+禁忌+效期等 |
| 🏆 **饼干背面标签** | 1 | **8.0字段** | 配料+营养表+生产商+地址+保质期+存储 |
| 🏆 **糖果/巧克力包装** | 1 | **7.0字段** | 7项营养表+许可证+配料完整提取 |
| 薯片背面标签 | 1 | 6.0字段 | 营养表3项+配料+生产商+保质期 |
| 酱料瓶标 | 1 | 5.0字段 | 许可证号+存储+营养表+规格 |
| 清洁用品瓶标 | 1 | 5.0字段 | 6项成分+用法+许可证+注意事项 |
| 酱油正面/茶饮/虾片 | 3 | 3.0字段 | 品牌+品名+规格 |
| 方便面正面 | 1 | 2.0字段 | 品牌+品名 |
| 酸奶/洗发水 | 2 | ~0.5字段 | 低质量生成图片导致OCR困难 |

### 质量分布
- **Excellent** (8+字段): 3张 (25%)
- **Good** (3-7字段): 7张 (58%)
- **Poor** (<3字段): 2张 (17%)

### 核心发现
1. **PackCV在结构化提取上远超纯OCR引擎**（Tesseract 0字段 vs PackCV 51字段）
2. **药品类是最大亮点** - 急支糖浆9字段完美提取（含药准字、分级用法等专业字段）
3. **优劣场景**：高对比度平面标签>曲面包裹>低质量生成图
4. **Tesseract局限**：纯OCR无法做结构化提取，对复杂包装图效果极差
- **中文模式**：日↔月、三产↔生产、改敏↔过敏、已二↔己二、特丁基↔叔丁基、双淀粉已二酸↔双淀粉己二酸、三梨酸钾↔山梨酸钾、0克↔〇克、贮芷↔贮藏
- **百分比修复**：254%→25.4%、110%→10%、20.0g→200g、5.0g→50g
- **英文品牌**：Dafl→Dafi、Daf→Dafi、Yili→伊利、Mengniu→蒙牛、Haier→海尔
- **OCR污渍移除**：连续`丨丨丨丨`、`一一一一`等模式的自动清除
- **场景预防**：`(?:日|月)\s*(?:期|份)`避免误替换"日期"中的"日"为"月"

## 测试验证场景
| 测试图片 | 产品类型 | 提取结果 |
|---------|---------|---------|
| 答菲湿巾（assets/测试识别图.jpg） | 日化-湿巾 | brand✅product_name✅ingredients✅features✅product_type✅ |
| 紫薯雪饼（assets/1111.jpg） | 食品-膨化 | brand✅product_name✅production_date✅shelf_life✅manufacturer✅ingredients✅standard✅license_number✅storage_condition✅ |

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

### API服务 (`src/web_server.py`)
启动方法: `python3 src/web_server.py` (端口8000)

| 端点 | 方法 | 说明 | 入参 |
|------|------|------|------|
| `/health` | GET | 健康检查 | - |
| `/api/ocr` | POST | 执行OCR识别 | `{image_url, question, platform, feishu_webhook}` |
| `/api/feishu/callback` | POST | 飞书事件回调（含challenge验证） | 飞书事件JSON格式 |

#### Feishu 机器人交互模式
1. **被动通知**：工作流执行完→自动推送到飞书群（`platform=feishu`）
2. **主动查询**：API服务接收飞书事件回调→解析图片→OCR识别→结果回复
3. **HTTP调用**：任何系统可调用`POST /api/ocr`获取JSON结果
4. **飞书卡片按钮**：飞书卡片中支持"重新分析"、"导出报表"等交互按钮

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
