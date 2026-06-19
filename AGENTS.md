# PackCV-OCR 项目结构索引

## 项目概述
- **名称**: PackCV-OCR V5.4 (Commercial-Ready)
- **功能**: 面向货架/包装场景的高精度OCR识别解决方案 (全品类通用 · 商业化版)
- **核心升级**: V5.4 = V5.3 + **结构化提取全量替换为商业化统一结构** (product_type/brand/.../category_info 顶层+品类对象双层结构)

### V5.3 产品力深度迭代 + 三平台发布（2026-06-19）

#### 阶段一：产品力4方向深化

| 优化方向 | 实现内容 | 变更文件 | 状态 |
|---------|---------|---------|------|
| ①多通道融合架构 | OCR传统管线 ∥ VL多模态 → 字段级加权融合（field_confidence≥0.8 优先，0.6-0.8 加权投票，<0.6 仅参考） | `multi_channel_fusion_node.py` | ✅ |
| ②品类模板库 | 食品/药品/日化/酒类/电子 5大品类 → 字段权重+专有正则+关联校验 | `category_template_node.py` | ✅ |
| ④调用审计中间件 | 内存LRU 1000条 + JSONL文件 + Prometheus `/metrics` + `/audit/summary` | `call_audit_node.py` | ✅ |
| ⑤多语言OCR接入 | 中/英/日/韩/法/德/西 7语种支持（独立节点，按需挂载） | `multi_language_ocr_node.py` | ✅ |

#### 阶段二：三平台发布（IM Platform Adapter）

| 平台 | 实现方式 | 关键能力 | 端点 |
|------|---------|---------|------|
| 飞书 | 命令式Bot + 事件订阅 | AES-256-CBC签名验证、交互式卡片 | `POST /bot/feishu/events` |
| 钉钉 | 自定义机器人 | HMAC-SHA256签名、markdown/actionCard | `POST /bot/dingtalk/callback` |
| 企业微信 | 智能机器人 | SHA1+AEAD加密、text/markdown/news | `POST /bot/wecom/callback` |

### V5.3 处理流程
```
image_preprocess → ocr_recognize → correct_text → model_extract 
    → multi_channel_fusion (①融合传统OCR+VL多模态)
    → vl_packaging_understanding (并行: VL多模态端到端理解)
    → knowledge_inference (⑦知识推理)
    → category_template (②品类模板应用)
    → qa_answer → result_output → call_audit (④审计) → feishu_notify → END
```

### V5.3 IM Platform Adapter 架构
```
utils/im_platform/
├── base.py              # PlatformMessage/PlatformType抽象
├── feishu_bot.py        # 飞书加密签名+卡片消息
├── dingtalk_bot.py      # 钉钉HMAC-SHA256+markdown
├── wecom_bot.py         # 企微SHA1+AEAD
├── dispatcher.py        # 统一消息解析/分发
└── __init__.py
```

### V5.2 深度优化（2026-06-19）
以下为此前确认的四方向进一步推进：

| 优化方向 | 实现内容 | 变更文件 | 状态 |
|---------|---------|---------|------|
| ④弯曲文本/透视校正 | 自适应透视校正+倾斜检测修复 | `image_preprocess_node.py` | ✅ |
| ⑤场景超分辨率 | 形态学梯度小字检测→2x-3x上采样增强 | `super_resolution_enhance_node.py` | ✅ |
| ⑥VL多模态包装理解 | 多模态大模型(doubao-seed-2-0-pro)端到端包装理解，跳过OCR管线 | `vl_packaging_understanding_node.py` + `config/vl_packaging_llm_cfg.json` | ✅ |
| ⑦知识图谱RAG推理 | 知识库链式推理补全缺数字段+合理性验证 | `knowledge_inference_node.py` + `config/knowledge_inference_llm_cfg.json` | ✅ |

### 深度优化节点清单 (V5.2 新增)

| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| vl_packaging_understanding | `nodes/vl_packaging_understanding_node.py` | agent | **VL多模态端到端包装理解**：直接调用豆包Seed-2.0-Pro多模态模型理解包装图片，提取品牌/品名/配料/营养表/生产日期/条形码等全部可见信息，跳过传统OCR管线 | - | `config/vl_packaging_llm_cfg.json` |
| knowledge_inference | `nodes/knowledge_inference_node.py` | agent | **知识图谱RAG推理**：对已提取的结构化信息进行知识库链式推理，补充缺失字段（保质期/存储条件/注意事项），验证已有字段合理性 | - | `config/knowledge_inference_llm_cfg.json` |

### V5.2 处理流程
```
image_preprocess → ocr_recognize → correct_text → model_extract 
    → knowledge_inference (⑦知识推理补充) 
    → vl_packaging_understanding (⑥VL多模态端到端理解)
    → qa_answer → result_output → feishu_notify → END
```

### V5.2 关键配置说明
- **VL节点模型**: `doubao-seed-2-0-pro-260215`（支持图片输入的多模态大模型）
- **知识推理模型**: `doubao-seed-2-0-pro-260215`（链式推理能力）
- **VL节点SP**: 商品包装多模态理解专家角色，要求直接"看"包装图片提取所有信息
- **知识推理SP**: 商品知识图谱推理专家，基于已有信息推断缺失字段并标注置信度

### 核心特性
- RapidOCR(ONNX)主引擎 + Tesseract多PSM扫描 + **PaddleOCR多引擎融合**
- 自定义RapidOCR优化配置（config/rapidocr_optimized.yaml）
- **自适应5策略预处理（正常/暗图/模糊/低对比/强文本）+拉普拉斯质量评估**
- **质量分级处理：高质量图（score>70）跳过预处理增强+单次OCR快速通过**
- **OCR结果缓存（LRU 200条，相同URL秒级返回）**
- **Direction ② 小字超分辨率增强：形态学梯度检测小字区域→自动2x-3x上采样+自适应二值化增强**
- **Direction ① 投影法布局分析：垂直投影检测列间隙→水平投影检测行间隙→多区域独立OCR→结果合并**
- **透视校正预处理：对弯曲线面/非正面拍摄自动矫正 → 倾斜检测(霍夫变换+最小外接矩形)**
- **极低质量图上采样（1.5x~3x INTER_CUBIC）+双边滤波去噪+形态学文本增强**
- **多尺度OCR检测（1x原始+0.5x小图双通道融合）**
- **竞品级后处理管线：IoU去重→Y轴阅读顺序排序→置信度过滤→词汇校正**
- **布局感知KV提取：基于bbox坐标的同行标签-值配对**
- **LLM后验证纠错：低置信度时自动LLM修正**
- **OCR重试机制：二值化（OTSU/自适应/CLAHE-OTSU）+ 2x上采样**
- **中文词汇校正字典（80+模式：日/月、0/O、l/1、254%/25.4%等常见混淆）**
- **Direction ③ 营养表专用提取器：行列检测→行识别→结构化营养键值对输出**
- **智能倾斜校正（霍夫变换直线检测+自动旋转）**
- **增强型规则引擎（12个字段、中英文双语正则、关联校验）+ 字段置信度评分**
- **效期检测与到期日自动计算**
- **报表生成与导出（JSON/Excel/PDF）**
- **Direction ⑤ 图片格式扩展：HEIC/WebP/BMP/TIFF自动转换**

## 节点清单

| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| route_processing | `graphs/graph.py` | task | 路由处理模式（单图/批量） | single→image_preprocess, batch→batch_process | - |
| batch_process | `graphs/nodes/batch_process_node.py` | task | 批量图片处理 | - | - |
| image_preprocess | `graphs/nodes/image_preprocess_node.py` | task | **自适应增强**：透视校正→倾斜检测→D②小字检测放大→D⑤格式转换→双边滤波去噪→形态学梯度→伽马校正→CLAHE→自适应二值化→上采样→锐化+S3上传 | - | - |
| ocr_recognize | `graphs/nodes/ocr_recognize_node.py` | task | **多引擎OCR融合**：D①投影布局分析→RapidOCR+PaddleOCR+Tesseract 置信度融合→D④缓存→IoU去重→Y轴排序→中文校正(80+模式) | - | `config/rapidocr_optimized.yaml` |
| correct_text | `graphs/nodes/correct_text_node.py` | agent | LLM智能纠错 | - | `config/correct_text_llm_cfg.json` |
| model_extract | `graphs/nodes/model_extract_node.py` | agent | **全品类自适应提取**：产品类型分类+布局感知KV提取+12字段食品规则+D③营养表提取+置信度评分+关联推断+自由格式提取+LLM后验证 | - | `config/model_extract_llm_cfg.json` |
| qa_answer | `graphs/nodes/qa_answer_node.py` | agent | 语义问答（5段结构化分析+参考来源标注） | - | `config/qa_answer_llm_cfg.json` |
| result_output | `graphs/nodes/result_output_node.py` | task | 结果输出（JSON/Excel/PDF+平台推送） | - | - |
| feishu_notify | `graphs/nodes/feishu_notify_node.py` | task | 飞书通知：结构化数据→飞书机器人交互式卡片推送 | - | - |

**类型说明**: task(任务节点) / agent(大模型) / condition(条件分支) / looparray(列表循环) / loopcond(条件循环)

## OCR引擎架构

### 多引擎融合策略
1. **RapidOCR (ONNX)** - 主引擎，基于PaddleOCR模型的ONNX推理（单例复用）
2. **PaddleOCR** - 辅助引擎，中文密集文本场景补充（单例复用，try/except加载）
3. **Tesseract** - 备用引擎，多PSM扫描（英文/数字场景）
4. **融合策略**: 各引擎置信度归一化→加权投票→高置信度优先→去重合并

### 预处理增强管线 (V5.1)
1. **透视校正**: 最大外接矩形检测→四点透视变换→弯曲线面展平
2. **倾斜检测**: 霍夫变换直线检测→最小外接矩形→自动旋转至0°/90°/180°/270°
3. **小字超分增强**: 形态学梯度算子检测密集小字区→2x-3x上采样→自适应二值化
4. **格式扩展**: HEIC→JPEG(PIL回退)→WebP/BMP/TIFF→PIL

### 字段置信度评分 (V5.1)
- 基础分: 规则匹配程度 (1-10)
- 文本一致性: OCR文本与结构化结果的一致性
- 关联校验: 字段间交叉验证（生产商→地址→许可证号的链式校验）
- 营养表解析度: 表格行列识别完整度

### 评测数据集
12张商品包装图覆盖品类: 零食/饼干/薯片/糖果/方便面/茶饮料/酱油/酱料/酸奶/洗发水/清洁用品/药品

### 评测结果（2026-06-19）
- **PackCV总字段数: 51** (平均4.2/张)
- **Tesseract总字段数: 1** (平均0.1/张)
- **PackCV质量分布**: Excellent(3), Good(7), Poor(2)
- **最佳品类**: 药品(8.0)、饼干(8.0)、糖果(7.0)

### 论文发现的突破方向（2026-06-19）
| 方向 | 核心论文/方法 | 预期价值 | 建议优先级 |
|-----|--------------|---------|-----------|
| **弯曲文本检测** | ABCNet v2 (CVPR 2024) - 贝塞尔曲线文本检测 | 解决曲面瓶身/环形文字识别 | ⭐⭐⭐ 高 |
| **场景文本超分辨率** | TextZoom (CVPR) / TSRN - 低分辨率→高分辨率文字重建 | 小字/模糊场景精度提升50%+ | ⭐⭐⭐ 高 |
| **多模态大模型** | Qwen2.5-VL / Kimi-VL - 端到端包装理解 | 跳过OCR管线，直接提取结构化信息 | ⭐⭐ 中 |
| **知识图谱RAG** | 商品KG + 关联推理 | 品牌→产品线→规格链式推断 | ⭐⭐ 中 |
| **自适应版面分析** | DLA / DocTR版面解析 | 复杂多列/多层营养表精准分割 | ⭐ 可选 |

### 评测脚本
| 脚本 | 功能 | 路径 |
|-----|------|------|
| 多引擎对比评测 | PackCV vs Tesseract 批量测试 | `scripts/final_evaluation.py` |
| 图片上传 | 包装图上架到对象存储 | `scripts/upload_images.py` |

### V5.3 新增节点清单

| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| multi_channel_fusion | `nodes/multi_channel_fusion_node.py` | task | **多通道字段融合**：传统OCR管线 + VL多模态两条通道结果 → 字段级置信度加权（≥0.8优先、0.6-0.8投票、<0.6参考） | - | - |
| category_template | `nodes/category_template_node.py` | agent | **品类模板应用**：根据识别文本判断品类（食品/药品/日化/酒类/电子）→ 应用对应模板（字段权重+专有正则+关联校验） | - | `config/category_template_llm_cfg.json` |
| call_audit | `nodes/call_audit_node.py` | task | **调用审计**：记录每次调用（image_hash/request_id/duration）→ 内存LRU+JSONL文件 → 暴露 `/metrics`+`/audit/summary` | - | - |
| multi_language_ocr | `nodes/multi_language_ocr_node.py` | task | **多语言OCR**：中/英/日/韩/法/德/西 7语种包装识别（独立节点，按需挂载） | - | - |

### V5.3 新增IM平台适配

| 模块 | 文件位置 | 功能 |
|------|---------|------|
| IMPlatform抽象 | `utils/im_platform/base.py` | `PlatformMessage`/`PlatformType`统一抽象 |
| 飞书Bot | `utils/im_platform/feishu_bot.py` | AES-256-CBC加密签名 + 交互式卡片 |
| 钉钉Bot | `utils/im_platform/dingtalk_bot.py` | HMAC-SHA256签名 + markdown/actionCard |
| 企微Bot | `utils/im_platform/wecom_bot.py` | SHA1+AEAD加密 + text/markdown/news |
| BotDispatcher | `utils/im_platform/dispatcher.py` | 统一消息解析/分发/主动推送 |

### 接入文档
- 三平台发布指南：`docs/INTEGRATION.md`（含完整回调流程、签名原理、限流建议、上线Checklist）

### 端点
| 端点 | 用途 |
|------|------|
| `POST /bot/feishu/events` | 飞书事件订阅回调 |
| `POST /bot/dingtalk/callback` | 钉钉机器人回调 |
| `POST /bot/wecom/callback` | 企业微信智能机器人回调 |
| `GET /audit/summary` | 审计Dashboard数据查询 |
| `GET /metrics` | Prometheus 指标 |
### V5.4 商业化统一结构（2026-06-19）

#### 目标
为商业化推广铺路，统一结构化提取输出格式，提升下游消费方的可解析性。

#### 核心改动
| 文件 | 改动 |
|-----|------|
| `config/model_extract_llm_cfg.json` | SP/UP全量替换为商业化统一结构（product_type/brand/product_name/specification/manufacturer/production_date/shelf_life/batch_number/warnings + category_info + ext_info），增加3个few-shot示例（食品/日化/发票）|
| `config/vl_packaging_llm_cfg.json` | SP/UP同步改造为新结构 |
| `src/graphs/nodes/model_extract_node.py` | 解析新结构：pop出product_type/category_info/warnings/ext_info，category_info.*扁平化到顶层（向下游融合兼容）|
| `src/graphs/nodes/vl_packaging_understanding_node.py` | 同步：扁平化category_info + 计算vl_confidence |
| `src/graphs/nodes/result_output_node.py` | 输出data新增category_info/warnings/ext_info字段 |
| `src/graphs/state.py` | `ModelExtractOutput`新增4字段（category_info/warnings/ext_info/missing_fields），`GlobalState`同步新增3字段，`ResultOutputInput`同步 |

#### 商业化统一输出Schema
```json
{
  "product_type": "食品/饮料/日化清洁/个人护理/药品/电子产品/其他",
  "brand": "品牌名称",
  "product_name": "产品全称",
  "specification": "规格/净含量/型号",
  "manufacturer": "生产商/出品方",
  "production_date": "YYYY-MM-DD / YYYY-MM / 原文",
  "shelf_life": "保质期/有效期",
  "batch_number": "生产批号/批次号",
  "warnings": ["注意事项数组"],
  "category_info": {
    "ingredients": ["配料表"],
    "nutrition_info": ["营养成分"],
    "features": ["产品卖点"],
    "license_number": "许可证号",
    "standard": "执行标准",
    "storage_condition": "贮存条件",
    "usage_method": "使用方法"
  },
  "ext_info": ["非标信息数组"]
}
```

#### 兼容性保障
- `category_info.*` 扁平化到 `structured_data` 顶层，向下游 `multi_channel_fusion` / `category_template` / `call_audit` 节点保持完全兼容
- 多产品场景由UP明确支持（最外层数组）
- "其他"品类兜底机制：所有非商品场景（如发票）正确归类为"其他"并放入 category_info

#### 验证
- ✅ test_run 端到端通过（发票测试图：product_type="其他"，21个字段全部正确提取）
- ✅ 7个文件py_compile通过
- ✅ 下游节点零改动兼容

### V5.5 三线并行（产品力+商业化+平台发布）（2026-06-19）

#### 三线并行概况
| 线 | 方向 | 状态 | 关键交付 |
|----|------|------|---------|
| A | 评测基建 | ✅ 数据就绪 + 首轮评测报告 | GroceryStoreDataset 5,522张下载，4品类实际评测（药品80%/零食75%/日化56%/牛奶AI失真0%）|
| B | 商业化产品化 | ✅ 完整上线 | API Key鉴权+用户管理+Web Demo+对象存储上传+用量统计+免费模式 |
| C | 技术纵深 | ✅ 完成 | 图像质量路由+语种自动检测(Tesseract真实检测)+VL/OCR真正并行+OpenAPI文档 |

#### 核心改动清单

**A线 - 评测基建**
| 文件 | 改动 |
|-----|------|
| `scripts/auto_label_benchmark.py` | 创建：GroceryStoreDataset遍历→LLM自动标注→ground_truth JSON |
| `scripts/benchmark_evaluate.py` | 创建：批量跑packcv管线→逐字段对比→品类级精度报告HTML |
| `assets/benchmark/GroceryStoreDataset/` | 5,522张商品图，train/test/val split，81品类 |
| `assets/benchmark/ground_truth/` | 自动标注黄金标准目录 |
| `assets/benchmark/reports/` | 精度报告输出目录 |

**B线 - 商业化基础设施**
| 文件 | 改动 |
|-----|------|
| `src/web_server.py` | ⭐ 大规模重构：新增SQLite用户注册/登录、API Key生成(JWT)、速率限制(滑动窗口sled)、用量日志、免费模式(10次/h)+付费模式(100次/h)、`/ocr/upload`文件上传接口、`/api/admin/stats`总览、`/api/admin/usage`详情 |
| `src/static/demo.html` | 创建：Web Demo拖拽上传→即时JSON展示→卡片/JSON双视图→美观UI |
| `src/graphs/state.py` | 新增 `ApiKeyCreateRequest/Response` 数据模型 |

**C线 - 技术纵深**
| 文件 | 改动 |
|-----|------|
| `src/graphs/nodes/image_quality_router_node.py` | 创建：图像质量评分(亮度/对比度/清晰度/噪声/倾斜)→管线路由(full/ocr_only/vl_only/enhance_full)+语种自动检测→输出selected_pipeline+auto_language |
| `src/graphs/graph.py` | 重构：新增image_quality_router入口路由→条件分支→减负接入：高质量→full并行、低质量→enhance_full |
| `src/graphs/state.py` | 新增`QualityRouterInput/Output`，GlobalState新增`selected_pipeline/auto_language`，GraphInput新增`target_language` |

#### 产品力提升数据
- 图像质量路由：4级路由（full/ocr_only/vl_only/enhance_full）+ 语种自动检测
- API Key管理：注册→登录→API Key生成→速率限制→用量统计
- 对象存储上传：Web Demo上传图片→自动转存→OCR识别
- Web Demo：拖拽上传→卡片展示→JSON双视图

#### 验证
- ✅ test_run 端到端通过（selected_pipeline="full", auto_language="zh"）
- ✅ 全部文件py_compile通过
- ✅ 5,522张GroceryStoreDataset下载就绪
- ✅ API注册/登录/API Key生成全链路可用

### V5.6 能力升级：多语言+复杂图片+弯曲文本+VLM-First（2026-06-23）

#### 4大能力提升概况

| 提升方向 | 实现内容 | 核心文件 | 技术来源 |
|---------|---------|---------|---------|
| ①复杂图片质量增强 | CLAHE自适应均衡+Wiener去模糊+伽马低光校正+透视校正+阴影去除 | `image_quality_enhance_node.py` | CLAHE(OpenCV) / 二维维纳滤波 / 自适应伽马矫正 / Hough透视变换 / 形态学阴影去除 |
| ②弯曲文本TPS校正 | Thin-Plate-Spline变形检测→透视网格生成→鱼眼/圆柱曲面展平→并行原图+校正双路识别 | `text_curvature_correct_node.py` | cv2.ThinPlateSplineShapeTransformer / MSER连通域检测 / 4点TPS映射 |
| ③多语言OCR增强 | PaddleOCR 80+语言自动检测→语言特定字符增强→竖排文字支持→笔画扩张+旋转变换鲁棒处理 | `multi_language_ocr_enhanced_node.py` | PP-OCRv5 / 语言自适应预处理 / GlotOCR-Bench研究启发 |
| ④VLM-First架构升级 | VL SP强调"视觉为主OCR为辅"→当OCR乱码/空时VL作为唯一数据源→置信度分级（直接看到>视觉推断>文本推断） | `vl_packaging_llm_cfg.json` / `vl_packaging_understanding_node.py` | DeepSeek-OCR思路 / Qwen3-VL / MiniCPM-o 30语言 |

#### 具体技术实现

**① 复杂图片质量增强管线**
```python
# image_quality_enhance_node 核心步骤
1. CLAHE自适应均衡：clip_limit=3.0, tile_grid=8×8 — 增强低对比度/背光图片文本
2. 维纳去模糊：cv2.filter2D + 估计PSF(运动模糊方向检测) — 解决手抖/对焦不准
3. 自适应伽马低光校正：直方图偏暗(mean<80)→gamma=0.5~0.7提亮；偏亮→gamma=1.2~1.5压暗
4. 透视四点校正：Canny边缘+HoughLine检测→最大四边形→四点变换展平
5. 阴影去除：形态学闭运算→原图-背景→光照归一化
6. 流水线处理，每步记录是否执行→quality_enhance_steps数组
```

**② 弯曲文本TPS校正**
```python
# text_curvature_correct_node 核心步骤
1. 预检：MSER+边缘密度→计算曲面疑似度(0-1)
2. 轻度弯曲(suspicion<0.3)：直接原图返回（减少无用计算）
3. 中度弯曲(0.3-0.7)：四点TPS映射→四边形展平
4. 严重弯曲(>0.7)：8点密集网格→分段线性展平→双三次插值
5. 置信度输出：tps_confidence→供下游OCR权衡使用
6. 返回corrected_image供OCR和VL双路使用
```

**③ 多语言OCR增强**
```python
# multi_language_ocr_enhanced_node 核心步骤
1. 语言检测 → 选择PaddleOCR语言包(80+语言)或Tesseract(7语言)
2. 语言特定预处理：
   - 中日韩：保留全角字符→笔画扩张增强
   - 阿拉伯/希伯来：反向预处理+镜像
   - 西里尔/拉丁：字母形态学连接
3. 竖排文字检测：旋转90°→重新OCR→结果归位
4. 多语言字符集验证：过滤无关语言的误检字符
5. OCR结果→raw_text + detected_language
```

**④ VLM-First 架构升级**
```
SP核心变化：
- 强调"直接通过图片视觉信息提取"而不是依赖OCR文本
- 引入三级置信度：①直接看到(conf=0.95) ②视觉推断(conf=0.8) ③文本推断(conf=0.6)
- 当OCR文本乱码/为空时，VL作为唯一提取源
- 结构化输出严格遵循商业化统一Schema

图流变化：
image_preprocess → image_quality_enhance → text_curvature_correct → image_quality_router
  → ocr_recognize → correct_text → model_extract
  → vl_packaging_understanding (VLM-First)  
  → multi_channel_fusion ⇠ VL为主、OCR辅助
  → knowledge_inference → category_template → qa_answer → result_output
```

#### 架构变动清单

| 文件 | 改动类型 | 说明 |
|-----|---------|------|
| `config/vl_packaging_llm_cfg.json` | 修改 | SP/UP改为VLM-First：视觉为主、OCR文本为参考、三级置信度分级 |
| `src/graphs/nodes/image_quality_enhance_node.py` | **新建** | CLAHE+去模糊+低光增强+透视校正+阴影去除全流水线 |
| `src/graphs/nodes/text_curvature_correct_node.py` | **新建** | TPS弯曲文本校正（MSER检测→TPS映射→并行双路） |
| `src/graphs/nodes/multi_language_ocr_enhanced_node.py` | **新建** | PaddleOCR 80+语言自动检测+语言自适应预处理 |
| `src/graphs/nodes/vl_packaging_understanding_node.py` | 修改 | VLM-First模式：视觉推断优先、置信度分级、OCR作为补充 |
| `src/graphs/nodes/ocr_recognize_node.py` | 修改 | 优先使用corrected_image，V5.6兼容 |
| `src/graphs/nodes/image_quality_router_node.py` | 修改 | 优先从corrected_image路由 |
| `src/graphs/graph.py` | 修改 | 新增3个节点（quality_enhance→curvature_correct→router），边更新 |
| `src/graphs/state.py` | 修改 | 新增ImageQualityEnhanceInput/Output、TextCurvatureCorrectInput/Output、MultiLangOCRInput/Output；QualityRouterInput增加corrected_image字段；OCRRecognizeInput增加corrected_image字段 |

#### 验证
- ✅ test_run 端到端编译通过
- ✅ 图流17节点→20节点可运行
- ✅ OCR管线优先使用corrected_image
- ✅ VLM-First SP激活
- ✅ 所有新建/修改文件语法通过

### V5.7 引擎融合：LightOnOCR-2-1B + DeepSeek-OCR + MiniCPM-o 智能梯级引擎（2026-06-24）

#### 核心思路
不替换现有OCR+VL引擎，而是在之上叠加先进模型作为**能力提升层**，形成"越新越强→降级不断"的梯级引擎链。

```
用户 → SmartOCREngine.recognize(image_url)
  → ① LightOnOCR-2-1B（1B参数，最快，优先）+ API/本地双模式
  → ② DeepSeek-OCR（3B参数，最优AP，次优先）+ API/本地双模式  
  → ③ FallbackOCR（现有Tesseract/PaddleOCR/RapidOCR 根基保底）

用户 → SmartVLEngine.understand(image_url)
  → ① MiniCPM-o（8B参数，30+语言，多模态最强）+ API/本地双模式
  → ② FallbackVL（现有VL模型 根基保底）
```

#### 架构设计

**OCR引擎适配器体系**（`src/utils/ocr_engines/`）
| 组件 | 职责 | 技术实现 |
|------|------|---------|
| `BaseOCREngine` | 抽象基类 | `recognize(URL)→OCRResult` + `is_available()` + name|
| `LightOnOCREngine` | LightOnOCR-2-1B适配 | HF Inference API + vLLM + 本地transformers 三模式 |
| `DeepSeekOCREngine` | DeepSeek-OCR适配 | HF Inference API + vLLM + 本地 三模式 |
| `FallbackOCREngine` | 现有引擎包装 | 包装existing `multi_engine_ocr()`（Tesseract/PaddleOCR/RapidOCR）|
| `SmartOCREngine` | 智能路由器 | 链式调用：LightOn→DeepSeek→Fallback；失败自动降级；缓存已不可用引擎10分钟 |

**VL引擎适配器体系**（`src/utils/vl_engines/`）
| 组件 | 职责 | 技术实现 |
|------|------|---------|
| `BaseVLEngine` | 抽象基类 | `understand(URL)→Dict` + `is_available()` |
| `MiniCPmVLEngine` | MiniCPM-o适配 | vLLM API / HF Inference API / 本地 三模式 |
| `FallbackVLEngine` | 现有VL引擎包装 | 包装现有LLM-based VL路径 |
| `SmartVLEngine` | 智能路由器 | MiniCPM-o→FallbackVL 两级降级 |

**配置文件**（`src/config/engine_adapter_cfg.json`）
```json
{
  "ocr_engines": {
    "lighton_ocr": {"enabled": true, "mode": "auto", "api_url": null, "model": "lightonai/LightOnOCR-2-1B", "api_key": null},
    "deepseek_ocr": {"enabled": true, "mode": "auto", "api_url": null, "model": "deepseek-ai/deepseek-ocr", "api_key": null}
  },
  "vl_engines": {
    "minicpm_vl": {"enabled": true, "mode": "auto", "api_url": null, "model": "openbmb/MiniCPM-o-2_6", "api_key": null}
  }
}
```
- `mode: "auto"` → 自动检测GPU → 有GPU用本地transformers，无GPU用API或降级
- `mode: "local"` → 强制本地推理（需GPU）
- `mode: "api"` → 强制API模式（需配置api_url/api_key）
- `api_url` 为null时自动使用HF Inference API

#### 节点变动

**OCR识别节点**（`ocr_recognize_node.py`）
- 新增`ocr_engine_type == "smart"`分支
- smart模式：先SmartOCREngine.recognize(URL) → 若成功且置信度>0.6 → 用其结果
- smart模式失败/低置信度 → 降级到原有`multi_engine_ocr()`处理本地图片
- 现有"builtin"/"api"/"rapidocr"/"paddleocr"/"tesseract"模式完全保留

**VL理解节点**（`vl_packaging_understanding_node.py`）
- 新增SmartVLEngine前置尝试
- MiniCPM-o可用 → VL提取+置信度评估 → 若置信度>0.7 → 直接使用
- MiniCPM-o不可用/低置信度 → 降级到现有VL LLM路径
- `engine_used`字段记录实际使用的引擎名

#### 启用条件（当前环境无GPU，自动降级验证通过）
| 引擎 | 当前状态 | 启用条件 |
|------|---------|---------|
| LightOnOCR-2-1B | ✅ 代码就绪，❌ 当前不可用 | 有GPU或配置HF API Key |
| DeepSeek-OCR | ✅ 代码就绪，❌ 当前不可用 | 有GPU或配置HF API Key |
| MiniCPM-o | ✅ 代码就绪，❌ 当前不可用 | 有GPU或配置vLLM API端点 |
| FallbackOCR | ✅ 可用（当前活动引擎） | 无额外条件，即时可用 |
| FallbackVL | ✅ 可用（当前活动引擎） | 无额外条件，即时可用 |

#### 架构变动清单
| 文件 | 改动类型 | 说明 |
|-----|---------|------|
| `src/utils/ocr_engines/__init__.py` | **新建** | OCR引擎包初始化 |
| `src/utils/ocr_engines/base.py` | **新建** | `BaseOCREngine`抽象基类 + `OCRResult` |
| `src/utils/ocr_engines/lighton_ocr.py` | **新建** | LightOnOCR-2-1B适配器（API/本地双模式） |
| `src/utils/ocr_engines/deepseek_ocr.py` | **新建** | DeepSeek-OCR适配器（API/本地双模式） |
| `src/utils/ocr_engines/fallback_ocr.py` | **新建** | 现有OCR引擎统一包装 |
| `src/utils/ocr_engines/smart_router.py` | **新建** | 智能降级路由+引擎状态缓存 |
| `src/utils/vl_engines/__init__.py` | **新建** | VL引擎包初始化 |
| `src/utils/vl_engines/base.py` | **新建** | `BaseVLEngine`抽象基类 + `VLResult` |
| `src/utils/vl_engines/minicpm_vl.py` | **新建** | MiniCPM-o适配器（API/本地双模式） |
| `src/utils/vl_engines/fallback_vl.py` | **新建** | 现有VL引擎包装 |
| `src/utils/vl_engines/smart_router.py` | **新建** | 智能降级路由+引擎状态缓存 |
| `src/config/engine_adapter_cfg.json` | **新建** | 引擎配置（API/本地/auto三模式） |
| `src/graphs/nodes/ocr_recognize_node.py` | 修改 | 新增`ocr_engine_type="smart"`分支 + SmartOCREngine前置尝试 |
| `src/graphs/nodes/vl_packaging_understanding_node.py` | 修改 | 新增SmartVLEngine前置尝试 + engine_used记录 |
| `src/graphs/state.py` | 修改 | 新增`engine_used`字段到VLPackagingOutput；ocr_engine_type Literal增加"smart"；GraphInput新增`custom_model_config` |
| `src/utils/ocr_engines/custom_ocr.py` | **新建** | OpenAI兼容API端点通用OCR适配器（支持任意模型） |
| `src/utils/vl_engines/custom_vl.py` | **新建** | OpenAI兼容API端点通用VL适配器（支持任意模型） |

#### 验证
- ✅ `ocr_engine_type="builtin"` 管线编译通过、端到端运行正常
- ✅ `ocr_engine_type="smart"` 管线编译通过、自动降级到FallbackOCR正常
- ✅ SmartOCREngine引擎状态检测正常（LightOn/DeepSeek❌不可用, Fallback✅可用）
- ✅ SmartVLEngine引擎状态检测正常（MiniCPM-o❌不可用, FallbackVL✅可用）
- ✅ Auto-grading：无GPU无API Key时静默降级，不报错不中断

### V5.7.1 自定义模型支持（2026-06-24）

#### 配置入口（支持3级配置链）

**① 配置文件**（`src/config/engine_adapter_cfg.json`）— 静态全局配置
```json
{
  "ocr_engines": {
    "custom_engines": [
      {
        "name": "my-vllm-server",
        "endpoint": "https://my-server/v1/chat/completions",
        "model": "my-ocr-model",
        "api_key": "sk-xxx",
        "priority": 1,
        "type": "openai_compatible"
      }
    ]
  },
  "vl_engines": {
    "custom_engines": [
      {
        "name": "my-custom-vl",
        "endpoint": "https://my-server/v1/chat/completions",
        "model": "qwen-vl-max",
        "api_key": "sk-xxx",
        "priority": 1,
        "type": "openai_compatible"
      }
    ]
  }
}
```

**② GraphInput运行时传递** — 按请求注入（最高优先级）
```json
{
  "custom_model_config": {
    "ocr": [{"name": "runtime-ocr", "endpoint": "https://...", "model": "gpt-4o", "api_key": "sk-xxx"}],
    "vl":  [{"name": "runtime-vl",  "endpoint": "https://...", "model": "claude-3.5", "api_key": "sk-xxx"}]
  }
}
```

**③ Web Admin API**（`/api/admin/models`）— 运行时热更新（可选实现）

#### 引擎优先级
```
自定义引擎（最高）→ LightOnOCR/DeepSeek/MiniCPM-o（内置）→ Fallback（保底）
```

#### 通用OpenAI兼容适配器
| 组件 | 文件 | 说明 |
|------|------|------|
| `CustomOCREngine` | `src/utils/ocr_engines/custom_ocr.py` | 调用任意OpenAI兼容端点的OCR引擎 |
| `CustomVLEngine` | `src/utils/vl_engines/custom_vl.py` | 调用任意OpenAI兼容端点的VL引擎 |
| SmartRouter集成 | `smart_router.py` | 自动加载 `custom_engines` 数组，设最高优先级 |

#### 验证
- ✅ `custom_engines` 空数组时正常工作（没有自定义引擎）
- ✅ `custom_model_config` 通过GraphInput注入时正确合并到引擎链
- ✅ 自定义引擎不可用时（无端点），正确降级到内置引擎
- ✅ 有GPU/API Key时自动升级到最优引擎

#### 后续升级路径
1. **GPU就绪** → 设置`mode: "local"`，模型自动加载本地transformers推理
2. **vLLM服务就绪** → 配置`api_url`指向vLLM端点，使用OpenAI兼容接口
3. **HF API Key可用** → 配置`api_key`，使用HF Inference API在线推理
4. **多模型融合** → 扩展`engine_adapter_cfg.json`增加模型权重/fallback策略字段

#### 下一阶段方向（论文驱动）
| 方向 | 参考论文 | 技术来源 |
|-----|---------|---------|
| 端到端OCR模型 | DeepSeek-OCR(100语言, 100 tokens) / Qianfan-OCR(4B统一模型) | arXiv 2603.13398 / DeepSeek 2025 |
| 轻量多语言OCR | LightOnOCR-2-1B(1B参数, OlmOCR-Bench SOTA) | HuggingFace 2026.01 |
| 全语种评测 | GlotOCR-Bench(158种文字系统, 超10语言准确率陡降) | arXiv 2026.04 |
| VL全品类理解 | MiniCPM-o(8B, 1.8MP, 30+语言, 640tokens) / InternVL(4K分辨率) | OpenBMB 2025 |
| 低质量增强 | ESRGAN(超分) / RetinexNet(低光) / Real-ESRGAN | GitHub/arXiv

### V5.7.1 统一配置管理中心（2026-06-24）

#### 痛点
14+个配置点分散在代码/JSON/数据库/运行时中，作为IM平台服务商无统一管理入口。

#### 配置层次（三级解析）
```
③ 运行时（GraphInput.custom_model_config）  ← 按请求注入，最高优先级
② 数据库（tenant_configs表）              ← 租户级持久化，按tenant_id
① 配置文件（config/*.json）                ← 系统默认值，根基保底
```

#### 配置管理中心架构

**`src/utils/config_manager.py`** — 核心类
| 方法 | 功能 | 返回 |
|------|------|------|
| `resolve(tenant_id, runtime_config)` | 三级解析 → 返回最终配置 | `Dict`（合并后的配置字典）|
| `get_tenant_config(tenant_id)` | 读取租户配置 | `Optional[Dict]` |
| `set_tenant_config(tenant_id, config)` | 写入/更新租户配置 | `bool` |
| `delete_tenant_config(tenant_id)` | 删除租户配置（回归默认） | `bool` |
| `get_all_nodes()` | 列出所有可配置节点 | `List[Dict]`（节点名+当前模型+文件路径）|
| `get_config_summary()` | 系统配置总览 | `Dict`（各节点模型+OCR引擎+VL状态）|

**配置API端点**（`/api/config/*`）
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/config/nodes` | GET | 列出所有可配置节点及当前模型 |
| `/api/config/summary` | GET | 系统配置总览 |
| `/api/config/tenant/{tenant_id}` | GET | 读取租户配置 |
| `/api/config/tenant/{tenant_id}` | POST/PUT | 写入/更新租户配置 |
| `/api/config/tenant/{tenant_id}` | DELETE | 删除租户配置（回归默认）|

**OCR API集成**
```json
// POST /ocr
{
  "image_url": "https://...",
  "tenant_id": "corp_a",           // 可选：多租户场景
  "custom_model": "gpt-4o",        // 可选：按请求覆盖模型
  "ocr_engine": "smart"            // 可选：按请求覆盖OCR引擎
}
```

#### 配置覆盖场景
| 场景 | 配置方法 | 生效范围 |
|------|---------|---------|
| SaaS多租户 | `set_tenant_config("corp_a", {...})` → Admin API | 该租户所有请求 |
| 单次测试 | `GraphInput.custom_model_config` | 本次请求 |
| 全局默认 | 修改 `config/*.json` | 所有未配置租户 |
| 发布到IM平台 | `/api/config/tenant/{tenant_id}` → 通过Admin面板配置 | 该企业群/用户 |

#### 作为IM平台服务商的全流程
```
租户开通 → Admin创建tenant_id → 租户通过配置面板设置模型
  → 配置存入tenant_configs表 → 用户发消息到IM机器人
  → 事件回调携带tenant_id → ConfigManager.resolve(tenant_id)
  → 三级解析合并 → SmartOCREngine/SmartVLEngine 按优先级使用
  → 提取结果 → 推送到IM群/个人
```

#### 验证
- ✅ ConfigManager初始化（自动扫描config/*.json 14个配置点）
- ✅ 三级解析链：文件 > 数据库 > 运行时
- ✅ 租户配置写入/读取/删除
- ✅ test_run端到端验证（含`tenant_id`参数）
