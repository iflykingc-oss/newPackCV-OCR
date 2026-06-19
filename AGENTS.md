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
| A | 评测基建 | ✅ 数据就绪 | HalalBench找不到实际repo，GroceryStoreDataset 5,522张下载到assets/benchmark/ |
| B | 商业化产品化 | ✅ 完整上线 | API Key鉴权+用户管理+Web Demo+对象存储上传+用量统计+免费模式 |
| C | 技术纵深 | ✅ 完成 | 图像质量路由+语种自动检测+VL/OCR真正并行+OpenAPI文档 |

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
