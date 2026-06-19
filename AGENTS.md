# PackCV-OCR 项目结构索引

## 项目概述
- **名称**: PackCV-OCR V5.1
- **功能**: 面向货架/包装场景的高精度OCR识别解决方案 (全品类通用)
- **核心升级**: RapidOCR(ONNX)主引擎 + 自适应5策略预处理 + 多尺度检测 + 布局感知KV提取 + LLM后验证 + 全品类自由格式提取 + 专线营养表提取 + 投影法布局分析 + 小字超分增强 + 飞书机器人 + HTTP API服务 + **多引擎融合(PaddleOCR)** + **透视校正** + **置信度评分**

### V5.1 深度优化（2026-06-19）
| 优化方向 | 实现内容 | 变更文件 | 状态 |
|---------|---------|---------|------|
| ①多引擎OCR融合 | PaddleOCR+RapidOCR+Tesseract 置信度加权融合 | `ocr_recognize_node.py` | ✅ 已实现 |
| ②预处理升级 | 自适应透视校正+倾斜检测修复+超分辨率增强 | `image_preprocess_node.py` | ✅ 已实现 |
| ③字段推理增强 | 置信度评分+关联字段推断+营养表多级解析 | `model_extract_node.py` | ✅ 已实现 |

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