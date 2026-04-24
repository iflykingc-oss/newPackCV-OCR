# PackCV-OCR 算法优化迭代方案 V1.1

## 一、调研背景

本次优化基于对以下先进OCR项目的深入调研：
- **Umi-OCR**（41.2k stars）：开源离线OCR软件，优秀的排版解析和忽略区域功能
- **PaddleOCR 3.0**（52.4k stars）：百度开源，PP-OCRv5多语言模型，PP-StructureV3复杂文档解析
- **MMOCR**（OpenMMLab）：完整的文本检测、识别、关键信息提取工具箱
- **货架场景应用**：STEP3-VL-10B、Qwen2.5-VL、HunyuanOCR等前沿模型
- **弯曲/旋转文本处理**：LCTP算法、EAST、DBNet++等先进检测算法

## 二、当前系统现状分析

### 2.1 已实现功能
✅ YOLOv8商品检测
✅ ROI分层裁切
✅ 并行处理引擎（10并发）
✅ 效期检测
✅ 库存分析
✅ 智能告警
✅ 自动报表生成
✅ 多平台推送（微信、飞书）
✅ 数据库存储

### 2.2 存在的不足
❌ 缺少图像预处理增强（去畸变、去噪、反光消除）
❌ 不支持文本方向分类（旋转文本识别）
❌ 不支持多语言OCR（仅支持中文）
❌ 缺少排版解析（多栏、自然段换行）
❌ 缺少忽略区域功能（无法排除水印、LOGO）
❌ 缺少文本后处理（半全角转换、文本纠错）
❌ 缺少关键信息提取（SDMGR算法）
❌ 缺少弯曲文本矫正
❌ 缺少文档方向分类

## 三、优化目标（V1.1）

### 3.1 核心目标
1. **提升识别准确率**：通过图像预处理增强和文本方向矫正，提升复杂场景下的识别准确率20%+
2. **扩展应用场景**：支持多语言、弯曲文本、旋转文本
3. **优化用户体验**：排版解析、忽略区域、文本纠错
4. **增强货架场景**：细粒度商品识别、合规性分析

### 3.2 技术指标
- 文本方向分类准确率：>99%（PP-LCNet）
- 多语言支持：5种语言（中、英、日、简繁中、拼音）
- 弯曲文本识别准确率提升：49.5%（LCTP算法）
- 任意角度文本检测：支持0-360度旋转
- 排版解析准确率：>95%（多栏、自然段换行）

## 四、优化方案详解

### 4.1 图像预处理增强节点（新增）

#### 功能特性
- **文档方向分类**：自动识别图像方向（0/90/180/270度）
- **文档去畸变**：解决拍摄倾斜问题
- **图像增强**：
  - 高斯模糊去噪
  - 直方图均衡化增强对比度
  - 反光消除（基于阈值处理）
  - 锐化增强（提高文字清晰度）

#### 技术实现
```python
# 文档方向分类
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 图像去畸变
# 使用透视变换矫正倾斜文档

# 图像增强
# - 高斯模糊：cv2.GaussianBlur
# - 直方图均衡化：cv2.equalizeHist
# - 反光消除：基于局部二值化
# - 锐化：cv2.filter2D with kernel
```

#### 状态定义
```python
class ImagePreprocessInput(BaseModel):
    image: File = Field(..., description="原始图片")
    enable_orientation_classify: bool = Field(default=True, description="是否启用方向分类")
    enable_dewarp: bool = Field(default=True, description="是否启用去畸变")
    enable_denoise: bool = Field(default=True, description="是否启用去噪")
    enable_enhance: bool = Field(default=True, description="是否启用增强")

class ImagePreprocessOutput(BaseModel):
    preprocessed_image: File = Field(..., description="预处理后的图片")
    orientation_angle: int = Field(default=0, description="检测到的方向角度")
    is_corrected: bool = Field(default=False, description="是否进行了矫正")
    enhancement_steps: List[str] = Field(default_factory=list, description="执行的增强步骤")
```

### 4.2 文本方向矫正节点（新增）

#### 功能特性
- **任意角度检测**：使用EAST/DBNet++支持0-360度旋转文本
- **自动旋转矫正**：基于检测到的角度自动旋转图片
- **边缘投影法**：小角度倾斜矫正（±45度以内）
- **文本行方向分类**：PP-LCNet（99.42%准确率）

#### 技术实现
```python
# 文本行方向分类
from paddleocr import PPStructureV3
pipeline = PPStructureV3(use_textline_orientation=True)

# 边缘投影法（小角度矫正）
def correct_rotation_edge_projection(image):
    angles = range(-45, 46, 1)
    best_angle = 0
    max_zero_count = 0
    for angle in angles:
        rotated = rotate_image(image, angle)
        projection = horizontal_projection(rotated)
        zero_count = np.sum(projection == 0)
        if zero_count > max_zero_count:
            max_zero_count = zero_count
            best_angle = angle
    return rotate_image(image, -best_angle), best_angle
```

#### 状态定义
```python
class TextDirectionCorrectInput(BaseModel):
    image: File = Field(..., description="输入图片")
    use_edge_projection: bool = Field(default=True, description="是否使用边缘投影法")
    use_cls_model: bool = Field(default=True, description="是否使用分类模型")

class TextDirectionCorrectOutput(BaseModel):
    corrected_image: File = Field(..., description="矫正后的图片")
    detected_angle: float = Field(default=0.0, description="检测到的旋转角度")
    correction_method: str = Field(default="", description="使用的矫正方法")
    confidence: float = Field(default=0.0, description="置信度")
```

### 4.3 智能排版解析节点（新增）

#### 功能特性
- **多栏布局识别**：自动识别单栏、双栏、多栏布局
- **自然段换行**：按段落规则智能换行
- **保留缩进**：适用于代码、表格等需要保留缩进的内容
- **单栏/多栏模式切换**：根据内容自动选择最佳解析模式
- **横排/竖排文本处理**：支持从右到左的竖排文本

#### 技术实现
```python
# 排版解析策略
class LayoutParser:
    def parse(self, ocr_result: List[Dict]) -> str:
        # 1. 检测布局类型（单栏/多栏）
        layout_type = self.detect_layout(ocr_result)

        # 2. 根据布局类型选择解析策略
        if layout_type == "multi_column":
            return self.parse_multi_column(ocr_result)
        elif layout_type == "single_column":
            return self.parse_single_column(ocr_result)

    def detect_layout(self, ocr_result: List[Dict]) -> str:
        # 基于文本框位置分析布局
        boxes = [item["box"] for item in ocr_result]
        x_positions = [box[0] for box in boxes]
        x_std = np.std(x_positions)

        # 如果x坐标标准差大，说明是多栏布局
        if x_std > threshold:
            return "multi_column"
        else:
            return "single_column"
```

#### 状态定义
```python
class LayoutParseInput(BaseModel):
    ocr_regions: List[Dict[str, Any]] = Field(..., description="OCR识别区域列表")
    parse_mode: Literal["auto", "multi_column", "single_column", "preserve_indent"] = Field(default="auto", description="解析模式")
    enable_paragraph_break: bool = Field(default=True, description="是否启用自然段换行")

class LayoutParseOutput(BaseModel):
    parsed_text: str = Field(..., description="解析后的文本")
    layout_type: str = Field(default="", description="检测到的布局类型")
    paragraph_count: int = Field(default=0, description="段落数量")
    column_count: int = Field(default=1, description="栏数")
```

### 4.4 忽略区域配置节点（新增）

#### 功能特性
- **矩形区域标注**：绘制多个矩形框指定忽略区域
- **排除水印、LOGO**：自动排除水印、LOGO、页眉页脚
- **智能匹配**：支持模板匹配和位置匹配
- **批量应用**：一次配置，批量应用到所有图片

#### 技术实现
```python
# 忽略区域管理
class IgnoreRegionManager:
    def __init__(self):
        self.regions = []

    def add_region(self, x1, y1, x2, y2):
        self.regions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    def filter_ocr_result(self, ocr_result: List[Dict]) -> List[Dict]:
        filtered = []
        for item in ocr_result:
            box = item["box"]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # 检查中心点是否在忽略区域内
            is_ignored = False
            for region in self.regions:
                if (region["x1"] <= center_x <= region["x2"] and
                    region["y1"] <= center_y <= region["y2"]):
                    is_ignored = True
                    break

            if not is_ignored:
                filtered.append(item)

        return filtered
```

#### 状态定义
```python
class IgnoreRegionInput(BaseModel):
    image: File = Field(..., description="输入图片")
    ocr_regions: List[Dict[str, Any]] = Field(..., description="OCR识别区域列表")
    ignore_regions: List[Dict[str, int]] = Field(default_factory=list, description="忽略区域列表")
    auto_detect_watermark: bool = Field(default=True, description="是否自动检测水印")

class IgnoreRegionOutput(BaseModel):
    filtered_regions: List[Dict[str, Any]] = Field(..., description="过滤后的OCR区域")
    ignored_count: int = Field(default=0, description="忽略的区域数量")
    detected_watermarks: List[Dict[str, Any]] = Field(default_factory=list, description="检测到的水印位置")
```

### 4.5 多语言OCR支持（优化现有节点）

#### 功能特性
- **PP-OCRv5多语言模型**：单模型支持5种语言
  - 简体中文
  - 繁体中文
  - 英文
  - 日语
  - 拼音
- **语言自动检测**：自动识别文本语言
- **多语言混合识别**：支持中英文混合文档
- **扩展支持**：支持37种语言（法、西、葡、俄、韩等）

#### 技术实现
```python
# 多语言OCR配置
from paddleocr import PaddleOCR

# 自动语言检测
ocr = PaddleOCR(lang='ch')  # 自动检测中文

# 指定语言
ocr_en = PaddleOCR(lang='en')  # 英文
ocr_jp = PaddleOCR(lang='japan')  # 日语
ocr_multi = PaddleOCR(lang='chinese_v2')  # 多语言（中英日）

# 多语言混合识别
def multi_lang_ocr(image_path):
    # 尝试多种语言模型
    langs = ['ch', 'en', 'japan']
    results = []
    for lang in langs:
        ocr = PaddleOCR(lang=lang)
        result = ocr.ocr(image_path, cls=True)
        if result and result[0]:
            results.extend(result[0])

    # 合并结果，去除重复
    return merge_results(results)
```

#### 状态定义
```python
class MultiLangOCRInput(BaseModel):
    image: File = Field(..., description="输入图片")
    languages: List[Literal["ch", "en", "japan", "korean", "fr", "es", "ru"]] = Field(default=["ch"], description="支持的语言列表")
    auto_detect: bool = Field(default=True, description="是否自动检测语言")

class MultiLangOCROutput(BaseModel):
    ocr_text: str = Field(..., description="识别的文本")
    detected_languages: List[str] = Field(default_factory=list, description="检测到的语言")
    regions: List[Dict[str, Any]] = Field(default_factory=list, description="识别区域列表")
    confidence: float = Field(default=0.0, description="平均置信度")
```

### 4.6 弯曲文本矫正节点（新增）

#### 功能特性
- **LCTP算法**：长弯曲文本预处理
- **关键拐点检测**：计算弯曲文本的关键拐点
- **切分与融合**：对文本行进行智能切分和融合
- **准确率提升**：相比PP-OCR提升49.5%

#### 技术实现
```python
# LCTP算法实现
class LCTPCorrector:
    def correct(self, text_line_image):
        # 1. 检测关键拐点
        keypoints = self.detect_keypoints(text_line_image)

        # 2. 基于拐点切分文本行
        segments = self.segment_by_keypoints(text_line_image, keypoints)

        # 3. 融合切分后的片段
        corrected = self.merge_segments(segments)

        return corrected

    def detect_keypoints(self, image):
        # 使用骨架化或轮廓检测关键拐点
        skeleton = self.skeletonize(image)
        keypoints = self.find_corners(skeleton)
        return keypoints
```

#### 状态定义
```python
class CurvedTextCorrectInput(BaseModel):
    image: File = Field(..., description="弯曲文本图片")
    correction_method: Literal["lctp", "thin_plate_spline", "polynomial"] = Field(default="lctp", description="矫正方法")

class CurvedTextCorrectOutput(BaseModel):
    corrected_image: File = Field(..., description="矫正后的图片")
    curvature_degree: float = Field(default=0.0, description="弯曲程度")
    keypoints: List[Dict[str, float]] = Field(default_factory=list, description="检测到的关键拐点")
```

### 4.7 文本后处理节点（新增）

#### 功能特性
- **半全角转换**：自动转换半全角字符
- **文本纠错**：基于规则和模型的文本纠错
- **格式规范化**：统一数字、日期、货币格式
- **去重与合并**：合并重复的文本块

#### 技术实现
```python
# 文本后处理
class TextPostProcessor:
    def process(self, text: str) -> str:
        # 1. 半全角转换
        text = self.convert_full_half(text)

        # 2. 文本纠错
        text = self.correct_spelling(text)

        # 3. 格式规范化
        text = self.normalize_format(text)

        # 4. 去除多余空格和换行
        text = self.cleanup_whitespace(text)

        return text

    def convert_full_half(self, text: str) -> str:
        # 全角转半角
        return text.translate({
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            # ... 更多映射
        })
```

#### 状态定义
```python
class TextPostProcessInput(BaseModel):
    text: str = Field(..., description="输入文本")
    enable_full_half_convert: bool = Field(default=True, description="是否启用半全角转换")
    enable_spell_correct: bool = Field(default=True, description="是否启用拼写纠错")
    enable_format_normalize: bool = Field(default=True, description="是否启用格式规范化")

class TextPostProcessOutput(BaseModel):
    processed_text: str = Field(..., description="处理后的文本")
    corrections: List[Dict[str, str]] = Field(default_factory=list, description="进行的纠错列表")
```

### 4.8 关键信息提取节点（优化）

#### 功能特性
- **SDMGR算法**：语义感知文档图推理
- **多模态融合**：结合视觉和文本特征
- **模板匹配**：支持自定义模板
- **大模型增强**：结合LLM提升准确率

#### 技术实现
```python
# SDMGR关键信息提取
from mmocr.apis import MMOCRInferencer

inferencer = MMOCRInferencer(det='DBNet', rec='SARNet', kie='SDMGR')

result = inferencer(
    'invoice.png',
    return_vis=True,
    save_vis=True
)

# 提取关键信息
key_info = {
    'invoice_no': result['predictions'][0]['kie_result']['invoice_no'],
    'date': result['predictions'][0]['kie_result']['date'],
    'amount': result['predictions'][0]['kie_result']['amount']
}
```

#### 状态定义
```python
class KeyInfoExtractInput(BaseModel):
    image: File = Field(..., description="输入图片")
    ocr_text: str = Field(default="", description="OCR文本")
    extraction_fields: List[str] = Field(default_factory=list, description="需要提取的字段列表")
    use_template: bool = Field(default=False, description="是否使用模板")

class KeyInfoExtractOutput(BaseModel):
    extracted_info: Dict[str, Any] = Field(default_factory=dict, description="提取的关键信息")
    confidence: Dict[str, float] = Field(default_factory=dict, description="各字段的置信度")
```

### 4.9 货架场景增强（优化现有节点）

#### 功能特性
- **多粒度特征融合**：包装主视觉+文字区域+条形码
- **细粒度商品识别**：区分规格、年份、批次
- **合规性分析**：
  - 先进先出检查
  - 批号一致性
  - 效期管理
  - 陈列规范
- **库存智能监控**：缺货检测、压货预警

#### 技术实现
```python
# 多粒度特征融合
class MultiGranularityRecognition:
    def recognize(self, image, bbox):
        # 1. 提取主视觉特征
        visual_feature = self.extract_visual_feature(image, bbox)

        # 2. 提取文字区域特征
        text_feature = self.extract_text_feature(image, bbox)

        # 3. 提取条形码特征
        barcode_feature = self.extract_barcode_feature(image, bbox)

        # 4. 融合特征
        fused_feature = self.fuse_features([
            visual_feature,
            text_feature,
            barcode_feature
        ])

        # 5. 分类识别
        result = self.classify(fused_feature)
        return result

# 合规性分析
class ComplianceAnalyzer:
    def analyze_fifo(self, products):
        # 检查先进先出
        dates = [p['expiry_date'] for p in products]
        is_fifo = dates == sorted(dates)
        return is_fifo

    def analyze_batch_consistency(self, products):
        # 检查批号一致性
        batch_numbers = [p['batch_number'] for p in products]
        return len(set(batch_numbers)) <= 2  # 允许最多2个批号
```

#### 状态定义
```python
class ShelfEnhanceInput(BaseModel):
    shelf_image: File = Field(..., description="货架图片")
    detected_objects: List[Dict[str, Any]] = Field(..., description="检测到的商品")
    enable_compliance_check: bool = Field(default=True, description="是否启用合规检查")

class ShelfEnhanceOutput(BaseModel):
    enhanced_objects: List[Dict[str, Any]] = Field(..., description="增强后的商品信息")
    compliance_report: Dict[str, Any] = Field(default_factory=dict, description="合规性报告")
    inventory_suggestions: List[str] = Field(default_factory=list, description="库存建议")
```

## 五、工作流编排优化

### 5.1 单图处理工作流（优化后）
```
GraphInput
  ↓
[图像预处理增强节点] ✨新增
  ├── 方向分类
  ├── 去畸变
  ├── 去噪
  └── 增强
  ↓
[文本方向矫正节点] ✨新增
  ├── 边缘投影法
  └── 分类模型
  ↓
[多语言OCR节点] 🔧优化
  ├── PP-OCRv5多语言
  ├── 语言自动检测
  └── 多语言混合
  ↓
[弯曲文本矫正节点] ✨新增
  ├── LCTP算法
  └── 拐点检测
  ↓
[忽略区域配置节点] ✨新增
  ├── 排除水印
  └── 排除LOGO
  ↓
[智能排版解析节点] ✨新增
  ├── 多栏布局
  ├── 自然段换行
  └── 保留缩进
  ↓
[文本后处理节点] ✨新增
  ├── 半全角转换
  ├── 文本纠错
  └── 格式规范化
  ↓
[关键信息提取节点] 🔧优化
  ├── SDMGR算法
  └── 模板匹配
  ↓
[模型结构化提取节点] 🔧优化（增强）
  ↓
[结果输出节点]
  ↓
GraphOutput
```

### 5.2 PackCV-OCR工作流（优化后）
```
GraphInput (Shelf Image)
  ↓
[图像预处理增强节点] ✨新增
  ↓
[CV目标检测节点] 🔧优化（多粒度特征融合）
  ↓
[ROI分层裁切节点] 🔧优化（弯曲文本矫正）
  ↓
[文本方向矫正节点] ✨新增
  ↓
[多语言OCR节点] 🔧优化
  ↓
[智能排版解析节点] ✨新增
  ↓
[并行处理引擎] 🔧优化（包含弯曲文本矫正）
  ↓
[文本后处理节点] ✨新增
  ↓
[关键信息提取节点] 🔧优化
  ↓
[告警引擎节点] 🔧优化（合规性分析）
  ↓
[报表生成节点]
  ↓
GraphOutput
```

## 六、实施计划

### 阶段一：图像预处理优化（Week 1）
- [x] 实现图像预处理增强节点
- [x] 实现文本方向矫正节点
- [x] 集成PaddleOCR方向分类
- [x] 测试倾斜文档识别

### 阶段二：文本识别优化（Week 2）
- [ ] 实现多语言OCR支持
- [ ] 实现弯曲文本矫正节点
- [ ] 集成PP-OCRv5模型
- [ ] 测试多语言和弯曲文本识别

### 阶段三：智能后处理（Week 3）
- [ ] 实现智能排版解析节点
- [ ] 实现忽略区域配置节点
- [ ] 实现文本后处理节点
- [ ] 测试复杂排版处理

### 阶段四：货架场景增强（Week 4）
- [ ] 优化CV检测节点（多粒度特征融合）
- [ ] 优化ROI裁切节点（弯曲文本矫正）
- [ ] 优化告警引擎（合规性分析）
- [ ] 测试货架场景

### 阶段五：集成测试与优化（Week 5）
- [ ] 集成所有新节点
- [ ] 端到端测试
- [ ] 性能优化
- [ ] 文档更新

## 七、预期效果

### 7.1 识别准确率提升
- 倾斜文档：+30%
- 弯曲文本：+49.5%
- 旋转文本：+25%
- 多语言文档：+20%
- 复杂排版：+15%

### 7.2 应用场景扩展
- ✅ 支持多语言文档识别
- ✅ 支持弯曲/旋转文本识别
- ✅ 支持复杂排版解析
- ✅ 支持货架细粒度商品识别
- ✅ 支持合规性分析

### 7.3 用户体验提升
- ✅ 自动排除水印、LOGO干扰
- ✅ 智能排版解析，输出格式化文本
- ✅ 文本后处理，自动纠错
- ✅ 关键信息提取，减少人工整理

## 八、技术选型

### 8.1 OCR引擎
- **PaddleOCR 3.1.0**：主力引擎
  - PP-OCRv5多语言模型
  - PP-StructureV3复杂文档解析
  - PP-ChatOCRv4智能信息提取
- **MMOCR**：备选引擎
  - DBNet++文本检测
  - SARNet文本识别
  - SDMGR关键信息提取

### 8.2 CV检测
- **YOLOv8**：保持现有实现
- **多粒度特征融合**：新增视觉+文字+条码融合

### 8.3 图像处理
- **OpenCV**：图像预处理
- **PaddleOCR方向分类**：PP-LCNet_x1_0_textline_ori

### 8.4 文本处理
- **正则表达式**：格式规范化
- **pycorrector**：中文文本纠错
- **jieba**：分词（如需要）

## 九、依赖安装

```bash
# PaddleOCR
uv add paddlepaddle paddleocr

# MMOCR
uv add mmdet mmocr

# 图像处理
uv add opencv-python-headless

# 文本处理
uv add pycorrector jieba

# 其他
uv add numpy pandas pillow
```

## 十、风险评估与应对

### 10.1 技术风险
- **风险**：PP-OCRv5模型较大，推理速度慢
- **应对**：提供轻量级模型选择（MobileNetV3）

### 10.2 兼容性风险
- **风险**：新节点与现有工作流不兼容
- **应对**：保持向后兼容，新增可选参数

### 10.3 性能风险
- **风险**：新增节点导致整体处理时间增加
- **应对**：优化节点并发处理，提供性能开关

## 十一、后续迭代方向

### V1.2 计划
- 集成STEP3-VL-10B多模态大模型
- 增强货架场景细粒度识别
- 优化端到端处理速度

### V1.3 计划
- 支持视频流实时处理
- 增强移动端部署
- 优化模型量化压缩

## 十二、总结

本次优化基于对业界先进OCR项目的深入调研，旨在将PackCV-OCR从"基础识别"提升到"智能理解"水平。通过集成PaddleOCR 3.0、MMOCR等先进技术，新增8个核心节点，优化4个现有节点，预期整体识别准确率提升20%+，应用场景扩展5类，用户体验显著提升。

优化方案遵循"渐进式增强"原则，保持向后兼容，确保现有功能不受影响。新功能通过可选参数控制，用户可根据需求灵活启用。
