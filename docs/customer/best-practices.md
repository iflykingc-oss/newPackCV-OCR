# 📈 最佳实践

> 提升准确率、降低成本、避免常见坑的生产经验。

## 🎯 准确率提升

### 1. 选对场景

`scenario=auto` 虽然方便,但**特定场景准确率平均高 8-12%**。

```python
# ❌ 错误: 永远用 auto
result = client.extract(image="r.jpg", scenario="auto")

# ✅ 正确: 业务逻辑判断后选择
if is_packaging_image(image):
    scenario = "packaging"
elif is_invoice(image):
    scenario = "finance_receipt"
result = client.extract(image=image, scenario=scenario)
```

### 2. 图片质量预处理

- **图片分辨率** < 1000x1000 像素 → 准确率下降 15%
- **模糊 / 倾斜** → 启用预处理 (`engine_tier="premium"`)
- **低光照** → 启用 CLAHE 增强

```python
result = client.extract(
    image="r.jpg",
    scenario="finance_receipt",
    engine_tier="premium",  # 启用高级引擎 + 预处理
)
```

### 3. 多语言混合

中文 + 英文混合文档 (如快递单) → 准确率会下降 5%。

```python
result = client.extract(
    image="waybill.jpg",
    scenario="logistics",
    locale="zh-CN",
    # 检测到英文字段时使用 LLM 翻译
    translate_to="zh-CN",
)
```

### 4. 表格识别

金融/合同场景的复杂表格 → 使用 `engine_tier="table"`:

```python
result = client.extract(
    image="contract.pdf",
    scenario="contract",
    engine_tier="table",  # 启用表格识别引擎
)
```

---

## 💰 成本控制

### 1. 选择合适引擎

| 引擎 | 单次成本 | 准确率 | 适用场景 |
|------|---------|-------|---------|
| Fast | ¥0.01 | 88% | 标准化文档、内部使用 |
| Standard | ¥0.05 | 92% | 通用业务 |
| Premium | ¥0.15 | 95% | 合同、医疗、关键业务 |
| Table | ¥0.20 | 97% (表格) | 复杂表格 |

```python
# 高价值文档用 Premium,普通票据用 Standard
high_value = client.extract(image, "contract", engine_tier="premium")
normal = client.extract(image, "finance_receipt", engine_tier="standard")
```

### 2. 批量处理

批量调用比分次调用**快 3-5 倍,省 30% 时间** (内部并发):

```python
# ❌ 错误: 串行调用
for image in images:
    result = client.extract(image, "finance_receipt")

# ✅ 正确: 批量
results = client.extract_batch(images, "finance_receipt")
```

### 3. 缓存

相同图片/相同场景 → 启用缓存 (`Cache-Control`):

```python
result = client.extract(
    image="r.jpg",
    scenario="finance_receipt",
    cache="aggressive",  # 24h 缓存
)
```

### 4. 监控告警

设置**预算告警**避免超额:

```python
from packcv import PackCVClient

client = PackCVClient(api_key="...", budget_alert=80)  # 用到 80% 告警
```

---

## 🔐 安全合规

### 1. 敏感数据脱敏

身份证、银行卡、密码字段**自动脱敏** (默认开启)。

可在 Web 控制台关闭 (不推荐)。

### 2. 数据隔离

- **生产环境**: 严禁使用 `pck_test_` 开头的 Key
- **测试环境**: 使用 `pck_test_` Key,数据不计入配额
- **开发环境**: 推荐使用本地沙箱

### 3. 数据保留

- **默认**: 数据保留 30 天 (用于问题排查)
- **可配置**: 7 / 30 / 90 / 365 天,或立即删除
- **Enterprise**: 自定义保留期,支持数据本地化

### 4. 审计日志

所有 API 调用记录在审计日志,可在 Web 控制台查看,或通过 API:

```bash
curl /api/v1/audit-logs?from=2024-01-01&to=2024-01-31
```

---

## 🚀 性能优化

### 1. 异步任务

> 30 秒以上的任务必须用异步 (否则会超时)。

```python
task = client.extract_async(image="big.pdf", scenario="contract")
# 业务逻辑...
result = client.get_task(task.task_id, wait=True, timeout=300)
```

### 2. Webhook 回调

> 避免长轮询,使用 Webhook 推送完成事件。

```python
task = client.extract_async(
    image="big.pdf",
    scenario="contract",
    webhook_url="https://my-server.com/webhook",
)
# 后台处理,完成后由 PackCV 主动推送
```

### 3. 客户端复用

**每次创建 Client 都要建立 HTTP 连接**, 复用可提速 5-10x。

```python
# ❌ 错误
def process(image):
    client = PackCVClient(api_key="...")  # 每次创建
    return client.extract(image, "...")

# ✅ 正确
client = PackCVClient(api_key="...")
def process(image):
    return client.extract(image, "...")
```

---

## 🐛 常见错误

### 1. "QUOTA_EXCEEDED"

**原因**: 配额耗尽。
**解决**:
- 等待下月重置
- 升级到 Pro/Enterprise
- 联系销售 (support@vibecoding.dev)

### 2. "INVALID_IMAGE"

**原因**: 图片格式不支持或损坏。
**解决**:
- 支持的格式: JPG / PNG / WebP / HEIC / TIFF / BMP / PDF
- 最大 50MB (Pro) / 200MB (Enterprise)
- 最小 200x200 像素

### 3. "TIMEOUT"

**原因**: 文档过大或太复杂。
**解决**:
- 用 `engine_tier="fast"` 减少处理时间
- 用异步任务 + Webhook
- 拆分大文档

### 4. 准确率低

**排查清单**:
- [ ] 图片分辨率 ≥ 1000x1000?
- [ ] 选择正确的 scenario?
- [ ] 启用预处理 (`engine_tier="premium"`)?
- [ ] 文档方向正确 (非倒置)?
- [ ] 单据非手写 (手写支持有限)?

---

## 📊 性能基准

| 文档类型 | 平均延迟 | P99 延迟 | 准确率 |
|---------|---------|---------|-------|
| 包装 | 1.2s | 3.5s | 95% |
| 票据 (电子) | 0.8s | 2.0s | 97% |
| 票据 (扫描) | 1.5s | 3.0s | 92% |
| 银行流水 (1页) | 1.0s | 2.5s | 95% |
| 银行流水 (10页) | 4.5s | 12s | 93% |
| 合同 (5页) | 3.0s | 8s | 94% |
| PDF (10页) | 5.0s | 15s | 96% |

*测试环境: 4 引擎梯级 + 8 场景 + 109 语言*

---

## 📚 进阶阅读

- [Webhook 集成](./webhooks.md)
- [异步任务模式](./async-tasks.md)
- [企业级 Schema 自定义](./enterprise-schema.md)
- [私有化部署](./self-hosting.md)
