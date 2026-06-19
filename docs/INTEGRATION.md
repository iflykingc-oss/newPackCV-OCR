# PackCV-OCR 三平台接入指南

本文档介绍如何将 PackCV-OCR（产品包装识别与信息提取服务）以**第三方应用**形式发布到飞书、钉钉、企业微信三大办公协同平台。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                业务方系统(工作流/企业)                      │
│   ┌────────┐  ┌────────┐  ┌────────┐                        │
│   │ 飞书Bot│  │ 钉钉Bot│  │企微Bot │   ← 用户/群消息入口    │
│   └────┬───┘  └────┬───┘  └────┬───┘                        │
└────────┼──────────┼──────────┼─────────────────────────────┘
         │ HTTPS回调│          │          │
         ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│                  PackCV-OCR 服务                            │
│                                                             │
│   ┌─────────────────────────────────────────────┐          │
│   │  IM Platform Adapter 层 (src/utils/im_platform)         │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │          │
│   │  │ FeishuBot│ │DingTalkBot│ │ WeComBot │    │          │
│   │  └──────────┘ └──────────┘ └──────────┘    │          │
│   │                  ↘ ↙ ↘ ↙                    │          │
│   │                Dispatcher                    │          │
│   └─────────────────────┬───────────────────────┘          │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────┐          │
│   │       LangGraph 工作流 (src/graphs)          │          │
│   │  image_preprocess → ocr_recognize → ...     │          │
│   │  → result_output → feishu_notify            │          │
│   └─────────────────────────────────────────────┘          │
│                         │                                   │
│   ┌─────────────────────▼───────────────────────┐          │
│   │   审计中间件 (CallAuditStore)                │          │
│   │   - 每次调用记录到内存+文件                    │          │
│   │   - 提供 Prometheus /metrics 端点             │          │
│   └─────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

核心思想：所有平台共用工单流，差异点全部封装在 `IM Platform Adapter` 层。

---

## 二、IM Platform Adapter API

### 2.1 抽象接口

```python
from utils.im_platform import (
    BotDispatcher,  # 总调度器
    FeishuBot,      # 飞书
    DingTalkBot,    # 钉钉
    WeComBot,       # 企业微信
    PlatformMessage,  # 消息结构
    PlatformType,     # 平台枚举
)
```

### 2.2 统一消息结构

```python
@dataclass
class PlatformMessage:
    platform: PlatformType     # FEISHU/DINGTALK/WECOM
    message_id: str            # 消息唯一ID
    sender_id: str             # 发送者ID
    chat_id: str               # 会话ID（群ID或用户ID）
    chat_type: str             # "group" / "direct"
    content: str               # 文本内容
    image_urls: List[str]      # 图片URL列表
    timestamp: int             # 毫秒时间戳
    raw_payload: Dict          # 原始回调payload（用于签名验证）
```

### 2.3 快速上手

```python
from utils.im_platform import BotDispatcher, PlatformType, send_extraction_result

# 1. 通过 Webhook 主动发送消息（无需回调URL）
dispatcher = BotDispatcher()
await dispatcher.send_text(
    platform=PlatformType.FEISHU,
    chat_id="oc_xxxxxxxxxxxx",
    text="📦 商品识别结果已生成..."
)

# 2. 解析回调（验证签名 → 抽取消息）
msg = await dispatcher.parse_incoming(
    platform=PlatformType.DINGTALK,
    headers=request.headers,
    raw_body=await request.body(),
    query_params=dict(request.query_params)
)
# msg.image_urls 拿到用户发来的包装图
# msg.content 拿到用户@机器人说的话

# 3. 调用工作流处理图片
result = await main_graph.ainvoke({
    "package_image": File(url=msg.image_urls[0], file_type="image"),
    ...
})

# 4. 把识别结果发回给用户
await send_extraction_result(
    dispatcher=dispatcher,
    platform=msg.platform,
    chat_id=msg.chat_id,
    result=result["final_result"]
)
```

---

## 三、HTTP回调端点

服务运行后（`uvicorn web_server:app --port 9000`），提供以下端点：

| 端点 | 用途 | 签名验证 |
|------|------|---------|
| `POST /bot/feishu/events` | 飞书事件订阅回调 | `Encrypt Key` + `Verification Token` |
| `POST /bot/dingtalk/callback` | 钉钉机器人回调 | `sign` 参数（HMAC-SHA256） |
| `POST /bot/wecom/callback` | 企业微信智能机器人回调 | `msg_signature` |
| `GET /bot/feishu/events` | 飞书URL验证握手 | 一次性challenge |
| `GET /audit/summary` | 审计Dashboard数据查询 | 无（建议内网访问） |
| `GET /metrics` | Prometheus 指标 | 无（建议内网访问） |

### 3.1 飞书接入

**步骤**：
1. 在[飞书开放平台](https://open.feishu.cn/)创建企业自建应用
2. 应用能力 → 机器人 → 启用
3. 事件订阅 → 请求URL：填 `https://your-domain/bot/feishu/events`
4. 权限管理 → 添加 `im:message` `im:message:receive`
5. 复制 Verification Token 和 Encrypt Key → 配置到 `secrets/feishu_credentials.json`

**消息交互**：
- 用户在群里 `@Bot` 发送"识别包装"+图片 → Bot 解析出图片URL → 调用工作流 → 返回结构化结果卡片
- 支持命令：`/ocr <image>` `/extract <image>` `/help`

### 3.2 钉钉接入

**步骤**：
1. 在[钉钉开放平台](https://open-dev.dingtalk.com/)创建机器人应用
2. 消息接收 → 机器人 → 设置机器人
3. 安全设置 → 勾选"加密" → 生成 AppSecret
4. Webhook URL 填：`https://your-domain/bot/dingtalk/callback`
5. 复制 AppKey/AppSecret → 配置到 `secrets/dingtalk_credentials.json`

**消息类型**：
- text：纯文本回复
- markdown：富文本回复
- actionCard：交互式卡片（含按钮）

### 3.3 企业微信接入

**步骤**：
1. 在[企业微信管理后台](https://work.weixin.qq.com/)创建自建应用
2. 应用 → 接收消息 → 设置API接收
3. URL 填：`https://your-domain/bot/wecom/callback`
4. Token / EncodingAESKey → 配置到 `secrets/wecom_credentials.json`
5. CorpID / CorpSecret → 用于主动推送API

**消息交互**：
- text：纯文本
- markdown：富文本
- news：图文消息

---

## 四、签名验证原理

### 4.1 飞书签名

```python
# 飞书事件订阅签名 = AES-256-CBC(Encrypt Key, Verification Token, 随机字符串)
# 解密后JSON包含: {challenge, event, type}
# 首次验证时返回 challenge 字段
```

### 4.2 钉钉签名

```python
# URL参数中包含 timestamp + sign
# sign = base64(HMAC-SHA256(AppSecret, f"{timestamp}\n{signature_token}"))
# 验证: 当前时间 - timestamp < 5分钟  &&  sign匹配
```

### 4.3 企业微信签名

```python
# msg_signature = SHA1(token, timestamp, nonce, encrypt_msg)
# 解密: AES-256-CBC(EncodingAESKey, message)
```

---

## 五、审计与监控

### 5.1 审计存储

每次调用自动记录到：
- **内存**：最近1000次调用（O(1)查询）
- **文件**：`logs/audit/audit-YYYYMMDD.jsonl`（追加写，永久保留）
- **扩展**：可对接 Sentry / DataDog / Prometheus

### 5.2 Prometheus 指标

```bash
# 查询最近1小时总调用数
curl http://localhost:9000/metrics

# 返回示例
packcv_calls_total{caller="api",status="success"} 1247
packcv_calls_total{caller="feishu",status="success"} 89
packcv_calls_total{caller="dingtalk",status="error"} 3
packcv_call_duration_seconds_sum 12405.3
packcv_call_duration_seconds_count 1339
packcv_node_duration_seconds{node="ocr_recognize",quantile="0.95"} 2.34
```

### 5.3 审计Dashboard

```bash
# 查询调用汇总
curl http://localhost:9000/audit/summary
# 返回
{
  "total": 1339,
  "success": 1336,
  "error": 3,
  "success_rate": 0.998,
  "avg_duration": 9.27,
  "p95_duration": 18.5,
  "calls_by_caller": {"api": 1247, "feishu": 89, "dingtalk": 3},
  "errors_by_type": {"image_decode_error": 2, "ocr_engine_error": 1},
  "node_avg_duration": {
    "ocr_recognize": 3.4,
    "multi_channel_fusion": 1.2,
    ...
  }
}
```

---

## 六、安全与限流建议

1. **签名验证必开**：所有回调端点必须验证签名（默认已开启）
2. **白名单IP**：钉钉/企微回调有固定出口IP，建议在网关层做白名单
3. **限流**：每个 chat_id 限 10次/分钟，IP 限 100次/分钟
4. **图片过期处理**：用户发送的图片URL有效期1小时，建议下载到本地后再处理
5. **HTTPS**：生产环境必须HTTPS
6. **审计日志**：保留至少90天，便于追溯

---

## 七、上线 Checklist

- [ ] 飞书应用已发布到企业应用商店
- [ ] 钉钉机器人已添加到目标群
- [ ] 企业微信应用已对目标部门可见
- [ ] 签名验证密钥已配置到 secrets/
- [ ] 公网域名 + HTTPS 证书
- [ ] Prometheus 监控接入告警
- [ ] 日志采集（audit jsonl）
- [ ] 限流配置生效
- [ ] 用户使用文档已发布（飞书Doc/钉钉Doc/企微Doc）
- [ ] 客服支持流程已建立

---

## 八、参考文档

- 飞书开放平台：https://open.feishu.cn/document/server-docs/event-subscription-guide/overview
- 钉钉开放平台：https://open-dev.dingtalk.com/apiExplorer
- 企业微信开发文档：https://developer.work.weixin.qq.com/document/path/90238
- PackCV-OCR API文档：`docs/API.md`
- 工作流节点手册：`AGENTS.md`
