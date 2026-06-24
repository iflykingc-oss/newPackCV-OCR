# VibeCoding-OCR API 参考

> 完整的 API 文档请参考 [OpenAPI 规范](./openapi.yaml) 和 [Postman 集合](./postman.json)。

## 快速开始

```bash
# 启动 API 服务
uv run python src/web_server.py

# 健康检查
curl http://localhost:9000/health

# OCR 识别 (包装场景)
curl -X POST http://localhost:9000/api/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "package_image": {
      "url": "https://example.com/package.jpg",
      "file_type": "image"
    }
  }'
```

## 端点列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 存活探针 |
| `/ready` | GET | 就绪探针 |
| `/metrics` | GET | Prometheus 指标 |
| `/api/ocr` | POST | OCR 识别主入口(支持图片+文档) |
| `/api/ocr/batch` | POST | 批量 OCR 识别 |
| `/api/config/summary` | GET | 配置中心总览 |
| `/api/config/tenant/{id}` | GET/PUT/DELETE | 租户配置管理 |
| `/api/scenarios` | GET | 支持的场景列表 |
| `/api/schemas/{scenario}` | GET | 场景字段 Schema |

详细参数见 [openapi.yaml](./openapi.yaml)。
