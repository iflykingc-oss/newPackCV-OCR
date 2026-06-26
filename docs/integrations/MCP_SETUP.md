# PackCV MCP Server 配置

## Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
`%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "packcv": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "env": {
        "PACKCV_BASE_URL": "http://localhost:9001",
        "PACKCV_API_KEY": "pk_live_your_key"
      }
    }
  }
}
```

## Cursor

`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "packcv": {
      "url": "http://localhost:9002/sse",
      "transport": "sse"
    }
  }
}
```

## 独立 SSE 模式启动

```bash
PACKCV_TRANSPORT=sse \
PACKCV_HOST=0.0.0.0 \
PACKCV_PORT=9002 \
PACKCV_BASE_URL=http://localhost:9001 \
PACKCV_API_KEY=pk_live_xxx \
python -m mcp_server
```

## stdio 模式（直连）

```bash
PACKCV_TRANSPORT=stdio \
PACKCV_BASE_URL=http://localhost:9001 \
PACKCV_API_KEY=pk_live_xxx \
python -m mcp_server
```

## 可用工具

| 工具名 | 功能 |
|--------|------|
| `extract_document` | 提取文档结构化数据 |
| `answer_question` | 基于图片问答 |
| `batch_extract` | 批量提取 |
| `list_scenarios` | 列出支持场景 |
| `health_check` | 服务健康检查 |

## 使用示例（对话中）

> "请使用 packcv 的 extract_document 工具，提取这张营业执照的法人信息。URL 是 https://..."

> "使用 list_scenarios 查看 PackCV 支持的所有场景。"

## 故障排除

1. **MCP Server 启动失败**: 检查 `PACKCV_API_KEY` 是否有效
2. **Claude Desktop 找不到 MCP**: 重启 Claude Desktop
3. **工具调用超时**: 增大 `timeout` 参数

## 协议版本

MCP Protocol: 2024-11-05
