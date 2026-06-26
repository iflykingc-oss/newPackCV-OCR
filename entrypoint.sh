#!/bin/bash
# PackCV-OCR Docker 入口脚本
set -e

echo "🚀 PackCV-OCR 启动中..."
echo "环境: ${ENV:-dev}"
echo "工作目录: $(pwd)"

# 等待依赖服务（Redis）
if [ -n "$REDIS_URL" ]; then
    echo "⏳ 等待 Redis: $REDIS_URL"
    for i in {1..30}; do
        if python -c "import redis; r=redis.from_url('$REDIS_URL'); r.ping()" 2>/dev/null; then
            echo "✅ Redis 已就绪"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "⚠️  Redis 30秒内未就绪，继续启动（限流功能将降级）"
        fi
        sleep 1
    done
fi

# 根据 ENV 决定 workers 数量
WORKERS=${WORKERS:-4}

echo "📦 启动 Uvicorn (workers=$WORKERS)..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 9000 \
    --workers "$WORKERS" \
    --log-level "${LOG_LEVEL:-info}" \
    --proxy-headers \
    --forwarded-allow-ips="*"
