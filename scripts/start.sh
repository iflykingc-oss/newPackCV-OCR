#!/bin/bash
###############################################################################
# PackCV-OCR 一键启动脚本
# 用途: 启动完整生产栈 (API + Redis + Nginx + 监控)
###############################################################################

set -e

WORKSPACE=$(cd "$(dirname "$0")/.." && pwd)
cd "$WORKSPACE"

echo "🚀 PackCV-OCR 启动脚本"
echo "=========================="
echo "工作目录: $WORKSPACE"
echo ""

# 1. 启动 Redis
echo "📦 1. 启动 Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    redis-server --daemonize yes --port 6379 --bind 0.0.0.0 \
                 --protected-mode no --maxmemory 512mb
    sleep 1
    echo "   ✅ Redis 已启动"
else
    echo "   ✅ Redis 已在运行"
fi

# 2. 启动 API 服务 (gunicorn + uvicorn workers)
echo ""
echo "📦 2. 启动 API 服务 (4 workers)..."

# 检查端口
PORT=9001
if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
    echo "   ⚠️  端口 $PORT 被占用, 停止旧进程"
    pkill -9 -f "uvicorn api.main" 2>/dev/null || true
    sleep 2
fi

cd "$WORKSPACE"
nohup env ENV=production \
         REDIS_URL=redis://localhost:6379/0 \
         PYTHONPATH=src \
         uvicorn api.main:app \
            --host 0.0.0.0 \
            --port $PORT \
            --workers 4 \
            --log-level info \
    > /tmp/packcv-api.log 2>&1 &
disown
echo "   ✅ API 启动中 (PID: $!)"
sleep 5

# 3. 健康检查
echo ""
echo "📦 3. 健康检查..."
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/api/v1/health || echo "000")
if [ "$HEALTH" = "200" ]; then
    echo "   ✅ API 健康: HTTP $HEALTH"
else
    echo "   ❌ API 异常: HTTP $HEALTH"
    echo "   查看日志: tail -30 /tmp/packcv-api.log"
    exit 1
fi

# 4. 初始化演示租户
echo ""
echo "📦 4. 初始化演示租户..."
curl -s -X POST http://localhost:$PORT/api/v1/admin/tenants/demo | \
    python3 -c "
import json, sys
data = json.load(sys.stdin)
if isinstance(data, list):
    for t in data:
        print(f\"   ✅ {t['tenant_name']:20} | {t.get('api_key','')[:30]}\")
else:
    print('   ✅ 演示数据已就绪')
" 2>/dev/null || echo "   ⚠️  租户可能已存在"

# 5. 验证指标
echo ""
echo "📦 5. Prometheus 指标..."
METRICS_COUNT=$(curl -s http://localhost:$PORT/metrics | grep -c "^# HELP packcv" || echo 0)
echo "   ✅ PackCV自定义指标: $METRICS_COUNT 个"

# 6. 显示访问信息
echo ""
echo "=========================="
echo "✅ 启动完成!"
echo ""
echo "📍 访问信息:"
echo "   API根路径:  http://localhost:$PORT/"
echo "   API文档:    http://localhost:$PORT/docs"
echo "   健康检查:   http://localhost:$PORT/api/v1/health"
echo "   Prometheus: http://localhost:$PORT/metrics"
echo "   演示租户:   POST /api/v1/admin/tenants/demo"
echo ""
echo "📋 后续操作:"
echo "   - 启动监控:  docker-compose -f docker-compose.monitoring.yml up -d"
echo "   - 启动Nginx: docker-compose -f docker-compose.nginx.yml up -d"
echo "   - 启动全栈:  docker-compose -f docker-compose.prod.yml up -d"
echo "   - 查看日志:  tail -f /tmp/packcv-api.log"
echo "=========================="
