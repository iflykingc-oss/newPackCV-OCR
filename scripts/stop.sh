#!/usr/bin/env bash
# ===================================================================
# PackCV-OCR 一键停止脚本
# ===================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "🛑 停止所有服务..."

# 按依赖逆序停止
for compose_file in docker-compose.nginx.yml docker-compose.monitoring.yml docker-compose.prod.yml docker-compose.base.yml; do
  if [ -f "$compose_file" ]; then
    echo "  停止 $compose_file ..."
    docker compose -f "$compose_file" down 2>/dev/null || true
  fi
done

echo "✅ 所有服务已停止"
