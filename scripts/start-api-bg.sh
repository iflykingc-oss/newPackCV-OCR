#!/bin/bash
# 后台启动 API 服务
cd /workspace/projects
export ENV=dev
export REDIS_URL=redis://localhost:6379/0
export PYTHONPATH=src
exec /workspace/projects/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 9001 --log-level info
