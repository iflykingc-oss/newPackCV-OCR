<<<<<<< HEAD
# ============= Stage 1: Builder =============
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install uv

WORKDIR /app

# 先复制依赖文件以利用 Docker 缓存
COPY pyproject.toml uv.lock* ./

# 安装依赖到 .venv
RUN uv sync --frozen --no-dev --no-install-project

# ============= Stage 2: Runtime =============
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# 从 builder 复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 复制项目代码
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 创建运行时所需目录
RUN mkdir -p /app/cache /tmp && chown -R appuser:appuser /app /tmp

USER appuser

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:9000/api/v1/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
=======
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装依赖
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart jinja2 \
    pydantic pillow opencv-python-headless \
    numpy langgraph langchain-core \
    pyjwt bcrypt aiofiles

# 创建数据目录
RUN mkdir -p /app/data /app/uploads /app/logs

# 暴露端口
EXPOSE 9000

# 启动
CMD ["uvicorn", "src.web_server:app", "--host", "0.0.0.0", "--port", "9000"]
>>>>>>> origin/main
