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
