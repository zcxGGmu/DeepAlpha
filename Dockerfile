# DeepAlpha Dockerfile
# 对应 Go 版本的 Dockerfile.brale

FROM python:3.10-slim as base

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建应用用户
RUN groupadd -r deepalpha && useradd -r -g deepalpha deepalpha

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt requirements-dev.txt ./

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /data/logs /data/db && \
    chown -R deepalpha:deepalpha /app /data

# 切换到应用用户
USER deepalpha

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9991/api/live/status || exit 1

# 暴露端口
EXPOSE 9991

# 启动命令
CMD ["python", "-m", "uvicorn", "deepalpha.main:app", "--host", "0.0.0.0", "--port", "9991"]