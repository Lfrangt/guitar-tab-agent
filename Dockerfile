FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements-web.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements-web.txt

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p output/logs output/tabs model

# 暴露端口
EXPOSE 8501

# 启动应用
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]