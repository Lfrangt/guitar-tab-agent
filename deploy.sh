#!/bin/bash
# 国内VPS一键部署脚本

set -e

echo "🎸 吉他谱AI系统 - 国内部署脚本"
echo "=================================="

# 检查系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ 检测到Linux系统"
else
    echo "❌ 此脚本仅支持Linux系统"
    exit 1
fi

# 更新系统包
echo "📦 更新系统包..."
sudo apt update -y

# 安装系统依赖
echo "🔧 安装系统依赖..."
sudo apt install -y python3 python3-pip ffmpeg git

# 检查Python版本
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python版本: $PYTHON_VERSION"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✅ Python版本符合要求"
else
    echo "❌ Python版本过低，需要3.8+"
    exit 1
fi

# 克隆项目（如果不存在）
if [ ! -d "guitar-tab-ai" ]; then
    echo "📥 克隆项目..."
    git clone https://github.com/Lfrangt/-agent.git guitar-tab-ai
fi

cd guitar-tab-ai

# 安装Python依赖
echo "📚 安装Python依赖..."
pip3 install -r requirements-cn.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p output/logs output/tabs model

# 检查端口
PORT=8501
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
    echo "⚠️  端口 $PORT 已被占用，将使用端口 8502"
    PORT=8502
fi

# 启动应用
echo "🚀 启动应用..."
echo "访问地址：http://$(curl -s ifconfig.me):$PORT"
echo "本地访问：http://localhost:$PORT"
echo "按 Ctrl+C 停止应用"

streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0