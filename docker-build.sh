#!/bin/bash
# 构建并推送Docker镜像到阿里云容器镜像服务

echo "🐳 构建Docker镜像..."

# 构建镜像
docker build -f Dockerfile-cn -t guitar-tab-ai:latest .

# 标记镜像（需要替换为你的镜像仓库地址）
# docker tag guitar-tab-ai:latest registry.cn-hangzhou.aliyuncs.com/你的命名空间/guitar-tab-ai:latest

# 推送镜像（需要先登录）
# docker login registry.cn-hangzhou.aliyuncs.com
# docker push registry.cn-hangzhou.aliyuncs.com/你的命名空间/guitar-tab-ai:latest

echo "✅ 镜像构建完成！"
echo ""
echo "📋 下一步："
echo "1. 登录阿里云容器镜像服务：https://cr.console.aliyun.com/"
echo "2. 创建命名空间和镜像仓库"
echo "3. 根据控制台指引推送镜像"
echo "4. 在函数计算中使用该镜像创建函数"