# 阿里云在线构建部署（无需本地Docker）

## 🚀 最简单方式 - 使用云效代码源

### 1. 准备代码仓库
✅ 已完成：https://github.com/Lfrangt/guitar-tab-agent

### 2. 阿里云容器镜像服务自动构建

#### 访问容器镜像服务
https://cr.console.aliyun.com/cn-hangzhou/instances

#### 创建镜像仓库
1. 点击"创建镜像仓库"
2. 配置：
   - **仓库名称**：guitar-tab-ai
   - **仓库类型**：公开
   - **摘要**：吉他谱AI系统
   - **代码源**：GitHub
   - **仓库地址**：https://github.com/Lfrangt/guitar-tab-agent
   - **Dockerfile路径**：Dockerfile-cn
   - **镜像版本**：latest

#### 触发构建
1. 在镜像仓库详情页点击"立即构建"
2. 等待构建完成（约5-10分钟）

### 3. 创建函数计算应用

#### 访问函数计算控制台
https://fc.console.aliyun.com/

#### 创建函数
1. 点击"创建函数"
2. 选择"容器镜像"
3. 配置：
   - **函数名称**：guitar-tab-ai
   - **镜像地址**：registry.cn-hangzhou.aliyuncs.com/[你的命名空间]/guitar-tab-ai:latest
   - **端口**：8501
   - **内存**：1024 MB
   - **超时时间**：300秒

#### 创建HTTP触发器
1. 在函数详情页点击"触发器"
2. 创建触发器：
   - **类型**：HTTP触发器
   - **认证**：anonymous
   - **方法**：GET, POST

### 4. 访问应用
触发器创建后会生成访问地址，直接访问即可！

## 🎯 全程预计时间：10-15分钟

## ⚡ 实时状态
- ✅ 代码已推送到GitHub
- ✅ 部署配置已准备完成
- ⏳ 等待你在阿里云控制台操作

## 📞 需要帮助？
如果遇到问题，可以截图发给我，我来帮你解决！