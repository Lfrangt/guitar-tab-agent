# 国内云平台部署指南

## 推荐平台（按易用性排序）

### 1. 阿里云 - 函数计算 FC 3.0 ⭐⭐⭐⭐⭐
**最推荐，支持容器部署，按使用量计费**

```bash
# 1. 安装 Serverless Devs 工具
npm install @serverless-devs/s -g

# 2. 配置阿里云账号
s config add

# 3. 部署应用
s deploy
```

**优势**：
- 免费额度充足
- 按使用量计费
- 支持HTTP触发
- 自动扩缩容

### 2. 腾讯云 - 轻量应用服务器 ⭐⭐⭐⭐
**稳定可靠，适合长期运行**

```bash
# 购买轻量应用服务器（最低配即可）
# Ubuntu 20.04 LTS
# 1核1GB内存，月付约24元

# SSH连接后执行：
git clone https://github.com/Lfrangt/-agent.git
cd -agent
sudo apt update && sudo apt install -y docker.io
sudo docker build -f Dockerfile-cn -t guitar-tab-ai .
sudo docker run -d -p 80:8501 guitar-tab-ai
```

### 3. 华为云 - 云耀云服务器 ⭐⭐⭐⭐
**性价比高，网络稳定**

### 4. 百度智能云 - 应用引擎 BAE ⭐⭐⭐
**支持Python应用直接部署**

### 5. Vercel（国外但国内可访问）⭐⭐⭐
**GitHub集成，但需要调整配置**

## 快速部署方案

### 方案一：阿里云函数计算（推荐）
1. 访问 https://fc.console.aliyun.com/
2. 创建应用 → 选择容器镜像
3. 上传本项目的 Dockerfile-cn
4. 配置触发器为HTTP
5. 部署完成

### 方案二：腾讯云轻量服务器
1. 购买1核1GB轻量服务器（月付约24元）
2. 安装Docker
3. 克隆项目并构建镜像
4. 运行容器

### 方案三：使用现有VPS
如果你有VPS服务器：

```bash
# 安装依赖
git clone https://github.com/Lfrangt/-agent.git
cd -agent
pip install -r requirements-cn.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 运行应用
streamlit run streamlit_app.py --server.port 8501
```

## 性能建议

- **最低配置**：1核1GB内存
- **推荐配置**：2核2GB内存
- **存储**：至少10GB
- **带宽**：1Mbps足够

## 费用预估

- **阿里云函数计算**：免费额度内基本免费，超出约￥0.1/万次调用
- **腾讯云轻量服务器**：￥24-50/月
- **华为云云耀服务器**：￥30-60/月

## 域名和备案

如果需要使用自定义域名：
1. 购买域名（建议使用.cn域名）
2. 完成ICP备案（约15-30天）
3. 配置域名解析

## 注意事项

1. 确保服务器位置选择国内
2. 使用国内镜像源加速安装
3. 音频文件处理比较消耗CPU，建议选择计算型实例
4. TensorFlow模型首次加载较慢，建议配置足够内存