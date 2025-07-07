# 阿里云函数计算手动部署指南

## 🚀 快速部署（推荐）

### 方式一：使用阿里云控制台（最简单）

#### 1. 容器镜像准备
```bash
# 在本地构建镜像
cd 和弦谱agent
docker build -f Dockerfile-cn -t guitar-tab-ai .
```

#### 2. 推送到阿里云容器镜像服务
1. 访问：https://cr.console.aliyun.com/cn-hangzhou/instances
2. 创建个人版实例（免费）
3. 创建命名空间：`guitar-ai`
4. 创建镜像仓库：`tab-generator`
5. 按照页面指引推送镜像：

```bash
# 登录阿里云Docker Registry
docker login --username=你的阿里云账号 registry.cn-hangzhou.aliyuncs.com

# 标记镜像
docker tag guitar-tab-ai registry.cn-hangzhou.aliyuncs.com/guitar-ai/tab-generator:latest

# 推送镜像
docker push registry.cn-hangzhou.aliyuncs.com/guitar-ai/tab-generator:latest
```

#### 3. 创建函数计算应用
1. 访问：https://fc.console.aliyun.com/
2. 点击"创建函数"
3. 选择"容器镜像"
4. 配置参数：
   - **函数名称**：guitar-tab-ai
   - **镜像地址**：registry.cn-hangzhou.aliyuncs.com/guitar-ai/tab-generator:latest
   - **端口**：8501
   - **内存规格**：1024 MB
   - **执行超时时间**：300秒
   - **实例并发度**：1

#### 4. 配置触发器
1. 在函数详情页，点击"触发器"
2. 创建触发器：
   - **触发器类型**：HTTP触发器
   - **认证方式**：anonymous
   - **请求方法**：GET, POST, PUT, DELETE

#### 5. 完成！
触发器创建后会生成一个公网访问地址，直接访问即可使用！

## ⚡ 预计费用

**函数计算**：
- 免费额度：每月100万次调用 + 40万CU-S计算时长
- 超出费用：约￥0.0133/万次调用
- 内存费用：约￥0.0101/万CU-S

**容器镜像服务**：
- 个人版免费
- 流量费用：约￥0.5/GB

**预计月费用**：轻度使用基本免费，重度使用约￥10-50/月

## 🔧 故障排查

### 常见问题：
1. **镜像推送失败**：检查Docker是否已启动，账号密码是否正确
2. **函数启动失败**：检查端口是否设置为8501
3. **访问超时**：增加超时时间到300秒
4. **内存不足**：调整到1024MB或更高

### 查看日志：
在函数计算控制台 → 函数详情 → 日志查询

## 🎯 性能优化建议

1. **冷启动优化**：设置预留实例（每月有免费额度）
2. **内存调优**：根据实际使用情况调整内存大小
3. **并发控制**：单实例并发度设为1，避免资源竞争