# 吉他谱AI - Guitar Tab AI

基于深度学习的吉他谱自动识别和生成系统，能够从音频/视频中提取音高信息并生成六线谱。

## 功能特性

- 🎵 音频处理和节拍检测
- 🎸 精确的音高识别（基于CREPE模型）
- 🎼 智能和弦分析
- 📝 六线谱自动生成
- 🤖 GPT驱动的演奏说明生成
- 🖥️ 友好的Web界面
- 📊 可视化分析结果

## 项目结构

```
guitar-tab-ai/
├── main.py                      # 主入口脚本
├── config.py                    # 配置参数
├── app/                         # 核心应用模块
│   ├── audio_processing.py      # 音频处理
│   ├── pitch_detection.py       # 音高识别
│   ├── chord_analysis.py        # 和弦分析
│   ├── tab_generator.py         # 六线谱生成
│   └── explanation_generator.py # 说明文字生成
├── webui/                       # Web界面
├── output/                      # 输出文件
└── model/                       # 模型权重
```

## 安装说明

1. 克隆项目
```bash
git clone <repository-url>
cd guitar-tab-ai
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 下载模型权重
```bash
# CREPE模型会在首次运行时自动下载
```

## 使用方法

### 命令行使用
```bash
python main.py --input audio.wav --output output/tabs/
```

### Web界面使用
```bash
streamlit run webui/app.py
```

## 配置说明

在 `config.py` 中可以调整以下参数：
- 音频采样率
- 模型路径
- 输出格式
- GPT API配置

## 依赖项

- Python 3.8+
- TensorFlow/PyTorch
- librosa
- crepe
- ffmpeg
- streamlit
- openai

## 许可证

MIT License 