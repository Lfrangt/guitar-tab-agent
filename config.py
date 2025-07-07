"""
配置文件 - 包含所有系统参数和设置
"""
import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TABS_DIR = OUTPUT_DIR / "tabs"
LOGS_DIR = OUTPUT_DIR / "logs"
MODEL_DIR = BASE_DIR / "model"

# 创建必要目录
for dir_path in [OUTPUT_DIR, TABS_DIR, LOGS_DIR, MODEL_DIR]:
    dir_path.mkdir(exist_ok=True)

# 音频处理配置
AUDIO_CONFIG = {
    "sample_rate": 22050,  # 音频采样率
    "hop_length": 512,     # 帧移长度
    "frame_size": 2048,    # 帧长度
    "n_fft": 2048,         # FFT窗口大小
    "win_length": 2048,    # 窗口长度
    "window": "hann",      # 窗口类型
    "center": True,        # 是否居中
    "pad_mode": "constant" # 填充模式
}

# 音高检测配置
PITCH_CONFIG = {
    "model_capacity": "full",  # CREPE模型容量: tiny, small, medium, large, full
    "viterbi": True,          # 是否使用维特比解码
    "confidence_threshold": 0.5,  # 置信度阈值
    "step_size": 10,          # 步长（毫秒）
    "fmin": 80.0,            # 最低频率（Hz）
    "fmax": 2000.0,          # 最高频率（Hz）
    "onset_threshold": 0.5    # 起始检测阈值
}

# 和弦分析配置
CHORD_CONFIG = {
    "window_size": 4096,      # 分析窗口大小
    "hop_length": 1024,       # 跳跃长度
    "chroma_bins": 12,        # 色度特征维度
    "chord_templates": "major_minor",  # 和弦模板类型
    "smoothing_window": 5,    # 平滑窗口大小
    "min_chord_duration": 0.5  # 最小和弦持续时间（秒）
}

# 六线谱生成配置
TAB_CONFIG = {
    "tuning": ["E", "A", "D", "G", "B", "E"],  # 标准调弦
    "fret_range": (0, 24),    # 品格范围
    "string_preference": [0, 1, 2, 3, 4, 5],  # 弦偏好顺序
    "max_stretch": 4,         # 最大手指跨度
    "open_string_bonus": 0.1, # 空弦奖励
    "position_smoothing": True, # 位置平滑
    "tab_format": "ascii"     # 输出格式: ascii, pdf, png
}

# 视频分析配置（可选）
VIDEO_CONFIG = {
    "fps": 30,               # 视频帧率
    "hand_detection": True,   # 是否检测手部
    "pose_estimation": False, # 是否进行姿态估计
    "roi_detection": True,    # 是否检测感兴趣区域
    "frame_skip": 1          # 帧跳跃间隔
}

# GPT配置
GPT_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": "gpt-3.5-turbo",
    "max_tokens": 500,
    "temperature": 0.7,
    "system_prompt": """你是一个专业的吉他演奏分析师。
    请根据提供的音乐分析结果，生成简洁明了的演奏说明，包括：
    1. 和弦进行分析
    2. 演奏技巧建议
    3. 节奏特点
    4. 难点提示
    请用中文回答，语言要通俗易懂。"""
}

# 输出配置
OUTPUT_CONFIG = {
    "save_audio": True,       # 是否保存处理后的音频
    "save_plots": True,       # 是否保存可视化图表
    "save_midi": True,        # 是否保存MIDI文件
    "save_tab": True,         # 是否保存六线谱
    "save_explanation": True, # 是否保存说明文字
    "formats": ["txt", "pdf", "png"]  # 输出格式
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "guitar_tab_ai.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# 模型路径配置
MODEL_PATHS = {
    "crepe": MODEL_DIR / "crepe",
    "chord_model": MODEL_DIR / "chord_model.pkl",
    "onset_model": MODEL_DIR / "onset_model.pkl"
}

# Web界面配置
WEB_CONFIG = {
    "host": "localhost",
    "port": 8501,
    "debug": True,
    "upload_max_size": 100 * 1024 * 1024,  # 100MB
    "allowed_extensions": [".wav", ".mp3", ".flac", ".m4a", ".mp4", ".avi", ".mov"]
}

# 性能配置
PERFORMANCE_CONFIG = {
    "use_gpu": True,          # 是否使用GPU
    "batch_size": 32,         # 批处理大小
    "num_workers": 4,         # 并行工作进程数
    "memory_limit": "8GB",    # 内存限制
    "cache_size": 1000        # 缓存大小
} 