"""
工具函数模块 - 提供系统通用功能
"""
import logging
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from config import LOGGING_CONFIG, WEB_CONFIG


def setup_logging(log_level: str = "INFO") -> None:
    """
    设置日志系统
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    """
    import logging.handlers
    
    # 创建日志目录
    log_file = Path(LOGGING_CONFIG["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 设置日志级别
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # 创建并配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 清除现有的handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加文件handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=LOGGING_CONFIG["max_size"],
        backupCount=LOGGING_CONFIG["backup_count"],
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 添加控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"日志系统已设置为 {log_level} 级别")


def validate_input_file(file_path: Union[str, Path]) -> bool:
    """
    验证输入文件是否有效
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 文件是否有效
    """
    file_path = Path(file_path)
    
    # 检查文件是否存在
    if not file_path.exists():
        logging.error(f"文件不存在: {file_path}")
        return False
    
    # 检查是否为文件
    if not file_path.is_file():
        logging.error(f"路径不是文件: {file_path}")
        return False
    
    # 检查文件大小
    file_size = file_path.stat().st_size
    if file_size == 0:
        logging.error(f"文件为空: {file_path}")
        return False
    
    # 检查文件扩展名
    allowed_extensions = WEB_CONFIG["allowed_extensions"]
    if file_path.suffix.lower() not in allowed_extensions:
        logging.error(f"不支持的文件格式: {file_path.suffix}")
        return False
    
    # 检查文件大小限制
    max_size = WEB_CONFIG["upload_max_size"]
    if file_size > max_size:
        logging.error(f"文件过大: {file_size} bytes (最大: {max_size} bytes)")
        return False
    
    logging.info(f"文件验证通过: {file_path}")
    return True


def save_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    output_format: str = "all",
    generate_visualization: bool = True
) -> List[str]:
    """
    保存处理结果到文件
    
    Args:
        results: 处理结果字典
        output_dir: 输出目录
        output_format: 输出格式
        generate_visualization: 是否生成可视化
        
    Returns:
        List[str]: 生成的文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件名（用于命名输出文件）
    input_file = Path(results["input_file"])
    base_name = input_file.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = f"{base_name}_{timestamp}"
    
    output_files = []
    
    # 1. 保存六线谱
    if results.get("tab_data") and output_format in ["txt", "all"]:
        tab_file = output_dir / f"{file_prefix}_tab.txt"
        try:
            with open(tab_file, 'w', encoding='utf-8') as f:
                f.write(results["tab_data"].get("ascii_tab", ""))
            output_files.append(str(tab_file))
            logging.info(f"六线谱已保存: {tab_file}")
        except Exception as e:
            logging.error(f"保存六线谱失败: {e}")
    
    # 2. 保存演奏说明
    if results.get("explanation") and output_format in ["txt", "all"]:
        explanation_file = output_dir / f"{file_prefix}_explanation.txt"
        try:
            with open(explanation_file, 'w', encoding='utf-8') as f:
                f.write(results["explanation"])
            output_files.append(str(explanation_file))
            logging.info(f"演奏说明已保存: {explanation_file}")
        except Exception as e:
            logging.error(f"保存演奏说明失败: {e}")
    
    # 3. 保存分析数据 (JSON)
    analysis_file = output_dir / f"{file_prefix}_analysis.json"
    try:
        analysis_data = {
            "input_file": results["input_file"],
            "timestamp": results.get("timestamp"),
            "duration": results.get("duration"),
            "audio_info": {
                "sample_rate": results.get("audio_data", {}).get("sample_rate"),
                "duration": results.get("audio_data", {}).get("duration"),
                "tempo": results.get("audio_data", {}).get("tempo")
            },
            "pitch_statistics": results.get("pitch_data", {}).get("statistics"),
            "chord_progression": results.get("chord_data", {}).get("progression"),
            "tab_info": {
                "tuning": results.get("tab_data", {}).get("tuning"),
                "total_measures": results.get("tab_data", {}).get("num_measures")
            }
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        output_files.append(str(analysis_file))
        logging.info(f"分析数据已保存: {analysis_file}")
    except Exception as e:
        logging.error(f"保存分析数据失败: {e}")
    
    # 4. 保存可视化图表
    if generate_visualization and output_format in ["png", "all"]:
        try:
            # 创建可视化图表
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # 音高曲线
            if results.get("pitch_data", {}).get("frequencies") is not None:
                frequencies = results["pitch_data"]["frequencies"]
                times = results["pitch_data"]["times"]
                axes[0].plot(times, frequencies, 'b-', linewidth=1)
                axes[0].set_title('音高检测结果')
                axes[0].set_xlabel('时间 (s)')
                axes[0].set_ylabel('频率 (Hz)')
                axes[0].grid(True, alpha=0.3)
            
            # 和弦进行
            if results.get("chord_data", {}).get("progression"):
                chord_names = [chord['chord'] for chord in results["chord_data"]["progression"]]
                chord_times = [chord['time'] for chord in results["chord_data"]["progression"]]
                axes[1].scatter(chord_times, range(len(chord_names)), c='red', s=50)
                axes[1].set_title('和弦进行')
                axes[1].set_xlabel('时间 (s)')
                axes[1].set_ylabel('和弦')
                axes[1].set_yticks(range(len(chord_names)))
                axes[1].set_yticklabels(chord_names)
                axes[1].grid(True, alpha=0.3)
            
            # 六线谱可视化（简化版）
            if results.get("tab_data", {}).get("positions"):
                positions = results["tab_data"]["positions"]
                strings = [pos['string'] for pos in positions[:50]]  # 前50个位置
                frets = [pos['fret'] for pos in positions[:50]]
                axes[2].scatter(range(len(strings)), strings, c=frets, cmap='viridis', s=30)
                axes[2].set_title('六线谱位置 (前50个音符)')
                axes[2].set_xlabel('音符序号')
                axes[2].set_ylabel('弦号')
                axes[2].set_ylim(-0.5, 5.5)
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            visualization_file = output_dir / f"{file_prefix}_visualization.png"
            plt.savefig(visualization_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_files.append(str(visualization_file))
            logging.info(f"可视化图表已保存: {visualization_file}")
            
        except Exception as e:
            logging.error(f"保存可视化图表失败: {e}")
    
    # 5. 保存原始数据 (pickle)
    raw_data_file = output_dir / f"{file_prefix}_raw_data.pkl"
    try:
        with open(raw_data_file, 'wb') as f:
            pickle.dump(results, f)
        output_files.append(str(raw_data_file))
        logging.info(f"原始数据已保存: {raw_data_file}")
    except Exception as e:
        logging.error(f"保存原始数据失败: {e}")
    
    return output_files


def load_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从文件加载处理结果
    
    Args:
        file_path: 结果文件路径
        
    Returns:
        Dict[str, Any]: 处理结果字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        logging.info(f"结果文件已加载: {file_path}")
        return results
    except Exception as e:
        logging.error(f"加载结果文件失败: {e}")
        raise


def format_duration(seconds: float) -> str:
    """
    格式化时长为可读字符串
    
    Args:
        seconds: 时长（秒）
        
    Returns:
        str: 格式化的时长字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}时{minutes}分{secs:.1f}秒"


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        Dict[str, Any]: 文件信息字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    stat = file_path.stat()
    
    return {
        "path": str(file_path),
        "name": file_path.name,
        "size": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "extension": file_path.suffix.lower()
    }


def clean_output_directory(output_dir: Union[str, Path], keep_recent: int = 5) -> None:
    """
    清理输出目录，保留最近的文件
    
    Args:
        output_dir: 输出目录
        keep_recent: 保留最近的文件数量
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        return
    
    # 获取所有文件，按修改时间排序
    files = []
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            files.append((file_path, file_path.stat().st_mtime))
    
    # 按修改时间排序（最新的在前）
    files.sort(key=lambda x: x[1], reverse=True)
    
    # 删除旧文件
    deleted_count = 0
    for file_path, _ in files[keep_recent:]:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            logging.warning(f"删除文件失败: {file_path} - {e}")
    
    if deleted_count > 0:
        logging.info(f"已清理 {deleted_count} 个旧文件")