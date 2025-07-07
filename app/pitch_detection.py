"""
音高检测模块 - 使用 CREPE 模型
功能包括：音高识别、置信度过滤、频率输出
"""

import logging
import numpy as np
import crepe
from typing import Tuple, Optional, Union
from pathlib import Path
from datetime import datetime


def predict_pitch(audio: np.ndarray, 
                 sr: int,
                 model_capacity: str = 'medium',
                 viterbi: bool = True,
                 step_size: int = 10,
                 min_confidence: Optional[float] = None,
                 center: bool = True,
                 verbose: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 CREPE 模型进行音高检测
    
    Args:
        audio: 单声道音频数组
        sr: 采样率
        model_capacity: CREPE 模型容量，可选 'tiny', 'small', 'medium', 'large', 'full'
        viterbi: 是否使用维特比解码进行平滑
        step_size: 步长（毫秒），控制时间分辨率
        min_confidence: 最小置信度阈值，低于此值的结果将被过滤
        center: 是否对音频进行中心化
        verbose: 详细程度，0=静默，1=进度条，2=详细信息
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (时间轴, 频率, 置信度)
        
    Raises:
        ValueError: 输入参数无效
        RuntimeError: CREPE 模型运行失败
    """
    # 验证输入参数
    if len(audio) == 0:
        raise ValueError("音频数组不能为空")
    
    if sr <= 0:
        raise ValueError("采样率必须大于0")
    
    if audio.ndim != 1:
        raise ValueError("音频必须是单声道（1维数组）")
    
    # 验证模型容量参数
    valid_capacities = ['tiny', 'small', 'medium', 'large', 'full']
    if model_capacity not in valid_capacities:
        raise ValueError(f"model_capacity 必须是 {valid_capacities} 中的一个")
    
    try:
        logging.info(f"开始 CREPE 音高检测: 模型={model_capacity}, 步长={step_size}ms")
        
        # 调用 CREPE 模型
        time, frequency, confidence, activation = crepe.predict(
            audio=audio,
            sr=sr,
            model_capacity=model_capacity,
            viterbi=viterbi,
            step_size=step_size,
            center=center,
            verbose=verbose
        )
        
        # 应用置信度过滤
        if min_confidence is not None:
            frequency, confidence = apply_confidence_filter(
                frequency, confidence, min_confidence
            )
        
        logging.info(f"CREPE 检测完成: 检测到 {len(time)} 个时间点")
        logging.info(f"频率范围: {np.min(frequency[confidence > 0]):.1f} - {np.max(frequency[confidence > 0]):.1f} Hz")
        logging.info(f"平均置信度: {np.mean(confidence[confidence > 0]):.3f}")
        
        return time, frequency, confidence
        
    except Exception as e:
        raise RuntimeError(f"CREPE 音高检测失败: {str(e)}")


def apply_confidence_filter(frequency: np.ndarray, 
                          confidence: np.ndarray,
                          min_confidence: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用置信度过滤，将低置信度的频率值设为0
    
    Args:
        frequency: 频率数组
        confidence: 置信度数组
        min_confidence: 最小置信度阈值
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (过滤后的频率, 过滤后的置信度)
    """
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence 必须在 0.0 到 1.0 之间")
    
    # 创建副本以避免修改原始数组
    filtered_frequency = frequency.copy()
    filtered_confidence = confidence.copy()
    
    # 应用过滤
    mask = confidence < min_confidence
    filtered_frequency[mask] = 0.0
    filtered_confidence[mask] = 0.0
    
    filtered_count = np.sum(mask)
    if filtered_count > 0:
        logging.info(f"置信度过滤: 过滤了 {filtered_count}/{len(frequency)} 个低置信度点")
    
    return filtered_frequency, filtered_confidence


def smooth_pitch_contour(frequency: np.ndarray, 
                        confidence: np.ndarray,
                        window_size: int = 5,
                        min_confidence: float = 0.5) -> np.ndarray:
    """
    对音高轮廓进行平滑处理
    
    Args:
        frequency: 频率数组
        confidence: 置信度数组
        window_size: 平滑窗口大小
        min_confidence: 参与平滑的最小置信度
        
    Returns:
        np.ndarray: 平滑后的频率数组
    """
    if window_size < 1:
        raise ValueError("window_size 必须大于等于1")
    
    smoothed_frequency = frequency.copy()
    
    # 只对高置信度的点进行平滑
    valid_mask = confidence >= min_confidence
    
    for i in range(len(frequency)):
        if not valid_mask[i]:
            continue
            
        # 定义窗口范围
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(frequency), i + window_size // 2 + 1)
        
        # 在窗口内找到有效的频率值
        window_mask = valid_mask[start_idx:end_idx]
        window_freq = frequency[start_idx:end_idx]
        
        if np.sum(window_mask) > 0:
            # 计算加权平均（使用置信度作为权重）
            window_conf = confidence[start_idx:end_idx]
            weights = window_conf * window_mask
            
            if np.sum(weights) > 0:
                smoothed_frequency[i] = np.average(window_freq, weights=weights)
    
    return smoothed_frequency


def frequency_to_midi_note(frequency: np.ndarray, 
                          confidence: np.ndarray,
                          min_confidence: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    将频率转换为 MIDI 音符编号
    
    Args:
        frequency: 频率数组 (Hz)
        confidence: 置信度数组
        min_confidence: 最小置信度阈值
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (MIDI 音符编号, 音符名称数组)
    """
    # 过滤低置信度的点
    valid_mask = confidence >= min_confidence
    
    # 初始化结果数组
    midi_notes = np.zeros_like(frequency)
    note_names = np.array([''] * len(frequency), dtype='<U4')
    
    # 只处理有效的频率点
    valid_freq = frequency[valid_mask]
    
    if len(valid_freq) > 0:
        # 转换为 MIDI 音符编号 (A4 = 440Hz = MIDI 69)
        midi_valid = 69 + 12 * np.log2(valid_freq / 440.0)
        midi_notes[valid_mask] = np.round(midi_valid)
        
        # 转换为音符名称
        note_names_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i, midi_note in enumerate(midi_notes):
            if valid_mask[i] and midi_note > 0:
                octave = int(midi_note // 12) - 1
                note_idx = int(midi_note % 12)
                note_names[i] = f"{note_names_list[note_idx]}{octave}"
    
    return midi_notes, note_names


def detect_pitch_changes(frequency: np.ndarray, 
                        confidence: np.ndarray,
                        time: np.ndarray,
                        min_confidence: float = 0.5,
                        cent_threshold: float = 50.0) -> np.ndarray:
    """
    检测音高变化点
    
    Args:
        frequency: 频率数组
        confidence: 置信度数组
        time: 时间数组
        min_confidence: 最小置信度阈值
        cent_threshold: 音高变化阈值（音分）
        
    Returns:
        np.ndarray: 音高变化点的时间索引
    """
    if len(frequency) < 2:
        return np.array([])
    
    # 过滤低置信度的点
    valid_mask = confidence >= min_confidence
    
    # 计算音高变化（以音分为单位）
    freq_diff = np.diff(frequency)
    cent_diff = np.zeros_like(freq_diff)
    
    for i in range(len(freq_diff)):
        if valid_mask[i] and valid_mask[i+1] and frequency[i] > 0 and frequency[i+1] > 0:
            cent_diff[i] = 1200 * np.log2(frequency[i+1] / frequency[i])
    
    # 找到超过阈值的变化点
    change_points = np.where(np.abs(cent_diff) > cent_threshold)[0]
    
    return change_points


def get_pitch_statistics(frequency: np.ndarray, 
                        confidence: np.ndarray,
                        min_confidence: float = 0.5) -> dict:
    """
    计算音高统计信息
    
    Args:
        frequency: 频率数组
        confidence: 置信度数组
        min_confidence: 最小置信度阈值
        
    Returns:
        dict: 包含统计信息的字典
    """
    # 过滤有效的频率点
    valid_mask = (confidence >= min_confidence) & (frequency > 0)
    valid_freq = frequency[valid_mask]
    valid_conf = confidence[valid_mask]
    
    if len(valid_freq) == 0:
        return {
            "valid_frames": 0,
            "total_frames": len(frequency),
            "valid_ratio": 0.0,
            "mean_frequency": 0.0,
            "std_frequency": 0.0,
            "min_frequency": 0.0,
            "max_frequency": 0.0,
            "mean_confidence": 0.0,
            "std_confidence": 0.0
        }
    
    return {
        "valid_frames": len(valid_freq),
        "total_frames": len(frequency),
        "valid_ratio": len(valid_freq) / len(frequency),
        "mean_frequency": float(np.mean(valid_freq)),
        "std_frequency": float(np.std(valid_freq)),
        "min_frequency": float(np.min(valid_freq)),
        "max_frequency": float(np.max(valid_freq)),
        "mean_confidence": float(np.mean(valid_conf)),
        "std_confidence": float(np.std(valid_conf))
    }


def save_pitch_data(time: np.ndarray, 
                   frequency: np.ndarray, 
                   confidence: np.ndarray,
                   output_path: Union[str, Path],
                   format: str = 'csv') -> None:
    """
    保存音高检测结果到文件
    
    Args:
        time: 时间数组
        frequency: 频率数组
        confidence: 置信度数组
        output_path: 输出文件路径
        format: 输出格式，'csv' 或 'txt'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'csv':
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time(s)', 'Frequency(Hz)', 'Confidence'])
            for t, f, c in zip(time, frequency, confidence):
                writer.writerow([f"{t:.3f}", f"{f:.2f}", f"{c:.3f}"])
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Time(s)\tFrequency(Hz)\tConfidence\n")
            for t, f, c in zip(time, frequency, confidence):
                f.write(f"{t:.3f}\t{f:.2f}\t{c:.3f}\n")
    
    logging.info(f"音高数据已保存到: {output_path}")


class PitchDetector:
    """
    音高检测器类 - 为main.py提供类接口
    """
    
    def __init__(self, model_capacity: str = 'medium', viterbi: bool = True, 
                 step_size: int = 10, min_confidence: float = 0.5):
        """
        初始化音高检测器
        
        Args:
            model_capacity: CREPE模型容量
            viterbi: 是否使用维特比解码
            step_size: 步长（毫秒）
            min_confidence: 最小置信度阈值
        """
        self.model_capacity = model_capacity
        self.viterbi = viterbi
        self.step_size = step_size
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
        
    def detect_pitch(self, audio_data: dict) -> dict:
        """
        检测音高
        
        Args:
            audio_data: 包含音频数据的字典
            
        Returns:
            dict: 包含音高检测结果的字典
        """
        try:
            audio = audio_data["audio"]
            sr = audio_data["sample_rate"]
            
            # 执行音高检测
            time, frequency, confidence = predict_pitch(
                audio=audio,
                sr=sr,
                model_capacity=self.model_capacity,
                viterbi=self.viterbi,
                step_size=self.step_size,
                min_confidence=self.min_confidence
            )
            
            # 平滑处理
            smoothed_frequency = smooth_pitch_contour(frequency, confidence)
            
            # 转换为MIDI音符
            midi_notes, note_names = frequency_to_midi_note(frequency, confidence)
            
            # 检测音高变化点
            change_points = detect_pitch_changes(frequency, confidence, time)
            
            # 获取统计信息
            statistics = get_pitch_statistics(frequency, confidence)
            
            return {
                "times": time,
                "frequencies": frequency,
                "smoothed_frequencies": smoothed_frequency,
                "confidence": confidence,
                "midi_notes": midi_notes,
                "note_names": note_names,
                "change_points": change_points,
                "statistics": statistics,
                "timestamp": datetime.now().isoformat(),
                "model_capacity": self.model_capacity,
                "step_size": self.step_size,
                "min_confidence": self.min_confidence
            }
            
        except Exception as e:
            self.logger.error(f"音高检测失败: {e}")
            raise


# 示例用法
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例：生成测试音频（440Hz 正弦波）
    import numpy as np
    
    sr = 22050
    duration = 2.0  # 2秒
    t = np.linspace(0, duration, int(sr * duration))
    frequency_test = 440.0  # A4
    audio_test = 0.5 * np.sin(2 * np.pi * frequency_test * t)
    
    try:
        # 测试类接口
        detector = PitchDetector()
        audio_data = {"audio": audio_test, "sample_rate": sr}
        result = detector.detect_pitch(audio_data)
        
        print(f"音高统计: {result['statistics']}")
        print(f"检测到的主要音符: {result['note_names'][result['confidence'] > 0.8]}")
        
    except Exception as e:
        print(f"音高检测失败: {e}") 