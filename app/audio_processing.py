"""
音频处理模块 - 类与函数混合设计
功能包括：视频转音频、音频加载、采样率转换、节拍估计
"""

import os
import tempfile
import logging
import numpy as np
import librosa
import ffmpeg
from pathlib import Path
from typing import Tuple, List, Union, Optional
from datetime import datetime

# 确保/opt/homebrew/bin加入PATH，保证ffmpeg可被Python子进程找到
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"


def extract_audio_from_video(video_path: Union[str, Path], 
                           output_path: Optional[Union[str, Path]] = None,
                           sample_rate: int = 22050) -> str:
    """
    使用 ffmpeg 从视频文件中提取音频并保存为 .wav 格式
    
    Args:
        video_path: 输入视频文件路径
        output_path: 输出音频文件路径，如果为None则创建临时文件
        sample_rate: 目标采样率，默认22050Hz
        
    Returns:
        str: 输出的音频文件路径
        
    Raises:
        FileNotFoundError: 视频文件不存在
        RuntimeError: ffmpeg 处理失败
    """
    video_path = Path(video_path)
    
    # 检查输入文件是否存在
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 设置输出路径
    if output_path is None:
        # 创建临时文件
        temp_fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='extracted_audio_')
        os.close(temp_fd)  # 关闭文件描述符，但保留文件路径
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用 ffmpeg 提取音频
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(output_path),
                acodec='pcm_s16le',  # 16位PCM编码
                ac=1,                # 单声道
                ar=sample_rate,      # 采样率
                loglevel='error'     # 只显示错误信息
            )
            .overwrite_output()      # 覆盖已存在的输出文件
            .run()
        )
        
        logging.info(f"音频提取成功: {video_path} -> {output_path}")
        return str(output_path)
        
    except ffmpeg.Error as e:
        # 清理可能创建的临时文件
        if output_path and Path(output_path).exists():
            Path(output_path).unlink()
        raise RuntimeError(f"ffmpeg 处理失败: {e.stderr.decode() if e.stderr else str(e)}")


def load_audio(file_path: Union[str, Path], 
               target_sr: int = 22050,
               mono: bool = True,
               offset: float = 0.0,
               duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    使用 librosa 加载音频文件并进行采样率转换
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率，默认22050Hz
        mono: 是否转换为单声道，默认True
        offset: 开始时间偏移（秒），默认0.0
        duration: 加载时长（秒），None表示加载全部
        
    Returns:
        Tuple[np.ndarray, int]: (音频数组, 采样率)
        
    Raises:
        FileNotFoundError: 音频文件不存在
        RuntimeError: 音频加载失败
    """
    file_path = Path(file_path)
    
    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    
    try:
        # 使用 librosa 加载音频
        audio, sr = librosa.load(
            str(file_path),
            sr=target_sr,
            mono=mono,
            offset=offset,
            duration=duration
        )
        
        logging.info(f"音频加载成功: {file_path}, 采样率: {sr}Hz, 时长: {len(audio)/sr:.2f}秒")
        return audio, sr
        
    except Exception as e:
        raise RuntimeError(f"音频加载失败: {str(e)}")


def resample_audio(audio: np.ndarray, 
                  orig_sr: int, 
                  target_sr: int,
                  res_type: str = 'kaiser_best') -> np.ndarray:
    """
    对音频进行采样率转换
    
    Args:
        audio: 输入音频数组
        orig_sr: 原始采样率
        target_sr: 目标采样率
        res_type: 重采样类型，默认'kaiser_best'
        
    Returns:
        np.ndarray: 重采样后的音频数组
        
    Raises:
        ValueError: 采样率参数无效
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("采样率必须大于0")
    
    if orig_sr == target_sr:
        return audio
    
    try:
        resampled_audio = librosa.resample(
            audio, 
            orig_sr=orig_sr, 
            target_sr=target_sr,
            res_type=res_type
        )
        
        logging.info(f"采样率转换: {orig_sr}Hz -> {target_sr}Hz")
        return resampled_audio
        
    except Exception as e:
        raise RuntimeError(f"采样率转换失败: {str(e)}")


def estimate_tempo_and_beats(audio: np.ndarray, 
                           sr: int,
                           hop_length: int = 512,
                           start_bpm: float = 120.0) -> Tuple[float, np.ndarray]:
    """
    使用 librosa 进行节拍估计
    
    Args:
        audio: 音频数组
        sr: 采样率
        hop_length: 跳跃长度，默认512
        start_bpm: 起始BPM估计，默认120.0
        
    Returns:
        Tuple[float, np.ndarray]: (估计的BPM, 节拍时间点数组)
        
    Raises:
        RuntimeError: 节拍估计失败
    """
    if len(audio) == 0:
        raise ValueError("音频数组为空")
    
    try:
        # 使用 librosa 进行节拍跟踪
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            start_bpm=start_bpm,
            units='time'  # 返回时间单位而不是帧单位
        )
        
        # 确保tempo和beats是Python原生类型
        tempo_float = float(tempo) if tempo is not None else 0.0
        beats_array = np.array(beats) if beats is not None else np.array([])
        
        logging.info(f"节拍估计完成: BPM={tempo_float:.1f}, 检测到{len(beats_array)}个节拍点")
        return tempo_float, beats_array
        
    except Exception as e:
        raise RuntimeError(f"节拍估计失败: {str(e)}")


def process_video_to_audio_with_beats(video_path: Union[str, Path],
                                    target_sr: int = 22050,
                                    temp_audio_path: Optional[Union[str, Path]] = None,
                                    cleanup_temp: bool = True) -> Tuple[np.ndarray, int, float, np.ndarray]:
    """
    完整的视频处理流程：提取音频 -> 加载 -> 节拍估计
    
    Args:
        video_path: 输入视频文件路径
        target_sr: 目标采样率，默认22050Hz
        temp_audio_path: 临时音频文件路径，None表示自动创建
        cleanup_temp: 是否清理临时文件，默认True
        
    Returns:
        Tuple[np.ndarray, int, float, np.ndarray]: (音频数组, 采样率, BPM, 节拍时间点)
        
    Raises:
        FileNotFoundError: 视频文件不存在
        RuntimeError: 处理过程中发生错误
    """
    temp_file_created = temp_audio_path is None
    
    try:
        # 1. 从视频提取音频
        audio_path = extract_audio_from_video(
            video_path=video_path,
            output_path=temp_audio_path,
            sample_rate=target_sr
        )
        
        # 2. 加载音频
        audio, sr = load_audio(
            file_path=audio_path,
            target_sr=target_sr
        )
        
        # 3. 估计节拍
        tempo, beats = estimate_tempo_and_beats(audio, sr)
        
        # 4. 清理临时文件
        if cleanup_temp and temp_file_created and Path(audio_path).exists():
            Path(audio_path).unlink()
            logging.info(f"临时文件已清理: {audio_path}")
        
        return audio, sr, tempo, beats
        
    except Exception as e:
        # 清理临时文件
        if cleanup_temp and temp_file_created and temp_audio_path and Path(temp_audio_path).exists():
            Path(temp_audio_path).unlink()
        raise


def process_audio_file_with_beats(audio_path: Union[str, Path],
                                target_sr: int = 22050) -> Tuple[np.ndarray, int, float, np.ndarray]:
    """
    处理音频文件：加载 -> 采样率转换 -> 节拍估计
    
    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率，默认22050Hz
        
    Returns:
        Tuple[np.ndarray, int, float, np.ndarray]: (音频数组, 采样率, BPM, 节拍时间点)
        
    Raises:
        FileNotFoundError: 音频文件不存在
        RuntimeError: 处理过程中发生错误
    """
    try:
        # 1. 加载音频
        audio, sr = load_audio(
            file_path=audio_path,
            target_sr=target_sr
        )
        
        # 2. 估计节拍
        tempo, beats = estimate_tempo_and_beats(audio, sr)
        
        return audio, sr, tempo, beats
        
    except Exception as e:
        raise RuntimeError(f"音频文件处理失败: {str(e)}")


def get_audio_info(audio: np.ndarray, sr: int) -> dict:
    """
    获取音频基本信息
    
    Args:
        audio: 音频数组
        sr: 采样率
        
    Returns:
        dict: 包含音频信息的字典
    """
    duration = len(audio) / sr
    
    return {
        "duration": duration,
        "sample_rate": sr,
        "samples": len(audio),
        "channels": 1 if audio.ndim == 1 else audio.shape[0],
        "max_amplitude": float(np.max(np.abs(audio))),
        "rms_energy": float(np.sqrt(np.mean(audio ** 2)))
    }


class AudioProcessor:
    """
    音频处理器类 - 为main.py提供类接口
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 默认采样率
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
    def process_file(self, input_path: Union[str, Path]) -> dict:
        """
        处理输入文件（音频或视频）
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            dict: 包含音频数据的字典
        """
        input_path = Path(input_path)
        
        # 检查文件扩展名
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        
        file_ext = input_path.suffix.lower()
        
        try:
            if file_ext in video_extensions:
                # 处理视频文件
                audio, sr, tempo, beats = process_video_to_audio_with_beats(
                    input_path, 
                    target_sr=self.sample_rate
                )
            elif file_ext in audio_extensions:
                # 处理音频文件
                audio, sr, tempo, beats = process_audio_file_with_beats(
                    input_path,
                    target_sr=self.sample_rate
                )
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            # 获取音频信息
            audio_info = get_audio_info(audio, sr)
            
            # 返回结构化数据
            return {
                "audio": audio,
                "sample_rate": sr,
                "duration": audio_info["duration"],
                "tempo": tempo,
                "beats": beats,
                "rms_energy": audio_info["rms_energy"],
                "max_amplitude": audio_info["max_amplitude"],
                "timestamp": datetime.now().isoformat(),
                "source_file": str(input_path)
            }
            
        except Exception as e:
            self.logger.error(f"处理文件失败: {e}")
            raise


# 示例用法
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例：处理视频文件
    try:
        processor = AudioProcessor()
        result = processor.process_file("example.mp4")
        
        print(f"音频信息: 时长={result['duration']:.2f}秒, 采样率={result['sample_rate']}Hz")
        print(f"节拍信息: BPM={result['tempo']:.1f}, 节拍点数量={len(result['beats'])}")
        
    except Exception as e:
        print(f"处理失败: {e}") 