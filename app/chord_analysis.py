"""
和弦分析模块
功能：从音高时间序列中识别和弦
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, Counter
from dataclasses import dataclass


@dataclass
class ChordSegment:
    """和弦片段数据类"""
    start_time: float
    end_time: float
    chord_name: str
    confidence: float
    notes: List[str]
    duration: float


class ChordAnalyzer:
    """和弦分析器"""
    
    def __init__(self, 
                 time_window: float = 1.0,
                 min_confidence: float = 0.5,
                 min_notes: int = 2):
        """
        初始化和弦分析器
        
        Args:
            time_window: 时间窗口大小（秒），用于聚合音符
            min_confidence: 最小置信度阈值
            min_notes: 识别和弦所需的最少音符数量
        """
        self.time_window = time_window
        self.min_confidence = min_confidence
        self.min_notes = min_notes
        
        # 初始化和弦模板
        self.chord_templates = self._init_chord_templates()
        
        # 音符名称映射
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        logging.info(f"和弦分析器初始化完成: 时间窗口={time_window}s, 最小置信度={min_confidence}")
    
    def _init_chord_templates(self) -> Dict[str, List[int]]:
        """
        初始化和弦模板
        
        Returns:
            Dict[str, List[int]]: 和弦名称到音符间隔的映射
        """
        templates = {
            # 大三和弦 (Major)
            'C': [0, 4, 7], 'C#': [1, 5, 8], 'D': [2, 6, 9], 'D#': [3, 7, 10],
            'E': [4, 8, 11], 'F': [5, 9, 0], 'F#': [6, 10, 1], 'G': [7, 11, 2],
            'G#': [8, 0, 3], 'A': [9, 1, 4], 'A#': [10, 2, 5], 'B': [11, 3, 6],
            
            # 小三和弦 (Minor)
            'Cm': [0, 3, 7], 'C#m': [1, 4, 8], 'Dm': [2, 5, 9], 'D#m': [3, 6, 10],
            'Em': [4, 7, 11], 'Fm': [5, 8, 0], 'F#m': [6, 9, 1], 'Gm': [7, 10, 2],
            'G#m': [8, 11, 3], 'Am': [9, 0, 4], 'A#m': [10, 1, 5], 'Bm': [11, 2, 6],
            
            # 七和弦 (7th)
            'C7': [0, 4, 7, 10], 'C#7': [1, 5, 8, 11], 'D7': [2, 6, 9, 0],
            'D#7': [3, 7, 10, 1], 'E7': [4, 8, 11, 2], 'F7': [5, 9, 0, 3],
            'F#7': [6, 10, 1, 4], 'G7': [7, 11, 2, 5], 'G#7': [8, 0, 3, 6],
            'A7': [9, 1, 4, 7], 'A#7': [10, 2, 5, 8], 'B7': [11, 3, 6, 9],
            
            # 大七和弦 (Major 7th)
            'Cmaj7': [0, 4, 7, 11], 'C#maj7': [1, 5, 8, 0], 'Dmaj7': [2, 6, 9, 1],
            'D#maj7': [3, 7, 10, 2], 'Emaj7': [4, 8, 11, 3], 'Fmaj7': [5, 9, 0, 4],
            'F#maj7': [6, 10, 1, 5], 'Gmaj7': [7, 11, 2, 6], 'G#maj7': [8, 0, 3, 7],
            'Amaj7': [9, 1, 4, 8], 'A#maj7': [10, 2, 5, 9], 'Bmaj7': [11, 3, 6, 10],
            
            # 小七和弦 (Minor 7th)
            'Cm7': [0, 3, 7, 10], 'C#m7': [1, 4, 8, 11], 'Dm7': [2, 5, 9, 0],
            'D#m7': [3, 6, 10, 1], 'Em7': [4, 7, 11, 2], 'Fm7': [5, 8, 0, 3],
            'F#m7': [6, 9, 1, 4], 'Gm7': [7, 10, 2, 5], 'G#m7': [8, 11, 3, 6],
            'Am7': [9, 0, 4, 7], 'A#m7': [10, 1, 5, 8], 'Bm7': [11, 2, 6, 9],
            
            # 挂四和弦 (Sus4)
            'Csus4': [0, 5, 7], 'C#sus4': [1, 6, 8], 'Dsus4': [2, 7, 9],
            'D#sus4': [3, 8, 10], 'Esus4': [4, 9, 11], 'Fsus4': [5, 10, 0],
            'F#sus4': [6, 11, 1], 'Gsus4': [7, 0, 2], 'G#sus4': [8, 1, 3],
            'Asus4': [9, 2, 4], 'A#sus4': [10, 3, 5], 'Bsus4': [11, 4, 6],
            
            # 挂二和弦 (Sus2)
            'Csus2': [0, 2, 7], 'C#sus2': [1, 3, 8], 'Dsus2': [2, 4, 9],
            'D#sus2': [3, 5, 10], 'Esus2': [4, 6, 11], 'Fsus2': [5, 7, 0],
            'F#sus2': [6, 8, 1], 'Gsus2': [7, 9, 2], 'G#sus2': [8, 10, 3],
            'Asus2': [9, 11, 4], 'A#sus2': [10, 0, 5], 'Bsus2': [11, 1, 6],
            
            # 减三和弦 (Diminished)
            'Cdim': [0, 3, 6], 'C#dim': [1, 4, 7], 'Ddim': [2, 5, 8],
            'D#dim': [3, 6, 9], 'Edim': [4, 7, 10], 'Fdim': [5, 8, 11],
            'F#dim': [6, 9, 0], 'Gdim': [7, 10, 1], 'G#dim': [8, 11, 2],
            'Adim': [9, 0, 3], 'A#dim': [10, 1, 4], 'Bdim': [11, 2, 5],
            
            # 增三和弦 (Augmented)
            'Caug': [0, 4, 8], 'C#aug': [1, 5, 9], 'Daug': [2, 6, 10],
            'D#aug': [3, 7, 11], 'Eaug': [4, 8, 0], 'Faug': [5, 9, 1],
            'F#aug': [6, 10, 2], 'Gaug': [7, 11, 3], 'G#aug': [8, 0, 4],
            'Aaug': [9, 1, 5], 'A#aug': [10, 2, 6], 'Baug': [11, 3, 7],
        }
        
        return templates
    
    def frequency_to_note(self, frequency: float) -> Optional[str]:
        """
        将频率转换为音符名称
        
        Args:
            frequency: 频率 (Hz)
            
        Returns:
            Optional[str]: 音符名称，如果频率无效则返回None
        """
        if frequency <= 0:
            return None
        
        # 使用A4=440Hz作为参考
        # MIDI音符 = 69 + 12 * log2(freq / 440)
        midi_note = 69 + 12 * np.log2(frequency / 440.0)
        note_index = int(round(midi_note)) % 12
        
        return self.note_names[note_index]
    
    def aggregate_notes_in_window(self, 
                                 time: np.ndarray, 
                                 frequency: np.ndarray, 
                                 confidence: np.ndarray) -> List[ChordSegment]:
        """
        在时间窗口内聚合音符并识别和弦
        
        Args:
            time: 时间数组
            frequency: 频率数组
            confidence: 置信度数组
            
        Returns:
            List[ChordSegment]: 和弦片段列表
        """
        if len(time) == 0:
            return []
        
        # 过滤低置信度的点
        valid_mask = confidence >= self.min_confidence
        valid_time = time[valid_mask]
        valid_freq = frequency[valid_mask]
        valid_conf = confidence[valid_mask]
        
        if len(valid_time) == 0:
            return []
        
        chord_segments = []
        start_time = valid_time[0]
        end_time = valid_time[-1]
        
        # 按时间窗口分割
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + self.time_window
            
            # 找到当前窗口内的音符
            window_mask = (valid_time >= current_time) & (valid_time < window_end)
            window_freq = valid_freq[window_mask]
            window_conf = valid_conf[window_mask]
            
            if len(window_freq) >= self.min_notes:
                # 转换为音符
                notes = []
                for freq in window_freq:
                    note = self.frequency_to_note(freq)
                    if note:
                        notes.append(note)
                
                if len(notes) >= self.min_notes:
                    # 识别和弦
                    chord_name, chord_confidence = self._identify_chord(notes)
                    
                    segment = ChordSegment(
                        start_time=current_time,
                        end_time=window_end,
                        chord_name=chord_name,
                        confidence=chord_confidence,
                        notes=notes,
                        duration=window_end - current_time
                    )
                    chord_segments.append(segment)
            
            current_time = window_end
        
        return chord_segments
    
    def _identify_chord(self, notes: List[str]) -> Tuple[str, float]:
        """
        根据音符列表识别和弦
        
        Args:
            notes: 音符列表
            
        Returns:
            Tuple[str, float]: (和弦名称, 置信度)
        """
        if len(notes) < self.min_notes:
            return "Unknown", 0.0
        
        # 统计音符出现次数
        note_counts = Counter(notes)
        unique_notes = list(note_counts.keys())
        
        # 转换为音符索引
        note_indices = []
        for note in unique_notes:
            if note in self.note_names:
                note_indices.append(self.note_names.index(note))
        
        if len(note_indices) < self.min_notes:
            return "Unknown", 0.0
        
        # 计算与和弦模板的匹配度
        best_chord = "Unknown"
        best_score = 0.0
        
        for chord_name, template in self.chord_templates.items():
            score = self._calculate_chord_match_score(note_indices, template)
            if score > best_score:
                best_score = score
                best_chord = chord_name
        
        return best_chord, best_score
    
    def _calculate_chord_match_score(self, 
                                   note_indices: List[int], 
                                   template: List[int]) -> float:
        """
        计算音符与和弦模板的匹配度
        
        Args:
            note_indices: 音符索引列表
            template: 和弦模板
            
        Returns:
            float: 匹配度分数 (0-1)
        """
        if len(note_indices) == 0 or len(template) == 0:
            return 0.0
        
        # 检查音符是否在模板中
        matches = 0
        for note_idx in note_indices:
            if note_idx in template:
                matches += 1
        
        # 计算匹配度
        match_ratio = matches / len(template)
        coverage_ratio = matches / len(note_indices)
        
        # 综合分数
        score = (match_ratio + coverage_ratio) / 2
        
        return score
    
    def _analyze_chords_from_pitch(self, 
                                  time: np.ndarray, 
                                  frequency: np.ndarray, 
                                  confidence: np.ndarray) -> List[ChordSegment]:
        """
        分析音高时间序列并识别和弦
        
        Args:
            time: 时间数组
            frequency: 频率数组
            confidence: 置信度数组
            
        Returns:
            List[ChordSegment]: 和弦片段列表
        """
        logging.info(f"开始和弦分析: {len(time)} 个时间点")
        
        # 聚合音符并识别和弦
        chord_segments = self.aggregate_notes_in_window(time, frequency, confidence)
        
        # 合并相邻的相同和弦
        merged_segments = self._merge_adjacent_chords(chord_segments)
        
        logging.info(f"和弦分析完成: 识别到 {len(merged_segments)} 个和弦片段")
        
        return merged_segments
    
    def analyze_chords(self, audio_data: dict, pitch_data: dict) -> dict:
        """
        分析和弦（为main.py提供的接口）
        
        Args:
            audio_data: 音频数据字典
            pitch_data: 音高数据字典
            
        Returns:
            dict: 包含和弦分析结果的字典
        """
        try:
            # 从pitch_data中提取需要的数据
            time = pitch_data["times"]
            frequency = pitch_data["frequencies"]
            confidence = pitch_data["confidence"]
            
            # 执行和弦分析
            chord_segments = self._analyze_chords_from_pitch(time, frequency, confidence)
            
            # 获取统计信息
            statistics = self.get_chord_statistics(chord_segments)
            
            # 转换为字典格式便于处理
            progression = []
            for segment in chord_segments:
                progression.append({
                    "time": segment.start_time,
                    "chord": segment.chord_name,
                    "confidence": segment.confidence,
                    "duration": segment.duration,
                    "notes": segment.notes
                })
            
            return {
                "segments": chord_segments,
                "progression": progression,
                "statistics": statistics,
                "total_chords": len(chord_segments),
                "total_duration": sum(seg.duration for seg in chord_segments),
                "timestamp": pitch_data.get("timestamp", "")
            }
            
        except Exception as e:
            logging.error(f"和弦分析失败: {e}")
            raise
    
    def _merge_adjacent_chords(self, segments: List[ChordSegment]) -> List[ChordSegment]:
        """
        合并相邻的相同和弦
        
        Args:
            segments: 和弦片段列表
            
        Returns:
            List[ChordSegment]: 合并后的和弦片段列表
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            if (next_segment.chord_name == current.chord_name and 
                next_segment.start_time == current.end_time):
                # 合并相邻的相同和弦
                current.end_time = next_segment.end_time
                current.duration = current.end_time - current.start_time
                current.notes.extend(next_segment.notes)
                current.confidence = max(current.confidence, next_segment.confidence)
            else:
                # 添加当前片段，开始新的片段
                merged.append(current)
                current = next_segment
        
        merged.append(current)
        return merged
    
    def get_chord_statistics(self, segments: List[ChordSegment]) -> Dict:
        """
        获取和弦统计信息
        
        Args:
            segments: 和弦片段列表
            
        Returns:
            Dict: 统计信息字典
        """
        if not segments:
            return {
                "total_segments": 0,
                "total_duration": 0.0,
                "unique_chords": 0,
                "chord_frequencies": {},
                "average_confidence": 0.0
            }
        
        # 统计和弦出现频率
        chord_counts = Counter(segment.chord_name for segment in segments)
        
        # 计算总时长
        total_duration = sum(segment.duration for segment in segments)
        
        # 计算平均置信度
        avg_confidence = np.mean([segment.confidence for segment in segments])
        
        return {
            "total_segments": len(segments),
            "total_duration": total_duration,
            "unique_chords": len(chord_counts),
            "chord_frequencies": dict(chord_counts),
            "average_confidence": avg_confidence
        }
    
    def save_chord_analysis(self, 
                          segments: List[ChordSegment], 
                          output_path: str) -> None:
        """
        保存和弦分析结果到文件
        
        Args:
            segments: 和弦片段列表
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Start_Time(s)\tEnd_Time(s)\tDuration(s)\tChord\tConfidence\tNotes\n")
            for segment in segments:
                notes_str = ','.join(segment.notes)
                f.write(f"{segment.start_time:.3f}\t{segment.end_time:.3f}\t"
                       f"{segment.duration:.3f}\t{segment.chord_name}\t"
                       f"{segment.confidence:.3f}\t{notes_str}\n")
        
        logging.info(f"和弦分析结果已保存到: {output_path}")


def analyze_chords_simple(time: np.ndarray, 
                         frequency: np.ndarray, 
                         confidence: np.ndarray,
                         time_window: float = 1.0,
                         min_confidence: float = 0.5) -> List[Dict]:
    """
    简化的和弦分析函数
    
    Args:
        time: 时间数组
        frequency: 频率数组
        confidence: 置信度数组
        time_window: 时间窗口大小
        min_confidence: 最小置信度
        
    Returns:
        List[Dict]: 和弦分析结果列表
    """
    analyzer = ChordAnalyzer(
        time_window=time_window,
        min_confidence=min_confidence
    )
    
    segments = analyzer._analyze_chords_from_pitch(time, frequency, confidence)
    
    # 转换为字典格式
    results = []
    for segment in segments:
        results.append({
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'chord': segment.chord_name,
            'confidence': segment.confidence,
            'notes': segment.notes,
            'duration': segment.duration
        })
    
    return results


# 示例用法
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例：生成测试数据（C大三和弦）
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 生成C大三和弦的音符：C(261.63Hz), E(329.63Hz), G(392.00Hz)
    c_freq = 261.63
    e_freq = 329.63
    g_freq = 392.00
    
    # 创建时间序列
    time_points = np.arange(0, duration, 0.1)  # 每0.1秒一个点
    frequencies = []
    confidences = []
    
    for t_point in time_points:
        # 随机选择和弦中的一个音符
        if t_point < 1.0:
            freq = c_freq
        elif t_point < 2.0:
            freq = e_freq
        else:
            freq = g_freq
        
        frequencies.append(freq)
        confidences.append(0.8)  # 高置信度
    
    frequencies = np.array(frequencies)
    confidences = np.array(confidences)
    
    try:
        # 进行和弦分析
        analyzer = ChordAnalyzer(time_window=1.0, min_confidence=0.5)
        segments = analyzer.analyze_chords(time_points, frequencies, confidences)
        
        # 打印结果
        print("和弦分析结果:")
        for segment in segments:
            print(f"时间: {segment.start_time:.1f}-{segment.end_time:.1f}s, "
                  f"和弦: {segment.chord_name}, 置信度: {segment.confidence:.3f}, "
                  f"音符: {segment.notes}")
        
        # 获取统计信息
        stats = analyzer.get_chord_statistics(segments)
        print(f"\n统计信息: {stats}")
        
    except Exception as e:
        print(f"和弦分析失败: {e}") 