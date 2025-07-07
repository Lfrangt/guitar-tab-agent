"""
六线谱生成模块
功能：将频率信息转换为吉他六线谱
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


@dataclass
class GuitarNote:
    """吉他音符数据类"""
    time: float
    string: int  # 弦号 (1-6, 1=最细弦)
    fret: int    # 品位 (0-24)
    frequency: float
    confidence: float
    note_name: str


@dataclass
class TabMeasure:
    """小节数据类"""
    start_time: float
    end_time: float
    notes: List[GuitarNote]
    chord_name: Optional[str] = None


class TabGenerator:
    """六线谱生成器"""
    
    def __init__(self, 
                 tuning: str = 'standard',
                 time_signature: Tuple[int, int] = (4, 4),
                 measures_per_line: int = 4,
                 max_fret: int = 24):
        """
        初始化六线谱生成器
        
        Args:
            tuning: 调弦方式 ('standard', 'drop_d')
            time_signature: 拍号 (分子, 分母)
            measures_per_line: 每行小节数
            max_fret: 最大品位
        """
        self.tuning = tuning
        self.time_signature = time_signature
        self.measures_per_line = measures_per_line
        self.max_fret = max_fret
        
        # 设置调弦
        self.string_tunings = self._get_tuning(tuning)
        
        # 音符名称
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        logging.info(f"六线谱生成器初始化: 调弦={tuning}, 拍号={time_signature}")
    
    def _get_tuning(self, tuning: str) -> List[str]:
        """
        获取调弦配置
        
        Args:
            tuning: 调弦方式
            
        Returns:
            List[str]: 各弦的调音音符 (从最细弦到最粗弦)
        """
        tunings = {
            'standard': ['E', 'B', 'G', 'D', 'A', 'E'],  # 标准调弦
            'drop_d': ['E', 'B', 'G', 'D', 'A', 'D'],    # Drop D调弦
            'open_g': ['D', 'B', 'G', 'D', 'G', 'D'],    # Open G调弦
            'open_d': ['D', 'A', 'F#', 'D', 'A', 'D'],   # Open D调弦
            'dadgad': ['D', 'A', 'D', 'G', 'A', 'D'],    # DADGAD调弦
        }
        
        return tunings.get(tuning, tunings['standard'])
    
    def frequency_to_guitar_position(self, 
                                   frequency: float,
                                   confidence: float = 1.0) -> Optional[GuitarNote]:
        """
        将频率转换为吉他指位
        
        Args:
            frequency: 频率 (Hz)
            confidence: 置信度
            
        Returns:
            Optional[GuitarNote]: 吉他音符，如果无法转换则返回None
        """
        if frequency <= 0 or confidence <= 0:
            return None
        
        # 计算MIDI音符编号
        midi_note = 69 + 12 * np.log2(frequency / 440.0)
        note_index = int(round(midi_note)) % 12
        octave = int(midi_note // 12) - 1
        
        note_name = self.note_names[note_index]
        
        # 寻找最佳指位
        best_string = None
        best_fret = None
        min_fret = float('inf')
        
        for string_idx, string_tuning in enumerate(self.string_tunings):
            # 计算该弦的空弦MIDI音符
            string_note_idx = self.note_names.index(string_tuning)
            string_midi = 69 + 12 * np.log2(440.0 / 440.0) + (string_idx * 5)  # 简化计算
            
            # 计算需要的品位
            fret_diff = note_index - string_note_idx
            if fret_diff < 0:
                fret_diff += 12
            
            # 考虑八度差异
            octave_diff = octave - (string_midi // 12)
            fret = fret_diff + octave_diff * 12
            
            # 检查是否在有效范围内
            if 0 <= fret <= self.max_fret:
                if fret < min_fret:
                    min_fret = fret
                    best_string = 6 - string_idx  # 转换为标准弦号 (1-6)
                    best_fret = fret
        
        if best_string is not None:
            return GuitarNote(
                time=0.0,  # 时间将在后续设置
                string=best_string,
                fret=best_fret,
                frequency=frequency,
                confidence=confidence,
                note_name=note_name
            )
        
        return None
    
    def convert_frequency_series(self, 
                                time: np.ndarray,
                                frequency: np.ndarray,
                                confidence: np.ndarray) -> List[GuitarNote]:
        """
        转换频率时间序列为吉他音符列表
        
        Args:
            time: 时间数组
            frequency: 频率数组
            confidence: 置信度数组
            
        Returns:
            List[GuitarNote]: 吉他音符列表
        """
        notes = []
        
        for i, (t, freq, conf) in enumerate(zip(time, frequency, confidence)):
            if freq > 0 and conf > 0:
                note = self.frequency_to_guitar_position(freq, conf)
                if note:
                    note.time = t
                    notes.append(note)
        
        logging.info(f"转换完成: {len(notes)} 个有效音符")
        return notes
    
    def group_notes_by_measures(self, 
                               notes: List[GuitarNote],
                               tempo: float = 120.0) -> List[TabMeasure]:
        """
        将音符按小节分组
        
        Args:
            notes: 音符列表
            tempo: 速度 (BPM)
            
        Returns:
            List[TabMeasure]: 小节列表
        """
        if not notes:
            return []
        
        # 计算小节时长
        beats_per_measure = self.time_signature[0]
        beat_duration = 60.0 / tempo
        measure_duration = beats_per_measure * beat_duration
        
        # 按时间分组
        measures = []
        current_measure_start = notes[0].time
        current_notes = []
        
        for note in notes:
            measure_end = current_measure_start + measure_duration
            
            if note.time >= measure_end:
                # 完成当前小节
                if current_notes:
                    measure = TabMeasure(
                        start_time=current_measure_start,
                        end_time=measure_end,
                        notes=current_notes.copy()
                    )
                    measures.append(measure)
                
                # 开始新小节
                current_measure_start = measure_end
                current_notes = []
            
            current_notes.append(note)
        
        # 添加最后一个小节
        if current_notes:
            measure = TabMeasure(
                start_time=current_measure_start,
                end_time=current_measure_start + measure_duration,
                notes=current_notes
            )
            measures.append(measure)
        
        logging.info(f"分组完成: {len(measures)} 个小节")
        return measures
    
    def generate_tab_text(self, measures: List[TabMeasure]) -> str:
        """
        生成六线谱文本格式
        
        Args:
            measures: 小节列表
            
        Returns:
            str: 六线谱文本
        """
        if not measures:
            return ""
        
        # 生成头部信息
        tab_text = []
        tab_text.append("=" * 60)
        tab_text.append(f"吉他六线谱 - 调弦: {self.tuning.upper()}")
        tab_text.append(f"拍号: {self.time_signature[0]}/{self.time_signature[1]}")
        tab_text.append("=" * 60)
        tab_text.append("")
        
        # 添加调弦信息
        tuning_line = "调弦: "
        for i, tuning in enumerate(self.string_tunings):
            tuning_line += f"{6-i}弦={tuning} "
        tab_text.append(tuning_line)
        tab_text.append("")
        
        # 按行生成小节
        for i in range(0, len(measures), self.measures_per_line):
            line_measures = measures[i:i + self.measures_per_line]
            tab_text.extend(self._generate_tab_line(line_measures))
            tab_text.append("")
        
        return "\n".join(tab_text)
    
    def _generate_tab_line(self, measures: List[TabMeasure]) -> List[str]:
        """
        生成一行六线谱
        
        Args:
            measures: 一行中的小节列表
            
        Returns:
            List[str]: 六线谱行
        """
        # 初始化6行（对应6根弦）
        lines = ["" for _ in range(6)]
        
        for measure in measures:
            # 为每个小节生成指位信息
            measure_lines = self._generate_measure_tab(measure)
            
            # 添加到对应行
            for i in range(6):
                lines[i] += measure_lines[i] + " | "
        
        return lines
    
    def _generate_measure_tab(self, measure: TabMeasure) -> List[str]:
        """
        生成单个小节的六线谱
        
        Args:
            measure: 小节
            
        Returns:
            List[str]: 6行指位信息
        """
        # 初始化6行
        lines = ["-" * 16 for _ in range(6)]
        
        # 按时间排序音符
        sorted_notes = sorted(measure.notes, key=lambda x: x.time)
        
        # 计算音符在小节中的位置
        measure_duration = measure.end_time - measure.start_time
        
        for note in sorted_notes:
            # 计算音符在小节中的相对位置
            relative_time = note.time - measure.start_time
            position = int((relative_time / measure_duration) * 16)
            
            if 0 <= position < 16:
                # 在对应弦上放置指位
                string_idx = 6 - note.string  # 转换为数组索引
                if string_idx < 6:
                    # 格式化指位数字
                    fret_str = str(note.fret)
                    if len(fret_str) == 1:
                        fret_str = " " + fret_str
                    
                    # 确保不超出边界
                    if position + len(fret_str) <= 16:
                        lines[string_idx] = (
                            lines[string_idx][:position] + 
                            fret_str + 
                            lines[string_idx][position + len(fret_str):]
                        )
        
        return lines
    
    def save_tab_to_file(self, 
                        measures: List[TabMeasure],
                        output_path: Union[str, Path]) -> None:
        """
        保存六线谱到文本文件
        
        Args:
            measures: 小节列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tab_text = self.generate_tab_text(measures)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tab_text)
        
        logging.info(f"六线谱已保存到: {output_path}")
    
    def render_tab_image(self, 
                        measures: List[TabMeasure],
                        output_path: Union[str, Path],
                        figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        渲染六线谱为图像
        
        Args:
            measures: 小节列表
            output_path: 输出文件路径
            figsize: 图像尺寸
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置坐标轴
        ax.set_xlim(0, len(measures) * 2)
        ax.set_ylim(0, 6)
        
        # 绘制六线谱线
        for i in range(6):
            ax.axhline(y=i + 0.5, color='black', linewidth=1)
        
        # 绘制小节线
        for i in range(len(measures) + 1):
            ax.axvline(x=i * 2, color='black', linewidth=2)
        
        # 绘制音符
        for measure_idx, measure in enumerate(measures):
            for note in measure.notes:
                # 计算位置
                x = measure_idx * 2 + 1
                y = 6 - note.string + 0.5
                
                # 绘制音符圆圈
                circle = plt.Circle((x, y), 0.3, fill=True, color='black')
                ax.add_patch(circle)
                
                # 添加指位数字
                ax.text(x, y, str(note.fret), ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
        
        # 设置标签
        ax.set_yticks(range(1, 7))
        ax.set_yticklabels([f'{6-i}弦' for i in range(6)])
        ax.set_xticks(range(0, len(measures) * 2 + 1, 2))
        ax.set_xticklabels([f'小节{i+1}' for i in range(len(measures) + 1)])
        
        # 设置标题
        ax.set_title(f'吉他六线谱 - {self.tuning.upper()}调弦', fontsize=14, fontweight='bold')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"六线谱图像已保存到: {output_path}")
    
    def _generate_tab_from_pitch(self, 
                                time: np.ndarray,
                                frequency: np.ndarray,
                                confidence: np.ndarray,
                                tempo: float = 120.0) -> Tuple[List[TabMeasure], str]:
        """
        完整的六线谱生成流程
        
        Args:
            time: 时间数组
            frequency: 频率数组
            confidence: 置信度数组
            tempo: 速度 (BPM)
            
        Returns:
            Tuple[List[TabMeasure], str]: (小节列表, 六线谱文本)
        """
        logging.info("开始生成六线谱...")
        
        # 1. 转换频率为吉他指位
        notes = self.convert_frequency_series(time, frequency, confidence)
        
        # 2. 按小节分组
        measures = self.group_notes_by_measures(notes, tempo)
        
        # 3. 生成六线谱文本
        tab_text = self.generate_tab_text(measures)
        
        logging.info("六线谱生成完成")
        return measures, tab_text
    
    def generate_tab(self, pitch_data: dict, chord_data: dict) -> dict:
        """
        生成六线谱（为main.py提供的接口）
        
        Args:
            pitch_data: 音高数据字典
            chord_data: 和弦数据字典
            
        Returns:
            dict: 包含六线谱数据的字典
        """
        try:
            # 从pitch_data中提取需要的数据
            time = pitch_data["times"]
            frequency = pitch_data["frequencies"]
            confidence = pitch_data["confidence"]
            
            # 获取节拍信息（如果有的话）
            tempo = pitch_data.get("tempo", 120.0)
            
            # 生成六线谱
            measures, tab_text = self._generate_tab_from_pitch(time, frequency, confidence, tempo)
            
            # 为每个小节添加和弦信息
            chord_progression = chord_data.get("progression", [])
            for measure in measures:
                # 找到这个小节时间范围内的和弦
                for chord_info in chord_progression:
                    if measure.start_time <= chord_info["time"] <= measure.end_time:
                        measure.chord_name = chord_info["chord"]
                        break
            
            # 转换为吉他位置信息
            positions = []
            for measure in measures:
                for note in measure.notes:
                    positions.append({
                        "time": note.time,
                        "string": note.string,
                        "fret": note.fret,
                        "frequency": note.frequency,
                        "confidence": note.confidence,
                        "note_name": note.note_name
                    })
            
            return {
                "ascii_tab": tab_text,
                "measures": measures,
                "positions": positions,
                "num_measures": len(measures),
                "tuning": self.string_tunings,
                "tuning_name": self.tuning,
                "time_signature": self.time_signature,
                "tempo": tempo,
                "total_notes": len(positions)
            }
            
        except Exception as e:
            logging.error(f"六线谱生成失败: {e}")
            raise


def generate_tab_simple(time: np.ndarray,
                       frequency: np.ndarray,
                       confidence: np.ndarray,
                       output_path: Union[str, Path],
                       tuning: str = 'standard',
                       tempo: float = 120.0,
                       save_image: bool = False) -> str:
    """
    简化的六线谱生成函数
    
    Args:
        time: 时间数组
        frequency: 频率数组
        confidence: 置信度数组
        output_path: 输出文件路径
        tuning: 调弦方式
        tempo: 速度
        save_image: 是否保存图像
        
    Returns:
        str: 六线谱文本
    """
    generator = TabGenerator(tuning=tuning)
    measures, tab_text = generator._generate_tab_from_pitch(time, frequency, confidence, tempo)
    
    # 保存文本文件（如果指定了路径）
    if output_path is not None:
        generator.save_tab_to_file(measures, output_path)
    
    # 可选保存图像
    if save_image and output_path is not None:
        image_path = Path(output_path).with_suffix('.png')
        generator.render_tab_image(measures, image_path)
    
    return tab_text


# 示例用法
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例：生成测试数据（C大三和弦的音符）
    duration = 4.0
    time_points = np.arange(0, duration, 0.1)
    
    # C大三和弦的频率：C(261.63Hz), E(329.63Hz), G(392.00Hz)
    frequencies = []
    confidences = []
    
    for t in time_points:
        # 模拟和弦进行
        if t < 1.0:
            freq = 261.63  # C
        elif t < 2.0:
            freq = 329.63  # E
        elif t < 3.0:
            freq = 392.00  # G
        else:
            freq = 261.63  # C (回到根音)
        
        frequencies.append(freq)
        confidences.append(0.9)
    
    frequencies = np.array(frequencies)
    confidences = np.array(confidences)
    
    try:
        # 生成六线谱
        tab_text = generate_tab_simple(
            time_points, frequencies, confidences,
            output_path="test_tab.txt",
            tuning="standard",
            tempo=120.0,
            save_image=True
        )
        
        print("六线谱生成成功!")
        print("\n生成的六线谱:")
        print(tab_text)
        
    except Exception as e:
        print(f"六线谱生成失败: {e}") 