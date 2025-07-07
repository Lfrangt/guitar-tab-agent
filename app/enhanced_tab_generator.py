"""
增强的六线谱生成器 - 生成专业级别的六线谱
功能：更美观的格式、和弦标记、小节线、拍号显示
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class EnhancedGuitarNote:
    """增强的吉他音符数据类"""
    time: float
    string: int  # 弦号 (0-5, 0=最细弦E)
    fret: int    # 品位 (0-24)
    frequency: float
    confidence: float
    note_name: str
    duration: float = 0.25  # 音符时长（四分音符=1.0）
    chord: Optional[str] = None  # 所属和弦


@dataclass
class EnhancedTabMeasure:
    """增强的小节数据类"""
    number: int  # 小节号
    start_time: float
    end_time: float
    notes: List[EnhancedGuitarNote]
    chord_name: Optional[str] = None
    time_signature: Tuple[int, int] = (4, 4)


class EnhancedTabGenerator:
    """增强的六线谱生成器"""
    
    def __init__(self, tuning: str = 'standard', measures_per_line: int = 4):
        """
        初始化增强六线谱生成器
        
        Args:
            tuning: 调弦方式
            measures_per_line: 每行小节数
        """
        self.tuning = tuning
        self.measures_per_line = measures_per_line
        self.string_tunings = self._get_tuning(tuning)
        self.logger = logging.getLogger(__name__)
        
        # 六线谱美化设置
        self.tab_width = 80  # 六线谱宽度
        self.fret_spacing = 3  # 品位间距
        self.measure_width = 16  # 小节宽度
        
    def _get_tuning(self, tuning_name: str) -> List[str]:
        """获取调弦设置"""
        tunings = {
            'standard': ['E', 'B', 'G', 'D', 'A', 'E'],  # 从细到粗
            'drop_d': ['E', 'B', 'G', 'D', 'A', 'D'],
            'dadgad': ['D', 'A', 'G', 'D', 'A', 'D'],
            'open_g': ['D', 'B', 'G', 'D', 'G', 'D'],
        }
        return tunings.get(tuning_name.lower(), tunings['standard'])
    
    def generate_professional_tab_text(self, measures: List[EnhancedTabMeasure], 
                                     title: str = "吉他六线谱") -> str:
        """
        生成专业级六线谱文本
        
        Args:
            measures: 小节列表
            title: 谱子标题
            
        Returns:
            str: 专业格式的六线谱文本
        """
        if not measures:
            return self._generate_empty_tab_template()
        
        tab_lines = []
        
        # 标题和信息
        tab_lines.extend(self._generate_header(title))
        
        # 按行生成六线谱
        for line_start in range(0, len(measures), self.measures_per_line):
            line_measures = measures[line_start:line_start + self.measures_per_line]
            tab_lines.extend(self._generate_professional_tab_line(line_measures))
            tab_lines.append("")  # 行间空白
        
        return "\n".join(tab_lines)
    
    def _generate_header(self, title: str) -> List[str]:
        """生成谱子头部信息"""
        lines = []
        
        # 标题框
        title_line = f"{'═' * 20} {title} {'═' * 20}"
        lines.append(title_line)
        lines.append("")
        
        # 调弦信息
        tuning_info = "调弦: "
        for i, note in enumerate(self.string_tunings):
            string_num = i + 1
            tuning_info += f"{string_num}弦={note}  "
        lines.append(tuning_info)
        lines.append("")
        
        # 说明
        lines.append("说明: 数字表示品位, 0表示空弦, |表示小节线")
        lines.append("      和弦标记显示在六线谱上方")
        lines.append("")
        
        return lines
    
    def _generate_professional_tab_line(self, measures: List[EnhancedTabMeasure]) -> List[str]:
        """生成专业格式的六线谱行"""
        lines = []
        
        # 收集这一行所有的和弦
        chord_line = self._generate_chord_line(measures)
        if chord_line.strip():
            lines.append(chord_line)
        
        # 生成小节号行
        measure_numbers = self._generate_measure_numbers(measures)
        lines.append(measure_numbers)
        
        # 生成6根弦的六线谱
        string_lines = self._generate_string_lines(measures)
        lines.extend(string_lines)
        
        return lines
    
    def _generate_chord_line(self, measures: List[EnhancedTabMeasure]) -> str:
        """生成和弦标记行"""
        chord_line = " " * 4  # 左边距
        
        for measure in measures:
            chord_display = measure.chord_name if measure.chord_name else ""
            # 和弦名称居中显示在小节上方
            chord_section = f"{chord_display:^{self.measure_width}}"
            chord_line += chord_section
        
        return chord_line
    
    def _generate_measure_numbers(self, measures: List[EnhancedTabMeasure]) -> str:
        """生成小节号行"""
        number_line = " " * 2  # 左边距
        
        for measure in measures:
            measure_num = f"M{measure.number}"
            number_section = f"{measure_num:^{self.measure_width}}"
            number_line += number_section
        
        return number_line
    
    def _generate_string_lines(self, measures: List[EnhancedTabMeasure]) -> List[str]:
        """生成六根弦的线"""
        string_lines = []
        
        # 为每根弦生成一行
        for string_idx in range(6):
            string_note = self.string_tunings[string_idx]
            line = f"{string_note}|"  # 弦名称和起始线
            
            # 为每个小节生成内容
            for measure in measures:
                measure_content = self._generate_measure_content(measure, string_idx)
                line += measure_content + "|"
            
            string_lines.append(line)
        
        return string_lines
    
    def _generate_measure_content(self, measure: EnhancedTabMeasure, string_idx: int) -> str:
        """生成单个小节在指定弦上的内容"""
        content = ["-"] * (self.measure_width - 1)  # 用破折号填充
        
        # 找到这根弦上的音符
        string_notes = [note for note in measure.notes if note.string == string_idx]
        
        if string_notes:
            # 按时间排序
            string_notes.sort(key=lambda x: x.time)
            
            # 计算音符在小节中的位置
            measure_duration = measure.end_time - measure.start_time
            
            for note in string_notes:
                relative_time = note.time - measure.start_time
                position = int((relative_time / measure_duration) * (self.measure_width - 1))
                position = max(0, min(position, self.measure_width - 2))
                
                # 格式化品位数字
                fret_str = str(note.fret)
                if len(fret_str) == 1:
                    content[position] = fret_str
                else:
                    # 双位数品位的处理
                    if position < len(content) - 1:
                        content[position] = fret_str[0]
                        content[position + 1] = fret_str[1]
        
        return "".join(content)
    
    def _generate_empty_tab_template(self) -> str:
        """生成空的六线谱模板"""
        lines = []
        lines.append("═" * 60)
        lines.append("                    空白六线谱模板")
        lines.append("═" * 60)
        lines.append("")
        
        # 调弦信息
        tuning_info = "调弦: "
        for i, note in enumerate(self.string_tunings):
            tuning_info += f"{i+1}弦={note}  "
        lines.append(tuning_info)
        lines.append("")
        
        # 4个空小节
        lines.append("     M1              M2              M3              M4")
        for i, string_note in enumerate(self.string_tunings):
            line = f"{string_note}|" + "-----------------|" * 4
            lines.append(line)
        
        lines.append("")
        lines.append("请上传音频/视频文件来生成六线谱内容")
        
        return "\n".join(lines)
    
    def convert_to_enhanced_measures(self, measures_data: List, chord_data: List) -> List[EnhancedTabMeasure]:
        """将普通小节数据转换为增强格式"""
        enhanced_measures = []
        
        if not measures_data:
            # 如果没有数据，创建4个空小节作为示例
            for i in range(4):
                measure = EnhancedTabMeasure(
                    number=i + 1,
                    start_time=i * 2.0,
                    end_time=(i + 1) * 2.0,
                    notes=[],
                    chord_name=None
                )
                enhanced_measures.append(measure)
            return enhanced_measures
        
        # 转换真实数据
        for i, measure_data in enumerate(measures_data):
            # 提取基本信息
            start_time = getattr(measure_data, 'start_time', i * 2.0)
            end_time = getattr(measure_data, 'end_time', (i + 1) * 2.0)
            notes = getattr(measure_data, 'notes', [])
            
            # 转换音符
            enhanced_notes = []
            for note in notes:
                enhanced_note = EnhancedGuitarNote(
                    time=getattr(note, 'time', start_time),
                    string=getattr(note, 'string', 0),
                    fret=getattr(note, 'fret', 0),
                    frequency=getattr(note, 'frequency', 0.0),
                    confidence=getattr(note, 'confidence', 0.0),
                    note_name=getattr(note, 'note_name', ''),
                    duration=0.25
                )
                enhanced_notes.append(enhanced_note)
            
            # 找到对应的和弦
            chord_name = None
            for chord in chord_data:
                chord_time = chord.get('time', 0)
                if start_time <= chord_time <= end_time:
                    chord_name = chord.get('chord', '')
                    break
            
            enhanced_measure = EnhancedTabMeasure(
                number=i + 1,
                start_time=start_time,
                end_time=end_time,
                notes=enhanced_notes,
                chord_name=chord_name
            )
            enhanced_measures.append(enhanced_measure)
        
        return enhanced_measures


def enhance_existing_tab_generator():
    """增强现有的六线谱生成器"""
    
    # 导入现有模块
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from tab_generator import TabGenerator
    
    # 为TabGenerator类添加增强方法
    def generate_enhanced_tab_text(self, measures, chord_data=None):
        """为现有TabGenerator添加增强格式"""
        enhancer = EnhancedTabGenerator(tuning=getattr(self, 'tuning', 'standard'))
        
        # 转换数据格式
        if chord_data is None:
            chord_data = []
        
        enhanced_measures = enhancer.convert_to_enhanced_measures(measures, chord_data)
        
        # 生成专业六线谱
        return enhancer.generate_professional_tab_text(enhanced_measures)
    
    # 动态添加方法
    TabGenerator.generate_enhanced_tab_text = generate_enhanced_tab_text
    
    return True


# 示例用法
if __name__ == "__main__":
    # 创建增强生成器
    generator = EnhancedTabGenerator()
    
    # 生成示例六线谱
    sample_tab = generator._generate_empty_tab_template()
    print(sample_tab)