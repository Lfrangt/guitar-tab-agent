"""
演奏说明生成模块
功能：使用GPT API生成吉他演奏的中文解释说明
"""

import os
import logging
import json
from typing import List, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import openai
from openai import OpenAI


@dataclass
class PlayingTechnique:
    """演奏技巧数据类"""
    name: str
    time_start: float
    time_end: float
    description: str
    difficulty: str  # 'easy', 'medium', 'hard'


@dataclass
class ChordProgression:
    """和弦进行数据类"""
    chords: List[str]
    time_start: float
    time_end: float
    key: Optional[str] = None
    style: Optional[str] = None


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    chord_progression: ChordProgression
    techniques: List[PlayingTechnique]
    tempo: float
    time_signature: str
    overall_difficulty: str


class ExplanationGenerator:
    """演奏说明生成器"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 1000,
                 temperature: float = 0.7):
        """
        初始化说明生成器
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: 使用的GPT模型
            max_tokens: 最大输出token数
            temperature: 生成温度
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 设置API密钥
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置，请通过参数传入或设置环境变量OPENAI_API_KEY")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key)
        
        # 预设的系统提示词
        self.system_prompt = self._get_system_prompt()
        
        logging.info(f"演奏说明生成器初始化完成: 模型={model}")
    
    def _get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        Returns:
            str: 系统提示词
        """
        return """你是一个专业的吉他演奏分析师和教育者。请根据提供的音乐分析数据，生成详细的中文演奏说明。

你的任务包括：
1. 分析和弦进行，解释音乐理论背景
2. 识别演奏技巧，提供练习建议
3. 评估难度等级，给出学习建议
4. 提供具体的练习方法和注意事项

请使用markdown格式输出，包含以下部分：
- 和弦分析
- 演奏技巧
- 难度评估
- 练习建议
- 注意事项

语言要通俗易懂，适合吉他学习者理解。"""
    
    def detect_playing_techniques(self, 
                                time: List[float],
                                frequency: List[float],
                                confidence: List[float],
                                chord_segments: List[Dict]) -> List[PlayingTechnique]:
        """
        检测演奏技巧
        
        Args:
            time: 时间数组
            frequency: 频率数组
            confidence: 置信度数组
            chord_segments: 和弦片段列表
            
        Returns:
            List[PlayingTechnique]: 检测到的演奏技巧列表
        """
        techniques = []
        
        # 检测滑音 (glissando)
        slide_techniques = self._detect_slides(time, frequency, confidence)
        techniques.extend(slide_techniques)
        
        # 检测和弦转换
        chord_change_techniques = self._detect_chord_changes(chord_segments)
        techniques.extend(chord_change_techniques)
        
        # 检测快速音符
        fast_note_techniques = self._detect_fast_notes(time, frequency, confidence)
        techniques.extend(fast_note_techniques)
        
        # 检测延音
        sustain_techniques = self._detect_sustains(time, frequency, confidence)
        techniques.extend(sustain_techniques)
        
        logging.info(f"检测到 {len(techniques)} 个演奏技巧")
        return techniques
    
    def _detect_slides(self, 
                      time: List[float],
                      frequency: List[float],
                      confidence: List[float],
                      threshold: float = 50.0) -> List[PlayingTechnique]:
        """
        检测滑音技巧
        
        Args:
            time: 时间数组
            frequency: 频率数组
            confidence: 置信度数组
            threshold: 滑音检测阈值（音分）
            
        Returns:
            List[PlayingTechnique]: 滑音技巧列表
        """
        slides = []
        
        for i in range(1, len(time)):
            if (frequency[i] > 0 and frequency[i-1] > 0 and 
                confidence[i] > 0.5 and confidence[i-1] > 0.5):
                
                # 计算音高变化（音分）
                cent_change = 1200 * (frequency[i] / frequency[i-1] - 1)
                
                if abs(cent_change) > threshold:
                    # 检测到滑音
                    slide = PlayingTechnique(
                        name="滑音",
                        time_start=time[i-1],
                        time_end=time[i],
                        description=f"从 {frequency[i-1]:.1f}Hz 滑到 {frequency[i]:.1f}Hz",
                        difficulty="medium" if abs(cent_change) < 200 else "hard"
                    )
                    slides.append(slide)
        
        return slides
    
    def _detect_chord_changes(self, 
                            chord_segments: List[Dict],
                            min_duration: float = 0.5) -> List[PlayingTechnique]:
        """
        检测和弦转换技巧
        
        Args:
            chord_segments: 和弦片段列表
            min_duration: 最小持续时间
            
        Returns:
            List[PlayingTechnique]: 和弦转换技巧列表
        """
        changes = []
        
        for i in range(1, len(chord_segments)):
            prev_chord = chord_segments[i-1]
            curr_chord = chord_segments[i]
            
            if (prev_chord['chord'] != curr_chord['chord'] and 
                curr_chord['duration'] < min_duration):
                
                change = PlayingTechnique(
                    name="快速和弦转换",
                    time_start=curr_chord['start_time'],
                    time_end=curr_chord['end_time'],
                    description=f"从 {prev_chord['chord']} 快速转换到 {curr_chord['chord']}",
                    difficulty="hard"
                )
                changes.append(change)
        
        return changes
    
    def _detect_fast_notes(self, 
                          time: List[float],
                          frequency: List[float],
                          confidence: List[float],
                          threshold: float = 0.1) -> List[PlayingTechnique]:
        """
        检测快速音符
        
        Args:
            time: 时间数组
            frequency: 频率数组
            confidence: 置信度数组
            threshold: 快速音符时间阈值（秒）
            
        Returns:
            List[PlayingTechnique]: 快速音符技巧列表
        """
        fast_notes = []
        
        for i in range(1, len(time)):
            time_diff = time[i] - time[i-1]
            
            if (time_diff < threshold and 
                frequency[i] > 0 and frequency[i-1] > 0 and
                confidence[i] > 0.5 and confidence[i-1] > 0.5):
                
                note = PlayingTechnique(
                    name="快速音符",
                    time_start=time[i-1],
                    time_end=time[i],
                    description=f"快速音符序列，间隔 {time_diff:.3f}秒",
                    difficulty="medium" if time_diff > 0.05 else "hard"
                )
                fast_notes.append(note)
        
        return fast_notes
    
    def _detect_sustains(self, 
                        time: List[float],
                        frequency: List[float],
                        confidence: List[float],
                        min_duration: float = 2.0) -> List[PlayingTechnique]:
        """
        检测延音
        
        Args:
            time: 时间数组
            frequency: 频率数组
            confidence: 置信度数组
            min_duration: 最小延音时长（秒）
            
        Returns:
            List[PlayingTechnique]: 延音技巧列表
        """
        sustains = []
        
        current_sustain_start = None
        current_freq = None
        
        for i, (t, freq, conf) in enumerate(zip(time, frequency, confidence)):
            if freq > 0 and conf > 0.5:
                if current_sustain_start is None:
                    current_sustain_start = t
                    current_freq = freq
                elif abs(freq - current_freq) < 10:  # 频率变化小于10Hz
                    continue
                else:
                    # 延音结束
                    if current_sustain_start is not None:
                        duration = t - current_sustain_start
                        if duration >= min_duration:
                            sustain = PlayingTechnique(
                                name="延音",
                                time_start=current_sustain_start,
                                time_end=t,
                                description=f"持续 {duration:.1f}秒 的延音",
                                difficulty="easy"
                            )
                            sustains.append(sustain)
                        current_sustain_start = None
        
        return sustains
    
    def analyze_music(self, 
                     chord_segments: List[Dict],
                     tempo: float,
                     time_signature: str,
                     techniques: List[PlayingTechnique]) -> AnalysisResult:
        """
        分析音乐特征
        
        Args:
            chord_segments: 和弦片段列表
            tempo: 速度
            time_signature: 拍号
            techniques: 演奏技巧列表
            
        Returns:
            AnalysisResult: 分析结果
        """
        # 提取和弦进行
        chords = [segment['chord'] for segment in chord_segments]
        
        # 确定调性（简化版）
        key = self._determine_key(chords)
        
        # 确定风格
        style = self._determine_style(chords, tempo, techniques)
        
        # 评估整体难度
        overall_difficulty = self._assess_difficulty(techniques, tempo, chords)
        
        chord_progression = ChordProgression(
            chords=chords,
            time_start=chord_segments[0]['start_time'] if chord_segments else 0,
            time_end=chord_segments[-1]['end_time'] if chord_segments else 0,
            key=key,
            style=style
        )
        
        return AnalysisResult(
            chord_progression=chord_progression,
            techniques=techniques,
            tempo=tempo,
            time_signature=time_signature,
            overall_difficulty=overall_difficulty
        )
    
    def _determine_key(self, chords: List[str]) -> str:
        """
        确定调性（简化版）
        
        Args:
            chords: 和弦列表
            
        Returns:
            str: 调性
        """
        # 统计和弦出现频率
        chord_counts = {}
        for chord in chords:
            if chord != "Unknown":
                chord_counts[chord] = chord_counts.get(chord, 0) + 1
        
        if not chord_counts:
            return "未知"
        
        # 找到最常见的和弦作为主和弦
        most_common = max(chord_counts.items(), key=lambda x: x[1])[0]
        
        # 简单的调性判断
        if most_common.endswith('m'):
            return most_common[:-1] + "小调"
        else:
            return most_common + "大调"
    
    def _determine_style(self, 
                        chords: List[str],
                        tempo: float,
                        techniques: List[PlayingTechnique]) -> str:
        """
        确定音乐风格
        
        Args:
            chords: 和弦列表
            tempo: 速度
            techniques: 演奏技巧列表
            
        Returns:
            str: 音乐风格
        """
        # 基于和弦和速度判断风格
        if tempo > 140:
            return "快节奏摇滚"
        elif tempo > 100:
            return "流行摇滚"
        elif tempo > 80:
            return "民谣"
        else:
            return "慢歌"
    
    def _assess_difficulty(self, 
                          techniques: List[PlayingTechnique],
                          tempo: float,
                          chords: List[str]) -> str:
        """
        评估整体难度
        
        Args:
            techniques: 演奏技巧列表
            tempo: 速度
            chords: 和弦列表
            
        Returns:
            str: 难度等级
        """
        difficulty_score = 0
        
        # 基于速度评分
        if tempo > 140:
            difficulty_score += 3
        elif tempo > 100:
            difficulty_score += 2
        elif tempo > 80:
            difficulty_score += 1
        
        # 基于技巧评分
        for technique in techniques:
            if technique.difficulty == "hard":
                difficulty_score += 3
            elif technique.difficulty == "medium":
                difficulty_score += 2
            else:
                difficulty_score += 1
        
        # 基于和弦复杂度评分
        complex_chords = [c for c in chords if '7' in c or 'maj7' in c or 'sus' in c]
        difficulty_score += len(complex_chords)
        
        if difficulty_score >= 8:
            return "困难"
        elif difficulty_score >= 5:
            return "中等"
        else:
            return "简单"
    
    def _generate_explanation_from_analysis(self, 
                                           analysis_result: AnalysisResult,
                                           tab_data: Optional[Dict] = None) -> str:
        """
        生成演奏说明
        
        Args:
            analysis_result: 分析结果
            tab_data: 六线谱数据（可选）
            
        Returns:
            str: 演奏说明（markdown格式）
        """
        try:
            # 构建用户提示词
            user_prompt = self._build_user_prompt(analysis_result, tab_data)
            
            # 调用GPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            explanation = response.choices[0].message.content
            
            logging.info("演奏说明生成成功")
            return explanation
            
        except Exception as e:
            logging.error(f"生成演奏说明失败: {str(e)}")
            return self._generate_fallback_explanation(analysis_result)
    
    def generate_explanation(self, pitch_data: dict, chord_data: dict, tab_data: dict) -> str:
        """
        生成演奏说明（为main.py提供的接口）
        
        Args:
            pitch_data: 音高数据字典
            chord_data: 和弦数据字典
            tab_data: 六线谱数据字典
            
        Returns:
            str: 演奏说明
        """
        try:
            # 构建分析结果
            from app.chord_analysis import ChordSegment
            
            # 提取基本信息
            tempo = pitch_data.get("tempo", 120.0)
            time_signature = tab_data.get("time_signature", (4, 4))
            
            # 构建和弦进行
            chord_progression = ChordProgression(
                segments=chord_data.get("segments", []),
                key="C",  # 默认调性，可以后续改进
                style="流行",  # 默认风格
                complexity=len(chord_data.get("progression", []))
            )
            
            # 构建技巧数据
            techniques = self.detect_techniques(pitch_data, chord_data, tab_data)
            
            # 构建分析结果
            analysis_result = AnalysisResult(
                tempo=tempo,
                time_signature=f"{time_signature[0]}/{time_signature[1]}",
                chord_progression=chord_progression,
                techniques=techniques,
                overall_difficulty=self._calculate_overall_difficulty(techniques, chord_data)
            )
            
            # 生成说明
            return self._generate_explanation_from_analysis(analysis_result, tab_data)
            
        except Exception as e:
            logging.error(f"生成演奏说明失败: {e}")
            # 生成简化的说明
            return self._generate_simple_explanation(pitch_data, chord_data, tab_data)
    
    def _generate_simple_explanation(self, pitch_data: dict, chord_data: dict, tab_data: dict) -> str:
        """生成简化的演奏说明"""
        explanation_parts = []
        
        explanation_parts.append("# 演奏说明\n")
        
        # 基本信息
        explanation_parts.append("## 基本信息")
        explanation_parts.append(f"- 速度: {pitch_data.get('tempo', 120)} BPM")
        explanation_parts.append(f"- 总时长: {len(pitch_data.get('times', []))} 个采样点")
        explanation_parts.append(f"- 检测到的和弦数量: {len(chord_data.get('progression', []))}")
        explanation_parts.append(f"- 六线谱小节数: {tab_data.get('num_measures', 0)}")
        
        # 和弦进行
        if chord_data.get("progression"):
            explanation_parts.append("\n## 和弦进行")
            for i, chord in enumerate(chord_data["progression"][:5]):  # 只显示前5个
                explanation_parts.append(f"- {chord.get('time', 0):.1f}s: {chord.get('chord', '未知')}")
        
        # 演奏建议
        explanation_parts.append("\n## 演奏建议")
        explanation_parts.append("- 注意和弦转换的时机")
        explanation_parts.append("- 保持稳定的节拍")
        explanation_parts.append("- 根据六线谱上的指位安排手指")
        
        return "\n".join(explanation_parts)
    
    def _build_user_prompt(self, 
                          analysis_result: AnalysisResult,
                          tab_data: Optional[Dict]) -> str:
        """
        构建用户提示词
        
        Args:
            analysis_result: 分析结果
            tab_data: 六线谱数据
            
        Returns:
            str: 用户提示词
        """
        prompt_parts = []
        
        # 基本信息
        prompt_parts.append(f"## 基本信息")
        prompt_parts.append(f"- 速度: {analysis_result.tempo} BPM")
        prompt_parts.append(f"- 拍号: {analysis_result.time_signature}")
        prompt_parts.append(f"- 调性: {analysis_result.chord_progression.key}")
        prompt_parts.append(f"- 风格: {analysis_result.chord_progression.style}")
        prompt_parts.append(f"- 整体难度: {analysis_result.overall_difficulty}")
        
        # 和弦进行
        prompt_parts.append(f"\n## 和弦进行")
        chords_str = " → ".join(analysis_result.chord_progression.chords)
        prompt_parts.append(f"和弦序列: {chords_str}")
        
        # 演奏技巧
        if analysis_result.techniques:
            prompt_parts.append(f"\n## 演奏技巧")
            for technique in analysis_result.techniques:
                prompt_parts.append(f"- {technique.name}: {technique.description} (难度: {technique.difficulty})")
        
        # 六线谱信息（如果有）
        if tab_data:
            prompt_parts.append(f"\n## 六线谱信息")
            prompt_parts.append(f"- 调弦: {tab_data.get('tuning', '标准调弦')}")
            prompt_parts.append(f"- 最大品位: {tab_data.get('max_fret', '未知')}")
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_explanation(self, analysis_result: AnalysisResult) -> str:
        """
        生成备用说明（当API调用失败时）
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            str: 备用说明
        """
        return f"""# 吉他演奏说明

## 基本信息
- **速度**: {analysis_result.tempo} BPM
- **拍号**: {analysis_result.time_signature}
- **调性**: {analysis_result.chord_progression.key}
- **风格**: {analysis_result.chord_progression.style}
- **整体难度**: {analysis_result.overall_difficulty}

## 和弦进行
和弦序列: {' → '.join(analysis_result.chord_progression.chords)}

## 演奏技巧
{chr(10).join([f"- **{t.name}**: {t.description} (难度: {t.difficulty})" for t in analysis_result.techniques])}

## 练习建议
1. 先慢速练习和弦转换
2. 注意节奏的准确性
3. 逐步提高演奏速度
4. 注意音色的控制

## 注意事项
- 保持正确的指法
- 注意手指的放松
- 练习时要有耐心
- 多听原曲找感觉"""
    
    def save_explanation(self, 
                        explanation: str,
                        output_path: Union[str, Path]) -> None:
        """
        保存演奏说明到文件
        
        Args:
            explanation: 演奏说明
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(explanation)
        
        logging.info(f"演奏说明已保存到: {output_path}")


def generate_explanation_simple(chord_segments: List[Dict],
                              tempo: float,
                              time_signature: str = "4/4",
                              api_key: Optional[str] = None,
                              output_path: Optional[Union[str, Path]] = None) -> str:
    """
    简化的演奏说明生成函数
    
    Args:
        chord_segments: 和弦片段列表
        tempo: 速度
        time_signature: 拍号
        api_key: OpenAI API密钥
        output_path: 输出文件路径
        
    Returns:
        str: 演奏说明
    """
    generator = ExplanationGenerator(api_key=api_key)
    
    # 检测演奏技巧（简化版）
    techniques = []
    
    # 分析音乐特征
    analysis_result = generator.analyze_music(
        chord_segments=chord_segments,
        tempo=tempo,
        time_signature=time_signature,
        techniques=techniques
    )
    
    # 生成说明
    explanation = generator.generate_explanation(analysis_result)
    
    # 保存文件（如果指定了路径）
    if output_path:
        generator.save_explanation(explanation, output_path)
    
    return explanation


# 示例用法
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例数据
    chord_segments = [
        {"chord": "C", "start_time": 0.0, "end_time": 2.0, "duration": 2.0},
        {"chord": "Am", "start_time": 2.0, "end_time": 4.0, "duration": 2.0},
        {"chord": "F", "start_time": 4.0, "end_time": 6.0, "duration": 2.0},
        {"chord": "G", "start_time": 6.0, "end_time": 8.0, "duration": 2.0},
    ]
    
    try:
        # 生成演奏说明
        explanation = generate_explanation_simple(
            chord_segments=chord_segments,
            tempo=120.0,
            output_path="playing_guide.md"
        )
        
        print("演奏说明生成成功!")
        print("\n生成的说明:")
        print(explanation)
        
    except Exception as e:
        print(f"演奏说明生成失败: {e}") 