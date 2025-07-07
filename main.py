#!/usr/bin/env python3
"""
吉他谱AI - 主入口脚本
从音频/视频文件生成吉他六线谱
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from config import (
    LOGGING_CONFIG, 
    OUTPUT_CONFIG, 
    TABS_DIR, 
    LOGS_DIR
)
from app.audio_processing import AudioProcessor
from app.pitch_detection import PitchDetector
from app.chord_analysis import ChordAnalyzer
from app.tab_generator import TabGenerator
from app.explanation_generator import ExplanationGenerator
from app.utils import setup_logging, validate_input_file, save_results


def setup_argument_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="吉他谱AI - 从音频/视频生成六线谱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --input song.mp3
  python main.py --input video.mp4 --output ./my_tabs/
  python main.py --input audio.wav --format pdf --no-explanation
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入音频或视频文件路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(TABS_DIR),
        help=f"输出目录路径 (默认: {TABS_DIR})"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "pdf", "png", "all"],
        default="all",
        help="输出格式 (默认: all)"
    )
    
    parser.add_argument(
        "--no-explanation",
        action="store_true",
        help="不生成演奏说明"
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="不生成可视化图表"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出模式"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式"
    )
    
    return parser


class GuitarTabAI:
    """吉他谱AI主类"""
    
    def __init__(self, verbose: bool = False, debug: bool = False):
        """初始化吉他谱AI系统"""
        self.verbose = verbose
        self.debug = debug
        
        # 设置日志
        log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
        setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个模块
        self.logger.info("初始化吉他谱AI系统...")
        self.audio_processor = AudioProcessor()
        self.pitch_detector = PitchDetector()
        self.chord_analyzer = ChordAnalyzer()
        self.tab_generator = TabGenerator()
        
        # 只在需要时初始化说明生成器
        self.explanation_generator = None
        
        self.logger.info("系统初始化完成")
    
    def process_file(
        self, 
        input_path: str, 
        output_dir: str,
        output_format: str = "all",
        generate_explanation: bool = True,
        generate_visualization: bool = True
    ) -> dict:
        """
        处理音频/视频文件并生成吉他谱
        
        Args:
            input_path: 输入文件路径
            output_dir: 输出目录
            output_format: 输出格式
            generate_explanation: 是否生成说明
            generate_visualization: 是否生成可视化
            
        Returns:
            包含处理结果的字典
        """
        try:
            # 验证输入文件
            input_path = Path(input_path)
            if not validate_input_file(input_path):
                raise ValueError(f"无效的输入文件: {input_path}")
            
            # 创建输出目录
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"开始处理文件: {input_path}")
            
            # 1. 音频处理
            self.logger.info("步骤 1/5: 音频处理...")
            audio_data = self.audio_processor.process_file(input_path)
            
            # 2. 音高检测
            self.logger.info("步骤 2/5: 音高检测...")
            pitch_data = self.pitch_detector.detect_pitch(audio_data)
            
            # 3. 和弦分析
            self.logger.info("步骤 3/5: 和弦分析...")
            chord_data = self.chord_analyzer.analyze_chords(audio_data, pitch_data)
            
            # 4. 六线谱生成
            self.logger.info("步骤 4/5: 六线谱生成...")
            tab_data = self.tab_generator.generate_tab(pitch_data, chord_data)
            
            # 5. 生成说明（可选）
            explanation = None
            if generate_explanation:
                self.logger.info("步骤 5/5: 生成演奏说明...")
                try:
                    # 初始化说明生成器（如果还没有初始化）
                    if self.explanation_generator is None:
                        self.explanation_generator = ExplanationGenerator()
                    
                    explanation = self.explanation_generator.generate_explanation(
                        pitch_data, chord_data, tab_data
                    )
                except Exception as e:
                    self.logger.warning(f"说明生成失败，跳过此步骤: {e}")
                    explanation = "说明生成功能需要OpenAI API密钥"
            
            # 整理结果
            results = {
                "input_file": str(input_path),
                "output_dir": str(output_dir),
                "audio_data": audio_data,
                "pitch_data": pitch_data,
                "chord_data": chord_data,
                "tab_data": tab_data,
                "explanation": explanation,
                "timestamp": audio_data.get("timestamp"),
                "duration": audio_data.get("duration")
            }
            
            # 保存结果
            self.logger.info("保存结果...")
            output_files = save_results(
                results, 
                output_dir, 
                output_format,
                generate_visualization
            )
            
            results["output_files"] = output_files
            
            self.logger.info(f"处理完成! 输出文件: {output_files}")
            return results
            
        except Exception as e:
            self.logger.error(f"处理文件时发生错误: {str(e)}", exc_info=True)
            raise


def main():
    """主函数"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # 初始化系统
        guitar_ai = GuitarTabAI(
            verbose=args.verbose,
            debug=args.debug
        )
        
        # 处理文件
        results = guitar_ai.process_file(
            input_path=args.input,
            output_dir=args.output,
            output_format=args.format,
            generate_explanation=not args.no_explanation,
            generate_visualization=not args.no_visualization
        )
        
        # 输出结果摘要
        print("\n" + "="*50)
        print("🎸 吉他谱AI处理完成!")
        print("="*50)
        print(f"输入文件: {results['input_file']}")
        print(f"输出目录: {results['output_dir']}")
        print(f"处理时长: {results.get('duration', 'N/A')} 秒")
        print(f"输出文件: {len(results.get('output_files', []))} 个")
        
        if results.get('output_files'):
            print("\n生成的文件:")
            for file_path in results['output_files']:
                print(f"  📄 {file_path}")
        
        print("\n✅ 处理成功完成!")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 