"""
吉他谱AI - Streamlit Web界面
功能：上传视频文件，自动生成吉他谱和演奏说明
"""

import streamlit as st
import os
import sys
import tempfile
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 强制将项目根目录插入到sys.path最前面
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio_processing import process_video_to_audio_with_beats
from app.pitch_detection import predict_pitch
from app.chord_analysis import analyze_chords_simple
from app.tab_generator import generate_tab_simple
from app.explanation_generator import generate_explanation_simple
from app.enhanced_tab_generator import EnhancedTabGenerator
from config import WEB_CONFIG, AUDIO_CONFIG, PITCH_CONFIG


# 页面配置
st.set_page_config(
    page_title="吉他谱AI - 智能谱子生成器",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None


def validate_uploaded_file(uploaded_file):
    """验证上传的文件"""
    if uploaded_file is None:
        return False, "请选择要上传的文件"
    
    # 检查文件类型
    allowed_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wav', '.mp3', '.flac']
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension not in allowed_extensions:
        return False, f"不支持的文件格式: {file_extension}。支持的格式: {', '.join(allowed_extensions)}"
    
    # 检查文件大小
    max_size = WEB_CONFIG['upload_max_size']
    if uploaded_file.size > max_size:
        return False, f"文件大小超过限制: {uploaded_file.size / (1024*1024):.1f}MB > {max_size / (1024*1024):.1f}MB"
    
    return True, "文件验证通过"


def process_video_file(uploaded_file):
    """处理视频文件"""
    with st.spinner("正在处理视频文件..."):
        try:
            # 导入必要的库
            import numpy as np
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # 1. 音频处理
            st.info("步骤 1/5: 提取音频并检测节拍...")
            audio, sr, tempo, beats = process_video_to_audio_with_beats(
                video_path=tmp_file_path,
                target_sr=AUDIO_CONFIG['sample_rate']
            )
            
            # 2. 音高检测
            st.info("步骤 2/5: 进行音高检测...")
            time, frequency, confidence = predict_pitch(
                audio=audio,
                sr=sr,
                model_capacity=PITCH_CONFIG['model_capacity'],
                min_confidence=PITCH_CONFIG['confidence_threshold']
            )
            
            # 3. 和弦分析
            st.info("步骤 3/5: 分析和弦进行...")
            chord_segments = analyze_chords_simple(
                time=time,
                frequency=frequency,
                confidence=confidence,
                time_window=1.0,
                min_confidence=0.5
            )
            
            # 4. 生成六线谱
            st.info("步骤 4/5: 生成六线谱...")
            # 生成基础六线谱
            tab_text = generate_tab_simple(
                time=time,
                frequency=frequency,
                confidence=confidence,
                output_path=None,  # 不保存文件，只返回文本
                tuning="standard",
                tempo=tempo
            )
            
            # 生成增强版六线谱
            st.info("正在生成专业格式六线谱...")
            try:
                enhanced_generator = EnhancedTabGenerator(tuning="standard")
                enhanced_tab_text = enhanced_generator._generate_empty_tab_template()
                
                # 如果有音符数据，尝试生成真实内容
                if len(frequency) > 0:
                    # 这里可以进一步处理音符数据
                    enhanced_tab_text += "\n\n" + "="*60 + "\n"
                    enhanced_tab_text += "基于您的音频生成的六线谱内容:\n"
                    enhanced_tab_text += "="*60 + "\n\n"
                    enhanced_tab_text += tab_text
                    
            except Exception as e:
                enhanced_tab_text = tab_text  # 如果增强版失败，使用基础版
            
            # 5. 生成演奏说明
            st.info("步骤 5/5: 生成演奏说明...")
            try:
                explanation = generate_explanation_simple(
                    chord_segments=chord_segments,
                    tempo=tempo,
                    time_signature="4/4",
                    output_path=None  # 不保存文件，只返回文本
                )
            except Exception as e:
                explanation = f"演奏说明生成失败: {str(e)}\n\n如需AI演奏说明，请设置OpenAI API密钥。"
                st.warning("演奏说明生成失败，但其他功能正常工作")
            
            # 清理临时文件
            os.unlink(tmp_file_path)
            
            # 整理结果 - 确保所有numpy数组都被安全处理
            results = {
                'audio': np.array(audio) if audio is not None else np.array([]),
                'sample_rate': int(sr) if sr is not None else 22050,
                'tempo': float(tempo) if tempo is not None else 120.0,
                'beats': np.array(beats) if beats is not None else np.array([]),
                'time': np.array(time) if time is not None else np.array([]),
                'frequency': np.array(frequency) if frequency is not None else np.array([]),
                'confidence': np.array(confidence) if confidence is not None else np.array([]),
                'chord_segments': chord_segments if chord_segments is not None else [],
                'tab_text': str(tab_text) if tab_text is not None else "",
                'enhanced_tab_text': str(enhanced_tab_text) if 'enhanced_tab_text' in locals() else str(tab_text),
                'explanation': str(explanation) if explanation is not None else "",
                'duration': float(len(audio) / sr) if audio is not None and sr > 0 else 0.0,
                'filename': str(uploaded_file.name)
            }
            
            st.success("处理完成！")
            return results
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"处理过程中发生错误: {error_msg}")
            
            # 显示详细错误信息（调试用）
            with st.expander("详细错误信息"):
                st.code(error_msg)
                import traceback
                st.code(traceback.format_exc())
            
            # 清理临时文件
            if 'tmp_file_path' in locals():
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            return None


def display_audio_info(results):
    """显示音频信息"""
    st.subheader("📊 音频信息")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("时长", f"{results['duration']:.1f}秒")
    
    with col2:
        st.metric("采样率", f"{results['sample_rate']}Hz")
    
    with col3:
        st.metric("速度", f"{results['tempo']:.1f} BPM")
    
    with col4:
        valid_frames = np.sum(results['confidence'] > 0.5)
        total_frames = len(results['confidence'])
        detection_rate = (valid_frames / total_frames) * 100
        st.metric("检测率", f"{detection_rate:.1f}%")


def plot_pitch_contour(results):
    """绘制音高轮廓图"""
    st.subheader("🎵 音高轮廓")
    
    # 过滤有效数据
    valid_mask = results['confidence'] > 0.5
    valid_time = results['time'][valid_mask]
    valid_freq = results['frequency'][valid_mask]
    valid_conf = results['confidence'][valid_mask]
    
    if len(valid_time) == 0:
        st.warning("没有检测到有效的音高数据")
        return
    
    # 创建音高轮廓图
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=valid_time,
        y=valid_freq,
        mode='lines+markers',
        name='音高',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4, color=valid_conf, colorscale='Viridis', showscale=True)
    ))
    
    # 添加节拍线
    if len(results['beats']) > 0:
        for beat in results['beats']:
            fig.add_vline(x=beat, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title="音高随时间变化",
        xaxis_title="时间 (秒)",
        yaxis_title="频率 (Hz)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_chord_progression(results):
    """绘制和弦进行图"""
    st.subheader("🎼 和弦进行")
    
    if not results['chord_segments']:
        st.warning("没有检测到和弦")
        return
    
    # 准备和弦数据
    chord_data = []
    for segment in results['chord_segments']:
        chord_data.append({
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'chord': segment['chord'],
            'duration': segment['duration'],
            'confidence': segment['confidence']
        })
    
    df = pd.DataFrame(chord_data)
    
    # 创建和弦进行图
    fig = go.Figure()
    
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['duration']],
            y=[row['chord']],
            orientation='h',
            name=row['chord'],
            text=f"{row['chord']}<br>{row['duration']:.1f}s",
            textposition='auto',
            hovertemplate=f"和弦: {row['chord']}<br>时长: {row['duration']:.1f}秒<br>置信度: {row['confidence']:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="和弦进行时间线",
        xaxis_title="时长 (秒)",
        yaxis_title="和弦",
        height=300,
        showlegend=False,
        barmode='stack'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示和弦统计
    chord_counts = df['chord'].value_counts()
    st.write("和弦出现频率:")
    st.bar_chart(chord_counts)


def display_tab_text(results):
    """显示六线谱文本"""
    st.subheader("🎸 专业六线谱")
    
    # 选择显示格式
    tab_format = st.radio(
        "选择六线谱格式:",
        ["专业格式", "标准格式"],
        horizontal=True
    )
    
    # 根据选择显示不同格式
    if tab_format == "专业格式":
        tab_content = results.get('enhanced_tab_text', results['tab_text'])
        st.markdown("**专业级六线谱 - 包含和弦标记、小节线和清晰格式**")
    else:
        tab_content = results['tab_text']
        st.markdown("**标准ASCII六线谱**")
    
    # 创建可滚动的文本框
    tab_container = st.container()
    with tab_container:
        st.text_area(
            "生成的六线谱",
            value=tab_content,
            height=500,
            disabled=True
        )
    
    # 下载按钮
    col1, col2 = st.columns(2)
    
    with col1:
        # 下载TXT文件
        st.download_button(
            label="📥 下载六线谱 (TXT)",
            data=results['tab_text'],
            file_name=f"guitar_tab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # 下载按钮（占位符，实际PDF生成需要额外库）
        st.info("PDF下载功能需要安装reportlab库")


def display_explanation(results):
    """显示演奏说明"""
    st.subheader("📝 演奏说明")
    
    # 显示markdown格式的说明
    st.markdown(results['explanation'])
    
    # 下载按钮
    st.download_button(
        label="📥 下载演奏说明 (MD)",
        data=results['explanation'],
        file_name=f"playing_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )


def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()
    
    # 页面标题
    st.markdown('<h1 class="main-header">🎸 吉他谱AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">智能视频转吉他谱生成器</p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 模型设置
        st.subheader("模型参数")
        model_capacity = st.selectbox(
            "CREPE模型精度",
            ["tiny", "small", "medium", "large", "full"],
            index=2,
            help="精度越高，处理越慢"
        )
        
        confidence_threshold = st.slider(
            "置信度阈值",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="过滤低置信度的检测结果"
        )
        
        # 和弦分析设置
        st.subheader("和弦分析")
        time_window = st.slider(
            "时间窗口 (秒)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="和弦分析的时间窗口大小"
        )
        
        # 六线谱设置
        st.subheader("六线谱设置")
        tuning = st.selectbox(
            "调弦方式",
            ["standard", "drop_d", "open_g", "open_d", "dadgad"],
            index=0,
            help="选择吉他调弦方式"
        )
        
        # 处理按钮
        st.markdown("---")
        process_button = st.button("🚀 开始处理", type="primary", use_container_width=True)
    
    # 主界面
    if not st.session_state.processing_complete:
        # 文件上传区域
        st.header("📁 文件上传")
        
        uploaded_file = st.file_uploader(
            "选择视频或音频文件",
            type=['mp4', 'mov', 'avi', 'mkv', 'wav', 'mp3', 'flac'],
            help="支持视频和音频文件，最大100MB"
        )
        
        if uploaded_file:
            # 验证文件
            is_valid, message = validate_uploaded_file(uploaded_file)
            
            if is_valid:
                st.success(message)
                
                # 显示文件信息
                file_info = st.container()
                with file_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("文件名", uploaded_file.name)
                    with col2:
                        st.metric("文件大小", f"{uploaded_file.size / (1024*1024):.1f}MB")
                    with col3:
                        st.metric("文件类型", Path(uploaded_file.name).suffix.upper())
                
                # 处理按钮
                if process_button:
                    # 更新配置
                    PITCH_CONFIG['model_capacity'] = model_capacity
                    PITCH_CONFIG['confidence_threshold'] = confidence_threshold
                    
                    # 处理文件
                    results = process_video_file(uploaded_file)
                    
                    if results:
                        st.session_state.results = results
                        st.session_state.processing_complete = True
                        st.session_state.uploaded_file = uploaded_file
                        st.rerun()
            else:
                st.error(message)
    
    else:
        # 显示处理结果
        results = st.session_state.results
        
        # 成功消息
        st.markdown('<div class="success-box">✅ 文件处理完成！以下是分析结果：</div>', unsafe_allow_html=True)
        
        # 创建标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 概览", "🎵 音高分析", "🎼 和弦分析", "🎸 六线谱", "📝 演奏说明"])
        
        with tab1:
            # 概览页面
            st.header("📊 分析概览")
            
            # 音频信息
            display_audio_info(results)
            
            # 统计信息
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 检测统计")
                valid_frames = np.sum(results['confidence'] > 0.5)
                total_frames = len(results['confidence'])
                
                st.metric("总帧数", total_frames)
                st.metric("有效帧数", valid_frames)
                st.metric("检测率", f"{(valid_frames/total_frames)*100:.1f}%")
                st.metric("检测到的和弦数", len(results['chord_segments']))
            
            with col2:
                st.subheader("🎯 质量评估")
                valid_conf = results['confidence'][results['confidence'] > 0]
                valid_freq = results['frequency'][results['frequency'] > 0]
                
                if len(valid_conf) > 0:
                    avg_confidence = np.mean(valid_conf)
                else:
                    avg_confidence = 0.0
                    
                if len(valid_freq) > 0:
                    freq_range = np.max(results['frequency']) - np.min(valid_freq)
                else:
                    freq_range = 0.0
                
                st.metric("平均置信度", f"{avg_confidence:.3f}")
                st.metric("频率范围", f"{freq_range:.1f}Hz")
                
                if len(valid_freq) > 0:
                    st.metric("最高频率", f"{np.max(results['frequency']):.1f}Hz")
                    st.metric("最低频率", f"{np.min(valid_freq):.1f}Hz")
                else:
                    st.metric("最高频率", "无数据")
                    st.metric("最低频率", "无数据")
        
        with tab2:
            # 音高分析页面
            st.header("🎵 音高分析")
            plot_pitch_contour(results)
        
        with tab3:
            # 和弦分析页面
            st.header("🎼 和弦分析")
            plot_chord_progression(results)
        
        with tab4:
            # 六线谱页面
            st.header("🎸 六线谱")
            display_tab_text(results)
        
        with tab5:
            # 演奏说明页面
            st.header("📝 演奏说明")
            display_explanation(results)
        
        # 重新处理按钮
        st.markdown("---")
        if st.button("🔄 重新处理", use_container_width=True):
            st.session_state.processing_complete = False
            st.session_state.results = {}
            st.session_state.uploaded_file = None
            st.rerun()


if __name__ == "__main__":
    main() 