"""
Streamlit dashboard for real-time deepfake voice detection.
Modern UI with improved visualizations and user experience.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from datetime import datetime, timedelta
import librosa
import soundfile as sf

# Configure Streamlit
st.set_page_config(
    page_title="🎙️ Deepfake Voice Interceptor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Real-Time Deepfake Voice Detection & Audio Authentication System"
    }
)

# Modern Custom CSS - Enhanced UI
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #2c3e50;
        --success: #27ae60;
        --warning: #f39c12;
        --danger: #e74c3c;
        --info: #3498db;
    }
    
    /* Header styling */
    .header-main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-main h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    
    .header-main p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
        margin-right: 10px;
    }
    
    .status-online {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-offline {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk levels */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #c92a2a;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
        color: #2c3e50;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f59f00;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2f9e44;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 15px 25px;
        font-weight: 600;
        border-radius: 10px 10px 0 0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 2px dashed #667eea;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    </style>
""", unsafe_allow_html=True)

# API Base URL
API_URL = "http://localhost:8000/api/v1"


def get_api_status():
    """Check API health."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_and_analyze(audio_file):
    """Upload and analyze audio file."""
    try:
        files = {"file": audio_file}
        response = requests.post(
            f"{API_URL}/analyze/audio",
            files=files,
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def load_demo_audio():
    """Load demo audio files if available."""
    demo_dir = "demo_samples"
    if os.path.exists(demo_dir):
        return [f for f in os.listdir(demo_dir) if f.endswith(('.wav', '.mp3', '.m4a'))]
    return []


def display_deepfake_result(result):
    """Display deepfake detection results."""
    col1, col2, col3 = st.columns(3)
    
    deepfake_prob = result.get("deepfake_probability", 0)
    
    with col1:
        st.metric(
            "Deepfake Probability",
            f"{deepfake_prob*100:.1f}%",
            delta=None
        )
    
    with col2:
        prediction = result.get("deepfake_prediction", "Unknown")
        st.metric("Prediction", prediction)
    
    with col3:
        model_loaded = "✓ Loaded" if result.get("deepfake_model_loaded") else "✗ Not Loaded"
        st.metric("Model Status", model_loaded)
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=deepfake_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Deepfake Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#90EE90"},
                {'range': [33, 66], 'color': "#FFD700"},
                {'range': [66, 100], 'color': "#FF6B6B"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)


def display_transcript_result(result):
    """Display ASR transcript."""
    transcript = result.get("transcript", "")
    if transcript:
        st.text_area(
            "Live Transcript",
            value=transcript,
            height=150,
            disabled=True
        )
    else:
        st.warning("No transcript available")


def display_scam_result(result):
    """Display scam analysis results."""
    col1, col2 = st.columns(2)
    
    scam_prob = result.get("scam_probability", 0)
    
    with col1:
        st.metric(
            "Scam Probability",
            f"{scam_prob*100:.1f}%"
        )
    
    with col2:
        risk_cat = result.get("scam_indicators", [])
        st.metric("Indicators Found", len(risk_cat))
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=scam_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Scam Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 33], 'color': "#90EE90"},
                {'range': [33, 66], 'color': "#FFD700"},
                {'range': [66, 100], 'color': "#FF6B6B"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    # Display indicators
    indicators = result.get("scam_indicators", [])
    if indicators:
        st.subheader("🚨 Detected Suspicious Indicators")
        for indicator in indicators:
            st.write(f"• {indicator}")
    else:
        st.success("✓ No suspicious indicators detected")


def display_risk_result(result):
    """Display overall risk assessment."""
    risk_score = result.get("overall_risk_score", 0)
    risk_category = result.get("risk_category", "UNKNOWN")
    reasons = result.get("risk_reasons", [])
    recommendation = result.get("recommendation", "")
    should_alert = result.get("should_alert", False)
    
    # Color coding
    if risk_category == "HIGH":
        risk_color = "high-risk"
        bg_color = "#ffebee"
    elif risk_category == "MEDIUM":
        risk_color = "medium-risk"
        bg_color = "#fff3e0"
    else:
        risk_color = "low-risk"
        bg_color = "#e8f5e9"
    
    # Risk Score Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Risk Score", f"{risk_score*100:.1f}%")
    with col2:
        st.markdown(f'<p class="{risk_color}">{risk_category}</p>', unsafe_allow_html=True)
    
    # Alert if HIGH risk
    if should_alert and risk_category == "HIGH":
        st.markdown(f"""
            <div class="alert-box">
            <strong>⚠️ HIGH RISK ALERT</strong>
            <p>Potential voice fraud detected!</p>
            <p>Do NOT share personal or financial information.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Reasons
    if reasons:
        st.subheader("Risk Assessment Reasons")
        for reason in reasons:
            st.write(f"• {reason}")
    
    # Recommendation
    if recommendation:
        st.info(f"📋 **Recommendation:** {recommendation}")


def display_timing_metrics(result):
    """Display processing time metrics."""
    st.subheader("⏱️ Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = {
        "Preprocessing": result.get("preprocessing_time"),
        "Deepfake Detection": result.get("deepfake_time"),
        "Speech Recognition": result.get("asr_time"),
        "Scam Analysis": result.get("scam_time"),
        "Total Time": result.get("total_processing_time")
    }
    
    cols = [col1, col2, col3, col4, col5]
    for (name, value), col in zip(metrics.items(), cols):
        with col:
            if value:
                st.metric(name, f"{value:.2f}s")
            else:
                st.metric(name, "N/A")


def get_analysis_history():
    """Fetch recent analysis history."""
    try:
        response = requests.get(f"{API_URL}/results?limit=10", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


def get_statistics():
    """Fetch system statistics."""
    try:
        response = requests.get(f"{API_URL}/statistics?hours=24", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


# Main UI
st.title("🎙️ Real-Time Deepfake Voice Interceptor & Audio Authenticator")
st.markdown("**Advanced AI-powered voice authentication and fraud detection system**")

# Check API status
if not get_api_status():
    st.error("⚠️ Backend API is not running. Please start the backend with: `uvicorn app.main:app --reload`")
else:
    st.success("✓ Backend API connected")

# Sidebar
with st.sidebar:
    st.header("🔧 Settings")
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Upload Audio", "Demo Audio", "Real-Time Stream"]
    )

# Main content
if analysis_mode == "Upload Audio":
    st.header("📁 Upload Audio File")
    
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"]
    )
    
    if audio_file:
        st.audio(audio_file)
        
        if st.button("🔍 Analyze Audio", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                result = upload_and_analyze(audio_file)
            
            if result and "error" not in result:
                st.success("✓ Analysis Complete!")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Risk Assessment",
                    "Deepfake Detection",
                    "Transcript",
                    "Scam Analysis",
                    "Performance"
                ])
                
                with tab1:
                    display_risk_result(result)
                
                with tab2:
                    display_deepfake_result(result)
                
                with tab3:
                    display_transcript_result(result)
                
                with tab4:
                    display_scam_result(result)
                
                with tab5:
                    display_timing_metrics(result)

elif analysis_mode == "Demo Audio":
    st.header("🎯 Demo Mode")
    st.info(
        "Demo Mode allows you to analyze pre-recorded sample audio files. "
        "This uses the actual analysis pipeline with real model predictions."
    )
    
    demo_files = load_demo_audio()
    
    if demo_files:
        selected_demo = st.selectbox("Select demo audio", demo_files)
        demo_path = os.path.join("demo_samples", selected_demo)
        
        if os.path.exists(demo_path):
            st.audio(demo_path)
            
            if st.button("🔍 Analyze Demo Audio", use_container_width=True):
                try:
                    with st.spinner("Analyzing demo audio..."):
                        with open(demo_path, "rb") as f:
                            demo_file = st.UploadedFile(
                                f.read(),
                                name=selected_demo,
                                type="audio"
                            )
                        result = upload_and_analyze(demo_file)
                    
                    if result and "error" not in result:
                        st.success("✓ Analysis Complete!")
                        
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "Risk Assessment",
                            "Deepfake Detection",
                            "Transcript",
                            "Scam Analysis",
                            "Performance"
                        ])
                        
                        with tab1:
                            display_risk_result(result)
                        with tab2:
                            display_deepfake_result(result)
                        with tab3:
                            display_transcript_result(result)
                        with tab4:
                            display_scam_result(result)
                        with tab5:
                            display_timing_metrics(result)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("No demo audio files found. Place audio files in the `demo_samples/` directory.")

elif analysis_mode == "Real-Time Stream":
    st.header("📡 Real-Time Streaming")
    st.info(
        "This feature allows streaming audio chunks over WebSocket for near-real-time analysis. "
        "Currently, this is a simulated streaming interface."
    )
    st.warning("⚠️ Real-time streaming requires WebSocket support (available in development mode)")

# Analytics Section
st.divider()
st.header("📊 Analytics & History")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Recent Analyses")
    history = get_analysis_history()
    if history:
        df = pd.DataFrame([
            {
                "Time": item.get("timestamp", "N/A"),
                "Risk": item.get("risk_category", "N/A"),
                "Deepfake": f"{item.get('deepfake_probability', 0)*100:.1f}%",
                "Scam": f"{item.get('scam_probability', 0)*100:.1f}%",
                "File": item.get("audio_filename", "N/A")
            }
            for item in history
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No analysis history yet")

with col2:
    st.subheader("System Statistics (24h)")
    stats = get_statistics()
    if stats:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Analyses", stats.get("total_analyses", 0))
        with col_b:
            st.metric("High Risk", stats.get("high_risk_count", 0))
        with col_c:
            st.metric("Avg Processing", f"{stats.get('average_processing_time', 0):.2f}s")
        
        # Risk distribution
        risk_data = {
            "LOW": stats.get("low_risk_count", 0),
            "MEDIUM": stats.get("medium_risk_count", 0),
            "HIGH": stats.get("high_risk_count", 0)
        }
        fig = px.pie(
            values=risk_data.values(),
            names=risk_data.keys(),
            title="Risk Distribution",
            color_discrete_map={
                "LOW": "#388e3c",
                "MEDIUM": "#f57c00",
                "HIGH": "#d32f2f"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Statistics not available")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 30px;'>
<p><strong>Real-Time Deepfake Voice Interceptor & Audio Authenticator</strong></p>
<p>Academic Project - VIT BCSE497J - Review-2 Prototype</p>
<p>⚠️ This is a research prototype. Not intended for production use.</p>
</div>
""", unsafe_allow_html=True)
