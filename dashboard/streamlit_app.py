"""
Enhanced Streamlit dashboard for real-time deepfake voice detection.
Modern UI with improved visualizations, better UX, and comprehensive features.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
from datetime import datetime, timedelta
import librosa
import soundfile as sf
from io import BytesIO
import base64
import json
import asyncio
import time
import websockets

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🎙️ Deepfake Voice Interceptor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Real-Time Deepfake Voice Detection System"}
)

# ============================================================================
# ENHANCED CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    /* Global dark theme */
    .stApp {
        background: linear-gradient(180deg, #07111f 0%, #0d1b2a 100%);
        color: #edf4ff;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="block-container"] {
        background: #07111f;
        color: #edf4ff;
    }
    div, p, li, span, label, h1, h2, h3, h4, h5, h6 {
        color: #edf4ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background: rgba(15, 23, 42, 0.9);
        color: #dfeaff;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px 12px 0 0;
        padding: 0.7rem 1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #2dd4bf 0%, #2563eb 100%);
        color: white;
        font-weight: 700;
    }
    .header-main {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        padding: 40px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.25);
        text-align: center;
    }
    .header-main h1 {
        margin: 0;
        font-size: 3em;
        font-weight: 800;
        letter-spacing: -1px;
        color: white;
    }
    .header-main p {
        margin: 15px 0 0 0;
        font-size: 1.2em;
        opacity: 0.95;
        color: white;
    }
    .status-online {
        display: inline-block;
        background: rgba(34, 197, 94, 0.18);
        color: #bbf7d0;
        padding: 8px 16px;
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 20px;
        font-weight: 600;
        margin: 10px 5px 0 0;
    }
    .status-offline {
        display: inline-block;
        background: rgba(239, 68, 68, 0.16);
        color: #fecaca;
        padding: 8px 16px;
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 20px;
        font-weight: 600;
        margin: 10px 5px 0 0;
    }
    /* Metric cards */
    .metric-container {
        background: rgba(15, 23, 42, 0.92);
        color: #edf4ff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.18);
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin: 10px 0;
    }
    
    /* Risk level boxes */
    .risk-high {
        background: linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.25);
        border-left: 5px solid #fecaca;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #d97706 0%, #92400e 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.25);
        border-left: 5px solid #fde68a;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #15803d 0%, #166534 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(34, 197, 94, 0.22);
        border-left: 5px solid #bbf7d0;
    }
    
    /* Info boxes */
    .info-box, .success-box, .warning-box {
        background: rgba(15, 23, 42, 0.9);
        color: #edf4ff;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #60a5fa;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .success-box {
        border-left-color: #34d399;
    }
    
    .warning-box {
        border-left-color: #f59e0b;
    }

    .call-card {
        background: linear-gradient(135deg, #111827 0%, #16233b 100%);
        border-radius: 16px;
        padding: 28px;
        color: white;
        box-shadow: 0 12px 30px rgba(16, 24, 39, 0.35);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .call-alert {
        background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%);
        color: white;
        border-radius: 14px;
        padding: 20px;
        font-weight: 700;
        box-shadow: 0 14px 30px rgba(180, 35, 24, 0.28);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #22c55e 100%);
        color: white !important;
        border: none;
        padding: 10px 18px;
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
    }
    
    .stButton > button:hover {
        filter: brightness(1.08);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
API_URL = "http://127.0.0.1:8000/api/v1"  # Match the backend startup port

# Session state initialization
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_api_status():
    """Check API health status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_and_analyze(audio_file):
    """Upload and analyze audio file via API."""
    try:
        files = {"file": audio_file}
        with st.spinner("🔄 Analyzing audio... This may take 4-7 seconds..."):
            response = requests.post(
                f"{API_URL}/analyze/audio",
                files=files,
                timeout=120
            )
        
        if response.status_code == 200:
            payload = response.json()
            return normalize_analysis_result(payload)
        else:
            st.error(f"❌ API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. The analysis took too long.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend at {API_URL}")
        st.info("💡 Make sure the backend is running: `python start_backend.py`")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def normalize_analysis_result(result):
    """Convert flat backend payloads into the nested structure expected by the dashboard UI."""
    if not isinstance(result, dict):
        return result

    normalized = dict(result)

    risk_category = result.get("risk_category", "UNKNOWN")
    risk_score = float(result.get("overall_risk_score", 0.0) or 0.0)
    if risk_category == "LOW":
        level = 1
    elif risk_category == "MEDIUM":
        level = 2
    elif risk_category == "HIGH":
        level = 3
    else:
        level = 1

    normalized["risk"] = {
        "risk_category": risk_category,
        "overall_risk_score": risk_score,
        "severity_level": level,
        "recommendation": result.get("recommendation", "No recommendation available"),
        "reasons": result.get("risk_reasons", [])
    }

    normalized["deepfake"] = {
        "deepfake_probability": float(result.get("deepfake_probability", 0.0) or 0.0),
        "prediction_class": result.get("deepfake_prediction", "Unknown"),
        "confidence_scores": [float(result.get("deepfake_probability", 0.0) or 0.0)],
        "model_loaded": bool(result.get("deepfake_model_loaded", False))
    }

    scam_indicators = result.get("scam_indicators", [])
    if isinstance(scam_indicators, str):
        try:
            import json
            scam_indicators = json.loads(scam_indicators)
        except Exception:
            scam_indicators = []

    normalized["scam"] = {
        "scam_probability": float(result.get("scam_probability", 0.0) or 0.0),
        "risk_category": risk_category,
        "detection_method": result.get("scam_detection_method", "unknown"),
        "indicators": {"scam": scam_indicators if isinstance(scam_indicators, list) else []}
    }

    normalized["timing_metrics"] = {
        "preprocessing_time": result.get("preprocessing_time"),
        "deepfake_time": result.get("deepfake_time"),
        "asr_time": result.get("asr_time"),
        "scam_time": result.get("scam_time"),
        "total_processing_time": result.get("total_processing_time")
    }

    return normalized


def load_demo_audio_files():
    """Load available demo audio files."""
    demo_dir = "demo_samples"
    if os.path.exists(demo_dir):
        files = [f for f in os.listdir(demo_dir) if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))]
        return sorted(files)
    return []


def create_risk_gauge(value, title="Risk Score"):
    """Create a beautiful risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        title={"text": title},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "rgba(102, 126, 234, 0.8)"},
            "steps": [
                {"range": [0, 39], "color": "rgba(39, 174, 96, 0.2)"},
                {"range": [39, 69], "color": "rgba(243, 156, 18, 0.2)"},
                {"range": [69, 100], "color": "rgba(231, 76, 60, 0.2)"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        }
    ))
    fig.update_layout(
        height=300,
        font={"size": 14},
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def get_risk_class(risk_category):
    """Get CSS class for risk category."""
    if risk_category == "HIGH":
        return "risk-high"
    elif risk_category == "MEDIUM":
        return "risk-medium"
    else:
        return "risk-low"


def stream_live_audio_to_backend(audio_file):
    """Stream a file in chunks to the backend WebSocket to simulate a live phone call."""
    try:
        file_bytes = audio_file.read() if hasattr(audio_file, 'read') else audio_file
        if not file_bytes:
            st.error("Uploaded file is empty.")
            return None

        audio, sr = librosa.load(BytesIO(file_bytes), sr=16000, mono=True)
        if len(audio) == 0:
            st.error("Could not read audio data from the uploaded file.")
            return None

        chunk_size = 16000
        total_chunks = max(1, len(audio) // chunk_size + (1 if len(audio) % chunk_size else 0))
        responses = []

        async def send_stream():
            uri = "ws://127.0.0.1:8000/api/v1/ws/analyze"
            async with websockets.connect(uri) as websocket:
                for i in range(total_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, len(audio))
                    chunk = audio[start:end]
                    payload = {
                        "type": "audio_chunk",
                        "data": base64.b64encode(np.asarray(chunk, dtype=np.float32).tobytes()).decode("utf-8"),
                        "sample_rate": sr,
                        "is_final": i == total_chunks - 1
                    }
                    await websocket.send(json.dumps(payload))
                    responses.append(json.loads(await websocket.recv()))

                await websocket.send(json.dumps({"type": "process"}))
                responses.append(json.loads(await websocket.recv()))

        asyncio.run(send_stream())

        final_result = None
        for item in reversed(responses):
            if item.get("type") == "result":
                final_result = item.get("data")
                break

        if final_result is not None:
            return normalize_analysis_result(final_result)

        if responses:
            return responses[-1]

        return {"status": "no_response"}
    except Exception as e:
        st.error(f"Live streaming error: {str(e)}")
        return {"error": str(e)}


def get_demo_live_call_file():
    """Return a demo file from demo_samples if available."""
    demo_files = load_demo_audio_files()
    if not demo_files:
        return None
    return os.path.join("demo_samples", demo_files[0])


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown("""
    <div class="header-main">
        <h1>🎙️ Deepfake Voice Interceptor</h1>
        <p>Real-Time Audio Analysis & Deepfake Detection</p>
    </div>
""", unsafe_allow_html=True)

# Status bar
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    api_status = get_api_status()
    if api_status:
        st.markdown('<div class="status-online">✅ Backend Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-offline">❌ Backend Disconnected</div>',
            unsafe_allow_html=True
        )
        st.error("Backend not running. Start it with: `python start_backend.py`")

with col2:
    st.metric("System Status", "Ready" if api_status else "Error")

with col3:
    st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))

st.divider()

# ============================================================================
# MAIN TABS
# ============================================================================

tab_upload, tab_live, tab_demo, tab_analytics, tab_api, tab_about = st.tabs([
    "📁 Upload Audio",
    "📡 Live Call Monitor",
    "🎵 Demo Samples",
    "📊 Analytics",
    "🔌 API Docs",
    "ℹ️ About"
])

# ============================================================================
# TAB 1: UPLOAD AUDIO
# ============================================================================
with tab_upload:
    st.header("Upload & Analyze Audio")
    
    st.markdown("""
    <div class="info-box">
    📢 <strong>How to use:</strong> Upload an audio file (WAV, MP3, M4A, FLAC) to analyze it for deepfake detection, transcription, and scam intent detection.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Supported formats: WAV, MP3, M4A, FLAC, OGG"
    )
    
    if uploaded_file is not None:
        # Display audio info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name[:30])
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("Duration", "Calculating...")
        
        # Audio player
        st.audio(uploaded_file)
        
        # Analyze button
        if st.button("🔍 Analyze Audio", use_container_width=True):
            if api_status:
                result = upload_and_analyze(uploaded_file)
                if result:
                    st.session_state.analysis_result = result
                    st.success("✅ Analysis complete!")
            else:
                st.error("❌ Backend is not running!")
    
    # Display results if available
    if st.session_state.analysis_result:
        st.divider()
        st.subheader("📊 Analysis Results")
        
        result = st.session_state.analysis_result
        risk_info = result.get("risk", {})
        risk_category = risk_info.get("risk_category", "UNKNOWN")
        risk_score = float(risk_info.get("overall_risk_score", 0.0) or 0.0)
        deepfake_prob = float(result.get("deepfake", {}).get("deepfake_probability", 0.0) or 0.0)
        scam_prob = float(result.get("scam", {}).get("scam_probability", 0.0) or 0.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-container'><div style='font-size:12px;letter-spacing:1px;color:#93c5fd;text-transform:uppercase;'>Deepfake Risk</div><div style='font-size:2.2rem;font-weight:800;margin-top:8px;'>{deepfake_prob * 100:.1f}%</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-container'><div style='font-size:12px;letter-spacing:1px;color:#93c5fd;text-transform:uppercase;'>Scam Intent</div><div style='font-size:2.2rem;font-weight:800;margin-top:8px;'>{scam_prob * 100:.1f}%</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-container'><div style='font-size:12px;letter-spacing:1px;color:#93c5fd;text-transform:uppercase;'>Overall Risk</div><div style='font-size:2.2rem;font-weight:800;margin-top:8px;'>{risk_score * 100:.1f}%</div></div>", unsafe_allow_html=True)

        risk_class = get_risk_class(risk_category)
        st.markdown(f"""
            <div class="{risk_class}">
                <h3 style="margin: 0;">Risk Level: {risk_category}</h3>
                <h1 style="margin: 10px 0 0 0;">{risk_score * 100:.1f}%</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">{risk_info.get('recommendation', 'No recommendation available')}</p>
            </div>
        """, unsafe_allow_html=True)

        res_tab1, res_tab2, res_tab3, res_tab4, res_tab5, res_tab6 = st.tabs([
            "🚨 Risk Assessment",
            "🤖 Deepfake Detection",
            "📝 Transcript",
            "🎯 Scam Analysis",
            "⏱️ Performance",
            "📋 Details"
        ])
        
        with res_tab1:
            st.write("**Risk reasons:**")
            reasons = risk_info.get("reasons", []) or result.get("risk_reasons", [])
            if reasons:
                for reason in reasons:
                    st.write(f"• {reason}")
            else:
                st.info("No specific risk reasons were returned.")

        with res_tab2:
            deepfake = result.get("deepfake", {})
            st.plotly_chart(create_risk_gauge(deepfake.get("deepfake_probability", 0), "Deepfake Score"), use_container_width=True)
            st.metric("Prediction", deepfake.get("prediction_class", "Unknown"))
            st.metric("Confidence", f"{max(deepfake.get('confidence_scores', [0]))*100:.1f}%")

        with res_tab3:
            transcript = result.get("transcript", "")
            if transcript:
                st.text_area("Transcription", value=transcript, height=180, disabled=True)
            else:
                st.warning("⚠️ No transcript available")

        with res_tab4:
            scam = result.get("scam", {})
            st.plotly_chart(create_risk_gauge(scam.get("scam_probability", 0), "Scam Risk"), use_container_width=True)
            st.metric("Category", scam.get("risk_category", "Unknown"))
            st.metric("Method", scam.get("detection_method", "Unknown"))
            indicators = scam.get("indicators", {})
            if indicators:
                for category, items in indicators.items():
                    if items:
                        with st.expander(f"{category.title()} ({len(items)})"):
                            for item in items:
                                st.write(f"• {item}")

        with res_tab5:
            timing = result.get("timing_metrics", {})
            perf_data = {
                "Component": ["Preprocessing", "Deepfake Detection", "ASR", "Scam Analysis", "Total"],
                "Time (ms)": [
                    float(timing.get("preprocessing_time", 0) or 0) * 1000,
                    float(timing.get("deepfake_time", 0) or 0) * 1000,
                    float(timing.get("asr_time", 0) or 0) * 1000,
                    float(timing.get("scam_time", 0) or 0) * 1000,
                    float(timing.get("total_processing_time", 0) or 0) * 1000,
                ]
            }
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

        with res_tab6:
            st.json(result)


# ============================================================================
# TAB 2: LIVE CALL MONITOR
# ============================================================================
with tab_live:
    st.header("📡 Live Call Monitor")
    st.markdown("""
    <div class="info-box">
    📢 <strong>Incoming call monitoring:</strong> This module is built for a real-time microphone-driven call interception workflow. The browser captures live audio, streams it to the backend, and evaluates the voice for deepfake and scam risk as the call progresses.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Live Monitoring Workflow")
    st.markdown("""
    1. Open the browser-based live call monitor.
    2. Allow microphone access.
    3. Start the live session.
    4. The system streams audio chunks to the backend in real time.
    5. The model evaluates deepfake, scam intent, and overall risk continuously.
    """)

    if get_api_status():
        st.success("✅ Backend is running and ready for live monitoring.")
    else:
        st.warning("⚠️ Backend is not running. Start it with `python start_backend.py` before opening the live monitor.")

    st.markdown("[Open Live Call Monitor in Browser](http://127.0.0.1:8000/api/v1/live-call)", unsafe_allow_html=True)

    st.caption("This is the implementation that matches the report. Demo audio files are only fallback testing data, not the main live-call workflow.")

    st.divider()

    st.subheader("Fallback Demo Test")
    st.info("Use this only if you want to test the backend pipeline with a sample file. The report-aligned system is the live microphone call monitor above.")

    demo_files = load_demo_audio_files()
    if not demo_files:
        st.warning("No demo files found in `demo_samples/`.")
        if st.button("Generate Demo Audio for Testing"):
            sr = 16000
            duration = 6
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.55 * np.sin(2 * np.pi * 220 * t)
            audio += 0.25 * np.sin(2 * np.pi * 330 * t)
            audio += 0.18 * np.sin(2 * np.pi * 500 * t)
            audio += np.random.normal(0, 0.03, len(audio))
            os.makedirs("demo_samples", exist_ok=True)
            sf.write("demo_samples/demo_live_call.wav", audio, sr)
            st.success("✅ Demo test audio created.")
            st.rerun()
    else:
        selected_demo = st.selectbox("Choose demo fallback file", demo_files)
        demo_path = os.path.join("demo_samples", selected_demo)
        with open(demo_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

        if st.button("Run Demo Fallback Analysis", use_container_width=True):
            if not get_api_status():
                st.error("❌ Backend is not running. Start it with `python start_backend.py` first.")
            else:
                with open(demo_path, "rb") as f:
                    live_stream = BytesIO(f.read())
                with st.spinner("Analyzing demo fallback audio..."):
                    result = stream_live_audio_to_backend(live_stream)
                if result and "error" not in result:
                    st.session_state.analysis_result = result
                    risk_data = result.get("risk", {})
                    st.success("✅ Demo fallback analysis complete")
                    st.metric("Deepfake Risk", f"{result.get('deepfake', {}).get('deepfake_probability', 0) * 100:.1f}%")
                    st.metric("Scam Intent", f"{result.get('scam', {}).get('scam_probability', 0) * 100:.1f}%")
                    st.metric("Overall Risk", f"{risk_data.get('overall_risk_score', 0) * 100:.1f}%")
                else:
                    st.error(f"❌ Demo fallback failed: {result.get('error', 'Unknown error') if result else 'Unknown error'}")

    if st.session_state.analysis_result:
        st.divider()
        st.subheader("Latest Analysis")
        result = st.session_state.analysis_result
        risk_info = result.get("risk", {})
        if result.get("transcript"):
            st.text_area("Transcript", value=result.get("transcript", ""), height=150, disabled=True)
        st.json({
            "Risk Category": risk_info.get("risk_category", "UNKNOWN"),
            "Overall Risk Score": risk_info.get("overall_risk_score", 0),
            "Recommendation": risk_info.get("recommendation", "No recommendation"),
            "Deepfake": result.get("deepfake", {}).get("deepfake_probability", 0),
            "Scam": result.get("scam", {}).get("scam_probability", 0),
            "Reasons": risk_info.get("reasons", [])
        })


# ============================================================================
# TAB 3: DEMO SAMPLES
# ============================================================================
with tab_demo:
    st.header("Demo Audio Samples")
    
    st.markdown("""
    <div class="success-box">
    🎵 <strong>Demo Mode:</strong> Use pre-loaded sample audio files to test the system without uploading your own.
    </div>
    """, unsafe_allow_html=True)
    
    demo_files = load_demo_audio_files()
    
    if not demo_files:
        st.warning("⚠️ No demo samples found in `demo_samples/` directory")
        st.info("""
        To add demo samples:
        1. Create a `demo_samples/` folder in the project root
        2. Add .wav, .mp3, .m4a, or .flac files
        3. Refresh this page
        
        **Or generate a test file:**
        """)
        
        if st.button("Generate Sample Audio"):
            sr = 16000
            duration = 3
            t = np.linspace(0, duration, int(sr*duration))
            
            # Create "speech-like" audio
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
            audio += 0.2 * np.sin(2 * np.pi * 200 * t)
            audio += 0.1 * np.sin(2 * np.pi * 600 * t)
            audio += np.random.normal(0, 0.02, len(audio))
            
            # Save
            os.makedirs("demo_samples", exist_ok=True)
            sf.write("demo_samples/test_sample.wav", audio, sr)
            st.success("✅ Generated test_sample.wav")
            st.rerun()
    else:
        st.success(f"✅ Found {len(demo_files)} demo sample(s)")
        
        selected_demo = st.selectbox("Choose a demo file:", demo_files)
        
        if selected_demo:
            demo_path = os.path.join("demo_samples", selected_demo)
            
            with open(demo_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
            
            if st.button("Analyze Demo Sample", use_container_width=True):
                if api_status:
                    with open(demo_path, "rb") as f:
                        result = upload_and_analyze(("demo.wav", f, "audio/wav"))
                    
                    if result:
                        st.session_state.analysis_result = result
                        st.success("✅ Analysis complete!")
                        if hasattr(st, "switch_to_tab"):
                            st.switch_to_tab("📁 Upload Audio")
                        else:
                            st.info("✅ Analysis complete. Please review the Upload Audio tab.")
                else:
                    st.error("❌ Backend not running!")


# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================
with tab_analytics:
    st.header("📊 Analytics & History")
    
    if st.button("🔄 Refresh Analytics", use_container_width=True):
        st.rerun()
    
    try:
        # Get statistics
        stats_response = requests.get(f"{API_URL}/statistics", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyses", stats.get("total_analyses", 0))
            with col2:
                st.metric("High Risk", stats.get("high_risk_count", 0))
            with col3:
                st.metric("Medium Risk", stats.get("medium_risk_count", 0))
            with col4:
                st.metric("Low Risk", stats.get("low_risk_count", 0))
            
            st.divider()
            
            # Risk distribution pie chart
            risk_data = {
                "Risk Level": ["High", "Medium", "Low"],
                "Count": [
                    stats.get("high_risk_count", 0),
                    stats.get("medium_risk_count", 0),
                    stats.get("low_risk_count", 0)
                ]
            }
            
            fig = px.pie(
                values=risk_data["Count"],
                names=risk_data["Risk Level"],
                color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analytics data available yet")
    except:
        st.warning("Could not fetch analytics data")


# ============================================================================
# TAB 5: API DOCS
# ============================================================================
with tab_api:
    st.header("🔌 API Documentation")
    
    st.markdown("""
    ### REST API Endpoints
    
    #### 1. Health Check
    ```
    GET /api/v1/health
    ```
    Returns API status and model information.
    
    #### 2. Analyze Audio (Upload)
    ```
    POST /api/v1/analyze/audio
    ```
    Upload an audio file for analysis.
    
    #### 3. Get Results
    ```
    GET /api/v1/results/{session_id}
    ```
    Retrieve a specific analysis result.
    
    #### 4. List Recent Results
    ```
    GET /api/v1/results?limit=10&risk_category=HIGH
    ```
    Get recent analyses with optional filtering.
    
    #### 5. Statistics
    ```
    GET /api/v1/statistics?hours=24
    ```
    Get aggregated statistics.
    
    ### WebSocket Endpoint
    
    ```
    WS /api/v1/ws/analyze
    ```
    Real-time streaming audio analysis.
    
    ### Example Usage
    ```bash
    # Analyze an audio file
    curl -X POST http://127.0.0.1:8001/api/v1/analyze/audio \\
      -F "file=@sample.wav"
    
    # Get health status
    curl http://127.0.0.1:8001/api/v1/health
    ```
    """)


# ============================================================================
# TAB 6: ABOUT
# ============================================================================
with tab_about:
    st.header("ℹ️ About")
    
    st.markdown("""
    ### Real-Time Deepfake Voice Interceptor & Audio Authenticator
    
    A comprehensive audio analysis system for detecting AI-generated speech and identifying scam intent.
    
    #### Features
    - 🤖 **Deepfake Detection** - CNN+GRU neural network for voice spoofing detection
    - 🎤 **Speech Recognition** - Whisper ASR for accurate transcription
    - 🎯 **Scam Analysis** - NLP-based fraud intent detection
    - 📊 **Risk Assessment** - Combined risk scoring and categorization
    - 📈 **Analytics** - Historical data and statistics
    - 🔌 **REST API** - Full API for integration
    - 💾 **Database** - SQLite logging for audit trails
    
    #### Technology Stack
    - **Backend**: FastAPI + Uvicorn
    - **Frontend**: Streamlit
    - **ML**: PyTorch, Transformers, scikit-learn
    - **Audio**: Librosa, Soundfile
    - **Database**: SQLite + SQLAlchemy
    
    #### Project Information
    - **Version**: 1.0.0-beta
    - **Status**: Ready for Review-2
    - **Institution**: VIT Vellore
    - **Course**: BCSE497J - Project-I
    - **Last Updated**: 2026-08-31
    
    #### Documentation
    - [README.md](README.md) - Complete setup guide
    - [QUICK_START.md](QUICK_START.md) - Quick setup instructions
    - [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - Implementation details
    
    #### Disclaimer
    ⚠️ **Academic Prototype**: This system is designed for research and demonstration purposes. 
    All predictions are generated from real models (not simulated), but the system has known 
    limitations. See documentation for complete details.
    """)


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
💡 For questions or issues, see the documentation files in the project root.
</div>
""", unsafe_allow_html=True)
