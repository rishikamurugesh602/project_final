"""
Enhanced Streamlit dashboard for real-time deepfake voice detection.
Modern UI with improved visualizations, honest model reporting, and WebRTC integration.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
from datetime import datetime
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
        padding: 35px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.25);
        text-align: center;
    }
    .header-main h1 {
        margin: 0;
        font-size: 2.8em;
        font-weight: 800;
        color: white;
    }
    .header-main p {
        margin: 12px 0 0 0;
        font-size: 1.15em;
        opacity: 0.95;
        color: #bfdbfe;
    }
    .status-online {
        display: inline-block;
        background: rgba(34, 197, 94, 0.18);
        color: #bbf7d0;
        padding: 8px 16px;
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 20px;
        font-weight: 600;
    }
    .status-offline {
        display: inline-block;
        background: rgba(239, 68, 68, 0.16);
        color: #fecaca;
        padding: 8px 16px;
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 20px;
        font-weight: 600;
    }
    .metric-container {
        background: rgba(15, 23, 42, 0.92);
        color: #edf4ff;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin: 8px 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #fecaca;
        margin: 14px 0;
    }
    .risk-medium {
        background: linear-gradient(135deg, #d97706 0%, #92400e 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #fde68a;
        margin: 14px 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #15803d 0%, #166534 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #bbf7d0;
        margin: 14px 0;
    }
    .info-box {
        background: rgba(15, 23, 42, 0.9);
        color: #edf4ff;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #60a5fa;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }
    .warning-card {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 10px;
        padding: 14px 18px;
        color: #fde68a;
        margin: 12px 0;
        font-size: 0.92rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
API_URL = "http://127.0.0.1:8000/api/v1"

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


def normalize_analysis_result(result):
    """Convert flat or nested backend payloads into the canonical structure expected by UI."""
    if not isinstance(result, dict):
        return result

    normalized = dict(result)

    risk_category = result.get("risk_category", "UNKNOWN")
    risk_score = float(result.get("overall_risk_score", 0.0) or 0.0)
    level = 3 if risk_category == "HIGH" else (2 if risk_category == "MEDIUM" else 1)

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
            scam_indicators = json.loads(scam_indicators)
        except Exception:
            scam_indicators = [scam_indicators]

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


def upload_and_analyze(audio_file):
    """Upload and analyze audio file via API."""
    try:
        files = {"file": audio_file}
        with st.spinner("🔄 Analyzing audio..."):
            response = requests.post(f"{API_URL}/analyze/audio", files=files, timeout=120)
        
        if response.status_code == 200:
            return normalize_analysis_result(response.json())
        else:
            st.error(f"❌ API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend at {API_URL}. Ensure backend is running.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def stream_live_audio_to_backend(audio_file):
    """Stream audio chunks to the backend WebSocket to simulate live streaming."""
    try:
        file_bytes = audio_file.read() if hasattr(audio_file, 'read') else audio_file
        if not file_bytes:
            st.error("Audio data is empty.")
            return None

        audio, sr = librosa.load(BytesIO(file_bytes), sr=16000, mono=True)
        if len(audio) == 0:
            st.error("Could not read audio data.")
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
                        "is_final": (i == total_chunks - 1),
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(payload))
                    resp = json.loads(await websocket.recv())
                    responses.append(resp)
                    if resp.get("type") == "status":
                        try:
                            extra = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            responses.append(json.loads(extra))
                        except asyncio.TimeoutError:
                            pass

                await websocket.send(json.dumps({"type": "process"}))
                resp = json.loads(await websocket.recv())
                responses.append(resp)

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


def load_demo_audio_files():
    """Load available demo audio files."""
    demo_dir = "demo_samples"
    if os.path.exists(demo_dir):
        files = [f for f in os.listdir(demo_dir) if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))]
        return sorted(files)
    return []


def create_risk_gauge(value, title="Risk Score"):
    """Create risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3b82f6"},
            "steps": [
                {"range": [0, 39], "color": "rgba(34, 197, 94, 0.25)"},
                {"range": [39, 69], "color": "rgba(245, 158, 11, 0.25)"},
                {"range": [69, 100], "color": "rgba(239, 68, 68, 0.25)"}
            ]
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf4ff"))
    return fig


def get_risk_class(risk_category):
    if risk_category == "HIGH":
        return "risk-high"
    elif risk_category == "MEDIUM":
        return "risk-medium"
    return "risk-low"


def render_unified_analysis_result(result):
    """Render canonical analysis results card used by upload and live call workflows."""
    if not result:
        return

    risk_info = result.get("risk", {})
    risk_category = risk_info.get("risk_category", "UNKNOWN")
    risk_score = float(risk_info.get("overall_risk_score", 0.0) or 0.0)
    deepfake_data = result.get("deepfake", {})
    deepfake_prob = float(deepfake_data.get("deepfake_probability", 0.0) or 0.0)
    scam_data = result.get("scam", {})
    scam_prob = float(scam_data.get("scam_probability", 0.0) or 0.0)

    # Model status transparency notice
    df_loaded = deepfake_data.get("model_loaded", False)
    df_status = str(result.get("deepfake_model_status", ""))
    if not df_loaded or "untrained" in df_status.lower() or "random" in df_status.lower():
        st.warning(
            "⚠️ **Model Integrity Notice:** The deepfake acoustic detector is running with untrained / random weights "
            "(demo mode). Scores are illustrative and cannot be used as evidence of real-world deepfake detection accuracy."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-container'><div style='font-size:12px;letter-spacing:1px;color:#93c5fd;text-transform:uppercase;'>Deepfake Risk</div><div style='font-size:2.2rem;font-weight:800;margin-top:8px;'>{deepfake_prob * 100:.1f}%</div><div style='font-size:11px;color:#94a3b8;'>{deepfake_data.get('prediction_class', 'Unknown')}</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='metric-container'><div style='font-size:12px;letter-spacing:1px;color:#93c5fd;text-transform:uppercase;'>Scam Intent</div><div style='font-size:2.2rem;font-weight:800;margin-top:8px;'>{scam_prob * 100:.1f}%</div><div style='font-size:11px;color:#94a3b8;'>{scam_data.get('detection_method', 'NLP')}</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='metric-container'><div style='font-size:12px;letter-spacing:1px;color:#93c5fd;text-transform:uppercase;'>Overall Risk</div><div style='font-size:2.2rem;font-weight:800;margin-top:8px;'>{risk_score * 100:.1f}%</div><div style='font-size:11px;color:#94a3b8;'>Category: {risk_category}</div></div>",
            unsafe_allow_html=True
        )

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
        "⏱️ Performance & Latency",
        "📋 Canonical Schema"
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
        st.plotly_chart(create_risk_gauge(deepfake_prob, "Deepfake Score"), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Prediction Class", deepfake_data.get("prediction_class", "Unknown"))
        with col_b:
            st.metric("Model Checkpoint Loaded", "Yes" if df_loaded else "No (Untrained / Demo)")
        if df_status:
            st.caption(f"Status: {df_status}")

    with res_tab3:
        transcript = result.get("transcript", "")
        if transcript:
            st.text_area("Transcription", value=transcript, height=180, disabled=True)
        else:
            st.info("No spoken transcript text detected in this audio.")

    with res_tab4:
        st.plotly_chart(create_risk_gauge(scam_prob, "Scam Risk"), use_container_width=True)
        st.metric("Detection Method", scam_data.get("detection_method", "Unknown"))
        indicators = scam_data.get("indicators", {})
        if isinstance(indicators, dict):
            for category, items in indicators.items():
                if items:
                    with st.expander(f"{category.title()} ({len(items)})"):
                        for item in items:
                            st.write(f"• {item}")
        elif isinstance(indicators, list) and indicators:
            with st.expander("Detected Scam Indicators"):
                for item in indicators:
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

        latency = result.get("latency_metrics")
        if latency:
            st.divider()
            st.subheader("Streaming Latency & Queue Metrics")
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            with l_col1:
                st.metric("Measured E2E Latency", f"{latency.get('end_to_end_latency_ms', 0)} ms")
            with l_col2:
                st.metric("Inference Time", f"{latency.get('inference_duration_ms', 0)} ms")
            with l_col3:
                st.metric("Queue Depth", latency.get("queue_depth", 0))
            with l_col4:
                st.metric("Dropped Chunks (Backpressure)", latency.get("dropped_chunks", 0))

    with res_tab6:
        st.json(result)


# ============================================================================
# MAIN APP HEADER & STATUS BAR
# ============================================================================
st.markdown("""
    <div class="header-main">
        <h1>🎙️ Deepfake Voice Interceptor</h1>
        <p>Real-Time Deepfake Detection, ASR, and Scam Intent Analysis</p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    api_status = get_api_status()
    if api_status:
        st.markdown('<div class="status-online">✅ Backend Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-offline">❌ Backend Disconnected</div>', unsafe_allow_html=True)
with col2:
    st.metric("System Mode", "WebRTC & Audio Analysis")
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
    Upload an audio file (WAV, MP3, M4A, FLAC, OGG). The pipeline extracts Mel spectrograms for deepfake detection, performs Whisper ASR transcription, and runs NLP scam intent analysis.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Supported formats: WAV, MP3, M4A, FLAC, OGG"
    )

    if uploaded_file is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("File Name", uploaded_file.name[:30])
        with c2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

        st.audio(uploaded_file)

        if st.button("🔍 Analyze Audio", use_container_width=True):
            if api_status:
                result = upload_and_analyze(uploaded_file)
                if result:
                    st.session_state.analysis_result = result
                    st.success("✅ Analysis complete!")
            else:
                st.error("❌ Backend is not running! Start it with `python start_backend.py`.")

    if st.session_state.analysis_result:
        st.divider()
        st.subheader("📊 Analysis Results")
        render_unified_analysis_result(st.session_state.analysis_result)

# ============================================================================
# TAB 2: LIVE CALL MONITOR
# ============================================================================
with tab_live:
    st.header("📡 Live Call Monitor (WebRTC & Streaming)")

    st.markdown("""
    <div class="warning-card">
    <strong>⚠️ Telephony / PSTN Limitation Notice:</strong> This project implements browser-to-browser voice calling via WebRTC and analyzes the <em>received remote audio track</em> with mutual participant consent. It does <strong>not</strong> intercept PSTN/cellular phone network calls, which requires telecommunication provider integration (SIP trunking) and credentials not available in this academic prototype.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>How to use Live WebRTC Audio Call:</strong>
    <ol>
      <li>Click the button below to open the <strong>Live WebRTC Call Interface</strong> in a new browser window.</li>
      <li>To test a 2-party call, open the same link in a <em>second</em> browser window.</li>
      <li>Participant A clicks <strong>Create New Call</strong> (generates a Call ID like <code>CALL-1234</code>) and checks the Consent box.</li>
      <li>Participant B enters that Call ID, checks Consent, and clicks <strong>Join Call</strong>.</li>
      <li>Once connected, the receiver's browser analyzes the <strong>remote audio track</strong> in real time.</li>
      <li>Alternatively, use <strong>Local Microphone Test Mode</strong> for single-user acoustic smoke-testing.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.link_button(
        "🚀 Open Live WebRTC Call Interface",
        "http://127.0.0.1:8000/api/v1/live-call",
        use_container_width=True
    )

    st.divider()
    st.subheader("Fallback Streaming Simulation")
    st.caption("Stream a pre-recorded audio file in bounded chunks over the WebSocket streaming pipeline:")

    demo_files = load_demo_audio_files()
    if demo_files:
        selected_demo = st.selectbox("Choose audio file to simulate streaming:", demo_files)
        demo_path = os.path.join("demo_samples", selected_demo)
        with open(demo_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

        if st.button("▶️ Run Streaming Simulation", use_container_width=True):
            if not api_status:
                st.error("❌ Backend is not running.")
            else:
                with open(demo_path, "rb") as f:
                    live_stream = BytesIO(f.read())
                with st.spinner("Streaming chunks over WebSocket to backend..."):
                    stream_res = stream_live_audio_to_backend(live_stream)
                if stream_res and "error" not in stream_res:
                    st.session_state.analysis_result = stream_res
                    st.success("✅ Streaming analysis complete!")
                    render_unified_analysis_result(stream_res)
                else:
                    st.error(f"❌ Streaming failed: {stream_res.get('error') if stream_res else 'Unknown error'}")

# ============================================================================
# TAB 3: DEMO SAMPLES
# ============================================================================
with tab_demo:
    st.header("🎵 Demo Audio Samples")
    st.markdown("""
    <div class="info-box">
    Pre-loaded audio samples for rapid pipeline verification. Note that generated sinusoidal waveforms in <code>demo_samples/</code> are synthetic test tones to verify pipeline execution, not authentic human speech.
    </div>
    """, unsafe_allow_html=True)

    demo_files = load_demo_audio_files()
    if not demo_files:
        st.warning("No demo files found in demo_samples/. Run `python populate_audio_datasets.py` to create test tones.")
    else:
        selected_demo = st.selectbox("Select sample to inspect:", demo_files)
        demo_path = os.path.join("demo_samples", selected_demo)
        with open(demo_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

        if st.button("🔍 Analyze Selected Sample", use_container_width=True):
            if api_status:
                with open(demo_path, "rb") as f:
                    res = upload_and_analyze(("demo.wav", f, "audio/wav"))
                if res:
                    st.session_state.analysis_result = res
                    st.success("✅ Sample analyzed.")
                    render_unified_analysis_result(res)
            else:
                st.error("❌ Backend is not running.")

# ============================================================================
# TAB 4: ANALYTICS & PRIVACY
# ============================================================================
with tab_analytics:
    st.header("📊 Analytics & History")

    top_c1, top_c2 = st.columns([4, 1])
    with top_c2:
        if st.button("🗑️ Clear History", use_container_width=True):
            try:
                del_res = requests.delete(f"{API_URL}/history", timeout=5)
                if del_res.status_code == 200:
                    st.success("✅ Analysis history purged from database.")
                    st.rerun()
                else:
                    st.error("Failed to purge history.")
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("🔒 Data Storage & Privacy Guarantees", expanded=False):
        st.markdown("""
        - **No Raw Audio Persistence**: Audio streams and uploaded recordings are processed entirely in memory. Raw audio is **never stored on disk** or external servers.
        - **Logged Metadata**: Only analytical records (timestamps, deepfake probabilities, transcript text, scam indicators, and risk scores) are logged in the local SQLite database for auditing.
        - **Data Deletion**: Users may purge all historical analysis records at any time using the **Clear History** button above.
        """)

    try:
        stats_response = requests.get(f"{API_URL}/statistics", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Total Analyses", stats.get("total_analyses", 0))
            with s2:
                st.metric("High Risk", stats.get("high_risk_count", 0))
            with s3:
                st.metric("Medium Risk", stats.get("medium_risk_count", 0))
            with s4:
                st.metric("Low Risk", stats.get("low_risk_count", 0))

            st.divider()
            risk_data = {
                "Risk Level": ["High", "Medium", "Low"],
                "Count": [
                    stats.get("high_risk_count", 0),
                    stats.get("medium_risk_count", 0),
                    stats.get("low_risk_count", 0)
                ]
            }
            if sum(risk_data["Count"]) > 0:
                fig = px.pie(
                    values=risk_data["Count"],
                    names=risk_data["Risk Level"],
                    color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf4ff"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recorded analyses in database yet.")
    except Exception as e:
        st.warning(f"Could not fetch analytics: {e}")

# ============================================================================
# TAB 5: API DOCS
# ============================================================================
with tab_api:
    st.header("🔌 REST & WebSocket API Documentation")
    st.markdown("""
    ### REST Endpoints
    - `GET /api/v1/health` - Check backend health and loaded model status.
    - `POST /api/v1/analyze/audio` - Upload audio file (WAV/MP3/FLAC/OGG) for complete analysis.
    - `POST /api/v1/analyze/transcript` - Analyze transcript text directly for scam indicators.
    - `GET /api/v1/results/{session_id}` - Retrieve analysis result by session UUID.
    - `GET /api/v1/results` - List recent analysis history.
    - `DELETE /api/v1/history` - Purge historical analysis records.
    - `GET /api/v1/privacy` - View privacy guarantees and data retention policy.
    - `GET /api/v1/live-call` - In-app WebRTC live call & remote track audio analyzer.
    - `POST /api/v1/call/session/create` - Create a WebRTC call session room ID.
    - `GET /api/v1/call/session/{call_id}` - Query participants and mutual consent status.

    ### WebSocket Endpoints
    - `WS /api/v1/ws/analyze` - Bounded sliding-window audio streaming analysis with backpressure handling.
    - `WS /api/v1/call/ws/{call_id}/{peer_id}` - WebRTC signaling socket (SDP offer/answer, ICE candidates, consent).
    """)

# ============================================================================
# TAB 6: ABOUT & CITATIONS
# ============================================================================
with tab_about:
    st.header("ℹ️ About & Academic Status")
    st.markdown("""
    ### Real-Time Deepfake Voice Interceptor & Audio Authenticator
    Academic prototype developed for **BCSE497J - Project-I**, VIT Vellore.

    #### Architecture Highlights
    - **Acoustic Anti-Spoofing**: CNN + GRU model inspired by RawNet2 on Mel spectrogram representations.
    - **Speech-to-Text**: Whisper ASR running on throttled cadences to preserve real-time streaming performance.
    - **Scam-Intent NLP**: Keyword and rule-based indicator extraction with optional transformer zero-shot classification.
    - **Risk Fusion Engine**: Configurable weighted combination of acoustic and linguistic threat probabilities.
    - **In-App WebRTC Calling**: Browser-to-browser audio call with mutual consent and remote audio track analysis.

    #### Scientific Integrity Statement
    ⚠️ **Academic Prototype Notice**: This system is designed for research, review, and demonstration. Detection metrics must be validated using official, held-out partitions from certified corpora (e.g. ASVspoof 2019/2021). The prototype does not perform cellular/PSTN telephone network interception, which requires carrier-level telephony gateway infrastructure.
    """)

st.divider()
st.caption("Real-Time Deepfake Voice Interceptor & Audio Authenticator | VIT Vellore BCSE497J Project-I")
