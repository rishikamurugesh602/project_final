# Real-Time Deepfake Voice Interceptor & Audio Authenticator

**Academic Project** | VIT BCSE497J - Project-I | B.Tech Final Year  
**Authors**: Rishika Murugesh (23BAI0005), Anish S.A (23BCB0143)  
**Supervisor**: Dr. Vijaya Kumar K, Associate Professor Senior Grade 1, SCOPE, VIT Vellore  

---

## 1. Project Overview

This project implements an intelligent, real-time audio authentication and fraud-detection pipeline designed to protect users during live voice conversations and pre-recorded audio analysis. The system combines:

1. **Acoustic Deepfake Detection**: A lightweight CNN + GRU neural architecture that inspects acoustic features (Mel spectrograms / RawNet2-inspired representations) to detect AI-synthesized or cloned speech.
2. **Automatic Speech Recognition (ASR)**: OpenAI Whisper converts speech into text at controlled intervals.
3. **NLP Scam Intent Analysis**: Rule-based keyword matching and zero-shot transformer classification (BART-MNLI) identify social engineering, urgency, credential requests, and financial coercion patterns.
4. **Risk Fusion Engine**: Merges acoustic spoof probability and semantic scam probability into a unified risk category (`LOW`, `MEDIUM`, `HIGH`) with transparent reasoning and actionable recommendations.
5. **In-App Browser-to-Browser WebRTC VoIP Calling**: Real-time peer-to-peer audio calls with room codes, explicit mutual consent verification, and remote audio track analysis.
6. **Streaming & Backpressure Management**: Bounded sliding audio windows, bounded worker queues, and event-loop offloading to maintain low-latency analysis.
7. **Privacy & Data Security**: No raw audio persisted by default, in-memory decoding with guaranteed temp cleanup, analysis history purging, and configurable CORS origins.

> [!IMPORTANT]
> **PSTN / Carrier Telephone Disclaimer**:  
> This system supports **in-app browser-to-browser WebRTC VoIP calling** and **local microphone test mode**. It does **NOT** intercept carrier/PSTN landline or cellular phone calls directly. Intercepting carrier telephony requires third-party SIP trunking credentials (e.g., Twilio SIP, FreePBX, or Asterisk), carrier interconnections, and specialized telephony hardware not included in this academic repository.

---

## 2. Technology Stack

- **Backend Framework**: FastAPI + Uvicorn (ASGI)
- **Real-Time Communication**: WebRTC (Web Audio API peer track capture) & WebSockets
- **Audio Processing**: librosa, soundfile, numpy, scipy
- **Deepfake Detection**: PyTorch (CNN + GRU with temporal pooling)
- **Speech Recognition**: OpenAI Whisper (`base`, `tiny`, `small`)
- **Scam Detection**: HuggingFace Transformers (`facebook/bart-large-mnli` / rule-based fallback)
- **Database & Persistence**: SQLite + SQLAlchemy ORM (metadata only, no raw audio)
- **Frontend Dashboard**: Streamlit + Plotly

---

## 3. Architecture & Streaming Design

```
[Peer A Browser] <====== WebRTC Audio Call ======> [Peer B Browser]
       │                                                   │
  (Local Mic)                                     (Remote Audio Track)
                                                           │
                                                  [Web Audio API Node]
                                                           │
                                                  [WebSocket Stream]
                                                           │
                                                           ▼
                                            ┌──────────────────────────────┐
                                            │ StreamingAnalysisSession     │
                                            │ ├── Bounded Queue (max 10)   │
                                            │ ├── Backpressure Chunk Drop  │
                                            │ └── Bounded Sliding Window   │
                                            │     (3.5s acoustic memory)   │
                                            └──────────────┬───────────────┘
                                                           │
                                      ┌────────────────────┴────────────────────┐
                                      │                                         │
                                      ▼ (asyncio.to_thread)                     ▼ (Cadence/Final)
                         ┌───────────────────────────┐             ┌───────────────────────────┐
                         │ Acoustic Deepfake Model   │             │ Whisper ASR & Transcribe  │
                         │ (CNN + GRU Spectrogram)   │             └─────────────┬─────────────┘
                         └─────────────┬─────────────┘                           │
                                       │                                         ▼
                                       │                           ┌───────────────────────────┐
                                       │                           │ NLP Scam Intent Detector  │
                                       │                           │ (Keywords + Transformer)  │
                                       │                           └─────────────┬─────────────┘
                                       │                                         │
                                       └─────────────────────┬───────────────────┘
                                                             ▼
                                                ┌───────────────────────────┐
                                                │ Risk Decision Engine      │
                                                │ (Weighted Fusion Scoring) │
                                                └─────────────┬─────────────┘
                                                              ▼
                                                [Canonical Analysis Result]
                                                ├── deepfake_probability
                                                ├── scam_probability
                                                ├── overall_risk_score & category
                                                ├── transcript & recommendations
                                                ├── latency_ms & timestamps
                                                └── model_status disclosures
```

### Key Streaming Safeguards
1. **Bounded Audio Buffer**: Audio chunks are appended to a sliding window capped at 3.5 seconds (56,000 samples @ 16kHz). Older samples are discarded to prevent unbounded memory growth.
2. **Bounded Queue & Backpressure**: Real-time WebSocket chunks are buffered in an `asyncio.Queue` capped at 10 chunks. If model inference lags, older chunks are safely dropped to prevent latency cascading.
3. **Offloading Blocking Model Inference**: All PyTorch, librosa, and Whisper inference calls are dispatched via `asyncio.to_thread` so FastAPI's asynchronous event loop remains responsive.
4. **ASR Cadence Throttling**: Whisper ASR runs on a throttled cadence (or upon call turn completion) rather than every fractional chunk, preserving CPU cycles and ensuring smooth streaming.

---

## 4. Privacy, Security & Data Handling

1. **Explicit Mutual Consent**:
   - WebRTC live calling requires both participants to explicitly acknowledge and check a consent prompt before audio capture or transmission begins.
   - Consent status is synchronized across peers via signaling.
2. **No Raw Audio Persistence**:
   - Audio is processed in-memory (`io.BytesIO`). When temporary files are strictly required for decoding exotic formats, they are enclosed in `try ... finally` blocks guaranteeing immediate deletion from disk.
   - The database stores only derived analytical metadata (probabilities, risk category, transcript, timestamps).
3. **Analysis History Purge**:
   - Users can purge all stored analysis records via `DELETE /api/v1/history` or the one-click **"Clear All Analysis History"** button in the Streamlit Analytics dashboard.
   - Endpoint `GET /api/v1/privacy` documents all data retention policies.
4. **Configurable CORS**:
   - Replaced permissive wildcard CORS (`*`) with configurable allowed origins (`ALLOWED_ORIGINS` in `.env` / `config.py`).
5. **Secure Transport**:
   - Production deployment requires HTTPS/WSS. Never commit or hardcode credentials in source code.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.9 - 3.11 (tested on Python 3.11.9)
- Virtual environment (`venv`)
- ~3GB free disk space for pre-trained ASR / NLP models

### 1. Clone & Setup Environment
```bash
git clone <repo-url>
cd project_1

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment (.env)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Key configuration parameters:
```env
MODEL_DEVICE=cpu                   # cpu or cuda
WHISPER_MODEL=base                 # tiny, base, small
ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000
DATABASE_URL=sqlite:///./data/app.db
ENABLE_AUDIO_STORAGE=false         # Guarantees raw audio is never written to disk
SCAM_LOCAL_ONLY=true               # Offline mode for zero-shot transformer
```

### 4. Initialize Database
```bash
python -c "from app.database.database import init_database; init_database()"
```

---

## 6. Running the System

### Option A: Start FastAPI Backend
```bash
python start_backend.py
# Or directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Privacy Policy: `http://localhost:8000/api/v1/privacy`
- In-App WebRTC Calling UI: `http://localhost:8000/call/app`

### Option B: Start Streamlit Dashboard
In a second terminal (with `.venv` active):
```bash
python start_dashboard.py
# Or directly:
streamlit run dashboard/streamlit_app.py
```
- Dashboard URL: `http://localhost:8501`

---

## 7. Using the Applications

### 1. In-App WebRTC Browser-to-Browser Call (`/call/app`)
1. Open `http://localhost:8000/call/app` in two browser tabs or on two devices.
2. Participant A enters their Name, enters or generates a Call ID, checks the **Mutual Consent Checkbox**, and clicks **"Create / Join Call"**.
3. Participant B enters the same Call ID, checks **Mutual Consent**, and clicks **"Create / Join Call"**.
4. Both participants click **"Establish Call"**. WebRTC peer-to-peer audio connects.
5. Click **"Start Live Call Analysis"**: the system taps the **remote audio track** using the Web Audio API and streams bounded chunks via WebSocket to the backend analyzer.
6. The panel displays live deepfake probability, transcript, scam indicators, overall risk, and measured round-trip latency.

### 2. Local Microphone Monitor Mode
On the WebRTC call page or dashboard, select **"Microphone Test Mode (Local Only)"** to test voice detection and Whisper transcription using your local microphone without creating a call room.

### 3. File Upload Analysis (Streamlit Dashboard)
1. Navigate to `http://localhost:8501`.
2. Select **"Upload Audio"** or **"Demo Audio"**.
3. Upload any standard audio file (WAV, MP3, FLAC).
4. Click **"Analyze Audio"** to view the canonical results: deepfake score, transcription, scam indicators, overall risk gauge, and honest model status disclosure.
5. In the **"Analytics"** tab, inspect past sessions or click **"Clear All Analysis History"** to purge records.

---

## 8. Dataset Preparation & Model Evaluation

### Honest Dataset & Evaluation Policy
- **Synthetic Demo Audio**: Audio files in `data/demo_audio/` (generated by `populate_audio_datasets.py`) are **synthetic test tones** designed strictly for verifying audio processing pipelines. They are **NOT real speech** and must not be used to claim model accuracy.
- **ASVspoof 2019 Corpus**: To train or evaluate genuine acoustic spoof detectors, researchers must independently register and obtain the official ASVspoof 2019 dataset from [asvspoof.org](https://www.asvspoof.org/).
- **No Manual Copying / Splitting**: Do not copy ASVspoof files into arbitrary `data/real` and `data/fake` folders or perform random splits. The official protocol files (`protocols/ASVspoof2019.LA.cm.train.trn.txt`, `protocols/ASVspoof2019.LA.cm.dev.trl.txt`, `protocols/ASVspoof2019.LA.cm.eval.trl.txt`) must be preserved to maintain speaker isolation across train/dev/eval splits.

### Validating ASVspoof Dataset Protocols
Run the protocol validation tool to verify protocol structure, check referenced FLAC/WAV files, and inspect class distribution:
```bash
python -m training.validate_protocol --protocol path/to/ASVspoof2019.LA.cm.dev.trl.txt --audio_dir path/to/flac/
```
The validator reports:
- Number of bonafide vs spoof utterances
- Missing audio files
- Label formatting validity

### Training & Evaluation on Held-Out Splits
```bash
# Train on official training protocol
python training/train_deepfake.py --data_dir data/asvspoof --epochs 30

# Evaluate ONLY on separate held-out evaluation protocol (do not invent accuracy)
python training/evaluate_deepfake.py --data_dir data/asvspoof --model_path models/checkpoints/deepfake_detector_best.pt
```

---

## 9. Canonical Result Structure

Both uploaded audio analysis and real-time streaming return identical, unified response structures:

```json
{
  "session_id": "uuid-v4",
  "timestamp": "2026-10-26T12:00:00",
  "deepfake_probability": 0.82,
  "deepfake_prediction": "AI-Generated",
  "transcript": "Urgent notification: verify your password immediately...",
  "scam_probability": 0.78,
  "scam_indicators": ["urgency", "credential_harvest"],
  "overall_risk_score": 0.80,
  "risk_category": "HIGH",
  "risk_reasons": [
    "High AI-generated voice probability (82%)",
    "High scam/fraud intent detected (78%)"
  ],
  "recommendation": "CRITICAL WARNING: High probability of AI voice synthesis and fraudulent intent.",
  "latency_ms": 412.5,
  "total_processing_time": 0.41,
  "model_status": {
    "deepfake_detector": "Untrained / Random Weights (Demo Only - No Verified Checkpoint)",
    "asr": "OpenAI Whisper (base)",
    "scam_detector": "Rule-Based + BART-MNLI Zero-Shot"
  }
}
```

> [!NOTE]
> When no trained `.pt` checkpoint is present in `models/checkpoints/`, `model_status.deepfake_detector` explicitly flags that random/demo weights are active so users are never misled into treating demo outputs as verified predictions.

---

## 10. Verification & Automated Testing

The project includes unit and integration tests covering the full pipeline:
- Bounded sliding window buffer enforcement
- Chunk boundary handling & empty chunk rejection
- Bounded queue limits and backpressure dropping
- Session disconnect and resource cleanup
- WebRTC signaling room management & mutual consent logic
- Analysis history purging and privacy verification
- ASVspoof protocol parsing and missing file detection
- Audio normalization, spectrogram extraction, model forward passes, and risk fusion

### Run the Full Test Suite:
```bash
pytest tests -v
```
**Current status**: 24 passed in ~35 seconds.

### Run End-to-End System Verification:
```bash
python verify_system.py
```
**Current status**: 10/10 subsystem checks passed.

---

## 11. Known Limitations & Future Work

1. **Carrier PSTN Telephony**: Intercepting traditional cellular or landline phone calls requires telecommunications carrier interconnection or SIP trunking hardware (Twilio, Asterisk). The current system implements browser-to-browser WebRTC VoIP calling.
2. **Acoustic Checkpoint**: Model weights are uninitialized/demo by default until trained against the official ASVspoof dataset.
3. **Multilingual NLP**: The rule-based scam indicator set is currently optimized for English conversation patterns; expansion to multilingual intent models is planned for future work.
4. **Speaker Separation**: In multi-speaker WebRTC calls with mixed tracks, diarization (e.g. PyAnnote) could be added to analyze individual speakers separately.

---

## 12. Academic Integrity Statement

This project adheres to strict academic honesty principles:
- No accuracy metrics or performance figures are fabricated or hardcoded.
- Generated demo tones are clearly identified as synthetic pipeline test audio, not human speech.
- Uninitialized models are explicitly labeled in API responses and the dashboard UI.
- Evaluation requires genuine held-out evaluation splits.
