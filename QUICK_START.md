# QUICK START GUIDE
**Real-Time Deepfake Voice Interceptor & Audio Authenticator**

---

## 1. Quick Setup

### Step 1: Environment Activation
```bash
cd project_1

# Activate existing virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Initialize Database
```bash
python -c "from app.database.database import init_database; init_database()"
```

### Step 3: Start FastAPI Backend
```bash
python start_backend.py
# Or directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- API Docs: `http://localhost:8000/docs`
- WebRTC Peer Calling UI: `http://localhost:8000/call/app`
- Privacy Policy: `http://localhost:8000/api/v1/privacy`

### Step 4: Start Streamlit Dashboard
Open a new terminal (with `.venv` active):
```bash
python start_dashboard.py
# Or directly:
streamlit run dashboard/streamlit_app.py
```
- Dashboard URL: `http://localhost:8501`

---

## 2. Using the System

### In-App Browser-to-Browser WebRTC Call
1. Navigate to `http://localhost:8000/call/app` in two browser tabs or two separate devices.
2. Enter a Caller Name and a shared Call ID (e.g. `call-101`).
3. Check the **Mutual Consent Checkbox** on both tabs.
4. Click **Create / Join Call** on both tabs, then **Establish Call**.
5. Once connected, click **Start Live Call Analysis** to intercept and analyze the **remote peer's audio track** in real time.
6. Observe live deepfake probability, transcription, scam indicators, and round-trip latency.

> [!NOTE]
> **PSTN Telephony Note**: This is in-browser WebRTC VoIP calling. Intercepting landline or cellular phone calls directly requires third-party SIP trunking credentials (e.g., Twilio SIP, Asterisk).

### Local Microphone Test Mode
If testing alone, select **"Microphone Test Mode (Local Only)"** on `http://localhost:8000/call/app` to test local speech capture and live analysis without creating a call room.

### Audio File Upload Analysis
1. Navigate to `http://localhost:8501`.
2. Upload any WAV, MP3, or FLAC audio file in the **Upload Audio** tab.
3. Click **Analyze Audio** to inspect the unified canonical result.
4. In the **Analytics** tab, view past sessions or click **"Clear All Analysis History"** to purge records.

---

## 3. Running Automated Tests

```bash
# Run all 24 unit and integration tests
pytest tests -v

# Run system subsystem checks
python verify_system.py

# Validate ASVspoof protocol format
python -m training.validate_protocol --protocol data/sample_protocol.txt --audio_dir data/demo_audio
```

---

## 4. Key Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/call/app` | GET | In-app WebRTC browser-to-browser calling interface |
| `/call/ws/{call_id}/{peer_id}` | WS | WebRTC signaling & mutual consent channel |
| `/api/v1/ws/analyze` | WS | Bounded streaming audio analysis channel |
| `/api/v1/analyze/audio` | POST | Audio file upload analysis |
| `/api/v1/history` | DELETE | Purge all analysis records from database |
| `/api/v1/privacy` | GET | Data retention policy (zero raw audio persistence) |
| `/api/v1/health` | GET | Backend health check & loaded model disclosures |
