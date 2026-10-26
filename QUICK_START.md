# QUICK START GUIDE

**Real-Time Deepfake Voice Interceptor & Audio Authenticator**

---

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd c:\Users\E.M.Rishika\OneDrive\Desktop\project_1

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell/CMD)
.venv\Scripts\activate.bat

# Install packages - Use the helper script:
.venv\Scripts\python setup_install.py

# Or install manually:
pip install -r requirements_minimal.txt
```

✓ **All packages installed successfully!**

Confirmed installed versions:
- FastAPI 0.141.1
- Uvicorn 0.52.4
- Streamlit 1.62.0
- LibROSA 1.0.0
- PyTorch 2.13.0
- Transformers 5.16.1

### Step 2: Initialize Database
```bash
.venv\Scripts\python -c "from app.database.database import init_database; init_database()"
```

✓ **Database initialized successfully!**

### Step 3: Start Backend
```bash
uvicorn app.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**API available at**: http://localhost:8000/docs

### Step 4: Start Dashboard
Open a **new terminal** (keep backend running) and run:
```bash
.venv\Scripts\python start_dashboard.py
```

Or directly:
```bash
streamlit run dashboard/streamlit_app.py
```

You should see:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

✓ **Dashboard is running on http://localhost:8501**

### Step 5: Use the System

1. Open browser to: **http://localhost:8501**
2. Select "Upload Audio" mode
3. Choose an audio file (WAV, MP3, etc.)
4. Click "Analyze Audio"
5. View results including:
   - Deepfake probability
   - Transcript
   - Scam risk
   - Overall risk assessment
   - Performance metrics

---

## Quick Test

### Test 1: API Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Test 2: Analyze File via API
```bash
curl -X POST http://localhost:8000/api/v1/analyze/audio \
  -F "file=@sample_audio.wav"
```

### Test 3: Run Tests
```bash
pytest tests/test_system.py -v
```

---

## Demo Audio Files

Create a sample to test:

**Option A: Use system sounds**
- Find audio files on your system
- Drag to upload in dashboard

**Option B: Generate synthetic audio**
```python
import numpy as np
import soundfile as sf

# Create 440Hz sine wave
sr = 16000
duration = 3
t = np.linspace(0, duration, int(sr*duration))
audio = np.sin(2 * np.pi * 440 * t)

# Add some "speech-like" variation
audio += 0.1 * np.sin(2 * np.pi * 200 * t)

sf.write('test_audio.wav', audio, sr)
```

**Option C: Use online resources**
- Download sample audio from internet
- Place in `demo_samples/` directory
- Use "Demo Audio" mode in dashboard

---

## Configuration

Edit `.env` file to customize:

```bash
# Use GPU if available
MODEL_DEVICE=cuda

# Smaller model for faster inference
WHISPER_MODEL=tiny

# Adjust risk thresholds
RISK_LOW_THRESHOLD=0.3
RISK_MEDIUM_THRESHOLD=0.6
RISK_HIGH_THRESHOLD=0.9
```

---

## Common Issues & Solutions

### Issue: "Module not found"
```
ModuleNotFoundError: No module named 'app'
```
**Solution**: Make sure you're in the project directory:
```bash
cd c:\Users\E.M.Rishika\OneDrive\Desktop\project_1
```

### Issue: "Port already in use"
```
Address already in use (:8000)
```
**Solution**: Use different port:
```bash
uvicorn app.main:app --reload --port 8001
```

### Issue: "Out of memory"
```
CUDA out of memory
```
**Solution**: Use CPU instead:
```bash
echo MODEL_DEVICE=cpu >> .env
```

### Issue: "Streamlit can't connect to backend"
**Solution**: Make sure backend is running:
```bash
# Check backend is running
curl http://localhost:8000/api/v1/health
```

### Issue: Download stuck on model initialization
**Solution**: This is normal on first run. Wait for completion.
- Whisper model: ~1.5GB
- Transformers: ~1-2GB
- Total: ~3-4GB

### Issue: "Cannot import 'setuptools.build_meta'"
```
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'
```
**Solution 1 (Quick fix)**:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Solution 2 (Start fresh)**:
```bash
rmdir /s .venv
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Solution 3 (Minimal packages)**:
```bash
pip install -r requirements_minimal.txt
```

---

## File Locations

| Component | Location |
|-----------|----------|
| Backend | `app/main.py` |
| Dashboard | `dashboard/streamlit_app.py` |
| Audio Data | `data/` |
| Demo Audio | `demo_samples/` |
| Database | `data/app.db` |
| Logs | `logs/app.log` |
| Models | `models/checkpoints/` |
| Tests | `tests/test_system.py` |

---

## Documentation

- **README.md** - Complete setup and API docs
- **PROJECT_IMPLEMENTATION_STATUS.md** - Implementation details
- **IMPLEMENTATION_COMPLETE.md** - Project summary
- **data/README.md** - Dataset preparation
- **models/README.md** - Model management

---

## Keyboard Shortcuts

### Streamlit Dashboard
- Press `r` to rerun
- Press `c` to clear cache
- Press `s` to save screenshot

### Terminal
- `Ctrl+C` to stop server
- `Ctrl+L` to clear screen

---

## Next Steps

1. ✓ Get system running (above)
2. ✓ Test with sample audio
3. □ Add real audio data (optional)
4. □ Train deepfake detector (optional)
5. □ View evaluation metrics
6. □ Demo for Review-2

---

## Support

**See full README for:**
- Detailed API documentation
- Training instructions
- Troubleshooting guide
- Performance tuning
- Dataset preparation

---

**Version**: 1.0.0-beta  
**Status**: Ready for Demo  
**Last Updated**: 2026-08-31
