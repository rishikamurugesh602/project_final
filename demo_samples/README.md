# Demo Samples

This directory contains sample audio files for demonstration purposes.

## Usage

Place audio files here to use the "Demo Audio" mode in the Streamlit dashboard:

1. Place audio files (WAV, MP3, etc.) in this directory
2. Run the dashboard: `streamlit run dashboard/streamlit_app.py`
3. Select "Demo Audio" mode
4. Choose a sample and click "Analyze"

## Example Files

Create sample files by:

```bash
# Using FFmpeg to extract audio
ffmpeg -i input_video.mp4 -q:a 0 -map a demo_sample.mp3

# Using Python to generate synthetic audio
python -c "
import numpy as np
import soundfile as sf
sr = 16000
duration = 5
t = np.linspace(0, duration, int(sr*duration))
# Generate 440 Hz sine wave
audio = np.sin(2 * np.pi * 440 * t)
sf.write('sine_440hz.wav', audio, sr)
"
```

## File Format

Supported formats:
- *.wav (recommended)
- *.mp3
- *.m4a
- *.flac
- *.ogg

Audio should be:
- 2-30 seconds duration
- 16kHz sample rate (will be auto-converted)
- Mono channel (will be auto-converted)

## Privacy

⚠️ Only place audio files you have permission to use in this directory.

---

**Created**: 2026-08-31
