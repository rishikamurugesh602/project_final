# Dataset Structure

This directory should contain audio files for training and evaluating the deepfake detector.

## Directory Layout

```
data/
├── real/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   ├── audio_003.mp3
│   └── ...
└── fake/
    ├── deepfake_001.wav
    ├── deepfake_002.wav
    ├── synthetic_001.mp3
    └── ...
```

## Getting Data

### Option 1: ASVspoof 2019 Dataset (Recommended)

The ASVspoof 2019 dataset is the largest publicly available anti-spoofing dataset.

**Download**: https://www.asvspoof.org/

**Steps**:
1. Register on ASVspoof website
2. Download the LA (Logical Access) subset
3. Extract LA/ASVspoof2019_LA_train/flac/ files:
   - Copy `genuine_U_*.flac` files to `data/real/`
   - Copy `spoof_U_*.flac` files to `data/fake/`

### Option 2: TIMIT Dataset

**Download**: https://catalog.ldc.upenn.edu/LDC93S1

**Usage**:
- Contains genuine speech samples
- Use for `data/real/` directory
- Combine with synthetic deepfakes for `data/fake/`

### Option 3: Generate Synthetic Deepfakes

Create synthetic audio using:

**A. Google Text-to-Speech**
```python
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()
input_text = texttospeech.SynthesisInput(text="Your text here")
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
)
response = client.synthesize_speech(request={
    "input": input_text,
    "voice": voice,
    "audio_config": audio_config,
})

with open("output.wav", "wb") as out:
    out.write(response.audio_content)
```

**B. Microsoft Azure**
```python
import azure.cognitiveservices.speech as speechsdk

speech_config = speechsdk.SpeechConfig(
    subscription=subscription_key, 
    region=service_region
)
audio_config = speechsdk.audio.AudioOutputConfig(
    filename="output.wav"
)
synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config, 
    audio_config=audio_config
)
synthesizer.speak_text("Text to synthesize")
```

**C. ElevenLabs API**
```python
import requests

url = "https://api.elevenlabs.io/v1/text-to-speech/YOUR_VOICE_ID"
headers = {
    "xi-api-key": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
data = {
    "text": "Sample text",
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.75
    }
}
response = requests.post(url, json=data, headers=headers)
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### Option 4: VoxCeleb Dataset

**Website**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

**Usage**:
- VoxCeleb1/2 contains real speaker audio
- Use for `data/real/` directory

## File Format Requirements

### Supported Formats
- WAV (recommended)
- MP3
- M4A
- FLAC
- OGG

### Audio Specifications
- **Sample Rate**: 16kHz (will be auto-converted if different)
- **Channels**: Mono (will be auto-converted if stereo)
- **Duration**: 3-30 seconds recommended
- **Bit Depth**: 16-bit or higher

### Converting Audio Files

If you have audio in different formats, convert using FFmpeg:

```bash
# Convert to 16kHz mono WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav

# Batch convert MP3 to WAV
for file in *.mp3; do
  ffmpeg -i "$file" -ar 16000 -ac 1 "${file%.mp3}.wav"
done
```

Or using Python:

```python
import librosa
import soundfile as sf

# Load and resample
audio, sr = librosa.load("input.mp3", sr=16000, mono=True)

# Save as WAV
sf.write("output.wav", audio, sr)
```

## Dataset Statistics

For optimal training, aim for:

| Metric | Target | Minimum |
|--------|--------|---------|
| Total samples | 1000+ | 100+ |
| Real samples | 500+ | 50+ |
| Fake samples | 500+ | 50+ |
| Real/Fake ratio | 1:1 | 1:3 to 3:1 |
| Audio duration | 3-10s | 1-30s |

## Class Balance

The dataset should have roughly equal numbers of real and fake samples:

```python
# Check balance
import os

real_count = len([f for f in os.listdir("data/real") if f.endswith(('.wav', '.mp3', '.m4a'))])
fake_count = len([f for f in os.listdir("data/fake") if f.endswith(('.wav', '.mp3', '.m4a'))])

print(f"Real: {real_count}")
print(f"Fake: {fake_count}")
print(f"Ratio: {real_count/fake_count:.2f}")
```

## Training Data Tips

1. **Diversity**: Use samples from different speakers and microphone qualities
2. **Noise**: Include samples with background noise
3. **Accents**: Include various accents and languages (if applicable)
4. **Quality**: Avoid heavily corrupted or very low-quality files
5. **Augmentation**: System will apply noise and pitch shifts during training

## Evaluation Protocol

After adding data:

```bash
# Train model
python training/train_deepfake.py \
    --data_dir data \
    --epochs 50 \
    --batch_size 16

# Evaluate model
python training/evaluate_deepfake.py \
    --data_dir data \
    --model_path models/checkpoints/deepfake_detector_best.pt
```

Check results in `results/metrics.json`:
- Target Accuracy: >85%
- Target F1-Score: >0.85

## Privacy & Legal

⚠️ **Important**:
- Only use audio you have permission to use
- Respect copyright and privacy laws
- Follow data usage policies of datasets
- ASVspoof and similar datasets have specific usage terms
- Do not distribute copyrighted audio

## Demo Without Training Data

The system can run without real training data using:

1. **Randomly initialized model**: Makes predictions but accuracy will be low
2. **Rule-based scam detection**: Works without training
3. **Pre-generated results**: For demo purposes only

For actual demonstrations, it's recommended to train the model with real data.

---

**Created**: 2026-08-31
**Last Updated**: 2026-08-31
