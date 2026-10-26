# Dataset & Demo Samples Guide

**Quick Answer**: You have **3 options** - pick what works best for you!

---

## 📋 Quick Decision Matrix

| Option | Effort | Quality | Training Needed | Best For |
|--------|--------|---------|-----------------|----------|
| **Option A: Demo Mode** | ⭐ (5 min) | Medium | No | Quick testing |
| **Option B: Generate Samples** | ⭐⭐ (15 min) | Low | No | Rapid prototyping |
| **Option C: Real Audio Data** | ⭐⭐⭐⭐ (1-2 hrs) | High | Yes | Production demo |

---

## Option A: Demo Mode (Easiest - 5 minutes)

**No audio files needed!** The system works as-is:
- Dashboard will still function
- All components will work
- Will show real analysis results
- Deepfake detector uses random weights (lower accuracy)

### Steps:
1. Just use the dashboard with the **"Demo Samples" tab**
2. Click **"Generate Sample Audio"** button
3. System creates a test file automatically
4. Click **"Analyze Demo Sample"**

**Result**: ✅ Working demo, no setup needed

---

## Option B: Generate Synthetic Audio (15 minutes)

**Create fake audio files programmatically** to populate demo_samples/.

### Method 1: Using Python Script
```python
# Save as: generate_demo_audio.py
import numpy as np
import soundfile as sf
import os

os.makedirs("demo_samples", exist_ok=True)

# Generate 3 test audio files
for i in range(3):
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, int(sr*duration))
    
    # Create "speech-like" audio with various frequencies
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)      # Base frequency
    audio += 0.2 * np.sin(2 * np.pi * 200 * t)     # Lower frequency
    audio += 0.1 * np.sin(2 * np.pi * 600 * t)     # Higher frequency
    audio += np.random.normal(0, 0.02, len(audio)) # Noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Save
    filename = f"demo_samples/synthetic_{i+1}.wav"
    sf.write(filename, audio, sr)
    print(f"✓ Created {filename}")

print("\n✅ Demo audio samples created!")
```

### Method 2: Using Streamlit UI
1. Open dashboard
2. Go to "Demo Samples" tab
3. Click "Generate Sample Audio"
4. Done! ✅

### Result:
- Creates synthetic .wav files
- Good for testing UI and API
- **Not representative of real speech**

---

## Option C: Real Audio Data (Best - 1-2 hours)

**Use actual audio files for training and accurate detection.**

### 3 Ways to Get Real Audio:

#### C.1: Download Public Datasets

**ASVspoof 2019** (Recommended - Best for deepfake detection)
```bash
# Download (requires registration at: https://www.asvspoof.org/)
# After registration and download:

# Extract the archive while preserving its train/dev/evaluation partitions
# and protocol files. Do not copy files into data/real and data/fake.
# See "ASVspoof Protocol-Based Training" below for the project commands.
```

**TIMIT Dataset**
```bash
# Download from: https://catalog.ldc.upenn.edu/LDC93S1
# Requires institutional access or purchase

# After download:
cp TIMIT/TRAIN/DR*/*/*.wav data/real/
# Use TTS to generate fakes for data/fake/
```

**VoxCeleb**
```bash
# Download from: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

# After download:
mkdir -p data/real
# Extract all speaker videos and convert to audio
ffmpeg -i video.mp4 -q:a 9 -n audio.wav
mv audio.wav data/real/
```

#### C.2: Use Microphone Recording

**Record real speech samples:**

```python
# Save as: record_samples.py
import sounddevice as sd
import soundfile as sf
import os

os.makedirs("data/real", exist_ok=True)

print("Recording 5 samples of YOUR voice...")
sr = 16000

for i in range(5):
    print(f"\n📢 Recording {i+1}/5 - Say something for 3 seconds...")
    audio = sd.rec(int(3 * sr), samplerate=sr, channels=1)
    sd.wait()
    
    filename = f"data/real/voice_sample_{i+1}.wav"
    sf.write(filename, audio, sr)
    print(f"✓ Saved {filename}")

print("\n✅ Voice recordings complete!")
```

**Install recording capability:**
```bash
pip install sounddevice
```

#### C.3: Download from Online Sources

Find Creative Commons licensed audio:
- **Pixabay**: https://pixabay.com/sound-effects/
- **FreeSound**: https://freesound.org/
- **YouTube Audio Library**: https://www.youtube.com/audiolibrary

```bash
# Create directories
mkdir -p data/real data/fake demo_samples

# Place downloaded .wav/.mp3 files in these directories
# Real speech → data/real/
# Deepfake/synthetic speech → data/fake/
# Test samples → demo_samples/
```

---

## Directory Structure

## ASVspoof Protocol-Based Training

For meaningful spoof-detection experiments, use a properly sourced corpus such as
ASVspoof 2019 Logical Access (LA), subject to its access and usage terms. Do not
use the generated tone simulations in `data/real` and `data/fake` as evidence of
real-world accuracy.

Keep the official partitions separate after extracting the corpus. The exact
folder names may vary by archive; provide the matching audio and labeled protocol
paths explicitly:

```text
ASVspoof2019_LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── <labeled-train-protocol.txt>
    ├── <labeled-dev-protocol.txt>
    └── <labeled-evaluation-protocol.txt>
```

Protocol rows must contain the utterance ID in the second column and a
`bonafide` or `spoof` label in the final column. The loader accepts audio IDs
with or without `.flac`/`.wav` extensions and fails if a referenced file is
missing or unlabeled. Some evaluation labels/keys may be distributed separately;
only evaluate with an authorized labeled key, and never use evaluation data for
training or tuning.

Train using the official train and development partitions:

```powershell
python -m training.train_deepfake --dataset-format asvspoof `
  --train-protocol "PATH_TO_TRAIN_PROTOCOL" `
  --train-audio-dir "PATH_TO_TRAIN_FLAC" `
  --dev-protocol "PATH_TO_DEV_PROTOCOL" `
  --dev-audio-dir "PATH_TO_DEV_FLAC" `
  --epochs 30 --batch_size 16 --device cpu
```

Evaluate the resulting checkpoint only on a separate labeled evaluation partition:

```powershell
python -m training.evaluate_deepfake --dataset-format asvspoof `
  --protocol "PATH_TO_LABELED_EVALUATION_PROTOCOL" `
  --audio-dir "PATH_TO_EVALUATION_FLAC" `
  --model_path "models/checkpoints/deepfake_detector_best.pt"
```

Replace each `PATH_TO_...` value with the path from your extraction. The old
`data/real` and `data/fake` random-split workflow remains available for quick
smoke tests, but its metrics are not a substitute for held-out ASVspoof results.

### Option A (Demo Mode - No Files Needed)
```
project_1/
├── app/
├── dashboard/
├── data/
│   ├── real/          ← Empty, no files needed
│   └── fake/          ← Empty, no files needed
└── demo_samples/      ← Empty or auto-generated
```

### Option B (Synthetic Audio)
```
project_1/
├── app/
├── dashboard/
├── data/
│   ├── real/          ← Empty
│   └── fake/          ← Empty
└── demo_samples/
    ├── synthetic_1.wav
    ├── synthetic_2.wav
    └── synthetic_3.wav
```

### Option C (Real Audio - BEST)
```
project_1/
├── app/
├── dashboard/
├── data/
│   ├── real/          ← 50-500 genuine audio files
│   │   ├── speaker1_001.wav
│   │   ├── speaker2_001.wav
│   │   └── ...
│   └── fake/          ← 50-500 deepfake audio files
│       ├── tts_001.wav
│       ├── vocoder_001.wav
│       └── ...
└── demo_samples/      ← 3-5 test samples
    ├── sample_1.wav
    ├── sample_2.wav
    └── sample_3.wav
```

---

## File Format Requirements

### Accepted Audio Formats
- ✅ .wav (recommended)
- ✅ .mp3
- ✅ .m4a
- ✅ .flac
- ✅ .ogg

### Audio Specifications
- **Sample Rate**: 16 kHz (auto-resampled if different)
- **Channels**: Mono (stereo auto-converted)
- **Duration**: 3-30 seconds recommended
- **Bit Depth**: 16-bit (standard)

### File Naming Convention
```
# For training data:
data/real/        speaker_name_or_id_00X.wav
data/fake/        tts_or_synthesis_method_00X.wav

# For demo samples:
demo_samples/     descriptive_sample_name.wav

# Examples:
data/real/speaker_01_001.wav
data/fake/vocoder_gan_001.wav
demo_samples/test_legitimate.wav
```

---

## Training the Model (Optional)

After adding audio data to `data/real/` and `data/fake/`:

```bash
# Train deepfake detector
python training/train_deepfake.py \
  --data_dir data \
  --epochs 50 \
  --batch_size 16 \
  --device cpu

# Evaluate
python training/evaluate_deepfake.py \
  --data_dir data \
  --model_path models/checkpoints/deepfake_detector_best.pt
```

**Expected Duration**:
- CPU: 30 minutes - 2 hours
- GPU: 5-15 minutes

**Results depend on the dataset, split, and model; do not assume an accuracy target.**
The evaluation script creates:
  - `models/checkpoints/deepfake_detector_best.pt`
  - `results/metrics.json`
  - `results/confusion_matrix.png`
  - `results/roc_curve.png`

---

## What to Do Right Now

### Scenario 1: Just want to demo for Review-2
→ **Use Option A (Demo Mode)**
```bash
# Backend already running
streamlit run dashboard/streamlit_app.py
# Use "Generate Sample Audio" in Demo Samples tab
```

### Scenario 2: Want better-looking demo without real data
→ **Use Option B (Synthetic Audio)**
```bash
python -c "
import numpy as np
import soundfile as sf
import os
os.makedirs('demo_samples', exist_ok=True)
for i in range(3):
    sr = 16000
    t = np.linspace(0, 3, sr*3)
    audio = 0.3*np.sin(2*np.pi*440*t) + 0.1*np.random.randn(len(t))
    sf.write(f'demo_samples/sample_{i}.wav', audio/max(abs(audio)), sr)
print('✅ Demo samples created!')
"

streamlit run dashboard/streamlit_app.py
```

### Scenario 3: Want best demo with training
→ **Use Option C (Real Audio)**
```bash
# For ASVspoof, keep the official partitions and use the protocol-based
# training commands in the section above. Do not merge partitions into
# data/real and data/fake or report unverified accuracy.

# Run system after training and evaluation
python start_backend.py
streamlit run dashboard/streamlit_app.py
```

---

## Troubleshooting

### "No demo samples found"
- ✅ Use dashboard "Generate Sample Audio" button, OR
- ✅ Create `demo_samples/` folder and add audio files manually

### Training dataset not found
```bash
# Verify structure
dir data/real/
dir data/fake/

# Should show .wav files in both directories
```

### Audio format not supported
```bash
# Convert to WAV using FFmpeg
ffmpeg -i input.mp3 -acodec pcm_s16le -ar 16000 output.wav
```

### Model not improving (training accuracy)
- Use more data (minimum 100 files per class)
- Use diverse data (different speakers, conditions)
- Try longer training (increase epochs to 100+)

---

## Recommended Setup for Demo

**For best Review-2 demo** (1-2 hours total setup):

1. ✅ **System already set up** - dependencies installed, backend running
2. ⏳ **Option**: Add 10-15 real audio files manually
   - Find 5-10 genuine speech clips
   - Find 5-10 AI-generated speech clips
   - Place in `data/real/` and `data/fake/`
3. ⏳ **Optional**: Train model (30 min on CPU)
4. ✅ **Demo**: Show working UI with results

OR simply:
1. ✅ Run dashboard as-is
2. ✅ Use "Demo Samples" → "Generate Sample Audio"
3. ✅ Analyze and show results

---

## Key Points

✅ **System works without any data** - all components functional  
✅ **Demo samples auto-generated** - no manual file creation required  
✅ **Detection accuracy must be measured on a held-out labeled dataset**  
✅ **Training optional** - system demonstrates full pipeline either way  
✅ **Generated demo audio is not speech and does not establish model accuracy**  

---

**Need help?** See `README.md` or `QUICK_START.md`
