# Dataset & Demo Samples Guide

This guide describes how to configure demo test samples or utilize real voice datasets (such as ASVspoof 2019/2021) for training and rigorous held-out evaluation.

---

## ⚠️ Important Dataset Integrity & Licensing Notice

1. **No Proprietary Data Bundled**: This repository **does not bundle, download, or fabricate official ASVspoof data**. Researchers and users must obtain the official ASVspoof corpora directly from [https://www.asvspoof.org/](https://www.asvspoof.org/) subject to their research licensing terms.
2. **Demo Samples Are NOT Real Speech**: Any generated audio in `data/` or `demo_samples/` created by `populate_audio_datasets.py` consists of synthetic sinusoidal tones and harmonic simulations designed **solely for pipeline smoke testing, code compilation verification, and UI wiring**. They are **not real human speech and must never be cited as evidence of model accuracy**.
3. **No Fabricated Metrics**: Genuine evaluation requires running the evaluation pipeline against a separate labeled held-out protocol split (e.g. `ASVspoof2019.LA.cm.eval.trl.txt`). Never report random-weight or synthetic-tone results as benchmark detection performance.

---

## 📋 Evaluation & Usage Modes

| Mode | Purpose | Real Speech? | Evidentiary Value |
|------|---------|-------------|-------------------|
| **Demo / Local Test Mode** | UI verification & pipeline smoke-test | No (synthetic test tones) | Smoke-test only |
| **Official ASVspoof Protocol** | Academic training & evaluation | Yes (bonafide & cloned speech) | Scientifically valid benchmark |

---

## ASVspoof Protocol-Based Training & Evaluation

For meaningful deepfake anti-spoofing experiments, use the official protocol-based loader.

### Directory Organization

Keep the official partitions intact after extraction. **Do not copy files into flat `data/real` and `data/fake` folders**, and do not randomly re-split the official corpus:

```text
ASVspoof2019_LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

### 1. Validate Dataset Protocol and Files

Before training or evaluation, run the validation tool to detect missing audio files, corrupt rows, or class imbalances:

```powershell
python -m training.validate_protocol `
  --protocol "ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt" `
  --audio-dir "ASVspoof2019_LA/ASVspoof2019_LA_dev/flac"
```

The validator reports:
- Total protocol entries
- Bona-fide vs spoof class counts
- Verified audio files on disk
- Missing or malformed file entries

### 2. Train on Official Train and Dev Partitions

Train the CNN+GRU model using the official training protocol and validate on the dev set:

```powershell
python -m training.train_deepfake --dataset-format asvspoof `
  --train-protocol "PATH_TO/ASVspoof2019.LA.cm.train.trn.txt" `
  --train-audio-dir "PATH_TO/ASVspoof2019_LA_train/flac" `
  --dev-protocol "PATH_TO/ASVspoof2019.LA.cm.dev.trl.txt" `
  --dev-audio-dir "PATH_TO/ASVspoof2019_LA_dev/flac" `
  --epochs 30 --batch_size 16 --device cpu
```

### 3. Evaluate on Held-Out Split Only

Evaluate the saved model checkpoint strictly on a held-out evaluation partition:

```powershell
python -m training.evaluate_deepfake --dataset-format asvspoof `
  --protocol "PATH_TO/ASVspoof2019.LA.cm.eval.trl.txt" `
  --audio-dir "PATH_TO/ASVspoof2019_LA_eval/flac" `
  --model_path "models/checkpoints/deepfake_detector_best.pt"
```

Metrics computed:
- Accuracy, Precision, Recall, F1-Score
- Equal Error Rate (EER) / AUC-ROC
- Confusion Matrix (saved to `results/confusion_matrix.png`)

---

## Demo Fallback Mode (Smoke-Testing Only)

If you do not yet have access to ASVspoof 2019:
1. Run `python populate_audio_datasets.py` to generate lightweight synthetic tones.
2. Start the backend (`python start_backend.py`) and dashboard (`python start_dashboard.py`).
3. Use the **Upload Audio** or **Demo Samples** tab to verify that the waveform loading, Mel spectrogram extraction, FastAPI endpoints, and Streamlit charts execute without errors.
4. Note that the dashboard will display an explicit status warning:
   *"⚠️ Model checkpoint not found or using untrained weights — predictions are demo-only and not reliable."*
