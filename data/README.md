# Dataset & Evaluation Protocol Guide

This directory documents dataset standards, protocol validation, and evaluation guidelines for the Deepfake Voice Interceptor project.

---

## 1. Ethical & Academic Standards

- **Do NOT download, fabricate, or distribute proprietary audio datasets without proper licensing.** Users must register and download official datasets (such as ASVspoof 2019) directly under their respective terms.
- **Do NOT manually copy ASVspoof files into arbitrary `data/real` and `data/fake` directories.**
- **Do NOT perform random splits on official corpora.** Random splitting causes speaker leakage across training, development, and evaluation splits, producing falsely inflated accuracy metrics.
- **Synthetic Demo Audio**: Any audio files in `data/demo_audio/` are synthetic test tones generated exclusively for pipeline and unit testing. They do not represent human speech and must not be used to claim model accuracy.

---

## 2. ASVspoof 2019 Protocol Format

Official ASVspoof datasets use structured protocol files to define bona fide (genuine) and spoof (fake) speech samples while ensuring strictly disjoint speaker sets across splits:

### Protocol Columns
```
SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
```
Example entries:
```
LA_0079 LA_D_1047731 - - bonafide
LA_0079 LA_D_1105538 - A07 spoof
LA_0079 LA_D_1125973 - A08 spoof
```

### Official Split Structure
```
data/asvspoof/
├── protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt   # Training protocol
│   ├── ASVspoof2019.LA.cm.dev.trl.txt     # Validation/Dev protocol
│   └── ASVspoof2019.LA.cm.eval.trl.txt    # Held-out Evaluation protocol
└── flac/                                   # Audio files referenced by protocols
    ├── LA_T_1000137.flac
    ├── LA_D_1047731.flac
    └── ...
```

---

## 3. Protocol Validation Tool

Before training or evaluation, run the protocol validator to verify file integrity, confirm references, and inspect class distribution:

```bash
python -m training.validate_protocol --protocol path/to/protocol.txt --audio_dir path/to/audio/
```

### Sample Validator Output:
```
============================================================
ASVspoof Protocol Validation Summary
============================================================
Protocol File : data/sample_protocol.txt
Audio Directory: data/demo_audio
Total Rows    : 2
Valid Entries : 2
Bona Fide     : 1
Spoof         : 1
Missing Files : 0
Valid Status  : PASS
============================================================
```

---

## 4. Evaluation on Held-Out Splits

To evaluate detection performance objectively without inventing accuracy:

```bash
python training/evaluate_deepfake.py \
    --data_dir data/asvspoof \
    --model_path models/checkpoints/deepfake_detector_best.pt
```

Results are logged dynamically to `results/metrics.json` including:
- Accuracy, Precision, Recall, F1-Score
- Area Under ROC Curve (AUC-ROC)
- Equal Error Rate (EER)
