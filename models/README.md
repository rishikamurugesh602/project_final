# Model Checkpoints

This directory stores trained model weights and checkpoints.

## Directory Structure

```
models/
└── checkpoints/
    ├── deepfake_detector_best.pt          # Best deepfake detector checkpoint
    └── training_history.json              # Training metrics
```

## Model Files

### deepfake_detector_best.pt

PyTorch model checkpoint containing:
- Model architecture weights
- Training history
- Validation metrics
- Hyperparameters

**Size**: ~50-100MB

**How to create**:
```bash
python training/train_deepfake.py \
    --data_dir data \
    --epochs 50 \
    --save_dir models/checkpoints
```

### training_history.json

JSON file with training metrics:
```json
{
  "train_loss": [...],
  "val_loss": [...],
  "train_acc": [...],
  "val_acc": [...]
}
```

## Using Pre-trained Models

The system can use pre-trained models if available:

1. Place checkpoint in `models/checkpoints/`
2. Configure path in `.env`:
   ```
   DEEPFAKE_MODEL_PATH=models/checkpoints/deepfake_detector_best.pt
   ```
3. Run application - model will be loaded automatically

## Training New Models

```bash
# Basic training
python training/train_deepfake.py

# With custom parameters
python training/train_deepfake.py \
    --data_dir data \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --device cuda
```

## Evaluation

```bash
python training/evaluate_deepfake.py \
    --model_path models/checkpoints/deepfake_detector_best.pt \
    --data_dir data
```

Output files in `results/`:
- `metrics.json` - Evaluation metrics
- `confusion_matrix.png` - Visualization
- `roc_curve.png` - ROC curve

## Model Architecture

The deepfake detector uses:

```
Input (Mel Spectrogram)
  ↓
Conv2D + BatchNorm + ReLU
  ↓
MaxPool2D
  ↓
Conv2D + BatchNorm + ReLU
  ↓
MaxPool2D
  ↓
Conv2D + BatchNorm + ReLU
  ↓
MaxPool2D
  ↓
GRU (Bidirectional)
  ↓
Attention Pooling
  ↓
Dense(64) + ReLU + Dropout
  ↓
Dense(2) - Logits
  ↓
Softmax → [P(Genuine), P(Fake)]
```

## Inference

```python
from app.models.deepfake_detector import DeepfakeDetector
from app.services.audio_processor import AudioProcessor

# Load model
detector = DeepfakeDetector(
    device='cpu',
    model_path='models/checkpoints/deepfake_detector_best.pt'
)

# Load and preprocess audio
processor = AudioProcessor()
audio, sr = processor.load_audio('sample.wav')
mel_spec = processor.extract_mel_spectrogram(audio)

# Predict
result = detector.predict(mel_spec)
print(f"Deepfake probability: {result['deepfake_probability']:.2%}")
print(f"Prediction: {result['prediction_class']}")
```

## Transfer Learning

To fine-tune a pre-trained model:

```python
import torch
from app.models.deepfake_detector import DeepfakeDetectorModel

# Load checkpoint
model = DeepfakeDetectorModel()
checkpoint = torch.load('models/checkpoints/deepfake_detector_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune (unfreeze some layers)
for param in model.gru.parameters():
    param.requires_grad = True

for param in model.fc1.parameters():
    param.requires_grad = True

# Continue training with lower learning rate
```

## Model Size & Performance

| Device | Speed | Memory |
|--------|-------|--------|
| CPU | ~0.5-1s/sample | ~500MB |
| GPU (NVIDIA) | ~50-100ms/sample | ~2-3GB |

## Troubleshooting

### Model not loading
```
FileNotFoundError: Model file not found
```
- Check path in `.env`
- Train model first: `python training/train_deepfake.py --data_dir data`

### Out of memory
- Use smaller batch size
- Use CPU instead of GPU
- Reduce model complexity

### Poor accuracy
- Check training data quality
- Increase epochs
- Try data augmentation
- Use different learning rate

---

**Created**: 2026-08-31
