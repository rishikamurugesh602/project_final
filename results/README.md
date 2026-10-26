# Results Directory

This directory stores evaluation results and training artifacts.

## Contents

### Training Results
- `training_history.json` - Epoch-by-epoch training metrics
- `deepfake_detector_best.pt` - Best model checkpoint (linked from models/checkpoints/)

### Evaluation Results  
- `metrics.json` - Final evaluation metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve plot

## Metrics File Format

```json
{
  "accuracy": 0.92,
  "precision": 0.90,
  "recall": 0.94,
  "f1": 0.92,
  "auc_roc": 0.96,
  "confusion_matrix": [
    [47, 3],
    [2, 48]
  ]
}
```

### Interpreting Metrics

| Metric | Good Range | Interpretation |
|--------|-----------|-----------------|
| Accuracy | > 85% | Overall correctness |
| Precision | > 85% | False positive rate |
| Recall | > 85% | False negative rate |
| F1-Score | > 0.85 | Balance of precision/recall |
| AUC-ROC | > 0.85 | Discriminative ability |

## Visualization Files

### Confusion Matrix (PNG)
Shows classification results:
- True Negatives (TN) - Top-left
- False Positives (FP) - Top-right
- False Negatives (FN) - Bottom-left
- True Positives (TP) - Bottom-right

### ROC Curve (PNG)
Plots true positive rate vs false positive rate:
- Closer to top-left = better model
- Area under curve (AUC) indicates performance
- Diagonal line = random classifier (50% accuracy)

## Viewing Results

```python
import json
import matplotlib.pyplot as plt
from PIL import Image

# Load metrics
with open('results/metrics.json') as f:
    metrics = json.load(f)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1-Score: {metrics['f1']:.4f}")

# View confusion matrix
img = Image.open('results/confusion_matrix.png')
plt.imshow(img)
plt.show()

# View ROC curve
img = Image.open('results/roc_curve.png')
plt.imshow(img)
plt.show()
```

## Generating Results

### After Training
```bash
python training/train_deepfake.py --data_dir data
# Generates: training_history.json
```

### After Evaluation
```bash
python training/evaluate_deepfake.py --data_dir data
# Generates: metrics.json, confusion_matrix.png, roc_curve.png
```

## Result Interpretation

### Great Results (>90%)
- Model is well-trained
- Good generalization
- Ready for deployment evaluation

### Good Results (85-90%)
- Model works reasonably well
- May need more data or training
- Consider data augmentation

### Poor Results (<85%)
- Model needs improvement
- Check data quality
- Increase training time
- Try different architecture

### Very Poor (<50%)
- Check data labels (may be reversed)
- Verify data loading
- Check for data corruption
- Review training process

---

**Created**: 2026-08-31
