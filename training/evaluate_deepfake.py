"""
Evaluation script for deepfake detector model.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.deepfake_detector import DeepfakeDetectorModel
from training.dataset import AudioDataset, ASVspoofProtocolDataset, DataSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for model assessment."""
    
    def __init__(self, model, device):
        """Initialize evaluator."""
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for mel_specs, labels in test_loader:
                mel_specs = mel_specs.unsqueeze(1).to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(mel_specs)
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auc_roc': roc_auc_score(all_labels, all_probs),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        return metrics, all_preds, all_labels, all_probs
    
    def plot_confusion_matrix(self, cm, save_path='results/confusion_matrix.png'):
        """Plot and save confusion matrix."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Genuine', 'Fake'])
        plt.yticks(tick_marks, ['Genuine', 'Fake'])
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, labels, probs, save_path='results/roc_curve.png'):
        """Plot and save ROC curve."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Saved ROC curve to {save_path}")
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate deepfake detector')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--dataset-format', choices=['folders', 'asvspoof'], default='folders',
                        help='Use legacy real/fake folders or an official ASVspoof protocol file')
    parser.add_argument('--protocol', type=str,
                        help='Labeled ASVspoof evaluation protocol file')
    parser.add_argument('--audio-dir', type=str,
                        help='Directory containing audio referenced by the evaluation protocol')
    parser.add_argument('--model_path', type=str, default='models/checkpoints/deepfake_detector_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check paths
    if args.dataset_format == 'asvspoof' and (not args.protocol or not args.audio_dir):
        parser.error('ASVspoof mode requires --protocol and --audio-dir')
    if args.dataset_format == 'folders' and not os.path.exists(args.data_dir):
        logger.error(f"Dataset directory not found: {args.data_dir}")
        return
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading dataset...")
    if args.dataset_format == 'asvspoof':
        test_set = ASVspoofProtocolDataset(args.protocol, args.audio_dir)
    else:
        dataset = AudioDataset(args.data_dir)
        if len(dataset) == 0:
            logger.error("Dataset is empty!")
            return
        _, _, test_indices = DataSplitter.split_dataset(dataset)
        test_set = Subset(dataset, test_indices)

    logger.info("Evaluation samples: %d", len(test_set))
    
    # Create data loader
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = DeepfakeDetectorModel()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Evaluate
    logger.info("Evaluating model...")
    evaluator = Evaluator(model, device)
    metrics, preds, labels, probs = evaluator.evaluate(test_loader)
    
    # Log metrics
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    logger.info("="*50 + "\n")
    
    # Save metrics
    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(cm, cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(args.results_dir, 'roc_curve.png')
    evaluator.plot_roc_curve(labels, probs, roc_path)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
