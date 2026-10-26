"""
Training script for deepfake detector model.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse
import logging
from datetime import datetime

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.deepfake_detector import DeepfakeDetectorModel
from training.dataset import AudioDataset, ASVspoofProtocolDataset, DataSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for deepfake detector."""
    
    def __init__(
        self,
        model,
        device,
        learning_rate=0.001,
        weight_decay=1e-5
    ):
        """Initialize trainer."""
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (mel_specs, labels) in enumerate(train_loader):
            mel_specs = mel_specs.unsqueeze(1).to(self.device)  # Add channel dimension
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(mel_specs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for mel_specs, labels in val_loader:
                mel_specs = mel_specs.unsqueeze(1).to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(mel_specs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs=50,
        save_dir="models/checkpoints"
    ):
        """Train for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            logger.info(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'training_history': self.training_history
                }
                model_path = os.path.join(save_dir, "deepfake_detector_best.pt")
                torch.save(checkpoint, model_path)
                logger.info(f"Saved best model to {model_path}")
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        return self.training_history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train deepfake detector')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--dataset-format', choices=['folders', 'asvspoof'], default='folders',
                        help='Use legacy real/fake folders or official ASVspoof protocol files')
    parser.add_argument('--train-protocol', type=str,
                        help='Labeled ASVspoof training protocol file')
    parser.add_argument('--train-audio-dir', type=str,
                        help='Directory containing audio referenced by the training protocol')
    parser.add_argument('--dev-protocol', type=str,
                        help='Labeled ASVspoof development protocol file')
    parser.add_argument('--dev-audio-dir', type=str,
                        help='Directory containing audio referenced by the development protocol')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='models/checkpoints',
                        help='Directory to save model')
    
    args = parser.parse_args()
    
    if args.dataset_format == 'asvspoof':
        required = {
            '--train-protocol': args.train_protocol,
            '--train-audio-dir': args.train_audio_dir,
            '--dev-protocol': args.dev_protocol,
            '--dev-audio-dir': args.dev_audio_dir,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error('ASVspoof mode requires ' + ', '.join(missing))
    elif not os.path.exists(args.data_dir):
        logger.error(f"Dataset directory not found: {args.data_dir}")
        logger.info("Please create 'data/real' and 'data/fake' directories with audio files")
        return
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading dataset...")
    if args.dataset_format == 'asvspoof':
        train_set = ASVspoofProtocolDataset(args.train_protocol, args.train_audio_dir)
        val_set = ASVspoofProtocolDataset(args.dev_protocol, args.dev_audio_dir)
    else:
        dataset = AudioDataset(args.data_dir)
        if len(dataset) == 0:
            logger.error("Dataset is empty!")
            return
        train_indices, val_indices, _ = DataSplitter.split_dataset(dataset)
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

    logger.info("Training samples: %d; development samples: %d", len(train_set), len(val_set))
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = DeepfakeDetectorModel()
    model.to(device)
    
    # Create trainer
    trainer = Trainer(model, device, learning_rate=args.learning_rate)
    
    # Train
    trainer.train(train_loader, val_loader, epochs=args.epochs, save_dir=args.save_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
