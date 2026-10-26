"""
Deepfake voice detection model.
Implements CNN + GRU architecture for voice spoofing detection.
Based on RawNet2-style anti-spoofing approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from app.utils.config import config
from app.utils.logging import logger


class DeepfakeDetectorModel(nn.Module):
    """
    CNN + GRU architecture for deepfake voice detection.
    
    Architecture:
    Mel Spectrogram -> CNN -> GRU -> Attention -> FC -> Sigmoid
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        num_gru_hidden: int = 64,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2  # Binary: real (0) or fake (1)
    ):
        """
        Initialize deepfake detector.
        
        Args:
            n_mels: Number of Mel spectrogram bands
            num_gru_hidden: Number of hidden units in GRU
            num_gru_layers: Number of GRU layers
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.num_gru_hidden = num_gru_hidden
        self.num_gru_layers = num_gru_layers
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.cnn_out_mels = n_mels // 8
        self.cnn_out_channels = 128
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=self.cnn_out_channels * self.cnn_out_mels,
            hidden_size=num_gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(num_gru_hidden * 2, 1)
        
        # Classification head
        self.fc1 = nn.Linear(num_gru_hidden * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input Mel spectrogram (batch, 1, n_mels, time_steps)
        
        Returns:
            Output logits (batch, num_classes)
        """
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Reshape for GRU: (batch, time_steps, channels * mels)
        batch_size, channels, mels, time_steps = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channels, mels)
        x = x.reshape(batch_size, time_steps, -1)  # (batch, time_steps, channels * mels)
        
        # GRU temporal modeling
        x, _ = self.gru(x)  # (batch, time_steps, num_gru_hidden * 2)
        
        # Attention mechanism
        attention_weights = self.attention(x)  # (batch, time_steps, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        x_attended = torch.sum(x * attention_weights, dim=1)  # (batch, num_gru_hidden * 2)
        
        # Classification head
        x = self.fc1(x_attended)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class DeepfakeDetector:
    """Wrapper for deepfake detection with model management."""
    
    def __init__(
        self,
        device: str = config.MODEL_DEVICE,
        model_path: Optional[str] = None
    ):
        """
        Initialize detector.
        
        Args:
            device: Device to use (cpu/cuda)
            model_path: Path to pretrained model checkpoint
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.status_message = "Untrained / Random Weights"
        
        # Create model architecture
        self.model = DeepfakeDetectorModel().to(device)
        
        # Try to load pretrained weights if provided
        if model_path and self._check_model_exists(model_path):
            self.load_model(model_path)
        else:
            self.is_loaded = False
            self.status_message = "Untrained / Random Weights (Demo Only - No Verified Checkpoint)"
            logger.warning(
                f"Model checkpoint not found or not provided at: {model_path}. "
                "Operating with untrained random weights (demo only)."
            )
    
    def _check_model_exists(self, path: str) -> bool:
        """Check if model file exists."""
        import os
        return os.path.exists(path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.is_loaded = True
            self.status_message = f"Checkpoint loaded: {model_path}"
            logger.info(f"Loaded deepfake detector from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            self.is_loaded = False
            self.status_message = f"Failed to load checkpoint ({str(e)})"
    
    def predict(
        self,
        mel_spectrogram: np.ndarray,
        return_confidence: bool = True
    ) -> dict:
        """
        Predict if audio is deepfake.
        
        Args:
            mel_spectrogram: Mel spectrogram (n_mels, time_steps)
            return_confidence: Return confidence scores
        
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Add batch and channel dimensions
                x = torch.from_numpy(mel_spectrogram).unsqueeze(0).unsqueeze(0).float()
                x = x.to(self.device)
                
                # Forward pass
                logits = self.model(x)
                probabilities = F.softmax(logits, dim=1)
                
                confidence = float(probabilities[0, 1].item())  # Probability of being fake
                
                return {
                    "deepfake_probability": float(confidence),
                    "is_deepfake": bool(confidence > config.DEEPFAKE_THRESHOLD),
                    "prediction_class": "AI-Generated" if confidence > 0.5 else "Genuine",
                    "confidence_scores": {
                        "genuine": float(probabilities[0, 0].item()),
                        "ai_generated": float(probabilities[0, 1].item())
                    },
                    "model_loaded": self.is_loaded,
                    "model_status": self.status_message
                }
        except Exception as e:
            logger.error(f"Error in deepfake prediction: {str(e)}")
            return {
                "deepfake_probability": 0.5,
                "is_deepfake": False,
                "prediction_class": "Unknown (Error)",
                "error": str(e),
                "model_loaded": self.is_loaded,
                "model_status": f"Inference error: {str(e)}"
            }
    
    def save_model(self, output_path: str, metadata: dict = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            output_path: Path to save model
            metadata: Additional metadata to save
        """
        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'metadata': metadata or {}
            }
            torch.save(checkpoint, output_path)
            logger.info(f"Saved model to {output_path}")
        except Exception as e:
            logger.error(f"Error saving model to {output_path}: {str(e)}")
