"""
Dataset utilities for deepfake detection training.
"""
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import librosa
import logging

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """
    Custom dataset for audio files with labels.
    
    Expected directory structure:
    data/
        real/
            audio1.wav
            audio2.wav
            ...
        fake/
            fake1.wav
            fake2.wav
            ...
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mels: int = 128,
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing real/ and fake/ subdirectories
            sample_rate: Target sample rate
            duration: Audio duration in seconds
            n_mels: Number of Mel bands for spectrogram
            transform: Optional data augmentation transform
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.transform = transform
        self.samples = []
        
        # Load file list
        self._load_samples()
    
    def _load_samples(self):
        """Load audio file paths and labels."""
        # Real audio (label = 0)
        real_dir = os.path.join(self.data_dir, "real")
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    self.samples.append((
                        os.path.join(real_dir, file),
                        0  # Label: genuine
                    ))
        
        # Fake audio (label = 1)
        fake_dir = os.path.join(self.data_dir, "fake")
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    self.samples.append((
                        os.path.join(fake_dir, file),
                        1  # Label: fake
                    ))
        
        logger.info(f"Loaded {len(self.samples)} audio samples")
        logger.info(f"Real samples: {sum(1 for _, l in self.samples if l == 0)}")
        logger.info(f"Fake samples: {sum(1 for _, l in self.samples if l == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get audio sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (mel_spectrogram, label)
        """
        audio_path, label = self.samples[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                duration=self.duration
            )
            
            # Pad or truncate to fixed duration
            target_samples = int(self.sample_rate * self.duration)
            if len(audio) < target_samples:
                audio = np.pad(
                    audio,
                    (0, target_samples - len(audio)),
                    mode='constant'
                )
            else:
                audio = audio[:target_samples]
            
            # Extract Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=2048,
                hop_length=512,
                fmin=80,
                fmax=7600
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Apply transform if provided
            if self.transform:
                mel_spec_db = self.transform(mel_spec_db)
            
            return mel_spec_db, label
        
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset audio {audio_path}: {e}") from e


class ASVspoofProtocolDataset(AudioDataset):
    """Audio dataset built from an ASVspoof protocol file and its official labels."""

    def __init__(self, protocol_path, audio_dir, **kwargs):
        self.protocol_path = Path(protocol_path)
        self.audio_dir = Path(audio_dir)
        super().__init__(data_dir="", **kwargs)

    def _load_samples(self):
        if not self.protocol_path.is_file():
            raise FileNotFoundError(f"ASVspoof protocol not found: {self.protocol_path}")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"ASVspoof audio directory not found: {self.audio_dir}")

        missing_files = []
        with self.protocol_path.open("r", encoding="utf-8") as protocol:
            for line_number, line in enumerate(protocol, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 2:
                    raise ValueError(
                        f"Malformed protocol row at {self.protocol_path}:{line_number}"
                    )

                utterance_id = fields[1]
                raw_label = fields[-1].lower().replace("_", "").replace("-", "")
                if raw_label in {"bonafide", "genuine", "real"}:
                    label = 0
                elif raw_label in {"spoof", "fake"}:
                    label = 1
                else:
                    raise ValueError(
                        f"Unrecognized label {fields[-1]!r} at "
                        f"{self.protocol_path}:{line_number}"
                    )

                utterance_path = Path(utterance_id)
                candidates = [self.audio_dir / utterance_path]
                if not utterance_path.suffix:
                    candidates.extend(
                        self.audio_dir / f"{utterance_id}{extension}"
                        for extension in (".flac", ".wav", ".mp3", ".m4a")
                    )
                audio_path = next((path for path in candidates if path.is_file()), None)
                if audio_path is None:
                    missing_files.append(utterance_id)
                    continue

                self.samples.append((str(audio_path), label))

        if missing_files:
            examples = ", ".join(missing_files[:5])
            raise FileNotFoundError(
                f"{len(missing_files)} protocol audio files are missing from "
                f"{self.audio_dir}; examples: {examples}"
            )
        if not self.samples:
            raise ValueError(f"No labeled samples found in {self.protocol_path}")

        class_counts = {label: sum(sample_label == label for _, sample_label in self.samples)
                        for label in (0, 1)}
        if not all(class_counts.values()):
            raise ValueError(
                f"Protocol must include both bona-fide and spoof samples; found {class_counts}"
            )

        logger.info(
            "Loaded %d protocol samples (%d bona-fide, %d spoof) from %s",
            len(self.samples), class_counts[0], class_counts[1], self.protocol_path
        )


class DataSplitter:
    """Split dataset into train, validation, and test sets."""
    
    @staticmethod
    def split_dataset(
        dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Split dataset maintaining class balance.
        
        Args:
            dataset: AudioDataset instance
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        np.random.seed(seed)
        
        # Separate indices by class
        real_indices = [i for i, (_, l) in enumerate(dataset.samples) if l == 0]
        fake_indices = [i for i, (_, l) in enumerate(dataset.samples) if l == 1]
        
        # Shuffle
        np.random.shuffle(real_indices)
        np.random.shuffle(fake_indices)
        
        # Split each class
        def split_class(indices):
            n = len(indices)
            train_n = int(n * train_ratio)
            val_n = int(n * val_ratio)
            
            train = indices[:train_n]
            val = indices[train_n:train_n + val_n]
            test = indices[train_n + val_n:]
            
            return train, val, test
        
        real_train, real_val, real_test = split_class(real_indices)
        fake_train, fake_val, fake_test = split_class(fake_indices)
        
        # Combine
        train_indices = np.concatenate([real_train, fake_train])
        val_indices = np.concatenate([real_val, fake_val])
        test_indices = np.concatenate([real_test, fake_test])
        
        # Shuffle combined sets
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        logger.info(f"Train set: {len(train_indices)} samples")
        logger.info(f"Val set: {len(val_indices)} samples")
        logger.info(f"Test set: {len(test_indices)} samples")
        
        return train_indices, val_indices, test_indices
