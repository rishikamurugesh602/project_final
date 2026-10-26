"""
Audio preprocessing module.
Handles audio loading, normalization, resampling, and feature extraction.
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
from app.utils.config import config
from app.utils.logging import logger


class AudioProcessor:
    """Audio preprocessing and feature extraction."""
    
    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        chunk_duration: float = config.CHUNK_DURATION,
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate (Hz)
            chunk_duration: Duration of audio chunks (seconds)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
    
    def load_audio(
        self, 
        audio_path: str,
        sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate (defaults to self.sample_rate)
            mono: Convert to mono
        
        Returns:
            Tuple of (audio_array, sample_rate)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If audio is invalid
        """
        if not sr:
            sr = self.sample_rate
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=sr, mono=mono)
            logger.info(f"Loaded audio: {audio_path}, shape={audio.shape}, sr={sr}")
            
            # Validate audio
            if len(audio) == 0:
                raise ValueError("Audio is empty")
            
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {str(e)}")
            raise
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio array
        
        Returns:
            Normalized audio array
        """
        if len(audio) == 0:
            return audio
        
        # Compute max absolute value
        max_val = np.max(np.abs(audio))
        
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def remove_silence(
        self,
        audio: np.ndarray,
        top_db: float = 40,
        ref: float = 1.0
    ) -> np.ndarray:
        """
        Remove or trim silence from audio.
        
        Args:
            audio: Audio array
            top_db: Threshold for silence detection (dB)
            ref: Reference amplitude
        
        Returns:
            Audio with silence trimmed
        """
        try:
            # Trim silence from edges
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db, ref=ref)
            logger.debug(f"Removed silence: {len(audio)} -> {len(audio_trimmed)} samples")
            return audio_trimmed
        except Exception as e:
            logger.warning(f"Error trimming silence: {str(e)}")
            return audio
    
    def apply_noise_reduction(
        self,
        audio: np.ndarray,
        duration_sec: float = 1.0
    ) -> np.ndarray:
        """
        Apply simple noise reduction using spectral gating.
        
        Args:
            audio: Audio array
            duration_sec: Duration of noise profile (seconds)
        
        Returns:
            Noise-reduced audio
        """
        try:
            # For a real implementation, you would use specialized noise reduction
            # For now, we implement a simple approach
            
            # Compute STFT
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Estimate noise profile from first duration_sec
            noise_frames = int(duration_sec * self.sample_rate / 512)  # 512 is default hop length
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Apply spectral gating
            mask = magnitude > (noise_profile * 2)  # Simple threshold
            magnitude_denoised = magnitude * mask.astype(float)
            
            # Reconstruct
            D_denoised = magnitude_denoised * np.exp(1j * phase)
            audio_denoised = librosa.istft(D_denoised)
            
            logger.debug("Applied noise reduction")
            return audio_denoised
        except Exception as e:
            logger.warning(f"Error applying noise reduction: {str(e)}")
            return audio
    
    def preprocess(
        self,
        audio: np.ndarray,
        sr: int,
        remove_silence_flag: bool = True,
        noise_reduction_flag: bool = True,
    ) -> np.ndarray:
        """
        Complete audio preprocessing pipeline.
        
        Args:
            audio: Raw audio array
            sr: Sample rate of input audio
            remove_silence_flag: Whether to remove silence
            noise_reduction_flag: Whether to apply noise reduction
        
        Returns:
            Preprocessed audio array
        """
        logger.info("Starting audio preprocessing")
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            logger.debug(f"Resampled from {sr} to {self.sample_rate} Hz")
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Remove silence
        if remove_silence_flag:
            audio = self.remove_silence(audio)
        
        # Apply noise reduction
        if noise_reduction_flag and config.ENABLE_NOISE_REDUCTION:
            audio = self.apply_noise_reduction(audio)
        
        logger.info(f"Preprocessing complete: {len(audio)} samples")
        return audio
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 80,
        f_max: float = 7600
    ) -> np.ndarray:
        """
        Extract Mel spectrogram from audio.
        
        Args:
            audio: Audio array
            n_mels: Number of Mel bands
            n_fft: FFT window size
            hop_length: Hop length
            f_min: Minimum frequency
            f_max: Maximum frequency
        
        Returns:
            Mel spectrogram (n_mels, time_steps)
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=f_min,
                fmax=f_max
            )
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            logger.debug(f"Extracted Mel spectrogram: {mel_spec_db.shape}")
            return mel_spec_db
        except Exception as e:
            logger.error(f"Error extracting Mel spectrogram: {str(e)}")
            raise
    
    def chunk_audio(self, audio: np.ndarray) -> list:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Audio array
        
        Returns:
            List of audio chunks
        """
        chunks = []
        stride = self.chunk_samples // 2  # 50% overlap
        
        for i in range(0, len(audio), stride):
            chunk = audio[i:i + self.chunk_samples]
            if len(chunk) < self.chunk_samples:
                # Pad last chunk if necessary
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
            chunks.append(chunk)
        
        logger.debug(f"Split audio into {len(chunks)} chunks")
        return chunks
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sr: int = None
    ) -> None:
        """
        Save audio to file.
        
        Args:
            audio: Audio array
            output_path: Path to save audio
            sr: Sample rate
        """
        if sr is None:
            sr = self.sample_rate
        
        try:
            sf.write(output_path, audio, sr)
            logger.info(f"Saved audio to {output_path}")
        except Exception as e:
            logger.error(f"Error saving audio to {output_path}: {str(e)}")
            raise
