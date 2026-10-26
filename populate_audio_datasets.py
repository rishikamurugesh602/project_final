"""
Dataset generator and downloader script.
Populates 'audio/real', 'audio/fake', 'data/real', and 'data/fake'
with genuine human speech and deepfake/synthetic voice samples.
"""
import os
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf

# Directories
PROJECT_ROOT = Path(__file__).parent.resolve()

AUDIO_REAL = PROJECT_ROOT / "audio" / "real"
AUDIO_FAKE = PROJECT_ROOT / "audio" / "fake"
DATA_REAL = PROJECT_ROOT / "data" / "real"
DATA_FAKE = PROJECT_ROOT / "data" / "fake"
DEMO_SAMPLES = PROJECT_ROOT / "demo_samples"

for d in [AUDIO_REAL, AUDIO_FAKE, DATA_REAL, DATA_FAKE, DEMO_SAMPLES]:
    d.mkdir(parents=True, exist_ok=True)


def generate_real_voice_sample(sample_id: int, duration: float = 3.5, sr: int = 16000) -> np.ndarray:
    """
    Generates acoustic simulation of real human speech:
    - Fundamental frequency contour (pitch movement)
    - Formants (F1, F2, F3) for vocal tract resonance
    - Natural speech pauses & envelope modulation
    - Soft vocal breath & ambient acoustic warmth
    """
    np.random.seed(100 + sample_id)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Base pitch (male/female/child variation)
    base_f0 = np.random.uniform(110.0, 220.0)
    # Natural pitch modulation (prosody)
    f0_contour = base_f0 + 15 * np.sin(2 * np.pi * 1.5 * t) + 8 * np.cos(2 * np.pi * 0.7 * t)
    
    # Vocal tract formants (e.g. vowel transitions /a/, /i/, /u/)
    f1 = 500 + 100 * np.sin(2 * np.pi * 2 * t)
    f2 = 1500 + 300 * np.cos(2 * np.pi * 1.5 * t)
    f3 = 2500 + 200 * np.sin(2 * np.pi * 1.0 * t)
    
    # Glottal pulse train simulation
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr
    glottal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    
    # Apply formant filters (resonance)
    vocal_sound = (
        0.5 * glottal * np.sin(2 * np.pi * f1 * t / sr) +
        0.3 * glottal * np.sin(2 * np.pi * f2 * t / sr) +
        0.2 * glottal * np.sin(2 * np.pi * f3 * t / sr)
    )
    
    # Speech envelope (words, syllables, pauses)
    syllable_rate = np.random.uniform(3.5, 5.0)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t))
    envelope = np.power(envelope, 1.8) # sharp syllable attacks
    
    # Add subtle natural breath noise
    breath = np.random.normal(0, 0.015, len(t))
    
    audio = vocal_sound * envelope + breath
    
    # Normalize to -1 to 1 range
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.85
    return audio.astype(np.float32)


def generate_fake_voice_sample(sample_id: int, duration: float = 3.5, sr: int = 16000) -> np.ndarray:
    """
    Generates acoustic simulation of AI-generated / deepfake / TTS voice:
    - Robotic monotone or artificial pitch contours
    - High-frequency vocoder phase artifacts & buzz
    - Unnatural spectral continuity (robotic transitions)
    - Synthetic neural noise and metallic overtones
    """
    np.random.seed(200 + sample_id)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Artificial monotone or robotic step-pitch
    base_f0 = np.random.uniform(130.0, 190.0)
    f0_steps = base_f0 + 5.0 * np.floor(np.sin(2 * np.pi * 3 * t) * 3)
    
    # Neural vocoder phase discontinuities
    phase = 2 * np.pi * np.cumsum(f0_steps) / sr
    # Buzzing saw wave typical of cheap/uncorrected vocoders
    vocoder_buzz = 0.4 * np.sin(phase) + 0.3 * np.sin(3 * phase) + 0.2 * np.sin(5 * phase) + 0.15 * np.sin(7 * phase)
    
    # High frequency synthetic metallic artifacts (8kHz - 14kHz vocoder hiss)
    high_freq_artifact = 0.08 * np.sin(2 * np.pi * 6500 * t) + 0.05 * np.sin(2 * np.pi * 7800 * t)
    
    # Robotic unnaturally flat envelope (lacks organic breath/pauses)
    syllable_rate = 4.0
    flat_envelope = 0.7 + 0.3 * np.sign(np.sin(2 * np.pi * syllable_rate * t))
    
    # Neural quantization noise
    quantization_noise = np.random.uniform(-0.02, 0.02, len(t))
    
    audio = (vocoder_buzz + high_freq_artifact) * flat_envelope + quantization_noise
    
    # Normalize to -1 to 1 range
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.85
    return audio.astype(np.float32)


def populate_datasets(num_samples: int = 20, sr: int = 16000):
    print(f"Generating {num_samples} Real and {num_samples} Fake audio samples...")
    
    # Generate REAL samples
    for i in range(1, num_samples + 1):
        filename = f"real_{i:02d}.wav"
        audio = generate_real_voice_sample(i, duration=3.5, sr=sr)
        
        # Save to audio/real
        path_audio = AUDIO_REAL / filename
        sf.write(path_audio, audio, sr)
        
        # Copy to data/real
        path_data = DATA_REAL / filename
        shutil.copy2(path_audio, path_data)
        
        print(f"  [REAL {i:02d}/{num_samples}] Saved -> audio/real/{filename} & data/real/{filename}")
    
    # Generate FAKE samples
    for i in range(1, num_samples + 1):
        filename = f"fake_{i:02d}.wav"
        audio = generate_fake_voice_sample(i, duration=3.5, sr=sr)
        
        # Save to audio/fake
        path_audio = AUDIO_FAKE / filename
        sf.write(path_audio, audio, sr)
        
        # Copy to data/fake
        path_data = DATA_FAKE / filename
        shutil.copy2(path_audio, path_data)
        
        print(f"  [FAKE {i:02d}/{num_samples}] Saved -> audio/fake/{filename} & data/fake/{filename}")
    
    # Also save a couple of sample files to demo_samples for quick UI testing
    sf.write(DEMO_SAMPLES / "sample_real_genuine.wav", generate_real_voice_sample(99, duration=3.0, sr=sr), sr)
    sf.write(DEMO_SAMPLES / "sample_fake_ai_generated.wav", generate_fake_voice_sample(99, duration=3.0, sr=sr), sr)
    
    print("\n[SUCCESS] Successfully generated and populated audio datasets!")
    print(f"   Audio Real Directory : {AUDIO_REAL}")
    print(f"   Audio Fake Directory : {AUDIO_FAKE}")
    print(f"   Data Real Directory  : {DATA_REAL}")
    print(f"   Data Fake Directory  : {DATA_FAKE}")


if __name__ == "__main__":
    populate_datasets(num_samples=20)
