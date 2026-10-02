"""
Audio Utilities for loading, standardizing, normalizing, and transient-preserving signal processing.
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from src import config


def load_audio(filepath: Path | str, target_sr: int = config.SAMPLE_RATE, mono: bool = config.MONO) -> tuple[np.ndarray, int]:
    """
    Loads an audio file, converts to mono if needed, and resamples to target_sr.
    Returns:
        audio: 1D numpy float32 array in range [-1.0, 1.0]
        sr: int sample rate
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file does not exist: {filepath}")

    # librosa.load standardizes to [-1.0, 1.0] float32
    audio, sr = librosa.load(str(filepath), sr=target_sr, mono=mono)
    return audio.astype(np.float32), sr


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize peak amplitude to target_peak while preserving transient dynamics.
    Avoids clipping and prevents background noise explosion if signal is near silent.
    """
    max_val = np.max(np.abs(audio))
    if max_val > eps:
        audio = audio * (target_peak / max_val)
    return audio


def save_audio(filepath: Path | str, audio: np.ndarray, sr: int = config.SAMPLE_RATE):
    """
    Saves float32 audio as a standard 16-bit PCM WAV.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Clip to prevent wrap-around distortion
    audio_clipped = np.clip(audio, -1.0, 1.0)
    sf.write(str(filepath), audio_clipped, sr, subtype="PCM_16")


def compute_rms(audio: np.ndarray) -> float:
    """Computes Root Mean Square (RMS) energy."""
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
