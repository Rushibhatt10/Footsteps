"""
Audio Preprocessing and Window Segmentation Module.
Implements sliding window segmentation, padding, and transient preservation.
"""

from typing import List, Tuple
import numpy as np
from src import config
from src.audio_utils import peak_normalize, compute_rms


def pad_or_trim(audio: np.ndarray, target_length: int = config.WINDOW_SIZE) -> np.ndarray:
    """
    Ensures audio is exactly target_length samples.
    Pads with zeros if shorter, or trims if longer.
    """
    length = len(audio)
    if length == target_length:
        return audio
    elif length < target_length:
        padded = np.zeros(target_length, dtype=audio.dtype)
        padded[:length] = audio
        return padded
    else:
        return audio[:target_length]


def segment_audio(
    audio: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    hop_size: int = config.HOP_SIZE,
    min_rms_threshold: float = 0.001
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Splits an audio signal into overlapping windows.
    Returns:
        List of (window_audio, start_time_sec, end_time_sec)
    """
    total_samples = len(audio)
    if total_samples < window_size:
        padded = pad_or_trim(audio, window_size)
        return [(padded, 0.0, total_samples / config.SAMPLE_RATE)]

    windows = []
    start = 0
    while start + window_size <= total_samples:
        window = audio[start:start + window_size]
        start_time = start / config.SAMPLE_RATE
        end_time = (start + window_size) / config.SAMPLE_RATE

        # Filter out completely dead silence
        if compute_rms(window) >= min_rms_threshold:
            windows.append((window, start_time, end_time))
        start += hop_size

    # Ensure tail is captured if there's significant audio left
    if total_samples - start > hop_size // 2 and start < total_samples:
        tail = audio[-window_size:]
        start_time = max(0.0, (total_samples - window_size) / config.SAMPLE_RATE)
        end_time = total_samples / config.SAMPLE_RATE
        if compute_rms(tail) >= min_rms_threshold:
            windows.append((tail, start_time, end_time))

    return windows


def preprocess_signal(audio: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Standard preprocessing pipeline for a raw audio clip.
    Removes DC offset and applies peak normalization.
    """
    # Remove DC offset (center around 0)
    audio = audio - np.mean(audio)
    if normalize:
        audio = peak_normalize(audio, target_peak=0.95)
    return audio.astype(np.float32)


def preprocess_audio_window(
    audio: np.ndarray,
    target_length: int = config.WINDOW_SIZE,
    apply_gain_norm: bool = True,
    max_gain_boost: float = 6.0,
    target_peak: float = 0.95
) -> np.ndarray:
    """
    Common Preprocessing Function shared by training, evaluation, and live inference.
    Guarantees:
      1. Float32 1D array.
      2. DC offset removed.
      3. Soft-limited adaptive gain normalization:
         - Boosts quiet footsteps (up to max_gain_boost ~15.5 dB)
         - Prevents explosive noise amplification on quiet background
      4. Exact target_length padding/trimming.
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()

    # 1. DC offset removal
    if len(audio) > 0:
        audio = audio - np.mean(audio)

    # 2. Soft-limited gain normalization
    if apply_gain_norm and len(audio) > 0:
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-4:
            # Gentle scale, capped to avoid blowing up ambient room hiss
            scale = min(target_peak / peak, max_gain_boost)
            audio = np.clip(audio * scale, -1.0, 1.0)

    # 3. Ensure exact sample length
    audio = pad_or_trim(audio, target_length)
    return audio.astype(np.float32)

