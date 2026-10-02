"""
Feature Extraction Module for Audio Classification.
Extracts Log-Mel Spectrogram and MFCC features using configurable parameters.
"""

from typing import Optional
import numpy as np
import librosa
import torch
from src import config


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mels: int = config.N_MELS,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    f_min: float = config.F_MIN,
    f_max: float = config.F_MAX,
    power: float = config.POWER
) -> np.ndarray:
    """
    Computes Log-Mel Spectrogram (dB scale) from an audio array.
    Returns:
        np.ndarray of shape (n_mels, time_frames), float32
    """
    peak_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak_amp < 1e-4:
        # Avoid boosting near-zero noise floor to 1.0
        time_frames = 1 + int(len(audio) // hop_length)
        return np.zeros((n_mels, time_frames), dtype=np.float32)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=power
    )
    # Convert power to decibels (log scale)
    ref_val = max(float(np.max(mel_spec)), 1e-5)
    log_mel = librosa.power_to_db(mel_spec, ref=ref_val, top_db=80.0)
    # Normalize dB range [-80.0, 0.0] to [0.0, 1.0] for neural network input
    log_mel_norm = (log_mel + 80.0) / 80.0
    log_mel_norm = np.clip(log_mel_norm, 0.0, 1.0)
    return log_mel_norm.astype(np.float32)


def extract_mfcc(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mfcc: int = 20,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH
) -> np.ndarray:
    """
    Computes MFCC features.
    Returns:
        np.ndarray of shape (n_mfcc, time_frames)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc.astype(np.float32)


def audio_to_tensor(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Convenience pipeline: audio numpy array -> Log-Mel Spectrogram -> PyTorch Tensor (1, 1, n_mels, time).
    """
    spec = extract_log_mel_spectrogram(audio, sr=sr)
    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    if device is not None:
        tensor = tensor.to(device)
    return tensor
