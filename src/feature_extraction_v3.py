"""
Feature Extraction V3 — 3-Channel Log-Mel Spectrogram.

Returns a 3-channel representation:
  Channel 0: Log-Mel Spectrogram (static spectral energy)
  Channel 1: Delta (first-order temporal derivative — onset/attack shape)
  Channel 2: Delta-Delta (second-order — acceleration/decay shape)

This 3-channel representation lets EfficientFootstepNet distinguish the
unique transient attack + resonant decay envelope of a footstep from
confusors (knocks, water drops, keyboard typing) that share similar energy
but differ in temporal dynamics.
"""

from typing import Optional
import numpy as np
import librosa
import torch

from src import config


# =====================================================================
# V3 Feature Config (intentionally separate from V1/V2 config constants)
# =====================================================================
V3_N_MELS: int = 80          # More mel bands → better frequency resolution
V3_N_FFT: int = 1024
V3_HOP_LENGTH: int = 512
V3_F_MIN: float = 20.0
V3_F_MAX: float = 8000.0
V3_POWER: float = 2.0
V3_DELTA_WIDTH: int = 9      # librosa default delta filter half-width


def extract_log_mel_v3(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mels: int = V3_N_MELS,
    n_fft: int = V3_N_FFT,
    hop_length: int = V3_HOP_LENGTH,
    f_min: float = V3_F_MIN,
    f_max: float = V3_F_MAX,
    power: float = V3_POWER,
    delta_width: int = V3_DELTA_WIDTH,
) -> np.ndarray:
    """
    Computes a 3-channel (mel, delta, delta²) spectrogram from audio.

    Args:
        audio:  1-D float32 waveform, already trimmed to WINDOW_SIZE.
        sr:     Sample rate (default 16000).

    Returns:
        np.ndarray of shape (3, n_mels, time_frames), float32, values in [0, 1].
    """
    # --- Channel 0: Log-Mel (dB scale, normalized to [0, 1]) ----------
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=power,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize dB range [−80, 0] → [0, 1]
    log_mel_norm = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)

    # --- Channel 1: Delta (temporal 1st derivative) -------------------
    delta = librosa.feature.delta(log_mel_norm, width=delta_width, order=1)
    # Delta range is roughly [−0.5, +0.5]; center-shift + normalize to [0, 1]
    delta_norm = np.clip(delta / 0.5 * 0.5 + 0.5, 0.0, 1.0)

    # --- Channel 2: Delta-Delta (temporal 2nd derivative) -------------
    delta2 = librosa.feature.delta(log_mel_norm, width=delta_width, order=2)
    delta2_norm = np.clip(delta2 / 0.5 * 0.5 + 0.5, 0.0, 1.0)

    # Stack → shape (3, n_mels, time_frames)
    stacked = np.stack([log_mel_norm, delta_norm, delta2_norm], axis=0)
    return stacked.astype(np.float32)


def audio_to_tensor_v3(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Convenience pipeline: raw audio → 3-channel spectrogram → PyTorch tensor.

    Returns:
        Tensor of shape (1, 3, n_mels, time_frames).
    """
    spec = extract_log_mel_v3(audio, sr=sr)
    tensor = torch.from_numpy(spec).unsqueeze(0)  # [1, 3, H, W]
    if device is not None:
        tensor = tensor.to(device)
    return tensor
