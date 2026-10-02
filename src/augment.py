"""
Acoustic Domain-Gap and Robust Audio Augmentations for Training.
Simulates real-world speaker-to-microphone playback conditions:
1. Speaker bass roll-off (highpass 150-350 Hz simulating small laptop/phone speaker drivers)
2. Microphone high-frequency roll-off (lowpass 4500-7500 Hz)
3. Room impulse response / synthetic reverberation (early reflections + decaying tail)
4. Additive background noise at realistic signal-to-noise ratios (SNRs 8 dB - 28 dB)
5. Dynamic range gain and attenuation variation (0.2x to 1.3x)
6. Random cyclic time shifting and pitch variations

NEVER applied to validation or test datasets.
"""

import random
import numpy as np
import scipy.signal as signal
import librosa
from src import config


def random_time_shift(audio: np.ndarray, sr: int = config.SAMPLE_RATE, max_shift_sec: float = 0.25) -> np.ndarray:
    """Cyclic shift of audio within a time window."""
    shift_samples = int(random.uniform(-max_shift_sec, max_shift_sec) * sr)
    return np.roll(audio, shift_samples)


def random_gain(audio: np.ndarray, min_gain: float = 0.25, max_gain: float = 1.3) -> np.ndarray:
    """Volume/gain scaling across wide dynamic range."""
    gain = random.uniform(min_gain, max_gain)
    return np.clip(audio * gain, -1.0, 1.0)


def add_subtle_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Adds subtle Gaussian white noise to simulate acoustic background hiss."""
    noise = np.random.normal(0, noise_factor, len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def add_background_noise_snr(audio: np.ndarray, snr_min_db: float = 8.0, snr_max_db: float = 28.0) -> np.ndarray:
    """
    Adds noise at a controlled Signal-to-Noise Ratio (SNR).
    Simulates ambient room rumble, HVAC hum, and mic pre-amp thermal noise.
    """
    sig_power = float(np.mean(audio ** 2))
    if sig_power < 1e-8:
        return audio

    snr_db = random.uniform(snr_min_db, snr_max_db)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def simulate_speaker_filter(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Simulates miniature laptop/phone speaker frequency response.
    Small speakers cannot reproduce bass below 150-350 Hz.
    """
    nyquist = sr * 0.5
    cutoff_hz = random.uniform(150.0, 350.0)
    norm_cutoff = min(cutoff_hz / nyquist, 0.95)
    b, a = signal.butter(2, norm_cutoff, btype="highpass")
    filtered = signal.lfilter(b, a, audio).astype(np.float32)
    return filtered


def simulate_mic_filter(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Simulates budget microphone high-frequency roll-off (4.5 kHz - 7.5 kHz).
    """
    nyquist = sr * 0.5
    cutoff_hz = random.uniform(4500.0, 7500.0)
    norm_cutoff = min(cutoff_hz / nyquist, 0.95)
    b, a = signal.butter(2, norm_cutoff, btype="lowpass")
    filtered = signal.lfilter(b, a, audio).astype(np.float32)
    return filtered


def simulate_room_reverb(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    max_delay_ms: float = 40.0,
    max_tail_ms: float = 120.0
) -> np.ndarray:
    """
    Simulates room acoustics / reverberation via a fast synthetic Room Impulse Response (RIR).
    Adds early reflections + exponentially decaying diffuse tail.
    """
    rir_len = int((max_tail_ms / 1000.0) * sr)
    if rir_len < 16:
        return audio

    rir = np.zeros(rir_len, dtype=np.float32)
    rir[0] = 1.0  # Direct sound

    # Early reflections
    delay1 = int(random.uniform(12.0, max_delay_ms) * 1e-3 * sr)
    if delay1 < rir_len:
        rir[delay1] = random.uniform(0.15, 0.35)

    delay2 = int(delay1 * random.uniform(1.4, 2.0))
    if delay2 < rir_len:
        rir[delay2] = random.uniform(0.08, 0.20)

    # Diffuse decay tail
    t = np.arange(rir_len) / sr
    decay_rate = random.uniform(20.0, 45.0)
    tail = np.exp(-t * decay_rate) * np.random.normal(0, 0.08, rir_len).astype(np.float32)
    rir += tail

    # Normalize impulse response
    rir_sum = np.sum(np.abs(rir))
    if rir_sum > 0:
        rir /= rir_sum

    reverbed = signal.fftconvolve(audio, rir, mode="same").astype(np.float32)
    return reverbed


def subtle_pitch_shift(audio: np.ndarray, sr: int = config.SAMPLE_RATE, n_steps: int = 1) -> np.ndarray:
    """Slight pitch shift (±1 semitone) to simulate different footstep surface tones."""
    step = random.choice([-n_steps, n_steps])
    try:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=step)
    except Exception:
        return audio


def apply_augmentations(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    probability: float = 0.70
) -> np.ndarray:
    """
    Master augmentation pipeline:
    Applies combinations of domain-gap acoustic transforms (speaker EQ, mic EQ, reverb,
    noise, gain, and time shifts) to build robustness against live speaker-mic playback.
    """
    if random.random() > probability:
        return audio

    augmented = audio.copy()

    # 1. Speaker frequency response loss (attenuate sub-bass)
    if random.random() < 0.55:
        augmented = simulate_speaker_filter(augmented, sr=sr)

    # 2. Microphone lowpass roll-off
    if random.random() < 0.45:
        augmented = simulate_mic_filter(augmented, sr=sr)

    # 3. Room reverberation
    if random.random() < 0.50:
        augmented = simulate_room_reverb(augmented, sr=sr)

    # 4. Gain / Distance variation
    if random.random() < 0.65:
        augmented = random_gain(augmented, min_gain=0.25, max_gain=1.3)

    # 5. Background noise at realistic SNRs
    if random.random() < 0.50:
        augmented = add_background_noise_snr(augmented, snr_min_db=10.0, snr_max_db=26.0)

    # 6. Cyclic time shift
    if random.random() < 0.45:
        augmented = random_time_shift(augmented, sr=sr, max_shift_sec=config.TIME_SHIFT_MAX_SEC)

    return augmented
