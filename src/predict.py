"""
Offline Audio File Footstep Predictor.
Runs Model V2 (Balanced Sampler) on an audio file, evaluates sliding windows,
applies temporal validation, and displays the timeline of probabilities and events.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio
from src.preprocessing import preprocess_audio_window
from src.models import BaselineFootstepCNN
from src.feature_extraction import extract_log_mel_spectrogram
from src.live_event_detector import TemporalEventDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Predictor")


def predict_audio_file(
    filepath: Path | str,
    model_path: Optional[Path | str] = None,
    threshold: float = 0.40,
    hop_sec: float = 0.25,
    device: str = "cpu"
):
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return

    if model_path is None:
        robust_path = config.MODELS_DIR / "footstep_detector_robust.pth"
        v2_path = config.MODELS_DIR / "model_v2_balanced_sampler.pth"
        model_path = robust_path if robust_path.exists() else v2_path

    logger.info(f"Loading audio from: {filepath}")
    audio, sr = load_audio(filepath, target_sr=config.SAMPLE_RATE, mono=True)
    duration = len(audio) / sr

    # Load Model
    model = BaselineFootstepCNN(in_channels=1, dropout=0.0).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Event Detector
    detector = TemporalEventDetector(
        threshold=threshold,
        high_confidence_threshold=max(threshold + 0.20, 0.65),
        min_confirmations=2,
        history_len=4,
        cooldown_sec=0.70
    )

    win_samples = int(config.WINDOW_DURATION_SEC * sr)
    hop_samples = int(hop_sec * sr)

    print("\n" + "=" * 65)
    print(f"OFFLINE AUDIO PREDICTION: {filepath.name}")
    print("=" * 65)
    print(f"Duration:   {duration:.2f} seconds ({len(audio)} samples)")
    print(f"Model:      {Path(model_path).name}")
    print(f"Threshold:  {threshold:.2f}")
    print(f"Hop size:   {hop_sec * 1000:.0f} ms")
    print("-" * 65)
    print(f"{'Time Range':18s} | {'Probability':11s} | {'Status':19s} | {'Event'}")
    print("-" * 65)

    start = 0
    t = 0.0
    detected_events = []
    window_probs = []

    while start + win_samples <= len(audio):
        window = audio[start:start + win_samples]
        norm_window = preprocess_audio_window(window, target_length=win_samples, apply_gain_norm=True)
        spec = extract_log_mel_spectrogram(norm_window, sr=sr)
        tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logit = model(tensor)
            prob = float(torch.sigmoid(logit).item())

        window_probs.append(prob)
        status = "FOOTSTEP CANDIDATE" if prob >= threshold else "NON_FOOTSTEP"
        ev = detector.update(prob, timestamp=t)

        ev_str = ""
        if ev:
            ev_str = f"*** FOOTSTEP DETECTED (Conf: {ev['confidence']:.2f}) ***"
            detected_events.append(ev)

        t_start = start / sr
        t_end = (start + win_samples) / sr
        print(f"{t_start:5.2f}s - {t_end:5.2f}s | {prob:11.4f} | {status:19s} | {ev_str}")

        start += hop_samples
        t += hop_sec

    # Handle short audio
    if not window_probs:
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        spec = extract_log_mel_spectrogram(padded, sr=sr)
        tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            prob = float(torch.sigmoid(model(tensor)).item())
        status = "FOOTSTEP CANDIDATE" if prob >= threshold else "NON_FOOTSTEP"
        print(f"0.00s - {duration:5.2f}s | {prob:11.4f} | {status:19s} |")
        window_probs.append(prob)

    avg_p = sum(window_probs) / len(window_probs)
    max_p = max(window_probs)

    print("=" * 65)
    print("PREDICTION SUMMARY:")
    print(f"Peak Probability:    {max_p:.4f}")
    print(f"Average Probability: {avg_p:.4f}")
    print(f"Candidate Windows:   {sum(1 for p in window_probs if p >= threshold)} / {len(window_probs)}")
    print(f"Confirmed Events:    {len(detected_events)}")
    if max_p >= threshold:
        print("Final Verdict:       [+] FOOTSTEP DETECTED")
    else:
        print("Final Verdict:       [-] NON-FOOTSTEP")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict footstep presence in an audio file")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to audio file (WAV/MP3)")
    parser.add_argument("--threshold", "-t", type=float, default=0.45, help="Decision threshold")
    parser.add_argument("--hop", type=float, default=0.25, help="Hop in seconds")
    args = parser.parse_args()

    predict_audio_file(filepath=args.file, threshold=args.threshold, hop_sec=args.hop)
