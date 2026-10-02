"""
Phase 9 / 15-Case Acoustic Test Suite (Multi-Class & Binary Compatible).
Evaluates the real-time sliding-window detection engine against 15 key acoustic test cases:
Real footsteps (wood, tile, carpet, boots, hallway), quiet/distant steps, table knock,
door knock, clapping, finger tapping, mouse click, object drop, silence, and room noise.
Outputs reports/live_manual_test.csv and reports/live_detector_summary.txt.
"""

import sys
import logging
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio, peak_normalize
from src.preprocessing import preprocess_audio_window
from src.models import BaselineFootstepCNN
from src.models_multiclass import MultiClassAudioNet
from src.feature_extraction import extract_log_mel_spectrogram
from src.feature_extraction_v3 import extract_log_mel_v3
from src.live_event_detector import TemporalEventDetector, MultiClassTemporalDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LiveManualTests")


def evaluate_audio_through_live_engine(
    audio: np.ndarray,
    model,
    event_detector,
    is_multiclass: bool = False,
    hop_sec: float = 0.25
):
    """
    Feeds an audio clip through the exact live sliding-window engine with 250ms hop.
    """
    sr = config.SAMPLE_RATE
    win_samples = int(config.WINDOW_DURATION_SEC * sr)
    hop_samples = int(hop_sec * sr)

    if is_multiclass:
        event_detector.history.clear()
        event_detector.step_timestamps.clear()
        event_detector.last_event_time = -999.0
        event_detector.last_event_type = None
    else:
        event_detector.in_event = False
        event_detector.current_event_windows = []
        event_detector.last_event_time = -999.0

    max_p_step = 0.0
    max_p_clap = 0.0
    max_p_knock = 0.0
    detected_footstep = False
    detected_event_type = "NONE"

    # Pad if shorter than window
    if len(audio) < win_samples:
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded

    start = 0
    t = 0.0
    while start + win_samples <= len(audio):
        window = audio[start:start + win_samples]
        rms = float(np.sqrt(np.mean(window ** 2)))
        peak = float(np.max(np.abs(window)))

        norm_win = preprocess_audio_window(window, target_length=win_samples, apply_gain_norm=True)

        if is_multiclass:
            feat = extract_log_mel_v3(norm_win, sr=sr)
            tensor = torch.from_numpy(feat).unsqueeze(0).float()
            with torch.no_grad():
                probs = model.predict_proba(tensor).cpu().numpy()[0]
            p_other, p_step, p_clap, p_knock = probs[0], probs[1], probs[2], probs[3]

            max_p_step = max(max_p_step, float(p_step))
            max_p_clap = max(max_p_clap, float(p_clap))
            max_p_knock = max(max_p_knock, float(p_knock))

            ev, label_str = event_detector.update(probs, rms=rms, timestamp=t)
            if ev is not None:
                detected_event_type = ev.get("event_type", label_str)
                if ev.get("event_type") == "FOOTSTEP":
                    detected_footstep = True
        else:
            spec = extract_log_mel_spectrogram(norm_win, sr=sr)
            tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                logit = model(tensor)
                prob = float(torch.sigmoid(logit).item())

            max_p_step = max(max_p_step, prob)
            ev = event_detector.update(prob, timestamp=t)
            if ev is not None:
                detected_footstep = True
                detected_event_type = "FOOTSTEP"

        start += hop_samples
        t += hop_sec

    return {
        "max_p_step": max_p_step,
        "max_p_clap": max_p_clap,
        "max_p_knock": max_p_knock,
        "detected_footstep": detected_footstep,
        "event_type": detected_event_type
    }


def run_manual_test_suite(
    model_path: Optional[Path] = None,
    threshold: float = 0.38,
    require_cadence: bool = True
):
    logger.info("=" * 70)
    logger.info("RUNNING 15 ACOUSTIC LIVE TESTS ACROSS TRANSIENTS & FOOTSTEPS")
    logger.info("=" * 70)

    # Load Model (prefer multi-class v4 if present)
    if model_path is None:
        multi_p = config.MODELS_DIR / "footstep_multiclass_v4.pth"
        robust_p = config.MODELS_DIR / "footstep_detector_robust.pth"
        v2_p = config.MODELS_DIR / "model_v2_balanced_sampler.pth"
        if multi_p.exists():
            model_path = multi_p
        elif robust_p.exists():
            model_path = robust_p
        else:
            model_path = v2_p

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = ckpt.get("num_classes", 1)
    is_multiclass = (num_classes == 4) or ("multiclass" in str(model_path))

    if is_multiclass:
        model = MultiClassAudioNet(in_channels=3, num_classes=4, dropout=0.0)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        event_detector = MultiClassTemporalDetector(
            step_threshold=threshold,
            clap_threshold=0.40,
            knock_threshold=0.40,
            require_cadence=require_cadence,
            cooldown_sec=0.70
        )
        logger.info(f"Loaded Multi-Class Model: {model_path.name} (Cadence Logic: {'ENABLED' if require_cadence else 'DISABLED'})")
    else:
        model = BaselineFootstepCNN(in_channels=1, dropout=0.0)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        event_detector = TemporalEventDetector(
            threshold=threshold,
            high_confidence_threshold=0.65,
            min_confirmations=2,
            history_len=4,
            cooldown_sec=0.70
        )
        logger.info(f"Loaded Binary Model: {model_path.name}")

    manifest_df = pd.read_csv(config.PROJECT_ROOT / "dataset_v2" / "manifest.csv")

    def get_path_for_cat(cat_name):
        match = manifest_df[manifest_df["original_class"] == cat_name]
        if not match.empty:
            return config.PROJECT_ROOT / match.iloc[0]["relative_path"]
        return None

    tests = [
        {"test_name": "Walking on wood floor", "expected": "FOOTSTEP", "path": get_path_for_cat("footstep_wood"), "notes": "Low-frequency resonant thuds"},
        {"test_name": "Walking on tile", "expected": "FOOTSTEP", "path": get_path_for_cat("footstep_tile"), "notes": "Crisp heel impact on ceramic tile"},
        {"test_name": "Walking on carpet", "expected": "FOOTSTEP", "path": get_path_for_cat("footstep_carpet"), "notes": "Muffled, low-amplitude footstep"},
        {"test_name": "Boots walking", "expected": "FOOTSTEP", "path": get_path_for_cat("footstep_boots"), "notes": "Heavy sole impact"},
        {"test_name": "Hallway walking sequence", "expected": "FOOTSTEP", "path": get_path_for_cat("footsteps"), "notes": "Continuous walking gait in corridor"},
        {"test_name": "Quiet footsteps (scaled 0.35x)", "expected": "FOOTSTEP", "path": get_path_for_cat("footstep_wood"), "scale": 0.35, "notes": "Soft gentle step"},
        {"test_name": "Distant footsteps (scaled 0.15x)", "expected": "FOOTSTEP", "path": get_path_for_cat("footstep_carpet"), "scale": 0.15, "notes": "Low SNR, distant steps"},
        {"test_name": "Door knock (Acoustic Look-Alike)", "expected": "NON_FOOTSTEP", "path": get_path_for_cat("door_wood_knock"), "notes": "Rhythmic wooden door rapping"},
        {"test_name": "Table knock / Desk impact", "expected": "NON_FOOTSTEP", "path": get_path_for_cat("knocking"), "notes": "Sharp percussive strike"},
        {"test_name": "Clapping (Acoustic Look-Alike)", "expected": "NON_FOOTSTEP", "path": get_path_for_cat("clapping"), "notes": "Transient hand clap"},
        {"test_name": "Mouse click (Sharp Transient)", "expected": "NON_FOOTSTEP", "path": get_path_for_cat("mouse_click"), "notes": "High-frequency plastic click"},
        {"test_name": "Keyboard typing (Finger Tapping)", "expected": "NON_FOOTSTEP", "path": get_path_for_cat("keyboard_typing"), "notes": "Rapid mechanical key clatter"},
        {"test_name": "Glass breaking / Object drop", "expected": "NON_FOOTSTEP", "path": get_path_for_cat("glass_breaking"), "notes": "Impulsive shattering sound"},
        {"test_name": "Background room noise / Fan", "expected": "NON_FOOTSTEP", "path": get_path_for_cat("background_noise"), "notes": "Continuous low-level ambient hum"},
        {"test_name": "Room silence", "expected": "NON_FOOTSTEP", "path": None, "notes": "Ambient room baseline (<0.0005 RMS)"}
    ]

    results = []

    print("\n" + "=" * 90)
    print(f"LIVE TEST RESULTS: {model_path.name} (Threshold: {threshold:.2f})")
    print("=" * 90)
    print(f"{'Test Sound':32s} | {'Expected':12s} | {'Step Prob':10s} | {'Triggered':10s} | {'Status':8s}")
    print("-" * 90)

    for tc in tests:
        if tc["path"] is None:
            audio = np.random.normal(0, 0.00008, int(2.0 * config.SAMPLE_RATE)).astype(np.float32)
        else:
            audio, sr = load_audio(tc["path"], target_sr=config.SAMPLE_RATE, mono=True)
            if "scale" in tc:
                audio = audio * tc["scale"]

        res = evaluate_audio_through_live_engine(audio, model, event_detector, is_multiclass=is_multiclass)
        step_p = res["max_p_step"]
        detected = res["detected_footstep"]
        det_str = "YES" if detected else "NO"

        # Correctness
        if tc["expected"] == "FOOTSTEP":
            passed = detected
        else:
            passed = not detected
        status_str = "PASS" if passed else "FAIL"

        results.append({
            "test_name": tc["test_name"],
            "expected": tc["expected"],
            "step_probability": round(float(step_p), 4),
            "clap_probability": round(float(res["max_p_clap"]), 4),
            "knock_probability": round(float(res["max_p_knock"]), 4),
            "footstep_triggered": det_str,
            "passed": status_str,
            "event_type": res["event_type"],
            "notes": tc["notes"]
        })

        print(f"{tc['test_name']:32s} | {tc['expected']:12s} | {step_p:10.4f} | {det_str:10s} | {status_str:8s}")

    print("=" * 90 + "\n")

    # Save CSV
    df_res = pd.DataFrame(results)
    csv_path = config.REPORTS_DIR / "live_manual_test_multiclass.csv"
    df_res.to_csv(csv_path, index=False)
    logger.info(f"Saved 15-case test results to {csv_path}")

    return df_res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 15 acoustic live tests on multi-class model")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--threshold", "-t", type=float, default=0.38, help="Decision threshold")
    parser.add_argument("--no-cadence", action="store_true", help="Disable cadence verification")
    args = parser.parse_args()

    model_p = Path(args.model) if args.model else None
    run_manual_test_suite(model_path=model_p, threshold=args.threshold, require_cadence=not args.no_cadence)
