"""
Real-Time Live Microphone Footstep Detector (Multi-Class & Binary Compatible).
Continuously streams microphone audio into a thread-safe ring buffer,
runs sliding-window inference with shared preprocessing, applies sequence & cadence verification,
suppresses competing transient false positives (Clapping, Knocking),
and outputs verified footstep events with rich diagnostic logging.
"""

import sys
import time
import argparse
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.models import BaselineFootstepCNN
from src.models_multiclass import MultiClassAudioNet
from src.feature_extraction import extract_log_mel_spectrogram
from src.feature_extraction_v3 import extract_log_mel_v3
from src.preprocessing import preprocess_audio_window
from src.audio_utils import load_audio
from src.live_event_detector import TemporalEventDetector, MultiClassTemporalDetector
from src.microphone_utils import list_microphones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LiveDetector")


class ThreadSafeAudioBuffer:
    """Thread-safe circular ring buffer for streaming real-time audio."""
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.write_idx = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        with self.lock:
            n = len(data)
            if n >= self.size:
                self.buffer[:] = data[-self.size:]
                self.write_idx = 0
            else:
                end_idx = self.write_idx + n
                if end_idx <= self.size:
                    self.buffer[self.write_idx:end_idx] = data
                else:
                    first_part = self.size - self.write_idx
                    self.buffer[self.write_idx:] = data[:first_part]
                    self.buffer[:n - first_part] = data[first_part:]
                self.write_idx = end_idx % self.size
            self.total_written += n

    def get_latest(self) -> np.ndarray:
        """Returns the most recent window ordered chronologically."""
        with self.lock:
            if self.write_idx == 0:
                return self.buffer.copy()
            return np.concatenate((self.buffer[self.write_idx:], self.buffer[:self.write_idx]))


class LiveFootstepDetector:
    """
    Real-time streaming footstep detector with multi-class transient disambiguation.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        device_id: Optional[int] = None,
        simulate_file: Optional[str] = None,
        threshold: float = 0.38,
        high_conf_threshold: float = 0.65,
        min_confirmations: int = 2,
        history_len: int = 4,
        cooldown_sec: float = 0.70,
        hop_seconds: float = 0.25,
        window_seconds: float = 1.5,
        require_cadence: bool = True,
        sample_rate: int = config.SAMPLE_RATE,
        max_gain_boost: float = 6.0,
        record_debug: bool = False,
        debug_mode: bool = False
    ):
        # Default model selection: prefer newly trained multi-class model if present
        if model_path is None:
            multiclass_p = config.MODELS_DIR / "footstep_multiclass_v4.pth"
            robust_p = config.MODELS_DIR / "footstep_detector_robust.pth"
            v2_p = config.MODELS_DIR / "model_v2_balanced_sampler.pth"
            if multiclass_p.exists():
                model_path = multiclass_p
            elif robust_p.exists():
                model_path = robust_p
            else:
                model_path = v2_p

        self.model_path = Path(model_path)
        self.device_id = device_id
        self.simulate_file = Path(simulate_file) if simulate_file else None
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_samples = int(window_seconds * sample_rate)
        self.hop_samples = int(hop_seconds * sample_rate)
        self.hop_seconds = hop_seconds
        self.max_gain_boost = max_gain_boost
        self.record_debug = record_debug
        self.debug_mode = debug_mode
        self.require_cadence = require_cadence

        # Buffering
        self.ring_buffer = ThreadSafeAudioBuffer(self.window_samples)
        self.is_running = False
        self.total_events = 0
        self.total_claps = 0
        self.total_knocks = 0
        self.last_event_str = "None"
        self.stream_native_sr = self.sample_rate
        self.needs_resample = False

        # Setup logs
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.event_log_path = config.LOGS_DIR / "live_events.csv"
        self._init_event_log()

        if self.record_debug:
            self.debug_audio_dir = config.RECORDINGS_DIR / "live_debug"
            self.debug_audio_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Model (Detect Multi-Class vs Binary)
        logger.info(f"Loading Model from {self.model_path.name}...")
        self.device = "cpu"
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        num_classes = checkpoint.get("num_classes", 1)
        self.is_multiclass = (num_classes == 4) or ("multiclass" in str(self.model_path))

        if self.is_multiclass:
            self.model = MultiClassAudioNet(in_channels=3, num_classes=4, dropout=0.0).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.event_detector = MultiClassTemporalDetector(
                step_threshold=self.threshold,
                clap_threshold=0.40,
                knock_threshold=0.40,
                high_step_threshold=high_conf_threshold,
                require_cadence=require_cadence,
                cooldown_sec=cooldown_sec,
                history_len=history_len
            )
            logger.info(f"Multi-Class Model Loaded: 4 Classes [other, footstep, clap, knock]")
            logger.info(f"Walking Cadence Logic: {'ENABLED (Rejects isolated single impacts)' if require_cadence else 'DISABLED'}")
        else:
            self.model = BaselineFootstepCNN(in_channels=1, dropout=0.0).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.event_detector = TemporalEventDetector(
                threshold=self.threshold,
                high_confidence_threshold=high_conf_threshold,
                min_confirmations=min_confirmations,
                history_len=history_len,
                cooldown_sec=cooldown_sec
            )
            logger.info("Binary Footstep Detector Loaded.")

    def _init_event_log(self):
        if not self.event_log_path.exists():
            with open(self.event_log_path, "w") as f:
                f.write("timestamp,time_str,event_type,confidence,step_prob,clap_prob,knock_prob,other_prob,rms,notes\n")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio stream status: {status}")
        chunk = indata[:, 0].copy()

        if self.needs_resample and self.stream_native_sr != self.sample_rate:
            num_samples = int(len(chunk) * (self.sample_rate / self.stream_native_sr))
            chunk = signal.resample(chunk, num_samples).astype(np.float32)

        self.ring_buffer.write(chunk)

    def run_simulated_file(self):
        """Simulates real-time microphone stream using a test audio file."""
        logger.info(f"Streaming audio file through live detector: {self.simulate_file}")
        audio, _ = load_audio(self.simulate_file, target_sr=self.sample_rate, mono=True)
        duration = len(audio) / self.sample_rate

        print("\n" + "=" * 70)
        print("SIMULATED LIVE FOOTSTEP & TRANSIENT DETECTOR")
        print("=" * 70)
        print(f"Source File:     {self.simulate_file.name}")
        print(f"File Duration:   {duration:.2f}s ({len(audio)} samples)")
        print(f"Model:           {self.model_path.name}")
        print(f"Cadence Check:   {'ACTIVE' if self.require_cadence else 'OFF'}")
        print(f"Hop duration:    {self.hop_seconds * 1000:.0f} ms ({self.window_samples / self.sample_rate:.1f}s window)")
        print("=" * 70 + "\n")

        chunk_size = self.hop_samples
        idx = 0
        t = 0.0

        while idx < len(audio):
            loop_start = time.perf_counter()
            chunk = audio[idx:idx + chunk_size]
            if len(chunk) < chunk_size:
                pad = np.zeros(chunk_size - len(chunk), dtype=np.float32)
                chunk = np.concatenate((chunk, pad))

            self.ring_buffer.write(chunk)
            idx += chunk_size
            t += self.hop_seconds

            self._process_latest_window(timestamp=t)

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.005, self.hop_seconds - elapsed))

        print(f"\n[INFO] Simulation finished. Steps: {self.total_events} | Claps: {self.total_claps} | Knocks: {self.total_knocks}\n")

    def _process_latest_window(self, timestamp: Optional[float] = None):
        """Processes the current ring buffer window through model and temporal validator."""
        audio_window = self.ring_buffer.get_latest()
        rms = float(np.sqrt(np.mean(audio_window ** 2)))
        peak = float(np.max(np.abs(audio_window)))

        now = timestamp if timestamp is not None else time.time()
        time_str = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 100):02d}"

        # True dead silence filter
        if peak < 1e-4 or rms < 0.00015:
            if self.is_multiclass:
                probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                label_str = "QUIET_AMBIENT"
                event = None
            else:
                prob = 0.0
                event = None
                label_str = "BACKGROUND_QUIET"
            inf_ms = 0.0
        else:
            norm_window = preprocess_audio_window(
                audio_window,
                target_length=self.window_samples,
                apply_gain_norm=True,
                max_gain_boost=self.max_gain_boost,
                target_peak=0.95
            )

            t_inf0 = time.perf_counter()
            with torch.no_grad():
                if self.is_multiclass:
                    feat = extract_log_mel_v3(norm_window, sr=self.sample_rate)
                    tensor = torch.from_numpy(feat).unsqueeze(0).float().to(self.device)
                    probs = self.model.predict_proba(tensor).cpu().numpy()[0]
                    event, label_str = self.event_detector.update(probs, rms=rms, timestamp=now)
                else:
                    spec = extract_log_mel_spectrogram(norm_window, sr=self.sample_rate)
                    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    prob = float(torch.sigmoid(self.model(tensor)).item())
                    event = self.event_detector.update(prob, timestamp=now)
                    label_str = "FOOTSTEP" if prob >= self.threshold else "NON_FOOTSTEP"
                    probs = np.array([1.0 - prob, prob, 0.0, 0.0])

            inf_ms = (time.perf_counter() - t_inf0) * 1000.0

        # Event handling
        if event:
            ev_type = event.get("event_type", "FOOTSTEP")
            conf = event["confidence"]

            if ev_type == "FOOTSTEP":
                self.total_events += 1
                cadence_str = f" | Interval: {event.get('cadence_step_interval_sec', 0.0):.2f}s ({event.get('cadence_rate_hz', 0.0):.1f} steps/s)" if "cadence_rate_hz" in event else ""
                print("\n" + "=" * 65)
                print(f"*** FOOTSTEP DETECTED ***")
                print(f"Time:             {time_str}")
                print(f"Confidence:       {conf * 100:.1f}%")
                print(f"Rhythm / Cadence: {event.get('notes', 'Verified')}{cadence_str}")
                print(f"RMS Level:        {rms:.4f}")
                print(f"Total Footsteps:  {self.total_events}")
                print("=" * 65 + "\n")

            elif ev_type == "CLAP":
                self.total_claps += 1
                print(f"\n[CLAP]   Hand clap rejected from footstep alarms (Conf: {conf*100:.1f}%, RMS: {rms:.4f})")

            elif ev_type == "KNOCK":
                self.total_knocks += 1
                print(f"\n[KNOCK]  Knock / Rap rejected from footstep alarms (Conf: {conf*100:.1f}%, RMS: {rms:.4f})")

            # Write event to log
            with open(self.event_log_path, "a") as f:
                f.write(f"{now:.4f},{time_str},{ev_type},{conf:.4f},{probs[1]:.4f},{probs[2]:.4f},{probs[3]:.4f},{probs[0]:.4f},{rms:.5f},{event.get('notes', '')}\n")

            if self.record_debug and ev_type == "FOOTSTEP":
                fname = f"step_{int(now)}_{time_str.replace(':', '-')}.wav"
                fpath = self.debug_audio_dir / fname
                sf.write(str(fpath), audio_window, self.sample_rate)

        # Real-Time Visual Display
        if self.is_multiclass:
            log_line = self.event_detector.format_log_line(now, probs, rms, label_str)
            print(log_line, end="\r", flush=True)
        else:
            bar_len = int(probs[1] * 20)
            prob_bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"[{time_str}] [{prob_bar}] Prob: {probs[1]:4.2f} | {label_str:18s} | RMS: {rms:.4f} | Steps: {self.total_events}", end="\r", flush=True)

    def run(self, max_duration_sec: Optional[float] = None):
        """Runs the live microphone or simulation detection loop."""
        if self.simulate_file is not None and self.simulate_file.exists():
            self.run_simulated_file()
            return

        devices = sd.query_devices()
        dev_name = "Default"
        chosen_device = self.device_id

        if chosen_device is not None and chosen_device < len(devices):
            dev_name = devices[chosen_device]["name"]
        else:
            default_in = sd.default.device[0]
            if default_in is not None and default_in >= 0 and default_in < len(devices):
                chosen_device = default_in
                dev_name = devices[default_in]["name"]

        print("\n" + "=" * 70)
        print("REAL-TIME LIVE FOOTSTEP & TRANSIENT DETECTOR")
        print("=" * 70)
        print(f"Microphone:    {dev_name} (ID: {chosen_device})")
        print(f"Sample Rate:   {self.sample_rate} Hz (Mono)")
        print(f"Model:         {self.model_path.name} ({'4-Class MultiClassAudioNet' if self.is_multiclass else 'Binary CNN'})")
        print(f"Step Thresh:   {self.threshold:.2f}")
        print(f"Cadence Check: {'ACTIVE (Requires multiple walking steps spaced 0.25s-0.85s)' if self.require_cadence else 'OFF'}")
        print(f"Max AGC Gain:  +{20.0 * np.log10(self.max_gain_boost):.1f} dB")
        print(f"Status:        INITIALIZING MICROPHONE STREAM...")
        print("=" * 70 + "\n")

        stream_sr = self.sample_rate
        self.needs_resample = False
        try:
            sd.check_input_settings(device=chosen_device, channels=1, samplerate=self.sample_rate, dtype="float32")
        except Exception as e:
            logger.warning(f"Device does not support 16,000 Hz directly ({e}). Attempting fallback to device default rate...")
            dev_info = sd.query_devices(chosen_device, "input")
            stream_sr = int(dev_info["default_samplerate"])
            self.needs_resample = True
            logger.info(f"Using native device rate {stream_sr} Hz with real-time resampling to 16,000 Hz.")

        self.stream_native_sr = stream_sr

        try:
            stream = sd.InputStream(
                device=chosen_device,
                channels=1,
                samplerate=stream_sr,
                dtype="float32",
                callback=self._audio_callback
            )
        except Exception as e:
            logger.error(f"Failed to open microphone stream: {e}")
            logger.error("Run 'python src/list_microphones.py' to list valid input devices and specify with --device <ID>.")
            return

        self.is_running = True
        start_time = time.time()

        with stream:
            logger.info("Microphone active! Distinguishing Footsteps vs Claps vs Knocks... (Press Ctrl+C to stop)")
            time.sleep(self.hop_seconds * 2)

            try:
                while self.is_running:
                    loop_start = time.perf_counter()
                    self._process_latest_window()

                    if max_duration_sec and (time.time() - start_time) >= max_duration_sec:
                        break

                    elapsed = time.perf_counter() - loop_start
                    sleep_time = max(0.005, self.hop_seconds - elapsed)
                    time.sleep(sleep_time)
            except KeyboardInterrupt:
                print("\n[INFO] Stopped by user.")

        print(f"\n[INFO] Live detector finished. Verified footsteps: {self.total_events} | Claps: {self.total_claps} | Knocks: {self.total_knocks}")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Live Microphone Footstep Detector")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", "-d", type=int, default=None, help="Input microphone device ID")
    parser.add_argument("--file", "-f", type=str, default=None, help="Stream an audio file through live detector instead of mic")
    parser.add_argument("--threshold", "-t", type=float, default=0.38, help="Footstep candidate threshold (default: 0.38)")
    parser.add_argument("--high-threshold", type=float, default=0.65, help="High-confidence instant threshold (default: 0.65)")
    parser.add_argument("--confirmations", "-c", type=int, default=2, help="Min positive windows required (default: 2)")
    parser.add_argument("--history", type=int, default=4, help="Window history length for M-of-N check (default: 4)")
    parser.add_argument("--cooldown", type=float, default=0.70, help="Cooldown in seconds between events (default: 0.70s)")
    parser.add_argument("--hop", type=float, default=0.25, help="Hop duration in seconds (default: 0.25s)")
    parser.add_argument("--gain-boost", type=float, default=6.0, help="Max AGC gain multiplier for quiet audio (default: 6.0 = +15.5 dB)")
    parser.add_argument("--duration", type=float, default=None, help="Max run duration in seconds")
    parser.add_argument("--no-cadence", action="store_true", help="Disable walking cadence rhythm requirement")
    parser.add_argument("--record-debug", action="store_true", help="Record audio clips around detected events")
    parser.add_argument("--debug", action="store_true", help="Display detailed signal debug info")
    args = parser.parse_args()

    detector = LiveFootstepDetector(
        model_path=args.model,
        device_id=args.device,
        simulate_file=args.file,
        threshold=args.threshold,
        high_conf_threshold=args.high_threshold,
        min_confirmations=args.confirmations,
        history_len=args.history,
        cooldown_sec=args.cooldown,
        hop_seconds=args.hop,
        require_cadence=not args.no_cadence,
        max_gain_boost=args.gain_boost,
        record_debug=args.record_debug,
        debug_mode=args.debug
    )
    detector.run(max_duration_sec=args.duration)


if __name__ == "__main__":
    main()
