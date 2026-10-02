"""
Temporal Validation & Multi-Class Event Detection Logic.
Transforms raw multi-class window probabilities [other, footstep, clap, knock]
into robust, validated events.

Features:
1. Multi-class impact disambiguation: Compares Footstep vs Clap vs Knock probabilities.
2. Sequence & Walking Cadence Verification:
   A single isolated impact is NOT declared as footsteps.
   A confirmed footstep event requires multiple impacts with a plausible walking rhythm
   (approx. 1.2 to 4.0 steps per second; inter-step interval 0.25s to 0.85s).
3. Independent Clap and Knock detectors with their own debouncing thresholds.
4. Rich per-window logging of [p_step, p_clap, p_knock, p_other], RMS level, and decision rationale.
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import numpy as np


class TemporalEventDetector:
    """
    Backwards-compatible binary temporal validator.
    """
    def __init__(
        self,
        threshold: float = 0.40,
        high_confidence_threshold: float = 0.65,
        min_confirmations: int = 2,
        history_len: int = 4,
        cooldown_sec: float = 0.70,
        smoothing_alpha: float = 0.40
    ):
        self.threshold = threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.min_confirmations = min_confirmations
        self.history_len = history_len
        self.cooldown_sec = cooldown_sec
        self.smoothing_alpha = smoothing_alpha

        self.history: deque = deque(maxlen=history_len)
        self.last_event_time: float = -999.0
        self.smoothed_prob: float = 0.0
        self.in_event: bool = False
        self.current_event_windows: List[Dict[str, Any]] = []

    def update(self, prob: float, timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now = timestamp if timestamp is not None else time.time()
        is_candidate = (prob >= self.threshold)

        if len(self.history) == 0:
            self.smoothed_prob = prob
        else:
            self.smoothed_prob = self.smoothing_alpha * prob + (1.0 - self.smoothing_alpha) * self.smoothed_prob

        self.history.append({"prob": prob, "time": now, "candidate": is_candidate})

        time_since_last = now - self.last_event_time
        if time_since_last < self.cooldown_sec:
            if not is_candidate:
                self.in_event = False
                self.current_event_windows = []
            return None

        positive_count = sum(1 for w in self.history if w["candidate"])

        if is_candidate:
            self.current_event_windows.append({"prob": prob, "time": now})
            is_high_conf = (prob >= self.high_confidence_threshold)
            is_m_of_n = (positive_count >= self.min_confirmations)
            is_smoothed_high = (self.smoothed_prob >= self.threshold and len(self.history) >= 2)

            if (is_high_conf or is_m_of_n or is_smoothed_high) and not self.in_event:
                self.in_event = True
                self.last_event_time = now
                peak_conf = max(w["prob"] for w in self.history if w["candidate"])
                avg_conf = sum(w["prob"] for w in self.history) / len(self.history)

                start_t = self.history[0]["time"]
                duration = max(0.05, now - start_t)

                return {
                    "event_type": "FOOTSTEP",
                    "timestamp": round(now, 4),
                    "time_str": time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 100):02d}",
                    "confidence": round(float(peak_conf), 4),
                    "smoothed_confidence": round(float(self.smoothed_prob), 4),
                    "avg_confidence": round(float(avg_conf), 4),
                    "duration_sec": round(float(duration), 3),
                    "positive_windows": positive_count,
                    "threshold": self.threshold,
                    "trigger_type": "HIGH_CONF" if is_high_conf else ("M_OF_N" if is_m_of_n else "SMOOTHED")
                }
        else:
            if len(self.history) == self.history_len and positive_count == 0:
                self.in_event = False
                self.current_event_windows = []

        return None


class MultiClassTemporalDetector:
    """
    Advanced Multi-Class Event Detector with Sequence / Cadence Verification.
    
    Classes:
    0: OTHER
    1: FOOTSTEP
    2: CLAP
    3: KNOCK
    """
    def __init__(
        self,
        step_threshold: float = 0.38,
        clap_threshold: float = 0.40,
        knock_threshold: float = 0.40,
        high_step_threshold: float = 0.65,
        require_cadence: bool = True,
        min_cadence_interval: float = 0.25,
        max_cadence_interval: float = 0.85,
        cooldown_sec: float = 0.70,
        history_len: int = 5,
        smoothing_alpha: float = 0.35
    ):
        self.step_threshold = step_threshold
        self.clap_threshold = clap_threshold
        self.knock_threshold = knock_threshold
        self.high_step_threshold = high_step_threshold
        self.require_cadence = require_cadence
        self.min_cadence_interval = min_cadence_interval
        self.max_cadence_interval = max_cadence_interval
        self.cooldown_sec = cooldown_sec
        self.history_len = history_len
        self.smoothing_alpha = smoothing_alpha

        self.history: deque = deque(maxlen=history_len)
        self.step_timestamps: deque = deque(maxlen=6)
        self.last_event_time: float = -999.0
        self.last_event_type: Optional[str] = None
        self.smoothed_step_prob: float = 0.0

        self.total_footsteps = 0
        self.total_claps = 0
        self.total_knocks = 0

    def update(
        self,
        probs: np.ndarray,
        rms: float = 0.0,
        timestamp: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Processes new window probabilities: probs = [p_other, p_step, p_clap, p_knock]
        Returns:
            (event_dict_or_None, decision_label_str)
        """
        now = timestamp if timestamp is not None else time.time()
        p_other, p_step, p_clap, p_knock = float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3])

        # 1. Update Exponential Moving Average for step probability
        if len(self.history) == 0:
            self.smoothed_step_prob = p_step
        else:
            self.smoothed_step_prob = self.smoothing_alpha * p_step + (1.0 - self.smoothing_alpha) * self.smoothed_step_prob

        # 2. Determine instant winner class for this window
        max_idx = int(np.argmax(probs))
        winner_class = ["OTHER", "FOOTSTEP", "CLAP", "KNOCK"][max_idx]

        self.history.append({
            "probs": probs,
            "p_step": p_step,
            "p_clap": p_clap,
            "p_knock": p_knock,
            "time": now,
            "rms": rms,
            "winner": winner_class
        })

        time_since_last = now - self.last_event_time
        in_cooldown = (time_since_last < self.cooldown_sec)

        # 3. Check for CLAP (sharp broadband percussive impact)
        if p_clap >= self.clap_threshold and p_clap > p_step and p_clap > p_knock:
            if not in_cooldown or self.last_event_type != "CLAP":
                self.last_event_time = now
                self.last_event_type = "CLAP"
                self.total_claps += 1
                return {
                    "event_type": "CLAP",
                    "timestamp": round(now, 4),
                    "confidence": round(p_clap, 4),
                    "rms": round(rms, 5),
                    "notes": "Sharp broadband hand clap"
                }, "CLAP"
            return None, "CLAP"

        # 4. Check for KNOCK (mid-frequency wooden resonance)
        if p_knock >= self.knock_threshold and p_knock > p_step and p_knock > p_clap:
            if not in_cooldown or self.last_event_type != "KNOCK":
                self.last_event_time = now
                self.last_event_type = "KNOCK"
                self.total_knocks += 1
                return {
                    "event_type": "KNOCK",
                    "timestamp": round(now, 4),
                    "confidence": round(p_knock, 4),
                    "rms": round(rms, 5),
                    "notes": "Resonant knuckle / door impact"
                }, "KNOCK"
            return None, "KNOCK"

        # 5. Check for FOOTSTEP candidate
        is_step_candidate = (p_step >= self.step_threshold and p_step > p_clap and p_step > p_knock)

        if is_step_candidate:
            self.step_timestamps.append(now)

            if in_cooldown and self.last_event_type == "FOOTSTEP":
                return None, "FOOTSTEP (Cooldown)"

            # Check sequence / rhythm cadence logic
            if self.require_cadence:
                # Need at least 2 impacts with walking interval
                cadence_matched = False
                cadence_interval = 0.0

                if len(self.step_timestamps) >= 2:
                    dt = now - self.step_timestamps[-2]
                    if self.min_cadence_interval <= dt <= self.max_cadence_interval:
                        cadence_matched = True
                        cadence_interval = dt

                # Exception: exceptionally high single confidence
                is_super_confident = (p_step >= self.high_step_threshold and self.smoothed_step_prob >= 0.50)

                if cadence_matched or is_super_confident:
                    self.last_event_time = now
                    self.last_event_type = "FOOTSTEP"
                    self.total_footsteps += 1
                    cadence_rate = (1.0 / cadence_interval) if cadence_interval > 0 else 0.0
                    return {
                        "event_type": "FOOTSTEP",
                        "timestamp": round(now, 4),
                        "confidence": round(p_step, 4),
                        "smoothed_confidence": round(self.smoothed_step_prob, 4),
                        "cadence_step_interval_sec": round(cadence_interval, 3),
                        "cadence_rate_hz": round(cadence_rate, 2),
                        "rms": round(rms, 5),
                        "notes": "Verified walking cadence" if cadence_matched else "High-confidence footstep"
                    }, "FOOTSTEP"
                else:
                    return None, "STEP_CANDIDATE (Awaiting Cadence)"
            else:
                # No cadence required: standard M-of-N or instant trigger
                self.last_event_time = now
                self.last_event_type = "FOOTSTEP"
                self.total_footsteps += 1
                return {
                    "event_type": "FOOTSTEP",
                    "timestamp": round(now, 4),
                    "confidence": round(p_step, 4),
                    "rms": round(rms, 5),
                    "notes": "Direct threshold trigger"
                }, "FOOTSTEP"

        return None, "NON_FOOTSTEP"

    def format_log_line(self, now: float, probs: np.ndarray, rms: float, decision_label: str) -> str:
        """Returns formatted diagnostic line for real-time terminal display."""
        p_other, p_step, p_clap, p_knock = probs[0], probs[1], probs[2], probs[3]
        time_str = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 100):02d}"

        # Visual confidence bar for footstep probability
        bar_len = 10
        fill = int(round(p_step * bar_len))
        bar = "#" * fill + "-" * (bar_len - fill)

        return (
            f"[{time_str}] [{bar}] Step: {p_step:.2f} (Smooth: {self.smoothed_step_prob:.2f}) | "
            f"Clap: {p_clap:.2f} | Knock: {p_knock:.2f} | Other: {p_other:.2f} | "
            f"RMS: {rms:.4f} | {decision_label}"
        )
