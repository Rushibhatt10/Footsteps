"""
Temporal Validation & Event Detection Logic.
Transforms raw window probabilities into robust, validated footstep events,
preventing spurious impulse false alarms, handling quiet/distant footsteps with M-of-N debouncing,
and enforcing cooldown to prevent duplicate detections of the same step.
"""

import time
from typing import Optional, Dict, Any, List
from collections import deque


class TemporalEventDetector:
    """
    Validates footstep events across a rolling buffer of sliding windows.

    Detection Criteria (any condition met while not in cooldown):
    1. Fast Trigger: Window probability >= high_confidence_threshold (e.g., 0.65) triggers immediately.
    2. M-of-N Cluster Trigger: At least min_confirmations windows out of the last history_len windows
       exceed candidate threshold (e.g., 2 of the last 4 windows >= 0.40).
    3. Smoothed Probability Trigger: Exponential moving average or recent window average exceeds threshold.

    Debouncing & Cooldown:
    - Enforces cooldown_sec between distinct footstep events so overlapping sliding windows
      from a single step do not produce multiple spurious triggers.
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
        """
        Updates detector with the latest window probability.
        Returns:
            Dict containing event details if a verified footstep event is confirmed, else None.
        """
        now = timestamp if timestamp is not None else time.time()
        is_candidate = (prob >= self.threshold)

        # 1. Update Exponential Moving Average (EMA)
        if len(self.history) == 0:
            self.smoothed_prob = prob
        else:
            self.smoothed_prob = self.smoothing_alpha * prob + (1.0 - self.smoothing_alpha) * self.smoothed_prob

        # 2. Append to rolling window history
        self.history.append({"prob": prob, "time": now, "candidate": is_candidate})

        # 3. Check cooldown
        time_since_last = now - self.last_event_time
        if time_since_last < self.cooldown_sec:
            if not is_candidate:
                self.in_event = False
                self.current_event_windows = []
            return None

        # 4. Count candidate hits in the rolling window
        positive_count = sum(1 for w in self.history if w["candidate"])

        if is_candidate:
            self.current_event_windows.append({"prob": prob, "time": now})

            # Condition 1: High confidence instant trigger
            is_high_conf = (prob >= self.high_confidence_threshold)
            # Condition 2: M-of-N cluster trigger (e.g. 2 of last 4 windows positive)
            is_m_of_n = (positive_count >= self.min_confirmations)
            # Condition 3: Sustained smoothed probability above threshold
            is_smoothed_high = (self.smoothed_prob >= self.threshold and len(self.history) >= 2)

            if (is_high_conf or is_m_of_n or is_smoothed_high) and not self.in_event:
                self.in_event = True
                self.last_event_time = now

                peak_conf = max(w["prob"] for w in self.history if w["candidate"])
                avg_conf = sum(w["prob"] for w in self.history) / len(self.history)

                start_t = self.history[0]["time"]
                duration = max(0.05, now - start_t)

                event = {
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
                return event
        else:
            # Drop below candidate threshold
            if len(self.history) == self.history_len and positive_count == 0:
                self.in_event = False
                self.current_event_windows = []

        return None

    def get_state(self) -> str:
        """Returns current operational state string."""
        positive_count = sum(1 for w in self.history if w.get("candidate", False))
        if self.in_event:
            return "EVENT_CONFIRMED"
        elif positive_count > 0:
            return f"CANDIDATE ({positive_count}/{self.min_confirmations})"
        else:
            return "LISTENING"
