"""Signal engine: premium history, winsorize, rolling quantile thresholds."""
import logging
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

LONG, SHORT, FLAT = "LONG", "SHORT", "FLAT"


# ===========================================================================
# Premium history ring-buffer
# ===========================================================================

class PremiumHistory:
    """Keep last N hourly premium observations."""
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buf: Deque[Tuple[int, float]] = deque(maxlen=maxlen)   # (ts_ms, premium)

    def push(self, ts_ms: int, premium: float):
        if self.buf and self.buf[-1][0] == ts_ms:
            self.buf[-1] = (ts_ms, float(premium))  # overwrite stale warmup value
            return
        self.buf.append((ts_ms, float(premium)))

    def __len__(self):
        return len(self.buf)

    def values(self) -> np.ndarray:
        return np.array([v for _, v in self.buf], dtype=float)


# ===========================================================================
# Threshold computation
# ===========================================================================

class ThresholdEngine:
    """Compute winsorized rolling quantile thresholds."""
    def __init__(self, win_lo: float, win_hi: float, q_lo: float, q_hi: float):
        self.win_lo = win_lo
        self.win_hi = win_hi
        self.q_lo = q_lo
        self.q_hi = q_hi

    def compute(self, arr: np.ndarray) -> Tuple[float, float]:
        if len(arr) < 10:
            return float("-inf"), float("inf")
        lo = np.quantile(arr, self.win_lo)
        hi = np.quantile(arr, self.win_hi)
        winsorized = np.clip(arr, lo, hi)
        q_lo_v = float(np.quantile(winsorized, self.q_lo))
        q_hi_v = float(np.quantile(winsorized, self.q_hi))
        return q_lo_v, q_hi_v


# ===========================================================================
# Signal generation
# ===========================================================================

class SignalGenerator:
    """Given current premium + thresholds, emit LONG / SHORT / FLAT."""
    def __init__(self, history: PremiumHistory, engine: ThresholdEngine,
                 min_history: int):
        self.hist = history
        self.eng = engine
        self.min_history = min_history

    def generate(self, current_premium: float) -> Dict:
        result = {
            "signal": FLAT,
            "premium": current_premium,
            "q_lo": None,
            "q_hi": None,
            "hist_len": len(self.hist),
            "reason": "",
            "ts_ms": int(time.time() * 1000),
        }
        if len(self.hist) < self.min_history:
            result["reason"] = f"warmup {len(self.hist)}/{self.min_history}"
            return result

        arr = self.hist.values()
        q_lo, q_hi = self.eng.compute(arr)
        result["q_lo"], result["q_hi"] = q_lo, q_hi

        if current_premium <= q_lo:
            result["signal"] = LONG
            result["reason"] = f"premium {current_premium:.6f} <= q_lo {q_lo:.6f}"
        elif current_premium >= q_hi:
            result["signal"] = SHORT
            result["reason"] = f"premium {current_premium:.6f} >= q_hi {q_hi:.6f}"
        else:
            result["reason"] = "within band"
        return result
