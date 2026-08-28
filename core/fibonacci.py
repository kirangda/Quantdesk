"""Fibonacci retracement / extension built on an auto-detected swing leg.

The leg is found from confirmed fractal pivots so the levels are the ones a
human would draw: last major low -> last major high in an uptrend (and the
mirror image in a downtrend).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import swing_points

RETRACEMENTS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
EXTENSIONS = [1.272, 1.618, 2.0]
GOLDEN_POCKET = (0.5, 0.618)


@dataclass
class FibLeg:
    """One measured swing plus every level projected off it."""

    start_idx: pd.Timestamp
    end_idx: pd.Timestamp
    start_price: float
    end_price: float
    direction: str  # "up" = low->high, "down" = high->low
    levels: dict = field(default_factory=dict)
    extensions: dict = field(default_factory=dict)

    @property
    def span(self) -> float:
        return abs(self.end_price - self.start_price)

    def retrace_pct(self, price: float) -> float:
        """Where `price` sits on the leg: 0.0 = at the end, 1.0 = fully retraced."""
        if self.span == 0:
            return float("nan")
        if self.direction == "up":
            return (self.end_price - price) / self.span
        return (price - self.end_price) / self.span

    def in_golden_pocket(self, price: float, pad: float = 0.02) -> bool:
        r = self.retrace_pct(price)
        return GOLDEN_POCKET[0] - pad <= r <= GOLDEN_POCKET[1] + pad

    def nearest_level(self, price: float):
        """(label, price) of the closest fib line to `price`."""
        merged = {**self.levels, **self.extensions}
        label = min(merged, key=lambda k: abs(merged[k] - price))
        return label, merged[label]


def _build_levels(start_price: float, end_price: float, direction: str):
    span = end_price - start_price
    levels = {}
    for r in RETRACEMENTS:
        # retracing back from the end of the leg toward its start
        levels[f"{r:.3f}".rstrip("0").rstrip(".")] = end_price - span * r
    exts = {}
    for e in EXTENSIONS:
        exts[f"{e:.3f}".rstrip("0").rstrip(".")] = start_price + span * e
    return levels, exts


def _make_leg(window: pd.DataFrame, start_idx, end_idx, direction: str) -> FibLeg | None:
    start_price = float(window.loc[start_idx, "low" if direction == "up" else "high"])
    end_price = float(window.loc[end_idx, "high" if direction == "up" else "low"])
    if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price == end_price:
        return None
    levels, exts = _build_levels(start_price, end_price, direction)
    return FibLeg(start_idx, end_idx, start_price, end_price, direction, levels, exts)


def detect_leg(df: pd.DataFrame, left: int = 5, right: int = 5, lookback: int = 250) -> FibLeg | None:
    """Pick the most recent still-valid impulse leg and project fibs from it.

    A leg price has already retraced past ~115% is dead - the move it measured
    has been erased - so we walk back to the previous pivot pair instead of
    drawing levels nobody would trade.
    """
    window = df.tail(lookback)
    if len(window) < (left + right + 2):
        return None

    hi, lo = swing_points(window["high"], window["low"], left, right)
    highs, lows = hi.dropna(), lo.dropna()
    if highs.empty or lows.empty:
        return None

    price = float(window["close"].iloc[-1])
    fallback: FibLeg | None = None

    # candidate legs, most recent first: pair each swing with the latest
    # opposite swing that precedes it
    candidates = []
    for h_idx in reversed(highs.index):
        prior_lows = lows.index[lows.index < h_idx]
        if len(prior_lows):
            candidates.append((max(h_idx, prior_lows[-1]), prior_lows[-1], h_idx, "up"))
    for l_idx in reversed(lows.index):
        prior_highs = highs.index[highs.index < l_idx]
        if len(prior_highs):
            candidates.append((max(l_idx, prior_highs[-1]), prior_highs[-1], l_idx, "down"))
    candidates.sort(key=lambda t: t[0], reverse=True)

    for _, start_idx, end_idx, direction in candidates[:8]:
        leg = _make_leg(window, start_idx, end_idx, direction)
        if leg is None:
            continue
        if fallback is None:
            fallback = leg
        depth = leg.retrace_pct(price)
        if np.isfinite(depth) and -0.60 <= depth <= 1.15:
            return leg

    return fallback


def retracement_series(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.Series:
    """Bar-by-bar retracement depth of the then-current leg (for backtesting).

    Only pivots already confirmed at each bar are used, so there is no lookahead.
    """
    hi, lo = swing_points(df["high"], df["low"], left, right)
    # a pivot is only knowable `right` bars after it prints
    hi = hi.shift(right)
    lo = lo.shift(right)

    last_hi = hi.ffill()
    last_lo = lo.ffill()
    hi_time = pd.Series(np.where(hi.notna(), np.arange(len(df)), np.nan), index=df.index).ffill()
    lo_time = pd.Series(np.where(lo.notna(), np.arange(len(df)), np.nan), index=df.index).ffill()

    up_leg = hi_time > lo_time
    span = (last_hi - last_lo).replace(0, np.nan)
    close = df["close"]
    retr = np.where(up_leg, (last_hi - close) / span, (close - last_lo) / span)
    return pd.Series(retr, index=df.index)


# --------------------------------------------------------------------------
# Plain-language interpretation
# --------------------------------------------------------------------------
# (lower bound, upper bound, headline, what it means for a trader)
RETRACE_ZONES = [
    (float("-inf"), 0.0, "Extended beyond the leg",
     "Price has pushed past the end of the measured swing - this leg is being "
     "superseded by a new impulse. Watch the extension levels, not the retracements."),
    (0.0, 0.236, "Shallow pullback",
     "Barely a pause. The trend is firm but there is no discount here - entries "
     "this close to the high carry a poor reward-to-risk."),
    (0.236, 0.45, "Normal pullback",
     "A healthy, ordinary retracement. Not yet the high-probability zone, but "
     "constructive if the trend holds."),
    (0.45, 0.70, "Golden pocket",
     "The 0.5-0.618 zone where trend continuation most often resumes. This is the "
     "level the Fibonacci strategy waits for - but it needs a reversal bar to confirm."),
    (0.70, 1.0, "Deep retracement",
     "Past the 0.786. The move is giving back most of its gain and the original "
     "impulse is losing credibility. Reduce size or stand aside."),
    (1.0, float("inf"), "Leg invalidated",
     "The entire swing has been retraced. These levels no longer describe the "
     "market - wait for a new impulse leg to form."),
]


def describe(leg: FibLeg, price: float) -> dict:
    """Where price sits on the leg, in words a trader can act on."""
    depth = leg.retrace_pct(price)
    if not np.isfinite(depth):
        return {}

    headline, meaning = "", ""
    for lo, hi, head, mean in RETRACE_ZONES:
        if lo <= depth < hi:
            headline, meaning = head, mean
            break

    label, level_price = leg.nearest_level(price)
    merged = {**leg.levels, **leg.extensions}
    above = sorted((p for p in merged.values() if p > price))
    below = sorted((p for p in merged.values() if p < price), reverse=True)

    return {
        "depth": depth,
        "depth_pct": depth * 100.0,
        "headline": headline,
        "meaning": meaning,
        "in_pocket": leg.in_golden_pocket(price),
        "nearest_label": label,
        "nearest_price": level_price,
        "nearest_distance_pct": (level_price / price - 1.0) * 100.0 if price else float("nan"),
        "next_resistance": above[0] if above else float("nan"),
        "next_support": below[0] if below else float("nan"),
    }


def level_rows(leg: FibLeg, price: float) -> list[dict]:
    """Every level, sorted high to low, tagged for the table view."""
    merged = [(k, v, "retracement") for k, v in leg.levels.items()]
    merged += [(f"ext {k}", v, "extension") for k, v in leg.extensions.items()]
    nearest_label, _ = leg.nearest_level(price)

    rows = []
    for label, lvl, kind in sorted(merged, key=lambda t: t[1], reverse=True):
        rows.append({
            "label": label,
            "price": lvl,
            "kind": kind,
            "zone": "Resistance" if lvl > price else "Support",
            "distance_pct": (lvl / price - 1.0) * 100.0 if price else float("nan"),
            "is_pocket": label in ("0.5", "0.618"),
            "is_nearest": label == nearest_label,
        })
    return rows
