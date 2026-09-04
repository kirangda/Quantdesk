"""The strategy book.

Nineteen independent, well-documented systems. Each one publishes:

  * entry / exit boolean series for both directions (used by the backtester)
  * an ATR stop multiple and reward:risk profile (used by risk sizing)
  * a live "entry reference" price - the level to actually place the order at
  * a human-readable checklist explaining the current bar

Rules are evaluated on the *close* of each bar and the backtester fills on the
*next* bar's open, so nothing here can peek at the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ta
from .data import bars_per_year
from .fibonacci import retracement_series


# --------------------------------------------------------------------------
# Shared feature frame
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, intraday: bool = False,
                   interval: str = "1d") -> pd.DataFrame:
    """Compute every indicator once; all strategies read from this frame.

    `interval` lets the long-horizon features mean the same thing in calendar
    terms whether the bars are daily, weekly or monthly.
    """
    f = df.copy()
    o, h, l, c, v = f["open"], f["high"], f["low"], f["close"], f["volume"]

    f["ema9"] = ta.ema(c, 9)
    f["ema20"] = ta.ema(c, 20)
    f["ema50"] = ta.ema(c, 50)
    f["ema200"] = ta.ema(c, 200)
    f["sma5"] = ta.sma(c, 5)
    f["sma50"] = ta.sma(c, 50)
    f["sma200"] = ta.sma(c, 200)

    f["rsi2"] = ta.rsi(c, 2)
    f["rsi14"] = ta.rsi(c, 14)

    m = ta.macd(c)
    f["macd"], f["macd_signal"], f["macd_hist"] = m["macd"], m["signal"], m["hist"]

    st_full = ta.stochastic(h, l, c)
    f["stoch_k"], f["stoch_d"] = st_full["k"], st_full["d"]

    f["atr"] = ta.atr(h, l, c, 14)
    f["atr_pct"] = f["atr"] / c * 100.0

    bb = ta.bollinger(c, 20, 2.0)
    f["bb_upper"], f["bb_lower"], f["bb_mid"] = bb["upper"], bb["lower"], bb["mid"]
    f["bb_width"], f["bb_pctb"] = bb["width"], bb["pctb"]

    kc = ta.keltner(h, l, c, 20, 1.5)
    f["kc_upper"], f["kc_lower"] = kc["upper"], kc["lower"]

    dc = ta.donchian(h, l, 20)
    f["dc_upper"], f["dc_lower"] = dc["upper"], dc["lower"]
    dc10 = ta.donchian(h, l, 10)
    f["dc10_upper"], f["dc10_lower"] = dc10["upper"], dc10["lower"]

    adx_full = ta.adx(h, l, c, 14)
    f["adx"], f["plus_di"], f["minus_di"] = adx_full["adx"], adx_full["plus_di"], adx_full["minus_di"]

    stf = ta.supertrend(h, l, c, 10, 3.0)
    f["supertrend"], f["st_dir"] = stf["supertrend"], stf["direction"]

    f["vwap"] = ta.vwap(h, l, c, v, session_reset=intraday)
    f["rvol"] = ta.relative_volume(v, 20)
    f["obv"] = ta.obv(c, v)
    f["obv_slope"] = f["obv"].diff(5)

    # --- ichimoku ---------------------------------------------------------
    ich = ta.ichimoku(h, l, c)
    for col in ("tenkan", "kijun", "senkou_a", "senkou_b", "cloud_top", "cloud_bot"):
        f[col] = ich[col]

    # --- heikin-ashi ------------------------------------------------------
    ha = ta.heikin_ashi(o, h, l, c)
    for col in ("ha_open", "ha_close", "ha_bull", "ha_strong_up", "ha_strong_dn"):
        f[col] = ha[col]

    # --- divergence, gaps, 52-week position -------------------------------
    f["rsi_div"] = ta.divergence(l, h, f["rsi14"])
    f["prev_close"] = c.shift(1)
    f["gap_pct"] = (o - c.shift(1)) / c.shift(1) * 100.0
    f["high_52w"] = h.rolling(252, min_periods=40).max()
    f["from_52w_pct"] = (c / f["high_52w"] - 1.0) * 100.0

    # --- volatility / volume contraction (VCP) ----------------------------
    f["atr_pct_prev"] = f["atr_pct"].shift(20)
    f["vol_ma"] = v.rolling(20, min_periods=10).mean()
    f["vol_contract"] = (f["vol_ma"] / f["vol_ma"].shift(20)).replace([np.inf, -np.inf], np.nan)

    dc55 = ta.donchian(h, l, 55)
    f["dc55_upper"], f["dc55_lower"] = dc55["upper"], dc55["lower"]

    # --- opening range (intraday sessions only) ---------------------------
    if intraday and isinstance(f.index, pd.DatetimeIndex) and len(f) > 2:
        minutes = max(1, int(round(pd.Series(f.index).diff().dt.total_seconds().median() / 60)))
        k = max(1, int(round(30 / minutes)))          # the first 30 minutes
        day = f.index.normalize()
        g = f.groupby(day)
        f["or_high"] = g["high"].transform(lambda x: x.iloc[:k].max())
        f["or_low"] = g["low"].transform(lambda x: x.iloc[:k].min())
        f["bar_of_day"] = g.cumcount()
        f["or_bars"] = k
    else:
        f["or_high"] = np.nan
        f["or_low"] = np.nan
        f["bar_of_day"] = 0
        f["or_bars"] = 0

    # --- calendar-scaled long-horizon features ----------------------------
    # 10 months, 12 months and 1 month expressed in bars of THIS interval, so
    # the Faber and momentum rules mean the same thing on daily and monthly data
    bpy = bars_per_year(interval)
    per_month = max(1, int(round(bpy / 12)))
    m10 = max(3, int(round(bpy * 10 / 12)))
    m12 = max(4, int(round(bpy)))
    f["ma_10mo"] = ta.sma(c, m10)
    f["above_10mo"] = (c > f["ma_10mo"]).astype(float)
    # 12-month return skipping the most recent month - the classic momentum
    # window, which deliberately omits the short-term reversal effect
    f["ret_12_1"] = (c.shift(per_month) / c.shift(m12) - 1.0) * 100.0

    # distance from a long-run anchor, in standard deviations
    anchor = max(6, int(round(bpy * 3)))
    log_c = np.log(c.replace(0, np.nan))
    f["value_anchor"] = np.exp(log_c.rolling(anchor, min_periods=max(6, anchor // 3)).mean())
    dev = log_c - np.log(f["value_anchor"])
    f["value_z"] = dev / dev.rolling(anchor, min_periods=max(6, anchor // 3)).std(ddof=0)

    f["fib_retr"] = retracement_series(f)
    f["swing_low_20"] = l.rolling(20, min_periods=5).min()
    f["swing_high_20"] = h.rolling(20, min_periods=5).max()

    # candle anatomy used by reversal triggers
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    f["body_pct"] = body / rng
    f["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    f["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    f["bullish_bar"] = (c > o) & (c > h.shift(1))
    f["bearish_bar"] = (c < o) & (c < l.shift(1))
    return f


def cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _b(s: pd.Series) -> pd.Series:
    """Coerce a comparison result to a clean boolean series."""
    return s.fillna(False).astype(bool)


# a structural stop is only worth using while it stays within this many ATR
# stop-multiples of entry; past that the trade risk is no longer practical
MAX_STOP_ATR_FACTOR = 1.25


def resolve_stop(entry: float, atr: float, stop_atr: float, structural: float,
                 direction: str) -> float:
    """Combine an ATR stop with the nearest swing, capped at a sane distance."""
    if not np.isfinite(atr) or atr <= 0:
        atr = max(abs(entry) * 0.01, 1e-9)
    base = stop_atr * atr
    cap = MAX_STOP_ATR_FACTOR * base

    if direction == "LONG":
        stop = entry - base
        if np.isfinite(structural) and structural - 0.15 * atr < stop:
            stop = max(structural - 0.15 * atr, entry - cap)
        return stop

    stop = entry + base
    if np.isfinite(structural) and structural + 0.15 * atr > stop:
        stop = min(structural + 0.15 * atr, entry + cap)
    return stop


# --------------------------------------------------------------------------
# Strategy contract
# --------------------------------------------------------------------------
@dataclass
class Rules:
    long_entry: pd.Series
    long_exit: pd.Series
    short_entry: pd.Series
    short_exit: pd.Series


@dataclass
class Strategy:
    key: str
    name: str
    family: str
    blurb: str
    styles: set = field(default_factory=lambda: {"swing", "day"})
    stop_atr: float = 2.0
    targets_r: tuple = (1.0, 2.0, 3.0)
    max_hold: int = 40
    allow_short: bool = True

    # -- to be implemented by each concrete strategy -------------------------
    def rules(self, f: pd.DataFrame) -> Rules:  # pragma: no cover - interface
        raise NotImplementedError

    def entry_reference(self, f: pd.DataFrame, direction: str) -> float:
        """Price level to actually place the order at, for the latest bar."""
        return float(f["close"].iloc[-1])

    def checklist(self, f: pd.DataFrame, direction: str) -> list[tuple[str, bool, str]]:
        """(label, passed, detail) rows describing the current setup."""
        return []

    def stop_price(self, f: pd.DataFrame, entry: float, direction: str) -> float:
        """ATR stop, widened to sit behind structure - but never absurdly wide."""
        atr = float(f["atr"].iloc[-1])
        structural = float(
            f["swing_low_20"].iloc[-1] if direction == "LONG" else f["swing_high_20"].iloc[-1]
        )
        return resolve_stop(entry, atr, self.stop_atr, structural, direction)


# --------------------------------------------------------------------------
# 1. Trend pullback
# --------------------------------------------------------------------------
class TrendPullback(Strategy):
    def __init__(self):
        super().__init__(
            key="trend_pullback",
            name="Trend Pullback Rider",
            family="Trend following",
            blurb=(
                "Buys dips inside an established trend: price above the 200 EMA, "
                "20 EMA above the 50 EMA, ADX confirming direction, then a pullback "
                "into the 20 EMA that closes back above the previous bar's high."
            ),
            stop_atr=1.5,
            targets_r=(1.0, 2.0, 3.5),
            max_hold=30,
            styles={"swing", "day", "long"},
        )

    def rules(self, f):
        c, l, h = f["close"], f["low"], f["high"]
        up_regime = _b((c > f["ema200"]) & (f["ema20"] > f["ema50"]) & (f["adx"] > 18))
        dn_regime = _b((c < f["ema200"]) & (f["ema20"] < f["ema50"]) & (f["adx"] > 18))

        # pullback = touched the 20 EMA (or cooled off on RSI) within the last 3 bars
        touched_up = _b((l <= f["ema20"] * 1.005) | (f["rsi14"] < 45)).rolling(3, min_periods=1).max().astype(bool)
        touched_dn = _b((h >= f["ema20"] * 0.995) | (f["rsi14"] > 55)).rolling(3, min_periods=1).max().astype(bool)

        long_entry = up_regime & touched_up & _b(c > h.shift(1)) & _b(c > f["ema20"])
        short_entry = dn_regime & touched_dn & _b(c < l.shift(1)) & _b(c < f["ema20"])

        long_exit = _b(c < f["ema50"]) | cross_below(f["ema20"], f["ema50"])
        short_exit = _b(c > f["ema50"]) | cross_above(f["ema20"], f["ema50"])
        return Rules(long_entry, _b(long_exit), short_entry, _b(short_exit))

    def entry_reference(self, f, direction):
        last, ema20 = float(f["close"].iloc[-1]), float(f["ema20"].iloc[-1])
        # if price is extended above the EMA, the plan is to wait for the retest
        if direction == "LONG":
            return last if last <= ema20 * 1.02 else ema20 * 1.005
        return last if last >= ema20 * 0.98 else ema20 * 0.995

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        return [
            ("Price vs 200 EMA",
             bool(r["close"] > r["ema200"]) == up,
             f"{r['close']:.2f} vs {r['ema200']:.2f}"),
            ("20/50 EMA alignment",
             bool(r["ema20"] > r["ema50"]) == up,
             f"EMA20 {r['ema20']:.2f} / EMA50 {r['ema50']:.2f}"),
            ("ADX > 18 (trend has force)", bool(r["adx"] > 18), f"ADX {r['adx']:.1f}"),
            ("Pullback into 20 EMA",
             bool(abs(r["close"] - r["ema20"]) <= 1.5 * r["atr"]),
             f"{abs(r['close'] - r['ema20']) / r['atr']:.2f} ATR from EMA20"),
            ("Momentum not exhausted",
             bool(30 < r["rsi14"] < 72) if up else bool(28 < r["rsi14"] < 70),
             f"RSI14 {r['rsi14']:.1f}"),
        ]


# --------------------------------------------------------------------------
# 2. RSI(2) mean reversion
# --------------------------------------------------------------------------
class RSI2Reversion(Strategy):
    def __init__(self):
        super().__init__(
            key="rsi2_reversion",
            name="RSI(2) Mean Reversion",
            family="Mean reversion",
            blurb=(
                "The Connors short-term reversal: only trades with the 200 SMA trend, "
                "buys a 2-period RSI washout below 10, and exits fast on the snap-back "
                "above the 5 SMA. High hit-rate, short holding period."
            ),
            styles={"swing", "day"},
            stop_atr=2.0,
            targets_r=(0.8, 1.5, 2.5),
            max_hold=8,
        )

    def rules(self, f):
        c = f["close"]
        long_entry = _b((c > f["sma200"]) & (f["rsi2"] < 10) & (c < f["sma5"]))
        long_exit = _b((c > f["sma5"]) | (f["rsi2"] > 70))
        short_entry = _b((c < f["sma200"]) & (f["rsi2"] > 90) & (c > f["sma5"]))
        short_exit = _b((c < f["sma5"]) | (f["rsi2"] < 30))
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        # buy the next stab lower rather than chasing - roughly a 0.5 ATR limit
        if direction == "LONG":
            return float(min(r["close"], r["close"] - 0.25 * r["atr"]))
        return float(max(r["close"], r["close"] + 0.25 * r["atr"]))

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        return [
            ("Long-term trend filter (200 SMA)",
             bool(r["close"] > r["sma200"]) == up,
             f"{r['close']:.2f} vs SMA200 {r['sma200']:.2f}"),
            ("RSI(2) at an extreme",
             bool(r["rsi2"] < 10) if up else bool(r["rsi2"] > 90),
             f"RSI2 {r['rsi2']:.1f}"),
            ("Stretched from the 5 SMA",
             bool(r["close"] < r["sma5"]) == up,
             f"{r['close']:.2f} vs SMA5 {r['sma5']:.2f}"),
            ("Volatility is tradeable", bool(r["atr_pct"] > 0.4), f"ATR {r['atr_pct']:.2f}% of price"),
        ]


# --------------------------------------------------------------------------
# 3. MACD momentum with Supertrend confirmation
# --------------------------------------------------------------------------
class MACDMomentum(Strategy):
    def __init__(self):
        super().__init__(
            key="macd_momentum",
            name="MACD Momentum + Supertrend",
            family="Momentum",
            blurb=(
                "Momentum ignition: MACD crosses its signal line while the Supertrend "
                "is already pointing the same way and ADX is above 20. Rides the "
                "expansion phase and exits when the cross reverses or Supertrend flips."
            ),
            stop_atr=2.0,
            targets_r=(1.0, 2.0, 3.0),
            max_hold=35,
            styles={"swing", "day", "long"},
        )

    def rules(self, f):
        c = f["close"]
        long_entry = (
            cross_above(f["macd"], f["macd_signal"])
            & _b(f["st_dir"] > 0)
            & _b(f["adx"] > 20)
            & _b(c > f["ema50"])
        )
        short_entry = (
            cross_below(f["macd"], f["macd_signal"])
            & _b(f["st_dir"] < 0)
            & _b(f["adx"] > 20)
            & _b(c < f["ema50"])
        )
        long_exit = cross_below(f["macd"], f["macd_signal"]) | _b(f["st_dir"] < 0)
        short_exit = cross_above(f["macd"], f["macd_signal"]) | _b(f["st_dir"] > 0)
        return Rules(_b(long_entry), _b(long_exit), _b(short_entry), _b(short_exit))

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        # momentum entries are stop-orders through the current bar's extreme
        return float(r["high"] + 0.05 * r["atr"]) if direction == "LONG" else float(r["low"] - 0.05 * r["atr"])

    def checklist(self, f, direction):
        r, p = f.iloc[-1], f.iloc[-2]
        up = direction == "LONG"
        crossed = (r["macd"] > r["macd_signal"]) if up else (r["macd"] < r["macd_signal"])
        return [
            ("MACD on the right side of signal", bool(crossed),
             f"MACD {r['macd']:.4f} / signal {r['macd_signal']:.4f}"),
            ("Histogram expanding",
             bool(abs(r["macd_hist"]) > abs(p["macd_hist"])),
             f"hist {r['macd_hist']:.4f} (prev {p['macd_hist']:.4f})"),
            ("Supertrend agrees", bool(r["st_dir"] > 0) == up,
             f"flip level {r['supertrend']:.2f}"),
            ("ADX > 20", bool(r["adx"] > 20), f"ADX {r['adx']:.1f}"),
            ("Above/below 50 EMA", bool(r["close"] > r["ema50"]) == up,
             f"EMA50 {r['ema50']:.2f}"),
        ]


# --------------------------------------------------------------------------
# 4. Fibonacci golden pocket
# --------------------------------------------------------------------------
class FibGoldenPocket(Strategy):
    def __init__(self):
        super().__init__(
            key="fib_golden_pocket",
            name="Fibonacci Golden Pocket",
            family="Retracement",
            blurb=(
                "Measures the last impulse leg and waits for price to retrace into the "
                "0.5-0.618 golden pocket, then requires a reversal bar to confirm. "
                "Stop sits just under the 0.786; targets run to the 1.272/1.618 extension."
            ),
            stop_atr=1.2,
            targets_r=(1.2, 2.2, 3.6),
            max_hold=30,
            styles={"swing", "day", "long"},
        )

    def rules(self, f):
        c = f["close"]
        retr = f["fib_retr"]
        in_pocket = _b((retr >= 0.45) & (retr <= 0.70))
        up_trend = _b((f["ema50"] > f["ema200"]) | (c > f["ema200"]))
        dn_trend = _b((f["ema50"] < f["ema200"]) | (c < f["ema200"]))

        long_entry = in_pocket & up_trend & _b(f["bullish_bar"]) & _b(f["rsi14"] > 36)
        short_entry = in_pocket & dn_trend & _b(f["bearish_bar"]) & _b(f["rsi14"] < 64)

        long_exit = _b((retr < 0.0) | (f["rsi14"] > 74) | (c < f["ema200"] * 0.97))
        short_exit = _b((retr < 0.0) | (f["rsi14"] < 26) | (c > f["ema200"] * 1.03))
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def entry_reference(self, f, direction):
        """Place the order at the 0.618 line of the current leg when possible."""
        from .fibonacci import detect_leg

        leg = detect_leg(f)
        last = float(f["close"].iloc[-1])
        if leg is None:
            return last
        gp = leg.levels.get("0.618")
        if gp is None:
            return last
        gp = float(gp)
        # only use the fib level if we are not already through it
        if direction == "LONG" and leg.direction == "up" and last > gp:
            return gp
        if direction == "SHORT" and leg.direction == "down" and last < gp:
            return gp
        return last

    def checklist(self, f, direction):
        from .fibonacci import detect_leg

        r = f.iloc[-1]
        leg = detect_leg(f)
        up = direction == "LONG"
        depth = leg.retrace_pct(float(r["close"])) if leg else float(r["fib_retr"])
        rows = [
            ("Impulse leg identified", leg is not None,
             f"{leg.direction}-leg {leg.start_price:.2f} to {leg.end_price:.2f}" if leg else "no clean swing"),
            ("Retracement in 0.5-0.618 pocket",
             bool(0.45 <= depth <= 0.70) if np.isfinite(depth) else False,
             f"retraced {depth * 100:.1f}%" if np.isfinite(depth) else "n/a"),
            ("Leg direction matches trade",
             (leg.direction == "up") == up if leg else False,
             leg.direction if leg else "n/a"),
            ("Reversal bar printed",
             bool(r["bullish_bar"]) if up else bool(r["bearish_bar"]),
             f"body {r['body_pct'] * 100:.0f}% of range"),
            ("RSI holding the pocket",
             bool(r["rsi14"] > 36) if up else bool(r["rsi14"] < 64),
             f"RSI14 {r['rsi14']:.1f}"),
        ]
        return rows

    def stop_price(self, f, entry, direction):
        from .fibonacci import detect_leg

        leg = detect_leg(f)
        atr = float(f["atr"].iloc[-1])
        if leg is not None and "0.786" in leg.levels and np.isfinite(atr) and atr > 0:
            fib786 = float(leg.levels["0.786"])
            # use the 0.786 invalidation level only while it is a practical distance away
            if abs(entry - fib786) <= MAX_STOP_ATR_FACTOR * self.stop_atr * atr:
                if direction == "LONG" and fib786 < entry:
                    return fib786 - 0.2 * atr
                if direction == "SHORT" and fib786 > entry:
                    return fib786 + 0.2 * atr
        return super().stop_price(f, entry, direction)


# --------------------------------------------------------------------------
# 5. Volatility squeeze breakout
# --------------------------------------------------------------------------
class SqueezeBreakout(Strategy):
    def __init__(self):
        super().__init__(
            key="squeeze_breakout",
            name="Volatility Squeeze Breakout",
            family="Breakout",
            blurb=(
                "Bollinger Bands compressing inside the Keltner channel mark stored "
                "energy. The trade fires when that squeeze releases and price clears the "
                "20-bar Donchian edge on above-average volume."
            ),
            stop_atr=1.8,
            targets_r=(1.0, 2.0, 3.5),
            max_hold=30,
            styles={"swing", "day", "long"},
        )

    def rules(self, f):
        c = f["close"]
        squeeze = _b((f["bb_upper"] < f["kc_upper"]) & (f["bb_lower"] > f["kc_lower"]))
        # released within the last 5 bars, or bandwidth still in its lowest quartile
        recently_squeezed = squeeze.shift(1).rolling(5, min_periods=1).max().astype(bool)
        tight = _b(f["bb_width"] <= f["bb_width"].rolling(60, min_periods=20).quantile(0.30))
        primed = recently_squeezed | tight

        vol_ok = _b(f["rvol"] > 1.1)
        long_entry = primed & _b(c > f["dc_upper"].shift(1)) & vol_ok & _b(c > f["ema50"])
        short_entry = primed & _b(c < f["dc_lower"].shift(1)) & vol_ok & _b(c < f["ema50"])

        long_exit = _b((c < f["dc10_lower"].shift(1)) | (f["st_dir"] < 0))
        short_exit = _b((c > f["dc10_upper"].shift(1)) | (f["st_dir"] > 0))
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        # buy-stop just beyond the channel edge
        if direction == "LONG":
            return float(max(r["dc_upper"], r["high"]) + 0.05 * r["atr"])
        return float(min(r["dc_lower"], r["low"]) - 0.05 * r["atr"])

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        sq_now = bool(r["bb_upper"] < r["kc_upper"] and r["bb_lower"] > r["kc_lower"])
        recent = f.tail(6)
        sq_recent = bool(((recent["bb_upper"] < recent["kc_upper"]) & (recent["bb_lower"] > recent["kc_lower"])).any())
        width_rank = float(
            (f["bb_width"].tail(60) <= r["bb_width"]).mean() * 100
        ) if np.isfinite(r["bb_width"]) else float("nan")
        return [
            ("Squeeze present or just released", sq_now or sq_recent,
             "in squeeze" if sq_now else ("released < 5 bars ago" if sq_recent else "no squeeze")),
            ("Bandwidth compressed", bool(width_rank <= 40),
             f"band width in {width_rank:.0f}th percentile of 60 bars"),
            ("Breaking the 20-bar channel",
             bool(r["close"] > r["dc_upper"] * 0.999) if up else bool(r["close"] < r["dc_lower"] * 1.001),
             f"channel {r['dc_lower']:.2f} - {r['dc_upper']:.2f}"),
            ("Volume confirms", bool(r["rvol"] > 1.1), f"{r['rvol']:.2f}x 20-bar average"),
            ("Trend side of 50 EMA", bool(r["close"] > r["ema50"]) == up, f"EMA50 {r['ema50']:.2f}"),
        ]


# --------------------------------------------------------------------------
# 6. VWAP reclaim (intraday only)
# --------------------------------------------------------------------------
class VWAPReclaim(Strategy):
    def __init__(self):
        super().__init__(
            key="vwap_reclaim",
            name="VWAP Trend Reclaim",
            family="Intraday",
            blurb=(
                "The institutional day-trade workhorse: price holds above session VWAP, "
                "pulls back to test it, and reclaims with the 9 EMA above the 20 EMA. "
                "Stop goes below the VWAP test, targets are prior session extremes."
            ),
            styles={"day"},
            stop_atr=1.2,
            targets_r=(1.0, 1.8, 3.0),
            max_hold=24,
        )

    def rules(self, f):
        c, h, l = f["close"], f["high"], f["low"]
        above = _b(c > f["vwap"])
        below = _b(c < f["vwap"])
        tested_up = _b(l <= f["vwap"] * 1.002).rolling(4, min_periods=1).max().astype(bool)
        tested_dn = _b(h >= f["vwap"] * 0.998).rolling(4, min_periods=1).max().astype(bool)

        long_entry = above & tested_up & _b(c > h.shift(1)) & _b(f["ema9"] > f["ema20"]) & _b(f["rsi14"] > 45)
        short_entry = below & tested_dn & _b(c < l.shift(1)) & _b(f["ema9"] < f["ema20"]) & _b(f["rsi14"] < 55)

        long_exit = _b((c < f["vwap"]) & (c.shift(1) < f["vwap"].shift(1)))
        short_exit = _b((c > f["vwap"]) & (c.shift(1) > f["vwap"].shift(1)))
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        vw = float(r["vwap"])
        last = float(r["close"])
        if direction == "LONG":
            return last if last <= vw * 1.005 else vw * 1.002
        return last if last >= vw * 0.995 else vw * 0.998

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        return [
            ("Right side of session VWAP", bool(r["close"] > r["vwap"]) == up,
             f"{r['close']:.2f} vs VWAP {r['vwap']:.2f}"),
            ("9/20 EMA alignment", bool(r["ema9"] > r["ema20"]) == up,
             f"EMA9 {r['ema9']:.2f} / EMA20 {r['ema20']:.2f}"),
            ("Recent VWAP test", bool(abs(r["close"] - r["vwap"]) < 1.2 * r["atr"]),
             f"{abs(r['close'] - r['vwap']) / r['atr']:.2f} ATR from VWAP"),
            ("Momentum onside", bool(r["rsi14"] > 45) if up else bool(r["rsi14"] < 55),
             f"RSI14 {r['rsi14']:.1f}"),
            ("Participation", bool(r["rvol"] > 0.9), f"{r['rvol']:.2f}x average volume"),
        ]


# --------------------------------------------------------------------------
# 7. Golden cross regime
# --------------------------------------------------------------------------
class GoldenCross(Strategy):
    def __init__(self):
        super().__init__(
            key="golden_cross",
            name="Golden Cross Regime",
            family="Trend following",
            blurb=(
                "The most-quoted signal in the business: the 50 SMA above the 200 SMA "
                "defines the regime, and the trade is taken on the cross itself or on "
                "each reclaim of the 50 SMA while that regime holds. Slow and "
                "low-frequency, but it holds whole trends."
            ),
            styles={"swing", "long"},
            stop_atr=2.5, targets_r=(1.5, 3.0, 5.0), max_hold=120,
        )

    def rules(self, f):
        c = f["close"]
        bull = _b(f["sma50"] > f["sma200"])
        bear = _b(f["sma50"] < f["sma200"])
        long_entry = cross_above(f["sma50"], f["sma200"]) | (bull & cross_above(c, f["sma50"]))
        short_entry = cross_below(f["sma50"], f["sma200"]) | (bear & cross_below(c, f["sma50"]))
        long_exit = cross_below(f["sma50"], f["sma200"]) | _b(c < f["sma200"])
        short_exit = cross_above(f["sma50"], f["sma200"]) | _b(c > f["sma200"])
        return Rules(_b(long_entry), _b(long_exit), _b(short_entry), _b(short_exit))

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        last, sma50 = float(r["close"]), float(r["sma50"])
        if not np.isfinite(sma50):
            return last
        if direction == "LONG":
            return last if last <= sma50 * 1.02 else sma50
        return last if last >= sma50 * 0.98 else sma50

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        gap = (r["sma50"] / r["sma200"] - 1) * 100 if np.isfinite(r["sma200"]) else float("nan")
        rising = bool(r["sma200"] > f["sma200"].iloc[-20]) if len(f) > 20 else False
        return [
            ("50 SMA above 200 SMA", bool(r["sma50"] > r["sma200"]) == up,
             f"{gap:+.2f}% apart"),
            ("Price on the trend side of the 200 SMA",
             bool(r["close"] > r["sma200"]) == up,
             f"{r['close']:.2f} vs {r['sma200']:.2f}"),
            ("Near the 50 SMA (not extended)",
             bool(abs(r["close"] - r["sma50"]) < 2.5 * r["atr"]),
             f"{abs(r['close'] - r['sma50']) / r['atr']:.2f} ATR away"),
            ("200 SMA sloping the right way", rising == up,
             "rising" if rising else "flat or falling"),
        ]


# --------------------------------------------------------------------------
# 8. Bollinger band fade
# --------------------------------------------------------------------------
class BollingerFade(Strategy):
    def __init__(self):
        super().__init__(
            key="bollinger_fade",
            name="Bollinger Band Fade",
            family="Mean reversion",
            blurb=(
                "Fades a stretch outside the 2-sigma band back toward the mean, but "
                "only in the direction of the 200 SMA so it is never fighting the "
                "primary trend. Exits at the middle band instead of waiting for a "
                "full reversal."
            ),
            stop_atr=2.0, targets_r=(0.8, 1.5, 2.5), max_hold=12,
        )

    def rules(self, f):
        c = f["close"]
        long_entry = _b((f["bb_pctb"] < 0.05) & (c > f["sma200"]) & (f["rsi14"] < 38))
        long_exit = _b((c > f["bb_mid"]) | (f["bb_pctb"] > 0.55))
        short_entry = _b((f["bb_pctb"] > 0.95) & (c < f["sma200"]) & (f["rsi14"] > 62))
        short_exit = _b((c < f["bb_mid"]) | (f["bb_pctb"] < 0.45))
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        band = float(r["bb_lower"]) if direction == "LONG" else float(r["bb_upper"])
        return band if np.isfinite(band) else float(r["close"])

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        return [
            ("Outside the 2-sigma band",
             bool(r["bb_pctb"] < 0.05) if up else bool(r["bb_pctb"] > 0.95),
             f"%B {r['bb_pctb']:.2f}"),
            ("With the 200 SMA trend", bool(r["close"] > r["sma200"]) == up,
             f"{r['close']:.2f} vs {r['sma200']:.2f}"),
            ("RSI confirms the stretch",
             bool(r["rsi14"] < 38) if up else bool(r["rsi14"] > 62),
             f"RSI {r['rsi14']:.1f}"),
            ("Bands wide enough to be worth fading",
             bool(r["bb_width"] > 0.03),
             f"band width {r['bb_width'] * 100:.1f}% of price"),
        ]


# --------------------------------------------------------------------------
# 9. Turtle channel breakout
# --------------------------------------------------------------------------
class TurtleBreakout(Strategy):
    def __init__(self):
        super().__init__(
            key="turtle_breakout",
            name="Turtle Channel Breakout",
            family="Breakout",
            blurb=(
                "The Richard Dennis system, essentially unchanged: buy a new 55-bar "
                "high, exit on a 20-bar low, stop at 2 ATR. No trend filter and no "
                "confirmation - it accepts a low hit-rate to catch the rare huge move."
            ),
            styles={"swing", "long"},
            stop_atr=2.0, targets_r=(2.0, 4.0, 6.0), max_hold=90,
        )

    def rules(self, f):
        c = f["close"]
        long_entry = _b(c > f["dc55_upper"].shift(1))
        short_entry = _b(c < f["dc55_lower"].shift(1))
        long_exit = _b(c < f["dc_lower"].shift(1))
        short_exit = _b(c > f["dc_upper"].shift(1))
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        if direction == "LONG":
            return float(max(r["dc55_upper"], r["high"]) + 0.05 * r["atr"])
        return float(min(r["dc55_lower"], r["low"]) - 0.05 * r["atr"])

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        dist = ((r["dc55_upper"] - r["close"]) / r["atr"]) if up \
            else ((r["close"] - r["dc55_lower"]) / r["atr"])
        return [
            ("At the 55-bar channel edge", bool(dist <= 1.0),
             f"{dist:.2f} ATR from the 55-bar {'high' if up else 'low'}"),
            ("20-bar exit not already breached",
             bool(r["close"] > r["dc_lower"]) if up else bool(r["close"] < r["dc_upper"]),
             f"exit level {r['dc_lower'] if up else r['dc_upper']:.2f}"),
            ("Trend has force", bool(r["adx"] > 18), f"ADX {r['adx']:.1f}"),
            ("Volatility tradeable", bool(r["atr_pct"] > 0.5), f"ATR {r['atr_pct']:.2f}%"),
        ]


# --------------------------------------------------------------------------
# 10. Ichimoku cloud
# --------------------------------------------------------------------------
class IchimokuCloud(Strategy):
    def __init__(self):
        super().__init__(
            key="ichimoku_cloud",
            name="Ichimoku Cloud Break",
            family="Trend following",
            blurb=(
                "Trades the full Ichimoku alignment: price clear of the cloud, Tenkan "
                "above Kijun, and the forward cloud itself pointing the same way. The "
                "Kijun doubles as the trailing exit, which is what keeps it in long trends."
            ),
            stop_atr=2.0, targets_r=(1.5, 3.0, 4.5), max_hold=60,
            styles={"swing", "day", "long"},
        )

    def rules(self, f):
        c = f["close"]
        above = _b(c > f["cloud_top"])
        below = _b(c < f["cloud_bot"])
        bull_cloud = _b(f["senkou_a"] > f["senkou_b"])
        bear_cloud = _b(f["senkou_a"] < f["senkou_b"])

        long_entry = (cross_above(c, f["cloud_top"])
                      | (above & cross_above(f["tenkan"], f["kijun"]))) & above & bull_cloud
        short_entry = (cross_below(c, f["cloud_bot"])
                       | (below & cross_below(f["tenkan"], f["kijun"]))) & below & bear_cloud
        long_exit = _b((c < f["kijun"]) | (c < f["cloud_bot"]))
        short_exit = _b((c > f["kijun"]) | (c > f["cloud_top"]))
        return Rules(_b(long_entry), long_exit, _b(short_entry), short_exit)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        last = float(r["close"])
        edge = float(r["cloud_top"]) if direction == "LONG" else float(r["cloud_bot"])
        if not np.isfinite(edge):
            return last
        if direction == "LONG":
            return last if last > edge else edge * 1.002
        return last if last < edge else edge * 0.998

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        return [
            ("Price clear of the cloud",
             bool(r["close"] > r["cloud_top"]) if up else bool(r["close"] < r["cloud_bot"]),
             f"cloud {r['cloud_bot']:.2f} - {r['cloud_top']:.2f}"),
            ("Tenkan / Kijun aligned", bool(r["tenkan"] > r["kijun"]) == up,
             f"Tenkan {r['tenkan']:.2f} / Kijun {r['kijun']:.2f}"),
            ("Forward cloud agrees", bool(r["senkou_a"] > r["senkou_b"]) == up,
             "bullish cloud" if r["senkou_a"] > r["senkou_b"] else "bearish cloud"),
            ("Kijun exit not already hit", bool(r["close"] > r["kijun"]) == up,
             f"Kijun {r['kijun']:.2f}"),
        ]

    def stop_price(self, f, entry, direction):
        r = f.iloc[-1]
        atr = float(r["atr"])
        kijun = float(r["kijun"])
        if np.isfinite(kijun) and np.isfinite(atr) and atr > 0 \
                and abs(entry - kijun) <= MAX_STOP_ATR_FACTOR * self.stop_atr * atr:
            if direction == "LONG" and kijun < entry:
                return kijun - 0.2 * atr
            if direction == "SHORT" and kijun > entry:
                return kijun + 0.2 * atr
        return super().stop_price(f, entry, direction)


# --------------------------------------------------------------------------
# 11. Opening range breakout (intraday)
# --------------------------------------------------------------------------
class OpeningRangeBreakout(Strategy):
    def __init__(self):
        super().__init__(
            key="opening_range",
            name="Opening Range Breakout",
            family="Intraday",
            blurb=(
                "The classic day trade: mark the first 30 minutes of the session, then "
                "take the first clean break of that range on above-average volume. The "
                "opposite side of the range is the invalidation."
            ),
            styles={"day"},
            stop_atr=1.2, targets_r=(1.0, 2.0, 3.0), max_hold=26,
        )

    def rules(self, f):
        c = f["close"]
        armed = _b((f["or_bars"] > 0) & (f["bar_of_day"] >= f["or_bars"]))
        vol_ok = _b(f["rvol"] > 1.0)
        above = _b(c > f["or_high"])
        below = _b(c < f["or_low"])
        # only the FIRST break of each side counts, not every bar beyond it
        first_up = above & _b(~above.shift(1).fillna(False))
        first_dn = below & _b(~below.shift(1).fillna(False))
        new_session = _b(f["bar_of_day"] < f["bar_of_day"].shift(1))

        long_entry = armed & first_up & vol_ok
        short_entry = armed & first_dn & vol_ok
        long_exit = _b(c < f["or_low"]) | new_session
        short_exit = _b(c > f["or_high"]) | new_session
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        hi, lo = r.get("or_high", np.nan), r.get("or_low", np.nan)
        if not np.isfinite(hi) or not np.isfinite(lo):
            return float(r["close"])
        return float(hi + 0.05 * r["atr"]) if direction == "LONG" \
            else float(lo - 0.05 * r["atr"])

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        has_or = bool(np.isfinite(r.get("or_high", np.nan)))
        return [
            ("Opening range defined", has_or,
             f"{r['or_low']:.2f} - {r['or_high']:.2f}" if has_or else "needs intraday bars"),
            ("Past the opening-range window",
             bool(r["or_bars"] > 0 and r["bar_of_day"] >= r["or_bars"]),
             f"bar {int(r['bar_of_day'])} of the session"),
            ("Breaking the range",
             (bool(r["close"] > r["or_high"]) if up else bool(r["close"] < r["or_low"]))
             if has_or else False,
             f"last {r['close']:.2f}"),
            ("Volume confirms", bool(r["rvol"] > 1.0), f"{r['rvol']:.2f}x average"),
        ]


# --------------------------------------------------------------------------
# 12. RSI divergence
# --------------------------------------------------------------------------
class RSIDivergence(Strategy):
    def __init__(self):
        super().__init__(
            key="rsi_divergence",
            name="RSI Divergence Reversal",
            family="Reversal",
            blurb=(
                "Looks for the tell that a move is running out of fuel: price prints a "
                "lower low while RSI prints a higher low, or the mirror image at highs. "
                "Pivots are only counted once confirmed, so the signal is never early."
            ),
            stop_atr=1.5, targets_r=(1.5, 2.5, 4.0), max_hold=25,
            styles={"swing", "day", "long"},
        )

    def rules(self, f):
        c = f["close"]
        long_entry = _b((f["rsi_div"] > 0) & f["bullish_bar"] & (f["rsi14"] > 32))
        short_entry = _b((f["rsi_div"] < 0) & f["bearish_bar"] & (f["rsi14"] < 68))
        long_exit = _b((f["rsi14"] > 66) | (c < f["swing_low_20"]))
        short_exit = _b((f["rsi14"] < 34) | (c > f["swing_high_20"]))
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        recent = f["rsi_div"].tail(5)
        has_bull, has_bear = bool((recent > 0).any()), bool((recent < 0).any())
        return [
            ("Divergence detected", has_bull if up else has_bear,
             "bullish" if has_bull else ("bearish" if has_bear else "none in the last 5 bars")),
            ("Reversal bar printed",
             bool(r["bullish_bar"]) if up else bool(r["bearish_bar"]),
             f"body {r['body_pct'] * 100:.0f}% of range"),
            ("RSI has room to run", bool(32 < r["rsi14"] < 66), f"RSI {r['rsi14']:.1f}"),
            ("Not fighting a runaway trend", bool(r["adx"] < 40), f"ADX {r['adx']:.1f}"),
        ]


# --------------------------------------------------------------------------
# 13. Volatility contraction pattern
# --------------------------------------------------------------------------
class VCPBreakout(Strategy):
    def __init__(self):
        super().__init__(
            key="vcp_breakout",
            name="Volatility Contraction (VCP)",
            family="Pattern",
            blurb=(
                "The Minervini setup: a leader near its 52-week high builds a base in "
                "which volatility tightens and volume dries up, then breaks out on a "
                "volume surge. The stop is tight - the pattern is wrong the moment it "
                "loses the base. Long only, by design."
            ),
            styles={"swing", "long"},
            stop_atr=1.5, targets_r=(2.0, 4.0, 6.0), max_hold=45,
            allow_short=False,
        )

    def rules(self, f):
        c = f["close"]
        near_high = _b(f["from_52w_pct"] > -18)
        stage2 = _b((c > f["sma50"]) & (f["sma50"] > f["sma200"]))
        vol_dry = _b(f["vol_contract"] < 1.0)
        atr_tight = _b(f["atr_pct"] < f["atr_pct_prev"])
        base = near_high & stage2 & vol_dry & atr_tight

        breakout = _b(c > f["dc_upper"].shift(1)) & _b(f["rvol"] > 1.5)
        long_entry = base.rolling(10, min_periods=1).max().astype(bool) & breakout
        long_exit = _b((c < f["sma50"]) | (c < f["dc10_lower"].shift(1)))
        never = pd.Series(False, index=f.index)
        return Rules(_b(long_entry), long_exit, never, never)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        return float(max(r["dc_upper"], r["high"]) + 0.05 * r["atr"])

    def checklist(self, f, direction):
        r = f.iloc[-1]
        return [
            ("Near the 52-week high", bool(r["from_52w_pct"] > -18),
             f"{r['from_52w_pct']:.1f}% from the high"),
            ("Stage-2 uptrend (price > 50 SMA > 200 SMA)",
             bool(r["close"] > r["sma50"] > r["sma200"]),
             f"{r['close']:.2f} / {r['sma50']:.2f} / {r['sma200']:.2f}"),
            ("Volume drying up in the base", bool(r["vol_contract"] < 1.0),
             f"20-bar volume {r['vol_contract']:.2f}x its earlier average"),
            ("Volatility contracting", bool(r["atr_pct"] < r["atr_pct_prev"]),
             f"ATR {r['atr_pct']:.2f}% vs {r['atr_pct_prev']:.2f}% 20 bars ago"),
            ("Breakout on volume", bool(r["rvol"] > 1.5), f"{r['rvol']:.2f}x average"),
        ]


# --------------------------------------------------------------------------
# 14. Heikin-Ashi trend flip
# --------------------------------------------------------------------------
class HeikinAshiTrend(Strategy):
    def __init__(self):
        super().__init__(
            key="heikin_ashi",
            name="Heikin-Ashi Trend Flip",
            family="Trend following",
            blurb=(
                "Heikin-Ashi averages away the noise that shakes traders out of good "
                "trends. Entry is the first strong candle after a colour flip - "
                "flat-bottomed green for longs - and the exit is simply the first "
                "candle of the opposite colour."
            ),
            stop_atr=2.0, targets_r=(1.5, 3.0, 4.5), max_hold=40,
            styles={"swing", "day", "long"},
        )

    def rules(self, f):
        c = f["close"]
        flip_up = _b((f["ha_bull"] > 0) & (f["ha_bull"].shift(1) == 0))
        flip_dn = _b((f["ha_bull"] == 0) & (f["ha_bull"].shift(1) > 0))
        strong_up = _b(f["ha_strong_up"] > 0) & _b(f["ha_strong_up"].shift(1) == 0)
        strong_dn = _b(f["ha_strong_dn"] > 0) & _b(f["ha_strong_dn"].shift(1) == 0)

        long_entry = (flip_up | strong_up) & _b(c > f["ema50"])
        short_entry = (flip_dn | strong_dn) & _b(c < f["ema50"])
        return Rules(_b(long_entry), flip_dn, _b(short_entry), flip_up)

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        strong = bool(r["ha_strong_up"] > 0) if up else bool(r["ha_strong_dn"] > 0)
        return [
            ("Heikin-Ashi candle colour", bool(r["ha_bull"] > 0) == up,
             "green" if r["ha_bull"] > 0 else "red"),
            ("Strong (wickless) candle", strong,
             "no opposing wick" if strong else "has an opposing wick"),
            ("Trend side of the 50 EMA", bool(r["close"] > r["ema50"]) == up,
             f"EMA50 {r['ema50']:.2f}"),
            ("Trend has force", bool(r["adx"] > 18), f"ADX {r['adx']:.1f}"),
        ]


# --------------------------------------------------------------------------
# 15. Stochastic pullback
# --------------------------------------------------------------------------
class StochasticPullback(Strategy):
    def __init__(self):
        super().__init__(
            key="stoch_pullback",
            name="Stochastic Pullback",
            family="Mean reversion",
            blurb=(
                "A textbook oscillator entry kept honest by a trend filter: with price "
                "above the 200 EMA, wait for %K to dip into oversold and cross back up "
                "through %D. Exits when the oscillator reaches the opposite extreme."
            ),
            stop_atr=2.0, targets_r=(1.0, 2.0, 3.0), max_hold=20,
        )

    def rules(self, f):
        c = f["close"]
        long_entry = cross_above(f["stoch_k"], f["stoch_d"]) & _b(f["stoch_k"] < 35) \
            & _b(c > f["ema200"])
        short_entry = cross_below(f["stoch_k"], f["stoch_d"]) & _b(f["stoch_k"] > 65) \
            & _b(c < f["ema200"])
        long_exit = _b((f["stoch_k"] > 80) | (c < f["ema200"] * 0.97))
        short_exit = _b((f["stoch_k"] < 20) | (c > f["ema200"] * 1.03))
        return Rules(_b(long_entry), long_exit, _b(short_entry), short_exit)

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        return [
            ("Trend filter (200 EMA)", bool(r["close"] > r["ema200"]) == up,
             f"{r['close']:.2f} vs {r['ema200']:.2f}"),
            ("Stochastic at the right extreme",
             bool(r["stoch_k"] < 35) if up else bool(r["stoch_k"] > 65),
             f"%K {r['stoch_k']:.1f}"),
            ("%K crossing %D the right way", bool(r["stoch_k"] > r["stoch_d"]) == up,
             f"%K {r['stoch_k']:.1f} / %D {r['stoch_d']:.1f}"),
            ("Not a runaway trend", bool(r["adx"] < 40), f"ADX {r['adx']:.1f}"),
        ]


# --------------------------------------------------------------------------
# 16. Gap fade
# --------------------------------------------------------------------------
class GapFade(Strategy):
    def __init__(self):
        super().__init__(
            key="gap_fade",
            name="Gap Fade",
            family="Reversal",
            blurb=(
                "Most ordinary gaps close. This fades a gap of 2% or more once the bar "
                "starts reversing back into the prior range, targeting the gap fill. It "
                "needs a market that actually gaps, so it finds nothing on 24/7 crypto."
            ),
            stop_atr=1.5, targets_r=(1.0, 2.0, 3.0), max_hold=6,
        )

    def rules(self, f):
        c, o = f["close"], f["open"]
        gap_dn = _b(f["gap_pct"] < -2.0)
        gap_up = _b(f["gap_pct"] > 2.0)
        long_entry = gap_dn & _b(c > o) & _b(c > f["sma200"] * 0.92)
        short_entry = gap_up & _b(c < o) & _b(c < f["sma200"] * 1.08)
        long_exit = _b(c >= f["prev_close"])
        short_exit = _b(c <= f["prev_close"])
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        return [
            ("A real gap printed",
             bool(r["gap_pct"] < -2.0) if up else bool(r["gap_pct"] > 2.0),
             f"gap {r['gap_pct']:+.2f}%"),
            ("Bar reversing back into the range",
             bool(r["close"] > r["open"]) == up,
             f"open {r['open']:.2f} close {r['close']:.2f}"),
            ("Gap fill still ahead",
             bool(r["close"] < r["prev_close"]) if up else bool(r["close"] > r["prev_close"]),
             f"prior close {r['prev_close']:.2f}"),
            ("Volume behind the move", bool(r["rvol"] > 1.2), f"{r['rvol']:.2f}x average"),
        ]


# --------------------------------------------------------------------------
# 17. Faber tactical allocation
# --------------------------------------------------------------------------
class FaberTactical(Strategy):
    def __init__(self):
        super().__init__(
            key="faber_tactical",
            name="Faber 10-Month Tactical",
            family="Position",
            blurb=(
                "Meb Faber's tactical rule, and one of the most-replicated results "
                "in the literature: hold while the close is above its 10-month moving "
                "average, step aside when it drops below. It does not try to beat "
                "buy-and-hold on return - it aims to sidestep the deep drawdowns."
            ),
            styles={"long", "swing"},
            stop_atr=3.0, targets_r=(2.0, 4.0, 8.0), max_hold=400,
            allow_short=False,
        )

    def rules(self, f):
        c = f["close"]
        long_entry = cross_above(c, f["ma_10mo"])
        long_exit = cross_below(c, f["ma_10mo"])
        never = pd.Series(False, index=f.index)
        return Rules(_b(long_entry), _b(long_exit), never, never)

    def entry_reference(self, f, direction):
        r = f.iloc[-1]
        last, ma = float(r["close"]), float(r["ma_10mo"])
        if not np.isfinite(ma):
            return last
        return last if last > ma else ma * 1.002

    def stop_price(self, f, entry, direction):
        r = f.iloc[-1]
        ma, atr = float(r["ma_10mo"]), float(r["atr"])
        # The rule's own exit is the moving average, so the stop belongs there -
        # but only for the long side, which is the only side this strategy takes.
        # Anchoring a short to the same line would put the stop below entry.
        if direction == "LONG" and np.isfinite(ma) and np.isfinite(atr) and ma < entry:
            return ma - 0.25 * atr
        return super().stop_price(f, entry, direction)

    def checklist(self, f, direction):
        r = f.iloc[-1]
        above = bool(r["close"] > r["ma_10mo"])
        dist = (r["close"] / r["ma_10mo"] - 1) * 100 if np.isfinite(r["ma_10mo"]) else float("nan")
        return [
            ("Above the 10-month average", above,
             f"{r['close']:.2f} vs {r['ma_10mo']:.2f} ({dist:+.1f}%)"),
            ("Not stretched far above it", bool(np.isfinite(dist) and dist < 25),
             f"{dist:+.1f}% above the line"),
            ("Long-term trend intact", bool(r["close"] > r["sma200"]),
             f"200-period SMA {r['sma200']:.2f}"),
            ("Volatility survivable", bool(r["atr_pct"] < 8), f"ATR {r['atr_pct']:.2f}%"),
        ]


# --------------------------------------------------------------------------
# 18. 12-1 momentum
# --------------------------------------------------------------------------
class Momentum12_1(Strategy):
    def __init__(self):
        super().__init__(
            key="momentum_12_1",
            name="12-1 Momentum",
            family="Position",
            blurb=(
                "The academic momentum factor (Jegadeesh & Titman): rank on the last "
                "twelve months of return but skip the most recent month, because the "
                "very short term tends to mean-revert. Holds while momentum stays "
                "positive and price holds its 10-month average."
            ),
            styles={"long", "swing"},
            stop_atr=3.0, targets_r=(2.0, 4.0, 6.0), max_hold=300,
            allow_short=False,
        )

    def rules(self, f):
        c = f["close"]
        strong = _b(f["ret_12_1"] > 0) & _b(c > f["ma_10mo"])
        long_entry = strong & _b(~strong.shift(1).fillna(False))
        long_exit = _b((f["ret_12_1"] < 0) | (c < f["ma_10mo"]))
        never = pd.Series(False, index=f.index)
        return Rules(_b(long_entry), long_exit, never, never)

    def checklist(self, f, direction):
        r = f.iloc[-1]
        return [
            ("12-month momentum positive", bool(r["ret_12_1"] > 0),
             f"{r['ret_12_1']:+.1f}% over 12 months, last month excluded"),
            ("Above the 10-month average", bool(r["close"] > r["ma_10mo"]),
             f"{r['close']:.2f} vs {r['ma_10mo']:.2f}"),
            ("Momentum still building",
             bool(r["ret_12_1"] > f["ret_12_1"].iloc[-2]) if len(f) > 2 else False,
             "rising" if len(f) > 2 and r["ret_12_1"] > f["ret_12_1"].iloc[-2] else "cooling"),
            ("Not a blow-off top", bool(r["rsi14"] < 82), f"RSI {r['rsi14']:.1f}"),
        ]


# --------------------------------------------------------------------------
# 19. Long-run value reversion
# --------------------------------------------------------------------------
class LongRunValue(Strategy):
    def __init__(self):
        super().__init__(
            key="long_run_value",
            name="Long-Run Value Reversion",
            family="Value",
            blurb=(
                "The backtestable cousin of the fair-value panel. Instead of "
                "fundamentals - which are only available as today's snapshot and so "
                "cannot be honestly backtested - this anchors to the symbol's own "
                "three-year average price and buys statistically deep discounts to it, "
                "but only once price has started turning back up."
            ),
            styles={"long", "swing"},
            stop_atr=2.5, targets_r=(1.5, 3.0, 5.0), max_hold=200,
        )

    def rules(self, f):
        c = f["close"]
        cheap = _b(f["value_z"] < -1.25)
        rich = _b(f["value_z"] > 1.75)
        turning_up = _b(c > c.shift(1)) & _b(f["rsi14"] > 35)
        turning_dn = _b(c < c.shift(1)) & _b(f["rsi14"] < 65)

        long_entry = cheap & turning_up & _b(c > f["ema20"])
        short_entry = rich & turning_dn & _b(c < f["ema20"])
        long_exit = _b((f["value_z"] > -0.1) | (c < f["value_anchor"] * 0.75))
        short_exit = _b((f["value_z"] < 0.4) | (c > f["value_anchor"] * 1.25))
        return Rules(long_entry, long_exit, short_entry, short_exit)

    def entry_reference(self, f, direction):
        return float(f["close"].iloc[-1])

    def checklist(self, f, direction):
        r = f.iloc[-1]
        up = direction == "LONG"
        z = float(r["value_z"])
        return [
            ("Statistically far from the long-run mean",
             bool(z < -1.25) if up else bool(z > 1.75),
             f"{z:+.2f} standard deviations from the 3-year anchor"),
            ("Anchor price", bool(np.isfinite(r["value_anchor"])),
             f"3-year average {r['value_anchor']:.2f} vs price {r['close']:.2f}"),
            ("Price already turning back",
             bool(r["close"] > f["close"].iloc[-2]) == up if len(f) > 2 else False,
             "turning up" if len(f) > 2 and r["close"] > f["close"].iloc[-2] else "still falling"),
            ("Not a broken chart", bool(35 < r["rsi14"] < 65) or (bool(r["rsi14"] > 35) if up else bool(r["rsi14"] < 65)),
             f"RSI {r['rsi14']:.1f}"),
        ]


# --------------------------------------------------------------------------
ALL_STRATEGIES: list[Strategy] = [
    TrendPullback(),
    RSI2Reversion(),
    MACDMomentum(),
    FibGoldenPocket(),
    SqueezeBreakout(),
    VWAPReclaim(),
    GoldenCross(),
    BollingerFade(),
    TurtleBreakout(),
    IchimokuCloud(),
    OpeningRangeBreakout(),
    RSIDivergence(),
    VCPBreakout(),
    HeikinAshiTrend(),
    StochasticPullback(),
    GapFade(),
    FaberTactical(),
    Momentum12_1(),
    LongRunValue(),
]


def strategies_for(style: str) -> list[Strategy]:
    """`style` is 'swing' or 'day'."""
    return [s for s in ALL_STRATEGIES if style in s.styles]
