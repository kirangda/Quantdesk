"""Technical indicators - pure pandas/numpy, no native dependencies.

Every function takes and returns pandas objects aligned to the input index.
Wilder-smoothed indicators (RSI, ATR, ADX) use the classic alpha = 1/n
exponential form, so values match TradingView / StockCharts defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def wilder(series: pd.Series, length: int) -> pd.Series:
    """Wilder smoothing (RMA) - the average used inside RSI/ATR/ADX."""
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    pv = (close * volume).rolling(length, min_periods=length).sum()
    v = volume.rolling(length, min_periods=length).sum()
    return pv / v.replace(0, np.nan)


# --------------------------------------------------------------------------
# Momentum
# --------------------------------------------------------------------------
def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder(gain, length)
    avg_loss = wilder(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of up-closes -> RSI pinned at 100
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )


def stochastic(high, low, close, k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    hh = high.rolling(k, min_periods=k).max()
    ll = low.rolling(k, min_periods=k).min()
    raw = 100.0 * (close - ll) / (hh - ll).replace(0, np.nan)
    k_line = raw.rolling(smooth, min_periods=smooth).mean()
    return pd.DataFrame({"k": k_line, "d": k_line.rolling(d, min_periods=d).mean()})


def roc(close: pd.Series, length: int = 12) -> pd.Series:
    return close.pct_change(length) * 100.0


# --------------------------------------------------------------------------
# Volatility / range
# --------------------------------------------------------------------------
def true_range(high, low, close) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high, low, close, length: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), length)


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = sma(close, length)
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    upper, lower = mid + mult * sd, mid - mult * sd
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "width": (upper - lower) / mid.replace(0, np.nan),
            "pctb": (close - lower) / (upper - lower).replace(0, np.nan),
        }
    )


def keltner(high, low, close, length: int = 20, mult: float = 1.5) -> pd.DataFrame:
    mid = ema(close, length)
    rng = atr(high, low, close, length) * mult
    return pd.DataFrame({"mid": mid, "upper": mid + rng, "lower": mid - rng})


def donchian(high, low, length: int = 20) -> pd.DataFrame:
    upper = high.rolling(length, min_periods=length).max()
    lower = low.rolling(length, min_periods=length).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})


# --------------------------------------------------------------------------
# Trend strength
# --------------------------------------------------------------------------
def adx(high, low, close, length: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    atr_n = wilder(true_range(high, low, close), length).replace(0, np.nan)
    plus_di = 100.0 * wilder(plus_dm, length) / atr_n
    minus_di = 100.0 * wilder(minus_dm, length) / atr_n
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, length), "plus_di": plus_di, "minus_di": minus_di})


def supertrend(high, low, close, length: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """ATR channel that ratchets in the direction of trend; +1 long, -1 short."""
    hl2 = (high + low) / 2.0
    band = mult * atr(high, low, close, length)

    c = close.to_numpy(dtype=float)
    ub = (hl2 + band).to_numpy(dtype=float)
    lb = (hl2 - band).to_numpy(dtype=float)
    n = len(c)
    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    direction = np.full(n, 1.0)

    for i in range(n):
        if i == 0 or np.isnan(ub[i]) or np.isnan(final_ub[i - 1]):
            final_ub[i], final_lb[i] = ub[i], lb[i]
            continue
        # bands only tighten toward price until price breaks through them
        final_ub[i] = ub[i] if (ub[i] < final_ub[i - 1] or c[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = lb[i] if (lb[i] > final_lb[i - 1] or c[i - 1] < final_lb[i - 1]) else final_lb[i - 1]
        if c[i] > final_ub[i - 1]:
            direction[i] = 1.0
        elif c[i] < final_lb[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]

    line = np.where(direction > 0, final_lb, final_ub)
    return pd.DataFrame({"supertrend": line, "direction": direction}, index=close.index)


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()


def vwap(high, low, close, volume, session_reset: bool = True) -> pd.Series:
    """Volume-weighted average price. Intraday data resets each calendar day."""
    tp = (high + low + close) / 3.0
    if session_reset and isinstance(close.index, pd.DatetimeIndex):
        groups = close.index.normalize()
        pv = (tp * volume).groupby(groups).cumsum()
        vv = volume.groupby(groups).cumsum()
    else:
        pv, vv = (tp * volume).cumsum(), volume.cumsum()
    return pv / vv.replace(0, np.nan)


def relative_volume(volume: pd.Series, length: int = 20) -> pd.Series:
    return volume / volume.rolling(length, min_periods=length).mean().replace(0, np.nan)


# --------------------------------------------------------------------------
# Swing structure
# --------------------------------------------------------------------------
def swing_points(high: pd.Series, low: pd.Series, left: int = 5, right: int = 5):
    """Fractal pivots: a bar that is the extreme of its +/- window.

    Returns (swing_high, swing_low) - NaN except at confirmed pivot bars.
    """
    win = left + right + 1
    roll_max = high.rolling(win, center=True, min_periods=win).max()
    roll_min = low.rolling(win, center=True, min_periods=win).min()
    return high.where(high == roll_max), low.where(low == roll_min)


# --------------------------------------------------------------------------
# Ichimoku
# --------------------------------------------------------------------------
def ichimoku(high, low, close, conv: int = 9, base: int = 26, span_b: int = 52) -> pd.DataFrame:
    """Tenkan/Kijun/Senkou A/B/Chikou.

    The two spans are shifted FORWARD by `base`, which is what makes the cloud
    a leading indicator; chikou is the close shifted back.
    """
    def mid(n):
        return (high.rolling(n, min_periods=n).max() + low.rolling(n, min_periods=n).min()) / 2.0

    tenkan, kijun = mid(conv), mid(base)
    senkou_a = ((tenkan + kijun) / 2.0).shift(base)
    senkou_b = mid(span_b).shift(base)
    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b,
        "cloud_top": pd.concat([senkou_a, senkou_b], axis=1).max(axis=1),
        "cloud_bot": pd.concat([senkou_a, senkou_b], axis=1).min(axis=1),
        "chikou": close.shift(-base),
    })


# --------------------------------------------------------------------------
# Heikin-Ashi
# --------------------------------------------------------------------------
def heikin_ashi(o, h, l, c) -> pd.DataFrame:
    """Smoothed candles. ha_open is recursive, so this walks the array once."""
    ha_close = (o + h + l + c) / 4.0
    op = o.to_numpy(dtype=float)
    cl = c.to_numpy(dtype=float)
    hac = ha_close.to_numpy(dtype=float)
    n = len(op)
    hao = np.empty(n)
    hao[0] = (op[0] + cl[0]) / 2.0 if np.isfinite(op[0]) else np.nan
    for i in range(1, n):
        prev = hao[i - 1]
        hao[i] = (op[i] + cl[i]) / 2.0 if not np.isfinite(prev) else (prev + hac[i - 1]) / 2.0

    ha_open = pd.Series(hao, index=o.index)
    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
    body = ha_close - ha_open
    rng = (ha_high - ha_low).replace(0, np.nan)
    return pd.DataFrame({
        "ha_open": ha_open, "ha_high": ha_high, "ha_low": ha_low, "ha_close": ha_close,
        "ha_bull": (body > 0).astype(float),
        # a flat-bottomed green candle (no lower wick) is the classic strong-trend bar
        "ha_strong_up": ((body > 0) & ((pd.concat([ha_open, ha_close], axis=1).min(axis=1) - ha_low) / rng < 0.08)).astype(float),
        "ha_strong_dn": ((body < 0) & ((ha_high - pd.concat([ha_open, ha_close], axis=1).max(axis=1)) / rng < 0.08)).astype(float),
    })


# --------------------------------------------------------------------------
# Oscillator divergence
# --------------------------------------------------------------------------
def divergence(price_low, price_high, osc, left: int = 5, right: int = 5) -> pd.Series:
    """+1 bullish divergence, -1 bearish, 0 none - carried until the next pivot.

    Compares the last two confirmed pivots. Pivots are shifted by `right` bars
    so a signal only appears once the pivot could actually be known.
    """
    hi, lo = swing_points(price_high, price_low, left, right)
    hi, lo = hi.shift(right), lo.shift(right)
    osc_at_hi = osc.shift(right).where(hi.notna())
    osc_at_lo = osc.shift(right).where(lo.notna())

    p_lo = lo.dropna()
    o_lo = osc_at_lo.dropna()
    p_hi = hi.dropna()
    o_hi = osc_at_hi.dropna()

    out = pd.Series(0.0, index=price_low.index)

    # bullish: lower price low, higher oscillator low
    for i in range(1, len(p_lo)):
        t, prev_t = p_lo.index[i], p_lo.index[i - 1]
        if t not in o_lo.index or prev_t not in o_lo.index:
            continue
        if p_lo.iloc[i] < p_lo.iloc[i - 1] and o_lo.loc[t] > o_lo.loc[prev_t]:
            out.loc[t] = 1.0

    # bearish: higher price high, lower oscillator high
    for i in range(1, len(p_hi)):
        t, prev_t = p_hi.index[i], p_hi.index[i - 1]
        if t not in o_hi.index or prev_t not in o_hi.index:
            continue
        if p_hi.iloc[i] > p_hi.iloc[i - 1] and o_hi.loc[t] < o_hi.loc[prev_t]:
            out.loc[t] = -1.0

    # a divergence stays relevant for a few bars after it prints
    return out.replace(0.0, np.nan).ffill(limit=3).fillna(0.0)
