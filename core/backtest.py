"""Event-driven backtester used to rank the strategies on the loaded symbol.

Deliberately conservative so the numbers mean something:

  * signals are read on a bar's close, the fill happens at the NEXT bar's open
  * stops and targets are checked intrabar against high/low, stop wins ties
  * commission + slippage are charged on both sides
  * a walk-forward split reports out-of-sample expectancy separately

Position sizing is fixed-fractional risk, so every trade result is expressed
in R (multiples of the money risked) and strategies are comparable to each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import bars_per_year
from .strategies import Rules, Strategy, resolve_stop


@dataclass
class Trade:
    direction: str
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp
    exit: float
    stop: float
    target: float
    r_multiple: float
    bars_held: int
    reason: str


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    win_rate: float = float("nan")
    expectancy_r: float = float("nan")
    profit_factor: float = float("nan")
    avg_win_r: float = float("nan")
    avg_loss_r: float = float("nan")
    total_return: float = float("nan")
    cagr: float = float("nan")
    max_drawdown: float = float("nan")
    sharpe: float = float("nan")
    avg_bars: float = float("nan")
    oos_expectancy_r: float = float("nan")
    oos_trades: int = 0
    edge_score: float = 0.0
    equity: pd.Series = field(default_factory=pd.Series)
    trade_list: list = field(default_factory=list)

    @property
    def has_sample(self) -> bool:
        return self.trades >= 8


def _finalise_r(direction: str, entry: float, exit_px: float, risk: float,
                cost_frac: float) -> float:
    gross = (exit_px - entry) if direction == "LONG" else (entry - exit_px)
    cost = cost_frac * (entry + exit_px)
    return (gross - cost) / risk if risk > 0 else 0.0


def run_backtest(
    f: pd.DataFrame,
    strat: Strategy,
    interval: str = "1d",
    allow_short: bool = True,
    risk_frac: float = 0.01,
    cost_bps: float = 5.0,
    target_r: float = 2.0,
    trail_after_r: float = 1.0,
) -> Metrics:
    """Simulate `strat` over `f` and return performance metrics."""
    rules: Rules = strat.rules(f)

    o = f["open"].to_numpy(float)
    h = f["high"].to_numpy(float)
    l = f["low"].to_numpy(float)
    c = f["close"].to_numpy(float)
    atr = f["atr"].to_numpy(float)
    swing_lo = f["swing_low_20"].to_numpy(float)
    swing_hi = f["swing_high_20"].to_numpy(float)

    le = rules.long_entry.to_numpy(bool)
    lx = rules.long_exit.to_numpy(bool)
    se = rules.short_entry.to_numpy(bool)
    sx = rules.short_exit.to_numpy(bool)

    n = len(f)
    cost_frac = cost_bps / 10_000.0
    equity = np.ones(n)
    eq = 1.0

    trades: list[Trade] = []
    pos = 0            # 0 flat, +1 long, -1 short
    entry = stop = target = units = 0.0
    entry_i = 0
    peak_r = 0.0
    init_risk = 0.0

    for i in range(1, n):
        # ---- mark-to-market an open position -----------------------------
        if pos != 0:
            equity[i] = eq + units * (c[i] - entry) * pos
        else:
            equity[i] = eq

        if pos != 0:
            exit_px, reason = None, ""
            # stop first: if a bar touches both, assume the worse fill
            if pos == 1 and l[i] <= stop:
                exit_px, reason = stop, "stop"
            elif pos == -1 and h[i] >= stop:
                exit_px, reason = stop, "stop"
            elif pos == 1 and h[i] >= target:
                exit_px, reason = target, "target"
            elif pos == -1 and l[i] <= target:
                exit_px, reason = target, "target"
            elif (pos == 1 and lx[i]) or (pos == -1 and sx[i]):
                exit_px, reason = c[i], "signal"
            elif (i - entry_i) >= strat.max_hold:
                exit_px, reason = c[i], "time"

            if exit_px is not None:
                r = _finalise_r("LONG" if pos == 1 else "SHORT", entry, exit_px, init_risk, cost_frac)
                eq = eq * (1.0 + risk_frac * r)
                equity[i] = eq
                trades.append(
                    Trade(
                        "LONG" if pos == 1 else "SHORT",
                        f.index[entry_i], entry, f.index[i], exit_px,
                        stop, target, r, i - entry_i, reason,
                    )
                )
                pos = 0
            else:
                # trail the stop once the trade is up `trail_after_r`
                cur_r = ((c[i] - entry) if pos == 1 else (entry - c[i])) / init_risk
                peak_r = max(peak_r, cur_r)
                if peak_r >= trail_after_r and np.isfinite(atr[i]):
                    if pos == 1:
                        stop = max(stop, entry + 0.05 * init_risk, c[i] - 2.0 * atr[i])
                    else:
                        stop = min(stop, entry - 0.05 * init_risk, c[i] + 2.0 * atr[i])

        # ---- look for a new entry, filled at the next bar's open ----------
        if pos == 0 and i + 1 < n and np.isfinite(atr[i]) and atr[i] > 0:
            direction = 0
            if le[i]:
                direction = 1
            elif allow_short and strat.allow_short and se[i]:
                direction = -1

            if direction != 0:
                fill = o[i + 1]
                if direction == 1:
                    s = resolve_stop(fill, atr[i], strat.stop_atr, swing_lo[i], "LONG")
                    risk = fill - s
                else:
                    s = resolve_stop(fill, atr[i], strat.stop_atr, swing_hi[i], "SHORT")
                    risk = s - fill

                if risk > 0 and np.isfinite(risk):
                    pos = direction
                    entry, stop, init_risk, entry_i = fill, s, risk, i + 1
                    target = fill + direction * target_r * risk
                    units = (risk_frac * eq) / risk
                    peak_r = 0.0

    eq_series = pd.Series(equity, index=f.index)
    return _metrics(trades, eq_series, interval, f)


def _metrics(trades: list[Trade], eq: pd.Series, interval: str, f: pd.DataFrame) -> Metrics:
    m = Metrics(equity=eq, trade_list=trades)
    m.trades = len(trades)
    if not trades:
        return m

    rs = np.array([t.r_multiple for t in trades], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs <= 0]

    m.wins = int(len(wins))
    m.win_rate = len(wins) / len(rs) * 100.0
    m.expectancy_r = float(rs.mean())
    m.avg_win_r = float(wins.mean()) if len(wins) else 0.0
    m.avg_loss_r = float(losses.mean()) if len(losses) else 0.0
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    m.profit_factor = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
    m.avg_bars = float(np.mean([t.bars_held for t in trades]))

    m.total_return = float(eq.iloc[-1] - 1.0)
    dd = eq / eq.cummax() - 1.0
    m.max_drawdown = float(dd.min())

    bpy = bars_per_year(interval)
    years = max(len(eq) / bpy, 1e-9)
    if eq.iloc[-1] > 0:
        m.cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) > 5 and rets.std(ddof=0) > 0:
        m.sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(bpy))

    # walk-forward: how did it do on the last 30% of history it never "saw"?
    split_time = f.index[int(len(f) * 0.70)]
    oos = [t.r_multiple for t in trades if t.entry_time >= split_time]
    m.oos_trades = len(oos)
    if oos:
        m.oos_expectancy_r = float(np.mean(oos))

    m.edge_score = _edge_score(m)
    return m


def _edge_score(m: Metrics) -> float:
    """0-100 blend of expectancy, consistency and sample size.

    Expectancy dominates, profit factor and drawdown adjust it, and a small
    sample or a negative out-of-sample result pulls the score down hard.
    """
    if m.trades == 0 or not np.isfinite(m.expectancy_r):
        return 0.0

    # expectancy: 0R -> 50, +0.35R -> ~69, -0.35R -> ~31. The scale is wide
    # enough that a book of uniformly losing strategies still ranks internally
    # instead of every score flooring at zero.
    base = 100.0 / (1.0 + np.exp(-m.expectancy_r / 0.45))

    pf = m.profit_factor if np.isfinite(m.profit_factor) else 3.0
    pf_adj = np.clip((min(pf, 3.0) - 1.0) * 10.0, -15.0, 20.0)

    # drawdown is a penalty only - the equity curve is sized at 1% risk per
    # trade, so a small drawdown is the baseline, not an achievement
    dd_adj = -min(20.0, max(0.0, abs(m.max_drawdown) - 0.10) * 120.0)

    # confidence from sample size: 8 trades ~0.5, 40+ trades ~1.0
    conf = float(np.clip(np.sqrt(m.trades / 40.0), 0.25, 1.0))

    oos_adj = 0.0
    if m.oos_trades >= 3 and np.isfinite(m.oos_expectancy_r):
        oos_adj = float(np.clip(m.oos_expectancy_r * 25.0, -18.0, 12.0))

    # cap the combined modifiers so expectancy stays the dominant term and a
    # strategy cannot be pushed off the bottom of the scale by penalties alone
    raw = base + float(np.clip(pf_adj + dd_adj + oos_adj, -25.0, 30.0))
    # pull toward neutral 50 when the sample is thin
    return float(np.clip(50.0 + (raw - 50.0) * conf, 0.0, 100.0))


def trades_frame(m: Metrics) -> pd.DataFrame:
    if not m.trade_list:
        return pd.DataFrame(
            columns=["direction", "entry_time", "entry", "exit_time", "exit", "r", "bars", "reason"]
        )
    return pd.DataFrame(
        [
            {
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry": round(t.entry, 4),
                "exit_time": t.exit_time,
                "exit": round(t.exit, 4),
                "r": round(t.r_multiple, 3),
                "bars": t.bars_held,
                "reason": t.reason,
            }
            for t in m.trade_list
        ]
    )
