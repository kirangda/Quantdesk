"""Analysis engine: runs every strategy, backtests it, and fuses the results
into one ranked recommendation with concrete buy/sell prices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import Metrics, run_backtest
from .data import Instrument
from .fibonacci import FibLeg, detect_leg
from .strategies import Strategy, build_features, strategies_for

# how recent an entry signal has to be to count as "live"
FRESH_BARS = 2

# an edge score below this is not a tradeable edge, it is noise
MIN_TRADEABLE_EDGE = 45.0


@dataclass
class Plan:
    """One strategy's current read on the market, with an actionable order."""

    key: str
    name: str
    family: str
    blurb: str
    direction: str              # LONG | SHORT | FLAT
    state: str                  # TRIGGERED | SETUP FORMING | NO SETUP
    conviction: float           # 0-100
    entry: float
    stop: float
    targets: list
    targets_r: tuple
    risk_per_unit: float
    rr: float
    checklist: list
    metrics: Metrics
    bars_since_signal: int = -1
    setup_quality: float = 0.0

    @property
    def action(self) -> str:
        if self.state == "NO DATA":
            return "NOT APPLICABLE HERE"
        if self.direction == "FLAT" or self.state == "NO SETUP":
            return "WAIT"
        verb = "BUY" if self.direction == "LONG" else "SELL / SHORT"
        return verb if self.state == "TRIGGERED" else f"{verb} ON TRIGGER"

    @property
    def is_live(self) -> bool:
        return self.state == "TRIGGERED"


@dataclass
class Context:
    """Objective description of the tape, independent of any strategy."""

    price: float
    change_pct: float
    trend: str
    trend_detail: str
    volatility: str
    atr: float
    atr_pct: float
    rsi: float
    adx: float
    regime_score: float
    above_200: bool
    from_high_pct: float
    notes: list = field(default_factory=list)


@dataclass
class Recommendation:
    verdict: str                # STRONG BUY | BUY | NEUTRAL | SELL | STRONG SELL
    direction: str
    score: float                # -100..100
    agreement: str
    best: Plan | None
    entry: float = float("nan")
    stop: float = float("nan")
    targets: list = field(default_factory=list)
    rr: float = float("nan")
    reasoning: list = field(default_factory=list)
    caution: str = ""   # set when nothing in the book has a proven edge here


@dataclass
class Analysis:
    instrument: Instrument
    interval: str
    style: str
    features: pd.DataFrame
    plans: list
    context: Context
    recommendation: Recommendation
    fib: FibLeg | None


# --------------------------------------------------------------------------
def _market_context(f: pd.DataFrame) -> Context:
    r = f.iloc[-1]
    prev = f["close"].iloc[-2]
    price = float(r["close"])
    change = (price / float(prev) - 1.0) * 100.0

    above200 = bool(np.isfinite(r["ema200"]) and price > r["ema200"])
    stacked_up = bool(r["ema20"] > r["ema50"] > r["ema200"]) if np.isfinite(r["ema200"]) else False
    stacked_dn = bool(r["ema20"] < r["ema50"] < r["ema200"]) if np.isfinite(r["ema200"]) else False
    adx = float(r["adx"]) if np.isfinite(r["adx"]) else 0.0

    if stacked_up and adx > 20:
        trend, detail = "Strong uptrend", "EMAs stacked 20 > 50 > 200 with ADX confirming"
    elif stacked_dn and adx > 20:
        trend, detail = "Strong downtrend", "EMAs stacked 20 < 50 < 200 with ADX confirming"
    elif above200:
        trend, detail = "Uptrend", "Price above the 200 EMA but the stack is not clean"
    elif np.isfinite(r["ema200"]):
        trend, detail = "Downtrend", "Price below the 200 EMA"
    else:
        trend, detail = "Undefined", "Not enough history for a 200-period trend read"
    if adx < 18:
        trend = "Range / chop" if "Strong" not in trend else trend
        detail += f" - ADX {adx:.0f} means the trend lacks force"

    atr_pct = float(r["atr_pct"]) if np.isfinite(r["atr_pct"]) else float("nan")
    vol_rank = float((f["atr_pct"].tail(120) <= r["atr_pct"]).mean() * 100) if np.isfinite(atr_pct) else 50.0
    if vol_rank > 75:
        volatility = "High"
    elif vol_rank < 25:
        volatility = "Compressed"
    else:
        volatility = "Normal"

    hi = float(f["high"].tail(252).max())
    from_high = (price / hi - 1.0) * 100.0

    regime = 0.0
    regime += 25 if above200 else -25
    regime += 15 if stacked_up else (-15 if stacked_dn else 0)
    regime += float(np.clip((adx - 20) * 1.2, -12, 12)) * (1 if above200 else -1)
    regime += float(np.clip((r["rsi14"] - 50) * 0.6, -18, 18)) if np.isfinite(r["rsi14"]) else 0.0

    notes = []
    if volatility == "Compressed":
        notes.append("Volatility is compressed - breakout setups are favoured over mean reversion.")
    if volatility == "High":
        notes.append("Volatility is elevated - size down; stops need more room than usual.")
    if np.isfinite(r["rsi14"]) and r["rsi14"] > 72:
        notes.append(f"RSI {r['rsi14']:.0f} is overbought - chasing longs here has poor risk/reward.")
    if np.isfinite(r["rsi14"]) and r["rsi14"] < 28:
        notes.append(f"RSI {r['rsi14']:.0f} is oversold - short entries are late.")
    if from_high > -2:
        notes.append("Trading at or near the 1-year high - breakout continuation is in play.")
    if np.isfinite(r["rvol"]) and r["rvol"] > 1.8:
        notes.append(f"Volume is {r['rvol']:.1f}x normal - conviction behind the current move.")

    return Context(
        price=price, change_pct=change, trend=trend, trend_detail=detail,
        volatility=volatility, atr=float(r["atr"]), atr_pct=atr_pct,
        rsi=float(r["rsi14"]), adx=adx,
        regime_score=float(np.clip(regime, -100, 100)),
        above_200=above200, from_high_pct=from_high, notes=notes,
    )


def _build_plan(strat: Strategy, f: pd.DataFrame, metrics: Metrics, allow_short: bool) -> Plan:
    rules = strat.rules(f)
    tail = slice(-FRESH_BARS, None)
    long_live = bool(rules.long_entry.iloc[tail].any())
    short_live = bool(allow_short and strat.allow_short and rules.short_entry.iloc[tail].any())

    # how much of each side's checklist currently passes
    long_rows = strat.checklist(f, "LONG")
    short_rows = strat.checklist(f, "SHORT")
    long_q = float(np.mean([p for _, p, _ in long_rows]) * 100) if long_rows else 0.0
    short_q = float(np.mean([p for _, p, _ in short_rows]) * 100) if short_rows else 0.0

    if long_live and not short_live:
        direction, state, quality, rows = "LONG", "TRIGGERED", long_q, long_rows
    elif short_live and not long_live:
        direction, state, quality, rows = "SHORT", "TRIGGERED", short_q, short_rows
    else:
        if not allow_short:
            short_q = -1.0
        if max(long_q, short_q) < 60.0:
            direction = "LONG" if long_q >= short_q else "SHORT"
            state = "NO SETUP"
        else:
            direction = "LONG" if long_q >= short_q else "SHORT"
            state = "SETUP FORMING"
        quality = max(long_q, short_q)
        rows = long_rows if direction == "LONG" else short_rows

    # a strategy that never triggered once on this history cannot be scored -
    # e.g. Gap Fade on 24/7 crypto, which by construction has no gaps
    if metrics.trades == 0:
        state = "NO DATA"

    # bars since the last entry signal on the chosen side
    sig = rules.long_entry if direction == "LONG" else rules.short_entry
    idx = np.flatnonzero(sig.to_numpy(bool))
    bars_since = int(len(sig) - 1 - idx[-1]) if len(idx) else -1

    entry = float(strat.entry_reference(f, direction))
    stop = float(strat.stop_price(f, entry, direction))
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= 0:
        risk = float(f["atr"].iloc[-1]) * strat.stop_atr
        stop = entry - risk if direction == "LONG" else entry + risk

    sign = 1.0 if direction == "LONG" else -1.0
    targets = [round(entry + sign * m * risk, 6) for m in strat.targets_r]
    rr = strat.targets_r[1] if len(strat.targets_r) > 1 else strat.targets_r[0]

    edge = metrics.edge_score
    conviction = 0.55 * quality + 0.45 * edge
    if state == "TRIGGERED":
        conviction = min(100.0, conviction * 1.12)
    elif state == "NO SETUP":
        conviction *= 0.55
    elif state == "NO DATA":
        conviction = 0.0
    if not metrics.has_sample:
        conviction *= 0.85  # thin backtest sample -> less trust

    return Plan(
        key=strat.key, name=strat.name, family=strat.family, blurb=strat.blurb,
        direction=direction, state=state, conviction=float(np.clip(conviction, 0, 100)),
        entry=entry, stop=stop, targets=targets, targets_r=tuple(strat.targets_r),
        risk_per_unit=risk, rr=float(rr),
        checklist=rows, metrics=metrics, bars_since_signal=bars_since,
        setup_quality=quality,
    )


def _fuse(plans: list[Plan], ctx: Context) -> Recommendation:
    """Edge-weighted vote across the strategy book."""
    num, den = 0.0, 0.0
    for p in plans:
        if p.state == "NO DATA":
            continue                       # no history here, no vote
        weight = (p.metrics.edge_score / 100.0) * (p.conviction / 100.0)
        if p.state == "TRIGGERED":
            weight *= 1.0
        elif p.state == "SETUP FORMING":
            weight *= 0.45
        else:
            weight *= 0.10
        vote = 1.0 if p.direction == "LONG" else -1.0
        num += vote * weight
        den += weight if weight > 0 else 0.0

    raw = (num / den * 100.0) if den > 1e-9 else 0.0
    # blend the strategy vote with the objective regime read
    score = float(np.clip(0.7 * raw + 0.3 * ctx.regime_score, -100, 100))

    live = [p for p in plans if p.is_live]
    engaged = [p for p in plans if p.state not in ("NO SETUP", "NO DATA")]
    longs = sum(1 for p in engaged if p.direction == "LONG")
    shorts = sum(1 for p in engaged if p.direction == "SHORT")
    agreement = f"{longs} long / {shorts} short of {len(plans)} strategies engaged"

    if score >= 45 and live:
        verdict, direction = "STRONG BUY", "LONG"
    elif score >= 18:
        verdict, direction = "BUY", "LONG"
    elif score <= -45 and live:
        verdict, direction = "STRONG SELL", "SHORT"
    elif score <= -18:
        verdict, direction = "SELL", "SHORT"
    else:
        verdict, direction = "NEUTRAL", "FLAT"

    # the order to actually place: best-ranked strategy that agrees and is live
    def rank(p: Plan):
        return (p.is_live, p.metrics.edge_score, p.conviction)

    agreeing = [p for p in plans if direction != "FLAT" and p.direction == direction
                and p.state not in ("NO SETUP", "NO DATA")]
    best = max(agreeing, key=rank) if agreeing else (max(plans, key=rank) if plans else None)

    reasoning = [ctx.trend_detail]
    if best is not None:
        reasoning.append(
            f"{best.name} ranks highest here (edge score {best.metrics.edge_score:.0f}/100, "
            f"{best.metrics.trades} historical trades, expectancy {best.metrics.expectancy_r:+.2f}R)."
        )
    if live:
        reasoning.append(
            "Live triggers right now: " + ", ".join(f"{p.name} ({p.direction})" for p in live) + "."
        )
    else:
        reasoning.append("No strategy has triggered on the latest bar - these are standing orders, not market orders.")
    reasoning.extend(ctx.notes[:3])

    rec = Recommendation(
        verdict=verdict, direction=direction, score=score,
        agreement=agreement, best=best, reasoning=reasoning,
    )

    # Guard rail: never present an order from a book that has no measured edge.
    # On noisy intraday series, costs alone can make every strategy negative -
    # the honest answer there is to stand aside, not to pick the least bad one.
    positive = [p for p in plans if np.isfinite(p.metrics.expectancy_r)
                and p.metrics.expectancy_r > 0 and p.metrics.edge_score >= MIN_TRADEABLE_EDGE]
    if not positive:
        rec.caution = (
            "No strategy in the book shows a positive backtested edge on this symbol "
            "and timeframe after costs. The levels below are still the correct places "
            "to act IF you take the trade, but the measured edge does not support it - "
            "consider standing aside or moving to a higher timeframe."
        )
        rec.verdict, rec.direction = "STAND ASIDE", "FLAT"
        rec.score = float(np.clip(score, -17, 17))
    elif best is not None and best.metrics.expectancy_r <= 0:
        rec.caution = (
            f"{best.name} is the best fit for current conditions but its own backtest on "
            f"this symbol is negative ({best.metrics.expectancy_r:+.2f}R). Size down or wait "
            "for a setup from a strategy with a proven edge here."
        )
    if best is not None and direction != "FLAT":
        rec.entry, rec.stop, rec.targets, rec.rr = best.entry, best.stop, best.targets, best.rr
    elif best is not None:
        rec.entry, rec.stop, rec.targets, rec.rr = best.entry, best.stop, best.targets, best.rr
    return rec


def analyse(
    df: pd.DataFrame,
    instrument: Instrument,
    interval: str = "1d",
    style: str = "swing",
    allow_short: bool = True,
    cost_bps: float = 5.0,
    risk_frac: float = 0.01,
) -> Analysis:
    """Full pipeline: features -> backtests -> live plans -> fused verdict."""
    intraday = interval not in ("1d", "1wk", "1mo")
    f = build_features(df, intraday=intraday, interval=interval)

    plans: list[Plan] = []
    for strat in strategies_for(style):
        metrics = run_backtest(
            f, strat, interval=interval, allow_short=allow_short,
            risk_frac=risk_frac, cost_bps=cost_bps,
            target_r=strat.targets_r[1] if len(strat.targets_r) > 1 else 2.0,
        )
        plans.append(_build_plan(strat, f, metrics, allow_short))

    plans.sort(key=lambda p: (p.is_live, p.metrics.edge_score, p.conviction), reverse=True)
    ctx = _market_context(f)
    rec = _fuse(plans, ctx)
    return Analysis(
        instrument=instrument, interval=interval, style=style, features=f,
        plans=plans, context=ctx, recommendation=rec, fib=detect_leg(f),
    )


# --------------------------------------------------------------------------
# Position sizing
# --------------------------------------------------------------------------
def size_position(account: float, risk_pct: float, entry: float, stop: float) -> dict:
    risk_amount = account * risk_pct / 100.0
    per_unit = abs(entry - stop)
    if per_unit <= 0 or not np.isfinite(per_unit):
        return {"units": 0.0, "risk_amount": risk_amount, "notional": 0.0, "pct_of_account": 0.0}
    units = risk_amount / per_unit
    notional = units * entry
    return {
        "units": units,
        "risk_amount": risk_amount,
        "notional": notional,
        "pct_of_account": notional / account * 100.0 if account > 0 else 0.0,
        "risk_per_unit": per_unit,
    }
