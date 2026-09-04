"""Correctness tests for the strategy book and the analysis pipeline.

Run with:  python -m pytest tests -q        (or: python tests/test_quantdesk.py)

The important one is `test_no_lookahead`. It walks forward: for a series of cut
points it recomputes a strategy from ONLY the bars up to that point and asserts
the resulting live decision equals what the full-history run shows for that same
bar. Anything reading even one bar ahead disagrees, because the truncated run
cannot see it. `test_lookahead_detector_works` plants a deliberately cheating
rule to prove the check actually bites - a green test that cannot fail is worse
than no test, and the first version of this file had exactly that bug.

This matters more than any backtest number: a lookahead bug makes every metric
a lie.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import run_backtest                       # noqa: E402
from core.data import Instrument, currency_symbol, native_currency, normalise_symbol  # noqa: E402
from core.engine import analyse, size_position               # noqa: E402
from core.fibonacci import detect_leg, describe, level_rows  # noqa: E402
from core.strategies import (ALL_STRATEGIES, Rules, _b, build_features,  # noqa: E402
                             strategies_for)


# --------------------------------------------------------------------------
# Synthetic markets - no network, deterministic, and shaped to exercise
# trending, ranging and gappy conditions.
# --------------------------------------------------------------------------
def _series(n=700, seed=0, drift=0.0004, vol=0.014, gaps=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    if gaps:                       # inject occasional overnight gaps
        jumps = rng.choice(n, size=max(1, n // 60), replace=False)
        open_[jumps] *= 1 + rng.choice([-1, 1], len(jumps)) * rng.uniform(0.025, 0.05, len(jumps))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol_ = rng.integers(1_000_000, 9_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol_}, index=idx)


def _intraday(n=900, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-04 09:30", periods=n, freq="15min")
    idx = idx[(idx.hour * 60 + idx.minute >= 570) & (idx.hour * 60 + idx.minute <= 955)]
    n = len(idx)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.0008, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0015, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0015, n)))
    v = rng.integers(10_000, 90_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": v}, index=idx)


MARKETS = {
    "uptrend": _series(seed=1, drift=0.0010),
    "downtrend": _series(seed=2, drift=-0.0009),
    "choppy": _series(seed=3, drift=0.0, vol=0.020),
    "gappy": _series(seed=4, gaps=True),
}

FEATURES = {k: build_features(v, intraday=False, interval="1d") for k, v in MARKETS.items()}
INTRADAY_F = build_features(_intraday(), intraday=True, interval="15m")
# a long, low-frequency series for the position-trading book
MONTHLY = _series(n=300, seed=7, drift=0.008, vol=0.055)
MONTHLY.index = pd.date_range("2001-01-31", periods=300, freq="ME")
MONTHLY_F = build_features(MONTHLY, intraday=False, interval="1mo")
INST = Instrument("TEST", "TEST", "equity")


def _frame_for(strat, market="uptrend"):
    """Feed each strategy a series whose bar size matches the style it targets."""
    if strat.styles == {"day"}:
        return INTRADAY_F
    if "long" in strat.styles and "swing" not in strat.styles:
        return MONTHLY_F
    return FEATURES[market]


# --------------------------------------------------------------------------
# Strategy contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("strat", ALL_STRATEGIES, ids=lambda s: s.key)
def test_rules_shape(strat):
    """Every rule set is boolean, full-length and aligned to the input index."""
    f = _frame_for(strat, "uptrend")
    r = strat.rules(f)
    for name in ("long_entry", "long_exit", "short_entry", "short_exit"):
        s = getattr(r, name)
        assert len(s) == len(f), f"{strat.key}.{name} wrong length"
        assert s.index.equals(f.index), f"{strat.key}.{name} misaligned index"
        assert s.dtype == bool, f"{strat.key}.{name} is {s.dtype}, expected bool"
        assert not s.isna().any(), f"{strat.key}.{name} contains NaN"


@pytest.mark.parametrize("strat", ALL_STRATEGIES, ids=lambda s: s.key)
def test_no_shorts_when_disallowed(strat):
    """A long-only strategy must never emit a short entry."""
    if strat.allow_short:
        pytest.skip("strategy permits shorts")
    f = _frame_for(strat, "uptrend")
    assert not strat.rules(f).short_entry.any()


def _last_bar_signals(strat, raw, cut, intraday, interval="1d"):
    """What the strategy would have said in real time at bar `cut - 1`."""
    r = strat.rules(build_features(raw.iloc[:cut], intraday=intraday, interval=interval))
    return bool(r.long_entry.iloc[-1]), bool(r.short_entry.iloc[-1])


@pytest.mark.parametrize("strat", ALL_STRATEGIES, ids=lambda s: s.key)
def test_no_lookahead(strat):
    """Walk-forward: the live decision must equal the historical one.

    For several cut points, the signal computed with ONLY the bars up to that
    point must equal the signal the full-history run shows for that same bar.
    A rule that reads even one bar ahead disagrees, because the truncated run
    cannot see it. (Verified to fail against a deliberately planted
    `close.shift(-1)` rule - see test_lookahead_detector_works.)
    """
    intraday = strat.styles == {"day"}
    full_f = _frame_for(strat, "choppy")
    raw = full_f[["open", "high", "low", "close", "volume"]]
    full = strat.rules(full_f)

    n = len(raw)
    iv = "15m" if intraday else ("1mo" if full_f is MONTHLY_F else "1d")
    for cut in (n - 120, n - 90, n - 60, n - 30, n):
        live_long, live_short = _last_bar_signals(strat, raw, cut, intraday, iv)
        assert live_long == bool(full.long_entry.iloc[cut - 1]), (
            f"{strat.key}: long signal at bar {cut - 1} changed once future bars "
            "were available - the rule is reading ahead")
        assert live_short == bool(full.short_entry.iloc[cut - 1]), (
            f"{strat.key}: short signal at bar {cut - 1} changed once future bars "
            "were available - the rule is reading ahead")


def test_lookahead_detector_works():
    """The detector above must actually catch a rule that peeks. """
    class Cheater:
        key, styles, allow_short = "cheater", {"swing"}, True

        def rules(self, f):
            peek = _b(f["close"].shift(-1) > f["close"])   # reads tomorrow
            never = pd.Series(False, index=f.index)
            return Rules(peek, never, never, never)

    strat = Cheater()
    raw = FEATURES["choppy"][["open", "high", "low", "close", "volume"]]
    full = strat.rules(build_features(raw))
    n = len(raw)
    caught = False
    for cut in (n - 120, n - 90, n - 60, n - 30, n):
        live, _ = _last_bar_signals(strat, raw, cut, False)
        if live != bool(full.long_entry.iloc[cut - 1]):
            caught = True
            break
    assert caught, "the lookahead detector failed to catch a planted future-peeking rule"


@pytest.mark.parametrize("strat", ALL_STRATEGIES, ids=lambda s: s.key)
def test_stop_is_on_the_right_side(strat):
    """Stops sit below entry for longs and above for shorts, at sane distance."""
    f = _frame_for(strat, "uptrend")
    atr = float(f["atr"].iloc[-1])
    for direction in ("LONG", "SHORT"):
        entry = float(strat.entry_reference(f, direction))
        stop = float(strat.stop_price(f, entry, direction))
        assert np.isfinite(entry) and np.isfinite(stop), f"{strat.key} produced NaN"
        assert entry > 0, f"{strat.key} non-positive entry"
        if direction == "LONG":
            assert stop < entry, f"{strat.key} LONG stop {stop} not below entry {entry}"
        else:
            assert stop > entry, f"{strat.key} SHORT stop {stop} not above entry {entry}"
        risk = abs(entry - stop)
        assert risk <= 8 * atr, f"{strat.key} {direction} risk {risk / atr:.1f} ATR is absurd"


# --------------------------------------------------------------------------
# Backtester
# --------------------------------------------------------------------------
@pytest.mark.parametrize("market", sorted(MARKETS))
@pytest.mark.parametrize("strat", ALL_STRATEGIES, ids=lambda s: s.key)
def test_backtest_metrics_are_sane(strat, market):
    f = _frame_for(strat, market)
    m = run_backtest(f, strat, interval="1d")

    assert m.trades >= 0
    if m.trades == 0:
        return
    assert 0.0 <= m.win_rate <= 100.0
    assert m.profit_factor >= 0.0
    assert m.wins <= m.trades
    assert -1.0 <= m.max_drawdown <= 0.0
    assert 0.0 <= m.edge_score <= 100.0
    assert (m.equity > 0).all(), "equity curve went non-positive"
    for t in m.trade_list:
        assert t.exit_time >= t.entry_time, "trade exits before it enters"
        assert t.bars_held >= 0
        assert np.isfinite(t.r_multiple)


def test_costs_reduce_expectancy():
    """Charging more per trade must not improve results."""
    f = FEATURES["uptrend"]
    strat = ALL_STRATEGIES[0]
    cheap = run_backtest(f, strat, cost_bps=0.0)
    dear = run_backtest(f, strat, cost_bps=50.0)
    if cheap.trades and dear.trades:
        assert dear.expectancy_r <= cheap.expectancy_r + 1e-9


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
@pytest.mark.parametrize("market", sorted(MARKETS))
def test_analysis_end_to_end(market):
    a = analyse(MARKETS[market], INST, "1d", "swing")
    assert len(a.plans) == len(strategies_for("swing"))
    assert a.recommendation.verdict in {
        "STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL", "STAND ASIDE"}
    assert -100.0 <= a.recommendation.score <= 100.0

    for p in a.plans:
        assert 0.0 <= p.conviction <= 100.0
        assert p.state in {"TRIGGERED", "SETUP FORMING", "NO SETUP", "NO DATA"}
        assert np.isfinite(p.entry) and np.isfinite(p.stop)
        # targets must run away from entry, in order, on the correct side
        sign = 1.0 if p.direction == "LONG" else -1.0
        prev = p.entry
        for t in p.targets:
            assert np.isfinite(t)
            assert (t - prev) * sign > 0, f"{p.key} targets not ordered outward"
            prev = t


def test_zero_trade_strategy_is_flagged_not_voted():
    """A strategy with no history here must not influence the verdict."""
    a = analyse(MARKETS["uptrend"], INST, "1d", "swing")
    for p in a.plans:
        if p.metrics.trades == 0:
            assert p.state == "NO DATA"
            assert p.conviction == 0.0
            assert p.action == "NOT APPLICABLE HERE"


@pytest.mark.parametrize("style,expect_in,expect_out", [
    ("day", "opening_range", "golden_cross"),
    ("long", "faber_tactical", "opening_range"),
    ("swing", "golden_cross", "opening_range"),
])
def test_style_books_are_distinct(style, expect_in, expect_out):
    keys = {s.key for s in strategies_for(style)}
    assert expect_in in keys and expect_out not in keys


def test_long_horizon_features_are_calendar_scaled():
    """The 10-month average must mean 10 months on daily AND monthly bars."""
    raw = MARKETS["uptrend"]
    daily = build_features(raw, interval="1d")["ma_10mo"].iloc[-1]
    # resample the same prices to monthly and rebuild
    monthly = raw.resample("ME").agg({"open": "first", "high": "max", "low": "min",
                                      "close": "last", "volume": "sum"}).dropna()
    m = build_features(monthly, interval="1mo")["ma_10mo"].iloc[-1]
    assert np.isfinite(daily) and np.isfinite(m)
    # same window in calendar terms -> the two averages should be close
    assert abs(daily / m - 1.0) < 0.15, f"daily {daily:.2f} vs monthly {m:.2f}"


def test_day_style_uses_the_day_book():
    a = analyse(_intraday(), INST, "15m", "day")
    keys = {p.key for p in a.plans}
    assert "opening_range" in keys and "vwap_reclaim" in keys
    assert "golden_cross" not in keys      # swing-only


# --------------------------------------------------------------------------
# Risk sizing
# --------------------------------------------------------------------------
def test_position_size_risks_exactly_the_requested_amount():
    s = size_position(account=10_000, risk_pct=1.0, entry=100.0, stop=95.0)
    assert s["risk_amount"] == pytest.approx(100.0)
    assert s["units"] == pytest.approx(20.0)          # 100 risk / 5 per unit
    assert s["notional"] == pytest.approx(2_000.0)
    # the loss taken if the stop fills is exactly the risk budget
    assert s["units"] * (100.0 - 95.0) == pytest.approx(s["risk_amount"])


def test_position_size_handles_degenerate_stop():
    s = size_position(account=10_000, risk_pct=1.0, entry=100.0, stop=100.0)
    assert s["units"] == 0.0


# --------------------------------------------------------------------------
# Fibonacci
# --------------------------------------------------------------------------
def test_fib_levels_are_ordered_and_bracket_the_leg():
    leg = detect_leg(FEATURES["uptrend"])
    assert leg is not None
    lo, hi = sorted([leg.start_price, leg.end_price])
    for name, price in leg.levels.items():
        assert lo - 1e-6 <= price <= hi + 1e-6, f"fib {name} outside the leg"
    assert leg.levels["0"] == pytest.approx(leg.end_price)
    assert leg.levels["1"] == pytest.approx(leg.start_price)


def test_fib_description_matches_depth():
    f = FEATURES["uptrend"]
    leg = detect_leg(f)
    d = describe(leg, float(f["close"].iloc[-1]))
    assert set(d) >= {"depth", "headline", "meaning", "nearest_label"}
    rows = level_rows(leg, float(f["close"].iloc[-1]))
    prices = [r["price"] for r in rows]
    assert prices == sorted(prices, reverse=True), "level table not sorted high to low"


# --------------------------------------------------------------------------
# Symbols and currency
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,resolved,ccy", [
    ("aapl", "AAPL", "USD"),
    ("btc", "BTC-USD", "USD"),
    ("ETH-EUR", "ETH-EUR", "EUR"),
    ("sap.de", "SAP.DE", "EUR"),
    ("reliance.ns", "RELIANCE.NS", "INR"),
    ("dax", "^GDAXI", "EUR"),
    ("7203.T", "7203.T", "JPY"),
])
def test_symbol_and_currency_resolution(raw, resolved, ccy):
    inst = normalise_symbol(raw)
    assert inst.symbol == resolved
    assert native_currency(inst) == ccy
    assert currency_symbol(ccy)


def test_empty_symbol_rejected():
    with pytest.raises(ValueError):
        normalise_symbol("   ")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
