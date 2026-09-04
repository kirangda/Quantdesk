"""Beta / alpha risk analytics and fundamental fair-value estimation.

Two separate ideas live here, and the difference matters:

* **Beta** is measured from price history, so it is objective and backtestable
  in the same way everything else in this app is.
* **Fair value** is built from *current* fundamentals (EPS, book value, cash
  flow, analyst targets). yfinance exposes only today's snapshot, not what the
  numbers were five years ago, so a fair-value signal **cannot be honestly
  backtested here**. It is presented as analysis, never scored as a strategy.
  The backtestable cousin is `LongRunValue` in strategies.py, which anchors to
  price history instead of fundamentals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from .data import Instrument, bars_per_year, fetch

# Which index to measure beta against, by market.
BENCHMARKS = {
    "USD": "^GSPC", "EUR": "^STOXX50E", "GBP": "^FTSE", "INR": "^NSEI",
    "JPY": "^N225", "CHF": "^SSMI", "CAD": "^GSPTSE", "AUD": "^AXJO",
    "HKD": "^HSI", "CNY": "000001.SS", "SEK": "^OMX", "BRL": "^BVSP",
}
BENCHMARK_NAMES = {
    "^GSPC": "S&P 500", "^STOXX50E": "EURO STOXX 50", "^FTSE": "FTSE 100",
    "^NSEI": "NIFTY 50", "^N225": "Nikkei 225", "^SSMI": "SMI",
    "^GSPTSE": "TSX", "^AXJO": "ASX 200", "^HSI": "Hang Seng",
    "000001.SS": "SSE Composite", "^OMX": "OMX 30", "^BVSP": "Bovespa",
    "BTC-USD": "Bitcoin",
}


def benchmark_for(instrument: Instrument, native_ccy: str) -> str:
    """Alt-coins are measured against Bitcoin; equities against their index."""
    if instrument.asset_class == "crypto":
        return "^GSPC" if instrument.symbol.startswith("BTC-") else "BTC-USD"
    return BENCHMARKS.get(native_ccy.upper(), "^GSPC")


# --------------------------------------------------------------------------
# Beta / alpha
# --------------------------------------------------------------------------
@dataclass
class Beta:
    benchmark: str
    benchmark_name: str
    beta: float = float("nan")
    alpha_annual: float = float("nan")      # annualised excess return, %
    correlation: float = float("nan")
    r_squared: float = float("nan")
    observations: int = 0
    asset_vol: float = float("nan")         # annualised %
    bench_vol: float = float("nan")
    rolling: pd.Series = field(default_factory=pd.Series)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.observations > 30 and np.isfinite(self.beta)

    @property
    def label(self) -> str:
        if not self.ok:
            return "unavailable"
        b = self.beta
        if b < 0:
            return "moves against the market"
        if b < 0.7:
            return "defensive"
        if b <= 1.3:
            return "moves with the market"
        if b <= 2.0:
            return "aggressive"
        return "very high beta"

    @property
    def meaning(self) -> str:
        if not self.ok:
            return ""
        move = self.beta * 10.0
        return (
            f"A 10% move in the {self.benchmark_name} has historically come with "
            f"roughly a {move:+.1f}% move here. "
            f"R-squared {self.r_squared * 100:.0f}% means "
            f"{self.r_squared * 100:.0f}% of this symbol's movement is explained by "
            f"the index; the rest is its own story."
        )


def compute_beta(asset: pd.DataFrame, bench: pd.DataFrame, interval: str = "1d",
                 benchmark: str = "", window: int = 252) -> Beta:
    """Ordinary least-squares beta of the asset against the benchmark."""
    out = Beta(benchmark=benchmark,
               benchmark_name=BENCHMARK_NAMES.get(benchmark, benchmark))

    a = asset["close"].pct_change()
    b = bench["close"].pct_change()
    # normalise timezone so a tz-aware intraday index can join a tz-naive one
    for s in (a, b):
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna().tail(window)

    out.observations = len(joined)
    if out.observations < 30:
        out.error = (f"Only {out.observations} overlapping bars with "
                     f"{out.benchmark_name} - not enough to measure beta.")
        return out

    var_b = float(joined["b"].var(ddof=1))
    if not np.isfinite(var_b) or var_b <= 0:
        out.error = "Benchmark returns have no variance."
        return out

    cov = float(joined["a"].cov(joined["b"]))
    out.beta = cov / var_b
    out.correlation = float(joined["a"].corr(joined["b"]))
    out.r_squared = out.correlation ** 2

    bpy = bars_per_year(interval)
    # alpha: the part of the average return the index does not explain
    per_bar_alpha = float(joined["a"].mean()) - out.beta * float(joined["b"].mean())
    out.alpha_annual = per_bar_alpha * bpy * 100.0
    out.asset_vol = float(joined["a"].std(ddof=1)) * np.sqrt(bpy) * 100.0
    out.bench_vol = float(joined["b"].std(ddof=1)) * np.sqrt(bpy) * 100.0

    roll = min(60, max(20, out.observations // 4))
    cov_r = joined["a"].rolling(roll).cov(joined["b"])
    var_r = joined["b"].rolling(roll).var()
    out.rolling = (cov_r / var_r.replace(0, np.nan)).dropna()
    return out


def load_beta(instrument: Instrument, asset: pd.DataFrame, native_ccy: str,
              interval: str = "1d") -> Beta:
    """Fetch the right benchmark and measure beta against it."""
    symbol = benchmark_for(instrument, native_ccy)
    try:
        bench = fetch(Instrument(symbol, symbol, "index"), interval, min_bars=30)
    except Exception as exc:  # noqa: BLE001 - beta is optional, never fatal
        b = Beta(benchmark=symbol, benchmark_name=BENCHMARK_NAMES.get(symbol, symbol))
        b.error = f"Could not load {symbol}: {exc}"
        return b
    return compute_beta(asset, bench, interval, symbol)


# --------------------------------------------------------------------------
# Fundamental fair value
# --------------------------------------------------------------------------
@dataclass
class Estimate:
    method: str
    value: float
    detail: str


@dataclass
class FairValue:
    applicable: bool = False
    reason: str = ""
    price: float = float("nan")
    estimates: list = field(default_factory=list)
    blended: float = float("nan")
    low: float = float("nan")
    high: float = float("nan")
    sector: str = ""
    industry: str = ""
    metrics: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    spread_warning: str = ""

    @property
    def upside_pct(self) -> float:
        if not np.isfinite(self.blended) or not self.price:
            return float("nan")
        return (self.blended / self.price - 1.0) * 100.0

    @property
    def verdict(self) -> str:
        u = self.upside_pct
        if not np.isfinite(u):
            return "No estimate"
        if u > 25:
            return "Deeply undervalued"
        if u > 10:
            return "Undervalued"
        if u > -10:
            return "Around fair value"
        if u > -25:
            return "Overvalued"
        return "Richly valued"


def _num(info: dict, key: str):
    v = info.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def fair_value(instrument: Instrument, price: float,
               discount_rate: float = 0.09, terminal_growth: float = 0.025) -> FairValue:
    """Blend several classic valuation models into one range.

    Every model here is crude and assumption-heavy - that is the nature of a
    one-number fair value. The blend is a MEDIAN so a single wild estimate
    cannot drag the answer, and the full spread is always reported alongside.
    """
    fv = FairValue(price=price)

    if instrument.asset_class != "equity":
        fv.reason = (
            f"Fair value needs company fundamentals - earnings, book value, cash flow. "
            f"A {instrument.asset_class} has none, so there is nothing to discount. "
            "Use the technical levels and the Long-Run Value strategy instead."
        )
        return fv

    try:
        info = yf.Ticker(instrument.symbol).get_info()
    except Exception as exc:  # noqa: BLE001
        fv.reason = f"Could not load fundamentals for {instrument.symbol}: {exc}"
        return fv

    if not info or _num(info, "trailingEps") is None and _num(info, "targetMeanPrice") is None:
        fv.reason = f"No usable fundamentals returned for {instrument.symbol}."
        return fv

    fv.sector = info.get("sector", "") or ""
    fv.industry = info.get("industry", "") or ""

    eps = _num(info, "trailingEps")
    feps = _num(info, "forwardEps")
    bvps = _num(info, "bookValue")
    growth = _num(info, "earningsGrowth")
    target = _num(info, "targetMeanPrice")
    fcf = _num(info, "freeCashflow")
    shares = _num(info, "sharesOutstanding")
    debt = _num(info, "totalDebt") or 0.0
    cash = _num(info, "totalCash") or 0.0

    fv.metrics = {
        "Trailing P/E": _num(info, "trailingPE"),
        "Forward P/E": _num(info, "forwardPE"),
        "Price / book": _num(info, "priceToBook"),
        "PEG": _num(info, "pegRatio"),
        "Return on equity %": (_num(info, "returnOnEquity") or 0) * 100 or None,
        "Earnings growth %": (growth * 100) if growth is not None else None,
        "Revenue growth %": (_num(info, "revenueGrowth") or 0) * 100 or None,
        "Reported beta": _num(info, "beta"),
    }

    est: list[Estimate] = []
    skipped: list[str] = []

    # 1. analyst consensus
    if target and target > 0:
        n = info.get("numberOfAnalystOpinions")
        lo, hi = _num(info, "targetLowPrice"), _num(info, "targetHighPrice")
        rng = f", range {lo:,.2f}-{hi:,.2f}" if lo and hi else ""
        est.append(Estimate("Analyst consensus target", target,
                            f"mean of {n or '?'} analysts{rng}"))

    # 2. Graham number - only inside its domain. Graham assumed asset-heavy
    #    industrials; on a buyback-shrunk, asset-light balance sheet (Apple has
    #    a price/book above 40) the formula returns a number that is not wrong
    #    so much as meaningless, and it would drag the blend badly.
    pb = _num(info, "priceToBook")
    if eps and bvps and eps > 0 and bvps > 0:
        if pb is not None and pb > 8:
            skipped.append(
                f"Graham number skipped: price/book is {pb:.0f}. The formula assumes "
                "a tangible-asset-heavy company and is not meaningful for an "
                "asset-light or heavily-buyback business.")
        else:
            graham = float(np.sqrt(22.5 * eps * bvps))
            est.append(Estimate("Graham number", graham,
                                f"sqrt(22.5 x EPS {eps:.2f} x book {bvps:.2f}) - "
                                "deliberately conservative, ignores growth"))

    # 3. Lynch-style growth multiple: a fair P/E roughly equals the growth rate
    if feps and feps > 0 and growth is not None:
        fair_pe = float(np.clip(growth * 100.0, 8.0, 35.0))
        est.append(Estimate("Growth-adjusted P/E", feps * fair_pe,
                            f"forward EPS {feps:.2f} x fair P/E {fair_pe:.0f} "
                            f"(from {growth * 100:.1f}% earnings growth, capped 8-35)"))

    # 4. discounted free cash flow, perpetuity form
    if fcf and shares and fcf > 0 and shares > 0:
        g = float(np.clip((growth or terminal_growth), -0.02, terminal_growth))
        if discount_rate > g:
            ev = fcf * (1 + g) / (discount_rate - g)
            equity = ev - debt + cash
            if equity > 0:
                est.append(Estimate(
                    "Discounted cash flow", equity / shares,
                    f"FCF {fcf / 1e9:.1f}B growing {g * 100:.1f}% forever, "
                    f"discounted at {discount_rate * 100:.0f}%, net of debt"))

    if not est:
        fv.reason = ("Fundamentals were returned but none of the models could run - "
                     "typically negative earnings and no analyst coverage.")
        return fv

    vals = [e.value for e in est if np.isfinite(e.value) and e.value > 0]
    fv.estimates = est
    fv.skipped = skipped
    fv.applicable = True
    fv.blended = float(np.median(vals))
    fv.low, fv.high = float(min(vals)), float(max(vals))
    # when the models disagree wildly the median is not a "fair value" at all
    if fv.low > 0 and fv.high / fv.low > 2.5:
        fv.spread_warning = (
            f"These models disagree by {fv.high / fv.low:.1f}x "
            f"({fv.low:,.2f} to {fv.high:,.2f}). Treat the blend as a midpoint of "
            "wide disagreement, not a precise target - that spread is itself the "
            "useful signal about how uncertain this valuation is.")
    return fv
