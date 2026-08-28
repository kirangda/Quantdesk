"""Market data loading and symbol normalisation (stocks + crypto via Yahoo)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

# Yahoo rate-limits bursts of requests, so keep a small on-disk cache. It makes
# repeat runs instant and, more importantly, lets the app fall back to slightly
# stale bars instead of dying when a request is throttled.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_TTL = {"1m": 60, "5m": 120, "15m": 300, "30m": 600, "60m": 900, "1h": 900,
             "1d": 1800, "1wk": 3600}

# yfinance chatters about optional metadata endpoints; the download path
# reports failures through exceptions, which we handle explicitly below
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Bar interval -> (default history to pull, max history Yahoo allows)
INTERVAL_LIMITS = {
    "1m": ("7d", "7d"),
    "5m": ("30d", "60d"),
    "15m": ("60d", "60d"),
    "30m": ("60d", "60d"),
    "60m": ("180d", "730d"),
    "1h": ("180d", "730d"),
    "1d": ("3y", "max"),
    "1wk": ("10y", "max"),
}

# Timeframes the app offers per trading style
SWING_INTERVALS = ["1d", "1wk", "1h"]
DAY_INTERVALS = ["5m", "15m", "30m", "1h", "1m"]

COMMON_CRYPTO = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC", "LINK",
    "LTC", "BCH", "ATOM", "UNI", "ETC", "XLM", "NEAR", "APT", "ARB", "OP",
    "SHIB", "TRX", "TON", "ICP", "FIL", "HBAR", "SUI", "INJ", "RNDR", "PEPE",
}

INDEX_ALIASES = {
    "SPX": "^GSPC", "SP500": "^GSPC", "NDX": "^NDX", "NASDAQ": "^IXIC",
    "DOW": "^DJI", "DJIA": "^DJI", "VIX": "^VIX", "RUSSELL": "^RUT",
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "DAX": "^GDAXI", "FTSE": "^FTSE",
}


@dataclass
class Instrument:
    symbol: str          # resolved Yahoo ticker
    raw: str             # what the user typed
    asset_class: str     # equity | crypto | index | fx
    name: str = ""
    currency: str = "USD"
    exchange: str = ""

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "crypto"

    @property
    def label(self) -> str:
        return f"{self.name} ({self.symbol})" if self.name else self.symbol


def normalise_symbol(raw: str) -> Instrument:
    """Turn loose user input into a Yahoo ticker.

    'btc' -> BTC-USD, 'spx' -> ^GSPC, 'reliance.ns' -> RELIANCE.NS, 'aapl' -> AAPL
    """
    s = (raw or "").strip().upper().replace(" ", "")
    if not s:
        raise ValueError("Enter a symbol, e.g. AAPL, NVDA, BTC or ETH-USD")

    if s in INDEX_ALIASES:
        return Instrument(INDEX_ALIASES[s], raw, "index")
    if s.startswith("^"):
        return Instrument(s, raw, "index")
    if s.endswith("=X"):
        return Instrument(s, raw, "fx")

    # already a crypto pair
    if "-" in s and s.split("-")[-1] in {"USD", "USDT", "EUR", "GBP", "INR"}:
        return Instrument(s, raw, "crypto")
    if s.endswith("USDT"):
        return Instrument(f"{s[:-4]}-USD", raw, "crypto")
    if s.endswith("USD") and s[:-3] in COMMON_CRYPTO:
        return Instrument(f"{s[:-3]}-USD", raw, "crypto")
    if s in COMMON_CRYPTO:
        return Instrument(f"{s}-USD", raw, "crypto")

    return Instrument(s, raw, "equity")


def _flatten(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance returns MultiIndex columns for some calls - squash to OHLCV."""
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values(-1)
        if ticker in set(levels):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "adj close", "volume"] if c in df.columns]
    df = df[keep].copy()
    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


PERIOD_FALLBACK = {
    "3y": ["3y", "2y", "1y"],
    "10y": ["10y", "5y", "2y"],
    "180d": ["180d", "90d", "60d"],
    "60d": ["60d", "30d", "15d"],
    "30d": ["30d", "15d", "7d"],
    "7d": ["7d", "5d", "2d"],
}


def _cache_path(symbol: str, interval: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol)
    return CACHE_DIR / f"{safe}__{interval}.pkl"


def _read_cache(symbol: str, interval: str, max_age: float | None) -> pd.DataFrame | None:
    path = _cache_path(symbol, interval)
    try:
        if not path.exists():
            return None
        if max_age is not None and (time.time() - path.stat().st_mtime) > max_age:
            return None
        df = pd.read_pickle(path)
        return df if isinstance(df, pd.DataFrame) and not df.empty else None
    except Exception:  # noqa: BLE001 - a bad cache file must never be fatal
        return None


def _write_cache(symbol: str, interval: str, df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(_cache_path(symbol, interval))
    except Exception:  # noqa: BLE001
        pass


def fetch(instrument: Instrument, interval: str = "1d", period: str | None = None,
          use_cache: bool = True, min_bars: int = 60) -> pd.DataFrame:
    """Download OHLCV. Raises RuntimeError with a readable message on failure.

    `min_bars` guards the analysis pipeline, which needs enough history for a
    200-period EMA. Callers that only want a spot quote (FX) can lower it.
    """
    default_period, _ = INTERVAL_LIMITS.get(interval, ("2y", "max"))
    period = period or default_period

    if use_cache:
        fresh = _read_cache(instrument.symbol, interval, CACHE_TTL.get(interval, 900))
        if fresh is not None:
            return fresh

    # Yahoo intermittently returns an empty frame for a long period that would
    # succeed for a shorter one, so walk down instead of failing outright
    candidates = PERIOD_FALLBACK.get(period, [period])
    df = None
    for i, attempt in enumerate(candidates):
        try:
            df = yf.download(
                instrument.symbol,
                period=attempt,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception:  # noqa: BLE001 - retried / falls back to cache below
            df = None
        if df is not None and not df.empty:
            break
        if i < len(candidates) - 1:
            time.sleep(1.0 + i)  # back off before asking again

    if df is None or df.empty:
        # throttled or delisted - stale bars beat no app at all
        stale = _read_cache(instrument.symbol, interval, None) if use_cache else None
        if stale is not None:
            return stale
        raise RuntimeError(
            f"No data returned for '{instrument.raw}' (resolved to {instrument.symbol}) "
            f"at {interval}. Yahoo may be rate-limiting - wait a moment and retry, "
            f"or check the symbol."
        )

    df = _flatten(df, instrument.symbol)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "datetime"

    if len(df) < min_bars:
        raise RuntimeError(
            f"Only {len(df)} bars available for {instrument.symbol} at {interval} - "
            "not enough history to analyse. Try a shorter interval or another symbol."
        )

    if use_cache:
        _write_cache(instrument.symbol, interval, df)
    return df


def enrich(instrument: Instrument) -> Instrument:
    """Best-effort metadata lookup; never fatal."""
    try:
        info = yf.Ticker(instrument.symbol).get_info()
        instrument.name = info.get("shortName") or info.get("longName") or ""
        instrument.currency = info.get("currency", "USD") or "USD"
        instrument.exchange = info.get("exchange", "") or ""
    except Exception:
        pass
    return instrument


def bars_per_year(interval: str) -> float:
    """Annualisation factor for Sharpe / CAGR at a given bar size."""
    return {
        "1m": 252 * 390, "5m": 252 * 78, "15m": 252 * 26, "30m": 252 * 13,
        "60m": 252 * 7, "1h": 252 * 7, "1d": 252, "1wk": 52,
    }.get(interval, 252)


# --------------------------------------------------------------------------
# Currency
# --------------------------------------------------------------------------
# Listing currency by Yahoo suffix. Resolved offline so display never depends
# on the flaky metadata endpoint; enrich() can still override it when it works.
SUFFIX_CURRENCY = {
    ".NS": "INR", ".BO": "INR",
    ".DE": "EUR", ".F": "EUR", ".SG": "EUR", ".MU": "EUR", ".BE": "EUR",
    ".DU": "EUR", ".HM": "EUR", ".PA": "EUR", ".AS": "EUR", ".BR": "EUR",
    ".MI": "EUR", ".MC": "EUR", ".LS": "EUR", ".VI": "EUR", ".HE": "EUR",
    ".IR": "EUR",
    ".L": "GBP", ".SW": "CHF", ".TO": "CAD", ".V": "CAD", ".AX": "AUD",
    ".NZ": "NZD", ".HK": "HKD", ".T": "JPY", ".KS": "KRW", ".TW": "TWD",
    ".SS": "CNY", ".SZ": "CNY", ".SA": "BRL", ".MX": "MXN",
    ".ST": "SEK", ".OL": "NOK", ".CO": "DKK", ".WA": "PLN", ".JO": "ZAR",
}

INDEX_CURRENCY = {
    "^GSPC": "USD", "^NDX": "USD", "^IXIC": "USD", "^DJI": "USD",
    "^RUT": "USD", "^VIX": "USD",
    "^GDAXI": "EUR", "^FCHI": "EUR", "^STOXX50E": "EUR", "^IBEX": "EUR",
    "^FTSE": "GBP", "^N225": "JPY", "^HSI": "HKD",
    "^NSEI": "INR", "^NSEBANK": "INR", "^BSESN": "INR",
}

CURRENCY_SYMBOL = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
    "INR": "₹", "CHF": "CHF ", "CAD": "C$", "AUD": "A$", "CNY": "CN¥",
}

# what the app offers in the display-currency toggle
DISPLAY_CURRENCIES = ["USD", "EUR", "GBP", "INR", "CHF", "JPY"]


def native_currency(instrument: Instrument) -> str:
    """The currency the instrument is actually quoted in."""
    if instrument.currency and instrument.currency not in ("", "USD"):
        return instrument.currency          # a successful enrich() wins
    sym = instrument.symbol.upper()

    if instrument.asset_class == "crypto" and "-" in sym:
        return sym.split("-")[-1]           # BTC-EUR quotes in EUR
    if sym in INDEX_CURRENCY:
        return INDEX_CURRENCY[sym]
    if "." in sym:
        suffix = sym[sym.rindex("."):]
        if suffix in SUFFIX_CURRENCY:
            return SUFFIX_CURRENCY[suffix]
    return "USD"


def currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOL.get(code.upper(), f"{code.upper()} ")


def fx_rate(base: str, quote: str) -> float:
    """Units of `quote` per 1 unit of `base` (e.g. USD->EUR ~ 0.92).

    Uses the same on-disk cache and stale fallback as price data, so a throttled
    FX request degrades to a slightly old rate instead of breaking the app.
    """
    base, quote = base.upper(), quote.upper()
    if base == quote:
        return 1.0

    pair = Instrument(f"{base}{quote}=X", f"{base}{quote}", "fx")
    for symbol, invert in ((f"{base}{quote}=X", False), (f"{quote}{base}=X", True)):
        pair.symbol = symbol
        try:
            df = fetch(pair, "1d", period="1mo", min_bars=1)
        except Exception:  # noqa: BLE001 - try the inverse pair next
            continue
        rate = float(df["close"].iloc[-1])
        if rate > 0:
            return 1.0 / rate if invert else rate

    raise RuntimeError(
        f"Could not fetch an FX rate for {base}/{quote}. Prices stay in {base}."
    )
