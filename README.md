# QuantDesk

A multi-strategy technical analysis desk for stocks and crypto. Type a symbol, get a
candlestick chart with EMAs, Bollinger bands, Fibonacci levels, RSI and MACD — plus
**sixteen trading strategies evaluated side by side**, each backtested on that exact symbol
and timeframe, each with a concrete *buy at / stop at / sell at* price.

The point of the app is that it does not assert which strategy is best. It runs them
all, measures them, and ranks them on the data in front of you.

---

## Quickstart

```bash
pip install -r requirements.txt

streamlit run app.py                  # the app
python cli.py NVDA                    # same analysis, terminal report
python cli.py NVDA --currency EUR     # ...priced in EUR
```

Symbols accepted:

| You type | Resolves to | |
|---|---|---|
| `AAPL`, `NVDA`, `TSLA` | US equities | |
| `RELIANCE.NS`, `SAP.DE` | international equities | |
| `BTC`, `ETH`, `SOL` | `BTC-USD`, `ETH-USD`, … | crypto, 24/7 |
| `SPX`, `NIFTY`, `DAX` | `^GSPC`, `^NSEI`, `^GDAXI` | indices |

Pick a **display currency** (USD / EUR / GBP / INR / CHF / JPY) in the sidebar — see
[Currency](#currency) below.

Pick **Swing** (daily bars, multi-day holds) or **Day** (5m–1h bars, same-session holds)
in the sidebar. The strategy book changes with the mode — VWAP Trend Reclaim only
appears intraday, where session VWAP actually means something.

---

## The strategy book

Sixteen independent systems across eight families, chosen so they disagree usefully
rather than all firing at once. **14 run on swing, 13 on day** (some are timeframe-specific).

| # | Strategy | Family | The idea |
|---|----------|--------|----------|
| 1 | **Trend Pullback Rider** | Trend following | Price > 200 EMA, 20 > 50 EMA, ADX > 18, then a dip into the 20 EMA that closes back above the prior high |
| 2 | **RSI(2) Mean Reversion** | Mean reversion | The Connors reversal: with the 200 SMA trend, buy an RSI(2) washout below 10, exit on the snap back above the 5 SMA |
| 3 | **MACD Momentum + Supertrend** | Momentum | MACD crosses its signal while Supertrend already points the same way and ADX > 20 |
| 4 | **Fibonacci Golden Pocket** | Retracement | Measures the last impulse leg, waits for a retrace into the 0.5–0.618 pocket, requires a reversal bar |
| 5 | **Volatility Squeeze Breakout** | Breakout | Bollinger compresses inside Keltner, then price clears the 20-bar Donchian edge on volume |
| 6 | **VWAP Trend Reclaim** *(day)* | Intraday | Holds above session VWAP, pulls back to test it, reclaims with 9 EMA > 20 EMA |
| 7 | **Golden Cross Regime** *(swing)* | Trend following | 50 SMA over 200 SMA defines the regime; entry on the cross or each reclaim of the 50 SMA within it |
| 8 | **Bollinger Band Fade** | Mean reversion | Fades a push outside the 2σ band back to the middle band, only with the 200 SMA trend |
| 9 | **Turtle Channel Breakout** *(swing)* | Breakout | Richard Dennis, unchanged: buy a 55-bar high, exit on a 20-bar low, 2 ATR stop |
| 10 | **Ichimoku Cloud Break** | Trend following | Price clear of the cloud, Tenkan over Kijun, forward cloud agreeing; Kijun trails the exit |
| 11 | **Opening Range Breakout** *(day)* | Intraday | Marks the first 30 minutes, takes the first clean break of that range on volume |
| 12 | **RSI Divergence Reversal** | Reversal | Price makes a lower low while RSI makes a higher low (or the mirror at highs), on confirmed pivots only |
| 13 | **Volatility Contraction (VCP)** *(swing)* | Pattern | Minervini: a leader near its 52-week high, volatility tightening and volume drying up, then a volume breakout. Long only |
| 14 | **Heikin-Ashi Trend Flip** | Trend following | Enters on the first strong candle after a colour flip, exits on the first candle of the other colour |
| 15 | **Stochastic Pullback** | Mean reversion | %K dips oversold and crosses back up through %D, with price above the 200 EMA |
| 16 | **Gap Fade** | Reversal | Fades a gap of 2%+ once the bar reverses back into the prior range, targeting the gap fill |

Each reports one of four states:

- **LIVE** — the entry condition fired on the latest bar (or the one before).
- **WATCH** — the regime qualifies but the trigger has not fired. The quoted entry is
  the *level to place the order at*, not the current price.
- **STAND DOWN** — conditions do not support this strategy here.
- **NO HISTORY** — the strategy never triggered once on the loaded data, so it cannot be
  scored and gets **no vote** in the verdict. Gap Fade on 24/7 crypto is the honest
  example: there are no gaps to fade, so it reports nothing rather than inventing a signal.

---

## Currency

The sidebar has a **display currency** toggle. Every price on screen — last price,
entry, stop, targets, Fibonacci levels, the chart axis, notional and risk — is
converted at the live spot rate, and the header states the rate it used:

```
Equity · 1d bars · Swing · 753 bars loaded · shown in EUR at USD/EUR 0.8563 (listed in USD)
```

Two things worth understanding:

**It converts from the instrument's listing currency, not from USD.** `SAP.DE` is
listed in EUR, `RELIANCE.NS` in INR, `7203.T` in JPY. Picking USD on `SAP.DE`
converts EUR→USD; picking EUR is a no-op. The listing currency is resolved offline
from the exchange suffix, so it does not depend on a flaky metadata call.

**Analysis stays in the listing currency.** Only the presentation layer is converted,
at today's spot rate. That is deliberate: the levels you get still match a broker
ticket in the native currency, and scaling by a constant leaves every ratio-based
indicator (RSI, ADX, %B, ATR%) untouched while price-level ones (EMA, Bollinger,
ATR, MACD, fibs) move with price — the technical picture is identical, just relabelled.
The alternative — converting each historical bar at its own FX rate — would give a
genuinely different chart with FX volatility baked in, and different signals. This
app does not do that.

The consequence: **position size is computed in your display currency.** Set the
account to 10,000 with EUR selected and it sizes a €10,000 account, converting your
risk into the native currency to work out the share count. €10,000 risking 1% buys
more AAPL shares than $10,000 risking 1%, and the app reflects that.

FX rates are cached on disk with the same stale-fallback behaviour as price data. If
a rate cannot be fetched at all, the app stays in the native currency and says so
rather than showing a wrong number.

---

## How "best strategy" is decided

Every strategy is backtested over the loaded history before anything is recommended.
The simulation is deliberately unflattering:

- signals are read on a bar's **close** and filled at the **next bar's open** — no lookahead
- stops and targets are checked intrabar against high/low, and **the stop wins ties**
- commission + slippage are charged on **both** sides (5 bps default, adjustable)
- position size is fixed-fractional risk, so every result is in **R** (multiples of money
  risked) and strategies are directly comparable
- the stop trails to breakeven-plus after +1R, then by 2 ATR

That produces an **Edge score (0–100)** per strategy, blending:

```
expectancy in R   (dominant)
+ profit factor
- drawdown penalty
+/- out-of-sample agreement     (the most recent 30% of history, reported separately)
× a confidence factor from sample size   (8 trades ≈ 0.5, 40+ ≈ 1.0)
```

Thin samples get pulled toward neutral rather than being allowed to look brilliant off
six lucky trades. **If the out-of-sample column disagrees with the headline expectancy,
treat the edge as unproven** — that column exists to catch curve-fitting.

The **composite verdict** is an edge-weighted vote across the whole book (live signals
count fully, forming setups at 45%, stand-downs at 10%), blended 70/30 with an objective
regime read of trend, ADX, RSI and volatility. The recommended order comes from the
highest-ranked strategy that agrees with that verdict.

---

## Reading the output

**Order card** — entry, stop, three targets, and the position size for your account and
risk %. Size comes from `risk_amount / (entry − stop)`, so the money at risk is the same
whichever strategy is chosen and however wide its stop is.

**The chart** is built on **TradingView Lightweight Charts** — the library TradingView
open-sourced and that many exchange front-ends run on — so it behaves like a real terminal
rather than a plotted figure:

- **Maximise** (button, or press **F**) expands the chart to fill the entire browser
  window — sidebar and all — with the panes rescaling to the new height; **Esc** restores
- **grab anywhere and drag** to pan; wheel to zoom; drag the price or date axis to scale
  just that axis; double-click an axis to reset
- **move the pointer over any bar** and the top-left readout shows that bar's O/H/L/C,
  the change %, EMA 20/50/200, VWAP, RSI and MACD *at that moment* — the crosshair carries
  price and date labels onto both axes
- three synced panes (price + volume, RSI, MACD) share one time axis and one crosshair
- overlay buttons above the chart toggle EMAs, Bollinger, VWAP, volume, Fibonacci and the
  order levels instantly

The library is **vendored at `assets/lwc.js`** and inlined into the page, so the chart needs
no CDN and works offline. A **Plotly fallback** is still available under *Chart & model
settings → Chart engine* and is what `cli.py --chart` exports to standalone HTML.

**Toggling overlays** — every overlay is a legend entry: Bollinger, EMA 20/50/200,
**Fibonacci**, **Trade levels**, MACD, Signal. Click one to hide it instantly (client-side,
no reload); double-click to isolate it. The whole Fibonacci overlay — levels, labels,
pocket shading and the measured swing — sits under one legend entry, as does the whole
order overlay. The sidebar **Chart overlays** checkboxes do the same thing but persist as
the default for each reload.

**Colours** — EMA 20 / 50 / 200 are three distinct hues (blue / magenta / amber), not
shades of one colour, so they are told apart at a glance; the set passes every
colour-blindness and contrast check on both the dark and light surface. Candles use
reserved green/red status colours, and everything Fibonacci is violet.

**Chart** — labels are split into two zones so they never collide: **order levels on the
left** (entry, stop, targets), **Fibonacci on the right**. The violet line with dots is the
measured swing itself, and fib levels are drawn only from that swing forward — not across
history the swing predates. Every drawn line carries a label; a level that cannot be
labelled legibly is dropped rather than left as an anonymous dash. The 0.5 and 0.618 edges
share one "pocket" label so neither is ever lost, and the shaded band is that pocket.
Dotted lines past the swing are extension targets. 0.236 is omitted from the chart to cut
clutter but remains in the table. A leg price has already retraced past ~115% is discarded
and the previous pivot pair is used instead.

**Fibonacci panel** (Levels & context tab) — a retracement track shows where price sits on
the leg from 0% (at the extreme) to 100% (fully given back), with the golden pocket marked
and a "you are here" line. Below it, the depth is named in plain language — *shallow
pullback / normal pullback / golden pocket / deep retracement / leg invalidated* — with what
it means for a trade, plus the nearest level and the next support and resistance. The full
level table is in the "All levels" expander, sorted high to low and tagged support or
resistance.

**Strategies tab** — each card's "Why / why not" expander shows the full rule checklist
with the live reading beside each condition, so a WATCH state tells you exactly which
condition is missing.

**Backtests tab** — the full metrics table, plus the equity curve and trade log for any
strategy you select.

---

## Layout

```
app.py               Streamlit UI
cli.py               terminal report (same engine)
core/
  data.py            symbol normalisation, Yahoo OHLCV with period fallback,
                     disk cache, listing-currency resolution + FX rates
  indicators.py      EMA/SMA/RSI/MACD/ATR/ADX/Bollinger/Keltner/Donchian/
                     Supertrend/Stochastic/VWAP/OBV — pure pandas, no TA-Lib
  fibonacci.py       fractal pivot detection, retracement + extension levels
  strategies.py      the six systems: rules, entry levels, stops, checklists
  backtest.py        event-driven simulator + metrics + edge score
  engine.py          orchestration, composite verdict, position sizing
  charting.py        Plotly figures (fallback engine + CLI HTML export)
  tvchart.py         TradingView Lightweight Charts renderer (default engine)
assets/lwc.js        vendored Lightweight Charts build (Apache 2.0)
```

Indicators are computed once into a shared feature frame, so adding a strategy means
implementing `rules()`, `entry_reference()` and `checklist()` and appending it to
`ALL_STRATEGIES` — the backtest, ranking, chart and UI pick it up automatically.

---

## Limitations worth knowing

- Yahoo intraday history is capped (1m ≈ 7 days, 5m–30m ≈ 60 days), so intraday edge
  scores rest on smaller samples than daily ones.
- Backtests assume the stop is always honoured at its exact price. Gaps and thin books
  will do worse in reality.
- No fundamentals, news, earnings dates or order-book data. This is price and volume only.

**Not investment advice.** Backtested edge describes past behaviour of one symbol on one
timeframe. It is a measurement, not a forecast.

---

## Tests

```bash
python -m pytest tests -q          # 133 passed, 15 skipped
```

The suite runs entirely on deterministic synthetic markets (trending, falling, choppy,
gappy, and an intraday session series) — no network, so it is reproducible.

The one that matters is **`test_no_lookahead`**. For each of the 16 strategies it walks
forward: at several cut points it rebuilds the strategy from *only* the bars up to that
point and asserts the live decision equals what the full-history run shows for that same
bar. Anything reading even one bar ahead disagrees, because the truncated run cannot see
it. A lookahead bug would make every backtest number in this app a lie, so it is worth a
dedicated test.

**`test_lookahead_detector_works`** plants a deliberately cheating rule (`close.shift(-1)`)
and asserts the detector catches it. The first version of that check silently passed
against the planted bug — a green test that cannot fail is worse than no test — so the
detector is now verified rather than assumed.

The rest covers the strategy contract (boolean, aligned, NaN-free rule series), stop
placement on the correct side and within a sane ATR distance, backtest metric bounds and
trade-ordering invariants, monotonic cost impact, target ordering, position sizing risking
exactly the requested amount, Fibonacci level bracketing, and symbol/currency resolution.

---

## Licence & attribution

QuantDesk is released under the **MIT Licence** — see [LICENSE](LICENSE). You may use,
modify and redistribute it freely, including commercially, provided the copyright notice
is kept.

Third-party components and obligations are recorded in [NOTICE](NOTICE). Two are worth
calling out here:

**TradingView Lightweight Charts** (`assets/lwc.js`, v4.2.3) is bundled with this repo
under the Apache-2.0 licence, whose terms include a **mandatory attribution requirement**:
TradingView must be credited and a link to <https://www.tradingview.com/> must appear on
the user-facing page. This is satisfied by the library's built-in `attributionLogo` (the
mark on each chart pane) plus the footer credit. **Do not disable `attributionLogo` or
remove the footer attribution** without providing that link elsewhere — it would breach
the licence. The full Apache-2.0 text ships at
[`assets/LICENSE-lightweight-charts.txt`](assets/LICENSE-lightweight-charts.txt).

**Market data** comes from Yahoo Finance via yfinance, which is not affiliated with or
endorsed by Yahoo and whose authors state the API is intended for personal, research and
educational use. This project is offered on that basis. Anyone deploying it should check
Yahoo's terms for their own case, and obtain a licensed data feed before commercial use.

**Not financial advice.** QuantDesk is a technical analysis tool. Nothing it produces is
investment advice or a recommendation to trade. Backtested edge describes past behaviour,
not future returns, and trading carries risk of loss.
