"""QuantDesk - multi-strategy technical analysis desk for stocks and crypto.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import streamlit as st

from core.charting import (THEMES, beta_chart, equity_chart, price_chart,
                           score_gauge)
from core.data import (DAY_INTERVALS, DISPLAY_CURRENCIES, LONG_INTERVALS,
                       SWING_INTERVALS, currency_symbol, fetch, fx_rate,
                       native_currency, normalise_symbol)
from core.engine import analyse, size_position
from core.fibonacci import describe, level_rows
from core import tvchart
from core.valuation import fair_value, load_beta
from core.backtest import trades_frame

st.set_page_config(page_title="QuantDesk", page_icon="", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------
def fmt(x: float, currency: str = "") -> str:
    """Price formatting that survives both BRK.A and PEPE."""
    if x is None or not np.isfinite(x):
        return "-"
    a = abs(x)
    if a >= 1000:
        s = f"{x:,.2f}"
    elif a >= 1:
        s = f"{x:,.2f}"
    elif a >= 0.01:
        s = f"{x:,.5f}"
    else:
        s = f"{x:,.8f}"
    return f"{currency}{s}" if currency else s


@st.cache_data(ttl=900, show_spinner=False)
def get_rate(base: str, quote: str) -> float:
    return fx_rate(base, quote)


VERDICT_COLOR = {
    "STRONG BUY": "#0ca30c", "BUY": "#0ca30c", "NEUTRAL": "#fab219",
    "SELL": "#d03b3b", "STRONG SELL": "#d03b3b", "STAND ASIDE": "#fab219",
}
STATE_BADGE = {
    "TRIGGERED": ("LIVE", "#0ca30c"),
    "SETUP FORMING": ("WATCH", "#fab219"),
    "NO SETUP": ("STAND DOWN", "#83827b"),
    "NO DATA": ("NO HISTORY", "#83827b"),
}


def stat(value: float, fmt_spec: str = "+.2f", suffix: str = "") -> str:
    """Format a metric, or an em dash when the backtest produced nothing."""
    if value is None or not np.isfinite(value):
        return "&mdash;"
    return f"{value:{fmt_spec}}{suffix}"


@st.cache_data(ttl=300, show_spinner=False)
def load(symbol: str, interval: str):
    inst = normalise_symbol(symbol)
    return inst, fetch(inst, interval)


@st.cache_data(ttl=1800, show_spinner=False)
def get_beta(symbol: str, interval: str, native: str):
    inst, df = load(symbol, interval)
    return load_beta(inst, df, native, interval)


@st.cache_data(ttl=3600, show_spinner=False)
def _fair_value_raw(symbol: str, interval: str):
    """Fundamentals only. Cached per SYMBOL, not per price.

    Valuation does not depend on bar size, so keying the cache on the interval
    or the live price would refetch Yahoo every time you switch timeframe and
    let a moving analyst target make the answer look unstable when it is not.
    """
    inst, _ = load(symbol, interval)
    return fair_value(inst, float("nan"))


def get_fair_value(symbol: str, interval: str, price: float):
    """The cached valuation, compared against the price currently on screen."""
    fv = _fair_value_raw(symbol, interval)
    return dataclasses.replace(fv, price=price)


@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(symbol: str, interval: str, style: str, allow_short: bool, cost_bps: float):
    inst, df = load(symbol, interval)
    return analyse(df, inst, interval, style, allow_short=allow_short, cost_bps=cost_bps)


def css(theme: str):
    t = THEMES[theme]
    st.markdown(
        f"""
        <style>
          .stApp {{ background: {t['surface']}; }}
          .qd-card {{
            background: {t['surface']}; border: 1px solid {t['grid']};
            border-radius: 12px; padding: 16px 18px; height: 100%;
          }}
          .qd-badge {{
            display:inline-block; padding: 2px 9px; border-radius: 999px;
            font-size: 11px; font-weight: 700; letter-spacing: .06em;
          }}
          .qd-label {{ color:{t['muted']}; font-size:11px; letter-spacing:.07em;
                       text-transform:uppercase; margin-bottom:2px; }}
          .qd-value {{ color:{t['text']}; font-size:20px; font-weight:650;
                       font-variant-numeric: tabular-nums; }}
          .qd-sub   {{ color:{t['text_secondary']}; font-size:12px; line-height:1.5; }}
          .qd-bar   {{ height:6px; border-radius:3px; background:{t['grid']}; overflow:hidden; }}
          .qd-bar > div {{ height:100%; border-radius:3px; }}
          .fib-track {{ position:relative; height:26px; border-radius:6px;
                        background:{t['grid']}; margin:10px 0 4px; overflow:hidden; }}
          .fib-pocket {{ position:absolute; top:0; bottom:0; background:{t['pocket']};
                         border-left:1px solid {t['warning']}66;
                         border-right:1px solid {t['warning']}66; }}
          .fib-mark {{ position:absolute; top:-3px; bottom:-3px; width:3px;
                       background:{t['text']}; border-radius:2px; }}
          .fib-scale {{ display:flex; justify-content:space-between;
                        color:{t['muted']}; font-size:10px; }}
          .fib-chip {{ display:inline-block; padding:3px 10px; border-radius:999px;
                       font-size:12px; font-weight:650; }}
          div[data-testid="stMetricValue"] {{ font-size: 20px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## QuantDesk")
    st.caption("Multi-strategy technical desk for stocks & crypto")

    symbol = st.text_input("Symbol", value="AAPL",
                           help="Stocks: AAPL, NVDA, TSLA, RELIANCE.NS  |  Crypto: BTC, ETH, SOL  |  Index: SPX, NIFTY")
    style = st.radio(
        "Trading style", ["Swing", "Day", "Long-term"], horizontal=True,
        help="Day: intraday bars, same-session holds. Swing: daily bars, multi-day holds. "
             "Long-term: weekly/monthly bars, positions held months to years.")
    style_key = {"Swing": "swing", "Day": "day", "Long-term": "long"}[style]

    intervals = {"swing": SWING_INTERVALS, "day": DAY_INTERVALS,
                 "long": LONG_INTERVALS}[style_key]
    interval = st.selectbox("Timeframe", intervals, index=0)

    st.divider()
    st.markdown("**Risk**")
    display_ccy = st.selectbox(
        "Display currency", DISPLAY_CURRENCIES, index=0,
        help="Converts every price, level and position value at the live spot rate. "
             "Analysis itself stays in the instrument's listing currency.")
    account = st.number_input(f"Account size ({display_ccy})",
                              min_value=100.0, value=10_000.0, step=500.0)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25)
    allow_short = st.toggle("Allow short trades", value=True)

    st.divider()
    st.markdown("**Chart overlays**")
    st.caption("Or click any legend entry on the chart to toggle it instantly.")
    ov1, ov2 = st.columns(2)
    show_fib = ov1.checkbox("Fibonacci", value=True)
    show_bb = ov2.checkbox("Bollinger", value=True)
    show_vwap = ov1.checkbox("VWAP", value=(style_key == "day"))
    show_plan = ov2.checkbox("Trade levels", value=True)

    st.divider()
    with st.expander("Chart & model settings"):
        engine = st.selectbox(
            "Chart engine", ["Interactive (TradingView)", "Static (Plotly)"], index=0,
            help="Interactive uses TradingView's Lightweight Charts - free panning, "
                 "wheel zoom, axis scaling and a crosshair readout. Static is the "
                 "Plotly fallback.")
        theme = st.selectbox("Theme", ["dark", "light"], index=0)
        chart_bars = st.slider("Bars on chart", 80, 500, 240, 20)
        cost_bps = st.slider("Costs per side (bps)", 0.0, 25.0, 5.0, 0.5,
                             help="Commission + slippage charged on entry and exit inside the backtest.")

    run = st.button("Analyse", type="primary", width="stretch")

css(theme)
T = THEMES[theme]

# --------------------------------------------------------------------------
if not symbol:
    st.info("Enter a symbol in the sidebar to begin.")
    st.stop()

try:
    with st.spinner(f"Loading {symbol} and backtesting the strategy book..."):
        a = run_analysis(symbol, interval, style_key, allow_short, cost_bps)
except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
    st.error(str(exc))
    st.stop()

inst, ctx, rec, f = a.instrument, a.context, a.recommendation, a.features

# Display currency. Analysis runs in the instrument's listing currency; only the
# presentation layer is converted, so quoted levels still match a broker ticket
# in the native currency.
native_ccy = native_currency(inst)
fx_note = ""
try:
    RATE = get_rate(native_ccy, display_ccy)
except Exception as exc:  # noqa: BLE001 - fall back to native rather than fail
    RATE, display_ccy = 1.0, native_ccy
    fx_note = f"FX unavailable ({exc}) - showing {native_ccy}."
SYM = currency_symbol(display_ccy)


def money(x: float) -> str:
    """Format a native-currency price in the chosen display currency."""
    return fmt(x * RATE, SYM)


def shown(x: float) -> str:
    """Format a value that is already in the display currency."""
    return fmt(x, SYM)

# ---- header ---------------------------------------------------------------
h1, h2, h3, h4, h5 = st.columns([2.4, 1, 1, 1, 1])
with h1:
    st.markdown(f"### {inst.symbol}")
    caption = f"{inst.asset_class.title()} · {interval} bars · {style} · {len(f):,} bars loaded"
    if RATE != 1.0:
        caption += (f" · shown in {display_ccy} at {native_ccy}/{display_ccy} "
                    f"{RATE:,.4f} (listed in {native_ccy})")
    st.caption(caption)
    if fx_note:
        st.warning(fx_note, icon="⚠️")
h2.metric("Last", money(ctx.price), f"{ctx.change_pct:+.2f}%")
h3.metric("Trend", ctx.trend.split(" /")[0])
h4.metric("RSI (14)", f"{ctx.rsi:.1f}")
h5.metric("ATR", f"{ctx.atr_pct:.2f}%", ctx.volatility, delta_color="off")

st.divider()

# ---- the recommendation ---------------------------------------------------
left, right = st.columns([1, 2.1])

with left:
    colour = VERDICT_COLOR.get(rec.verdict, T["muted"])
    st.markdown(
        f"<div class='qd-label'>Composite verdict</div>"
        f"<div style='font-size:30px;font-weight:750;color:{colour};line-height:1.2'>{rec.verdict}</div>"
        f"<div class='qd-sub'>{rec.agreement}</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(score_gauge(rec.score, theme), width="stretch", theme=None,
                    config={"displayModeBar": False})

with right:
    best = rec.best
    if best is None:
        st.warning("No strategy produced a usable plan on this data.")
    else:
        side = "Buy" if best.direction == "LONG" else "Sell / short"
        st.markdown(
            f"<div class='qd-label'>Recommended order · {best.name}</div>",
            unsafe_allow_html=True,
        )
        o1, o2, o3, o4 = st.columns(4)
        o1.markdown(
            f"<div class='qd-label'>{side} at</div><div class='qd-value'>{money(rec.entry)}</div>"
            f"<div class='qd-sub'>{'limit / on trigger' if not best.is_live else 'live now'}</div>",
            unsafe_allow_html=True)
        o2.markdown(
            f"<div class='qd-label'>Stop loss</div>"
            f"<div class='qd-value' style='color:{T['critical']}'>{money(rec.stop)}</div>"
            f"<div class='qd-sub'>{abs(rec.entry - rec.stop) / rec.entry * 100:.2f}% away</div>",
            unsafe_allow_html=True)
        tgt_txt = "<br>".join(
            f"T{i + 1} {money(t)} <span style='color:{T['muted']}'>({mult:.1f}R)</span>"
            for i, (t, mult) in enumerate(zip(rec.targets, best.targets_r))
        )
        o3.markdown(
            f"<div class='qd-label'>Take profit</div>"
            f"<div class='qd-value' style='color:{T['good']};font-size:15px;line-height:1.6'>{tgt_txt}</div>",
            unsafe_allow_html=True)

        sz = size_position(account, risk_pct, rec.entry * RATE, rec.stop * RATE)
        units = f"{sz['units']:,.4f}".rstrip("0").rstrip(".")
        o4.markdown(
            f"<div class='qd-label'>Position size</div>"
            f"<div class='qd-value'>{units} units</div>"
            f"<div class='qd-sub'>{shown(sz['notional'])} notional<br>"
            f"risking {shown(sz['risk_amount'])} ({risk_pct:.2f}%)</div>",
            unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        for line in rec.reasoning:
            st.markdown(f"<div class='qd-sub'>• {line}</div>", unsafe_allow_html=True)
        if rec.caution:
            st.warning(rec.caution, icon="⚠️")

st.divider()

# ---- tabs -----------------------------------------------------------------
tab_chart, tab_strats, tab_bt, tab_levels, tab_val = st.tabs(
    ["Chart", f"Strategies ({len(a.plans)})", "Backtests", "Levels & context",
     "Valuation & risk"]
)

with tab_chart:
  if engine.startswith("Interactive"):
    st.iframe(
        tvchart.render(
            f, fib=a.fib, plan=rec.best if show_plan else None, theme=theme,
            rate=RATE, currency=SYM, symbol=inst.symbol, interval=interval,
            bars=chart_bars, height=760,
            show_bb=show_bb, show_vwap=show_vwap, show_fib=show_fib,
            show_plan=show_plan,
        ),
        height=810,
    )
    st.caption(
        "**Maximise** (or press **F**) fills the whole window; **Esc** returns. "
        "Drag anywhere to pan · wheel to zoom · drag the price or date axis to scale it · "
        "double-click an axis to reset. Move the pointer over any bar to read its OHLC and "
        "every indicator at that moment in the top-left. Buttons above the chart toggle "
        "overlays. Fibonacci levels are labelled inside the chart; the price axis is "
        "reserved for the order levels."
    )
  else:
    fig = price_chart(
          f, fib=a.fib, plan=rec.best if show_plan else None, theme=theme,
          show_bollinger=show_bb, show_vwap=show_vwap, show_fib=show_fib,
          title=f"{inst.symbol} · {interval}", bars=chart_bars,
          rate=RATE, currency=SYM,
          weekend_breaks=(inst.asset_class in ("equity", "index") and interval in ("1d", "1wk")),
      )
    st.plotly_chart(
        fig, width="stretch", theme=None,
        config={
            "scrollZoom": True,              # wheel zooms inside the plot
            "doubleClick": "reset+autosize",  # double-click restores the full view
            "displaylogo": False,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"],
            "toImageButtonOptions": {"format": "png", "scale": 2,
                                     "filename": f"{inst.symbol}_{interval}"},
        },
    )
    st.caption(
        "**Navigation** — drag the price axis (right edge) up or down to stretch or compress the "
        "price scale; drag the date axis to stretch time; drag inside the chart to pan; scroll to "
        "zoom; double-click to reset. "
        "**Left labels** are the recommended order (entry, stop, targets). **Right labels** are the "
        "Fibonacci levels, drawn only from the measured swing forward — the violet line with dots is "
        "that swing, and the shaded band is the 0.5–0.618 golden pocket. Dotted lines beyond the swing "
        "are extension targets. If the swing is recent it occupies a narrow strip; lower "
        "**Bars on chart** in the sidebar to zoom into it."
    )

with tab_strats:
    st.caption(
        "All strategies run on the same data at the same time. **Edge** is each system's "
        "backtested score on this symbol and timeframe - it is what decides the ranking."
    )
    cols = st.columns(2)
    for i, p in enumerate(a.plans):
        m = p.metrics
        badge, badge_col = STATE_BADGE[p.state]
        dir_col = T["good"] if p.direction == "LONG" else T["critical"]
        with cols[i % 2]:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(
                    f"**{p.name}**  \n<span class='qd-sub'>{p.family}</span>",
                    unsafe_allow_html=True)
                c2.markdown(
                    f"<div style='text-align:right'><span class='qd-badge' "
                    f"style='background:{badge_col}22;color:{badge_col}'>{badge}</span></div>",
                    unsafe_allow_html=True)

                st.markdown(
                    f"<div style='font-size:18px;font-weight:700;color:{dir_col}'>{p.action}"
                    f"<span style='color:{T['text_secondary']};font-weight:400;font-size:13px'>"
                    f" &nbsp;@ {money(p.entry)}</span></div>",
                    unsafe_allow_html=True)

                st.markdown(
                    f"<div class='qd-label'>Conviction {p.conviction:.0f}/100</div>"
                    f"<div class='qd-bar'><div style='width:{p.conviction:.0f}%;background:{dir_col}'></div></div>",
                    unsafe_allow_html=True)

                k1, k2, k3 = st.columns(3)
                k1.markdown(f"<div class='qd-label'>Stop</div><div class='qd-sub' "
                            f"style='color:{T['critical']};font-size:14px'>{money(p.stop)}</div>",
                            unsafe_allow_html=True)
                k2.markdown(f"<div class='qd-label'>Targets</div><div class='qd-sub' "
                            f"style='color:{T['good']};font-size:14px'>"
                            f"{' / '.join(money(t) for t in p.targets)}</div>",
                            unsafe_allow_html=True)
                k3.markdown(f"<div class='qd-label'>Reward:risk</div>"
                            f"<div class='qd-sub' style='font-size:14px'>1 : {p.rr:.1f}</div>",
                            unsafe_allow_html=True)

                st.markdown(
                    f"<div class='qd-sub' style='margin-top:8px'>"
                    f"<b>Edge {m.edge_score:.0f}</b>/100 · {m.trades} trades · "
                    f"win {stat(m.win_rate, '.0f', '%')} · "
                    f"expectancy {stat(m.expectancy_r)}R · PF {stat(m.profit_factor, '.2f')}"
                    + ("" if m.trades == 0 else
                       ("" if m.has_sample else " · <i>thin sample</i>"))
                    + ("<br><i>This strategy never triggered on the loaded history — "
                       "it does not apply to this market.</i>" if m.trades == 0 else "")
                    + "</div>",
                    unsafe_allow_html=True)

                with st.expander("Why / why not"):
                    st.markdown(f"<div class='qd-sub'>{p.blurb}</div><br>", unsafe_allow_html=True)
                    for label, ok, detail in p.checklist:
                        mark = "✅" if ok else "⬜"
                        st.markdown(
                            f"<div class='qd-sub'>{mark} {label} "
                            f"<span style='color:{T['muted']}'>— {detail}</span></div>",
                            unsafe_allow_html=True)

with tab_bt:
    rows = []
    for p in a.plans:
        m = p.metrics
        rows.append({
            "Strategy": p.name,
            "Edge": round(m.edge_score, 1),
            "Trades": m.trades,
            "Win %": round(m.win_rate, 1) if np.isfinite(m.win_rate) else None,
            "Expectancy (R)": round(m.expectancy_r, 3) if np.isfinite(m.expectancy_r) else None,
            "Profit factor": round(m.profit_factor, 2) if np.isfinite(m.profit_factor) else None,
            "Out-of-sample (R)": round(m.oos_expectancy_r, 3) if np.isfinite(m.oos_expectancy_r) else None,
            "Return %": round(m.total_return * 100, 1) if np.isfinite(m.total_return) else None,
            "Max DD %": round(m.max_drawdown * 100, 1) if np.isfinite(m.max_drawdown) else None,
            "Sharpe": round(m.sharpe, 2) if np.isfinite(m.sharpe) else None,
            "Avg bars held": round(m.avg_bars, 1) if np.isfinite(m.avg_bars) else None,
        })
    bt = pd.DataFrame(rows)
    st.dataframe(
        bt, width="stretch", hide_index=True,
        column_config={
            "Edge": st.column_config.ProgressColumn(
                "Edge", help="Backtested edge on this symbol and timeframe",
                format="%.0f", min_value=0, max_value=100),
            "Win %": st.column_config.NumberColumn(format="%.1f%%"),
            "Return %": st.column_config.NumberColumn(format="%.1f%%"),
            "Max DD %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(
        f"Simulated at {risk_pct:.2f}% risk per trade with {cost_bps:.1f} bps cost per side, "
        "signals read on the close and filled at the next bar's open. "
        "**Out-of-sample** is the expectancy over the most recent 30% of history only - "
        "if it disagrees with the headline number, treat the edge as unproven."
    )

    pick = st.selectbox("Equity curve & trade log", [p.name for p in a.plans])
    chosen = next(p for p in a.plans if p.name == pick)
    st.plotly_chart(
        equity_chart(chosen.metrics, theme, f"{pick} · cumulative return at {risk_pct:.2f}% risk/trade"),
        width="stretch", theme=None, config={"displayModeBar": False},
    )
    tf = trades_frame(chosen.metrics)
    if tf.empty:
        st.info("This strategy produced no trades on the loaded history.")
    else:
        st.dataframe(tf.tail(60).iloc[::-1], width="stretch", hide_index=True)

with tab_levels:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Market context")
        st.markdown(f"<div class='qd-sub'>{ctx.trend_detail}</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("ADX (trend force)", f"{ctx.adx:.1f}")
        m2.metric("Volatility regime", ctx.volatility, f"ATR {ctx.atr_pct:.2f}%", delta_color="off")
        m1.metric("From 1-year high", f"{ctx.from_high_pct:+.1f}%")
        m2.metric("Regime score", f"{ctx.regime_score:+.0f}")
        for n in ctx.notes:
            st.markdown(f"<div class='qd-sub'>• {n}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("#### Fibonacci")
        if a.fib is None:
            st.info("No clean impulse leg detected on this timeframe.")
        else:
            leg = a.fib
            d = describe(leg, ctx.price)
            arrow = "↗" if leg.direction == "up" else "↘"
            st.markdown(
                f"<div class='qd-sub'>Measured from the {arrow} <b>{leg.direction}-leg</b> "
                f"{money(leg.start_price)} ({leg.start_idx:%d %b}) → {money(leg.end_price)} "
                f"({leg.end_idx:%d %b})</div>",
                unsafe_allow_html=True)

            # where price sits on the leg: 0% = at the extreme, 100% = fully given back
            depth = float(np.clip(d.get("depth_pct", 0.0), 0.0, 100.0))
            tone = T["good"] if d.get("in_pocket") else (
                T["critical"] if d.get("depth_pct", 0) > 100 or d.get("depth_pct", 0) < 0
                else T["warning"])
            st.markdown(
                f"<div class='fib-track'>"
                f"<div class='fib-pocket' style='left:50%;width:11.8%'></div>"
                f"<div class='fib-mark' style='left:calc({depth:.1f}% - 1px)'></div>"
                f"</div>"
                f"<div class='fib-scale'><span>0% (leg extreme)</span>"
                f"<span>golden pocket</span><span>100% (fully retraced)</span></div>",
                unsafe_allow_html=True)

            st.markdown(
                f"<div style='margin-top:10px'>"
                f"<span class='fib-chip' style='background:{tone}22;color:{tone}'>"
                f"{d.get('depth_pct', float('nan')):.1f}% retraced · {d.get('headline', '')}</span></div>"
                f"<div class='qd-sub' style='margin-top:6px'>{d.get('meaning', '')}</div>",
                unsafe_allow_html=True)

            n1, n2, n3 = st.columns(3)
            n1.markdown(f"<div class='qd-label'>Nearest level</div>"
                        f"<div class='qd-sub' style='font-size:15px'>{d.get('nearest_label', '-')} · "
                        f"{money(d.get('nearest_price', float('nan')))}</div>"
                        f"<div class='qd-sub'>{d.get('nearest_distance_pct', float('nan')):+.2f}% away</div>",
                        unsafe_allow_html=True)
            n2.markdown(f"<div class='qd-label'>Next resistance</div>"
                        f"<div class='qd-sub' style='font-size:15px;color:{T['critical']}'>"
                        f"{money(d.get('next_resistance', float('nan')))}</div>",
                        unsafe_allow_html=True)
            n3.markdown(f"<div class='qd-label'>Next support</div>"
                        f"<div class='qd-sub' style='font-size:15px;color:{T['good']}'>"
                        f"{money(d.get('next_support', float('nan')))}</div>",
                        unsafe_allow_html=True)

            with st.expander("All levels"):
                rows = []
                for r in level_rows(leg, ctx.price):
                    tag = "◆ golden pocket" if r["is_pocket"] else (
                        "← nearest" if r["is_nearest"] else "")
                    rows.append({
                        "Level": r["label"],
                        "Price": money(r["price"]),
                        "Zone": r["zone"],
                        "Distance": f"{r['distance_pct']:+.2f}%",
                        "": tag,
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Key indicator readings")
    r = f.iloc[-1]
    grid = st.columns(6)
    readings = [
        ("EMA 20", money(r["ema20"])), ("EMA 50", money(r["ema50"])), ("EMA 200", money(r["ema200"])),
        ("SMA 200", money(r["sma200"])), ("MACD hist", f"{r['macd_hist'] * RATE:+.4f}"),
        ("Stoch %K", f"{r['stoch_k']:.1f}"), ("RSI (2)", f"{r['rsi2']:.1f}"),
        ("ADX", f"{r['adx']:.1f}"), ("Rel. volume", f"{r['rvol']:.2f}x"),
        ("BB %B", f"{r['bb_pctb']:.2f}"), ("Supertrend", "up" if r["st_dir"] > 0 else "down"),
        ("VWAP", money(r["vwap"])),
    ]
    for i, (k, v) in enumerate(readings):
        grid[i % 6].markdown(
            f"<div class='qd-label'>{k}</div><div class='qd-sub' style='font-size:15px'>{v}</div>",
            unsafe_allow_html=True)


with tab_val:
    vcol1, vcol2 = st.columns([1, 1])

    # ---- beta / alpha ----------------------------------------------------
    with vcol1:
        st.markdown("#### Market risk (beta)")
        try:
            bt = get_beta(symbol, interval, native_ccy)
        except Exception as exc:  # noqa: BLE001
            bt = None
            st.warning(f"Beta unavailable: {exc}")

        if bt is None or not bt.ok:
            if bt is not None and bt.error:
                st.info(bt.error)
        else:
            b1, b2, b3 = st.columns(3)
            b1.metric("Beta", f"{bt.beta:.2f}", bt.label, delta_color="off")
            b2.metric("Alpha (annual)", f"{bt.alpha_annual:+.1f}%")
            b3.metric("R-squared", f"{bt.r_squared * 100:.0f}%")
            st.markdown(f"<div class='qd-sub'>{bt.meaning}</div>", unsafe_allow_html=True)

            v1, v2 = st.columns(2)
            v1.markdown(
                f"<div class='qd-label'>Volatility (annual)</div>"
                f"<div class='qd-sub' style='font-size:15px'>{bt.asset_vol:.1f}% "
                f"vs {bt.bench_vol:.1f}% for the {bt.benchmark_name}</div>",
                unsafe_allow_html=True)
            v2.markdown(
                f"<div class='qd-label'>Correlation</div>"
                f"<div class='qd-sub' style='font-size:15px'>{bt.correlation:+.2f} "
                f"over {bt.observations} bars</div>",
                unsafe_allow_html=True)

            # what beta means for the position actually being recommended
            if np.isfinite(rec.entry) and np.isfinite(rec.stop) and abs(bt.beta) > 0.05:
                szb = size_position(account, risk_pct, rec.entry * RATE, rec.stop * RATE)
                beta_adj = szb["notional"] * bt.beta
                st.markdown(
                    f"<div class='qd-sub' style='margin-top:10px'>The recommended "
                    f"{shown(szb['notional'])} position carries "
                    f"<b>{shown(beta_adj)}</b> of {bt.benchmark_name}-equivalent exposure "
                    f"at a beta of {bt.beta:.2f}. If you hold other correlated positions, "
                    f"that is the number that stacks up, not the cash amount.</div>",
                    unsafe_allow_html=True)

            if not bt.rolling.empty:
                st.plotly_chart(
                    beta_chart(bt, theme), width="stretch",
                    config={"displayModeBar": False})

    # ---- fundamental fair value -----------------------------------------
    with vcol2:
        st.markdown("#### Fair value")
        try:
            fv = get_fair_value(symbol, interval, ctx.price)
        except Exception as exc:  # noqa: BLE001
            fv = None
            st.warning(f"Fair value unavailable: {exc}")

        if fv is None or not fv.applicable:
            if fv is not None:
                st.info(fv.reason)
        else:
            tone = (T["good"] if fv.upside_pct > 10 else
                    T["critical"] if fv.upside_pct < -10 else T["warning"])
            st.markdown(
                f"<div class='qd-label'>Blended estimate vs {money(fv.price)}</div>"
                f"<div style='font-size:26px;font-weight:700;color:{tone}'>"
                f"{money(fv.blended)}</div>"
                f"<div class='fib-chip' style='background:{tone}22;color:{tone}'>"
                f"{fv.upside_pct:+.1f}% · {fv.verdict}</div>",
                unsafe_allow_html=True)
            if fv.sector:
                st.markdown(f"<div class='qd-sub' style='margin-top:6px'>{fv.sector}"
                            f"{' · ' + fv.industry if fv.industry else ''}</div>",
                            unsafe_allow_html=True)

            st.markdown("<div class='qd-label' style='margin-top:12px'>How each model "
                        "gets there</div>", unsafe_allow_html=True)
            rows = [{"Model": e.method, "Value": money(e.value),
                     "vs price": f"{(e.value / fv.price - 1) * 100:+.1f}%",
                     "Basis": e.detail} for e in fv.estimates]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            for n in fv.notes:
                st.caption(f"· {n}")
            if fv.spread_warning:
                st.warning(fv.spread_warning, icon="⚠️")
            for sk in fv.skipped:
                st.caption(f"· {sk}")

            mets = {k: v for k, v in fv.metrics.items() if v is not None}
            if mets:
                st.markdown("<div class='qd-label' style='margin-top:10px'>Key multiples"
                            "</div>", unsafe_allow_html=True)
                mcols = st.columns(4)
                for i, (k, v) in enumerate(mets.items()):
                    mcols[i % 4].markdown(
                        f"<div class='qd-label'>{k}</div>"
                        f"<div class='qd-sub' style='font-size:15px'>{v:,.2f}</div>",
                        unsafe_allow_html=True)

    st.divider()
    st.caption(
        "**Beta and alpha are measured from price history** and update with the data. "
        "**Fair value is not backtested and cannot be.** It is built from *today's* "
        "fundamentals — yfinance exposes only the current snapshot, not what earnings or "
        "book value were five years ago — so there is no honest way to score it as a "
        "strategy, and it takes no part in the composite verdict. Treat it as one input "
        "among several. The backtestable equivalent is the **Long-Run Value Reversion** "
        "strategy, which anchors to price history instead of fundamentals. Every model "
        "here is crude and assumption-heavy; the spread between them is usually more "
        "informative than the blend."
    )

st.divider()
st.caption(
    "QuantDesk is a technical analysis tool, not investment advice. Backtested edge is measured on "
    "past data for this symbol and timeframe only and does not predict future returns. "
    "Every plan shown assumes the stop is honoured. Trading carries risk of loss."
)
st.caption(
    "Charts by [TradingView Lightweight Charts](https://www.tradingview.com/), used under the "
    "Apache-2.0 licence. Market data via yfinance from Yahoo Finance — not affiliated with or "
    "endorsed by Yahoo; provided for personal, research and educational use. "
    "QuantDesk itself is MIT licensed — see NOTICE."
)
