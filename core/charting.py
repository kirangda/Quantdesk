"""Plotly chart builders.

Layout rules kept deliberately strict:
  * one y-axis per panel - price, volume, RSI and MACD each get their own row,
    never a second scale stacked on the price plot
  * candles use the reserved status colours (up = good, down = critical); the
    EMA family uses one blue ramp light->dark because it is one measure at
    four lookbacks, not four unrelated entities
  * grid and axes are solid hairlines one shade off the surface; dashes are
    reserved for real thresholds (fib levels, entry, stop, target)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .fibonacci import GOLDEN_POCKET, FibLeg

# --------------------------------------------------------------------------
# Palette (validated reference instance, stepped per surface)
# --------------------------------------------------------------------------
THEMES = {
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#83827b",
        "grid": "#2e2e2c",
        "up": "#0ca30c",
        "down": "#d03b3b",
        # EMA 20 / 50 / 200 as three DISTINCT categorical hues - a single-hue
        # ramp was too easy to confuse at a glance. blue/magenta/yellow passes
        # every all-pairs check on this surface (worst CVD dE 13.2, normal 19.3).
        "ema": ["#3987e5", "#d55181", "#c98500"],
        "accent": "#3987e5",
        "band": "rgba(147,146,138,0.10)",
        "vwap": "#ffffff",
        "macd": "#3987e5",
        "signal": "#d95926",
        "fib": "#a89bf0",
        "pocket": "rgba(144,133,233,0.20)",
        "good": "#0ca30c",
        "critical": "#d03b3b",
        "warning": "#fab219",
    },
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#75746e",
        "grid": "#eceae5",
        "up": "#0ca30c",
        "down": "#d03b3b",
        "ema": ["#2a78d6", "#e87ba4", "#eda100"],
        "accent": "#2a78d6",
        "band": "rgba(117,116,110,0.10)",
        "vwap": "#0b0b0b",
        "macd": "#2a78d6",
        "signal": "#eb6834",
        "fib": "#4a3aa7",
        "pocket": "rgba(74,58,167,0.16)",
        "good": "#0ca30c",
        "critical": "#d03b3b",
        "warning": "#fab219",
    },
}


# Columns denominated in the instrument's price. Scaling these by an FX rate
# is exact: every level-based indicator moves with price, while ratio-based
# ones (RSI, ADX, %B, ATR%, rel. volume) are scale-invariant and must NOT move.
PRICE_COLUMNS = (
    "open", "high", "low", "close",
    "ema9", "ema20", "ema50", "ema200", "sma5", "sma50", "sma200",
    "atr", "bb_upper", "bb_lower", "bb_mid", "kc_upper", "kc_lower",
    "dc_upper", "dc_lower", "dc10_upper", "dc10_lower",
    "supertrend", "vwap", "swing_low_20", "swing_high_20",
    "macd", "macd_signal", "macd_hist",
)


def convert_frame(f: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Restate every price-level column in another currency."""
    if rate == 1.0:
        return f
    out = f.copy()
    for col in PRICE_COLUMNS:
        if col in out.columns:
            out[col] = out[col] * rate
    return out


def _x(v):
    """Shapes and annotations are serialised by strict JSON encoders that do
    not know pandas Timestamps - hand them a plain datetime."""
    return v.to_pydatetime() if isinstance(v, pd.Timestamp) else v


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i: i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------
def price_chart(
    f: pd.DataFrame,
    fib: FibLeg | None = None,
    plan=None,
    theme: str = "dark",
    show_bollinger: bool = True,
    show_vwap: bool = False,
    show_fib: bool = True,
    title: str = "",
    bars: int = 240,
    weekend_breaks: bool = False,
    rate: float = 1.0,
    currency: str = "",
) -> go.Figure:
    """Four-panel chart: price + overlays, volume, RSI, MACD.

    `rate` restates every price into the display currency (1.0 = native).
    """
    c = THEMES[theme]
    d = convert_frame(f, rate).tail(bars)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.54, 0.11, 0.175, 0.175],
        subplot_titles=("", "Volume", "RSI (14)", "MACD (12, 26, 9)"),
    )

    # ---- row 1: candles -------------------------------------------------
    fig.add_trace(
        go.Candlestick(
            x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
            name="Price",
            increasing=dict(line=dict(color=c["up"], width=1), fillcolor=c["up"]),
            decreasing=dict(line=dict(color=c["down"], width=1), fillcolor=c["down"]),
            showlegend=False,
        ),
        row=1, col=1,
    )

    if show_bollinger and "bb_upper" in d:
        fig.add_trace(
            go.Scatter(x=d.index, y=d["bb_upper"], line=dict(width=0), hoverinfo="skip",
                       showlegend=False, name="BB upper"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=d.index, y=d["bb_lower"], line=dict(width=0), fill="tonexty",
                       fillcolor=c["band"], hoverinfo="skip", showlegend=True,
                       name="Bollinger (20, 2)"),
            row=1, col=1,
        )

    # the EMA family is one measure at three lookbacks -> one hue, light to dark.
    # each line is also labelled at its right-hand end so identity never rests
    # on hue alone.
    overlays = [
        ("EMA 20", "ema20", c["ema"][0]),
        ("EMA 50", "ema50", c["ema"][1]),
        ("EMA 200", "ema200", c["ema"][2]),
    ]
    if show_vwap and "vwap" in d:
        overlays.append(("VWAP", "vwap", c["vwap"]))

    for label, col, colour in overlays:
        if col not in d:
            continue
        fig.add_trace(
            go.Scatter(x=d.index, y=d[col], name=label, mode="lines",
                       line=dict(color=colour, width=2),
                       hovertemplate=label + ": %{y:,.2f}<extra></extra>"),
            row=1, col=1,
        )
        endpoint = d[col].dropna()
        if not endpoint.empty:
            fig.add_annotation(
                x=_x(endpoint.index[-1]), y=float(endpoint.iloc[-1]), text=f" {label}",
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=10, color=colour), row=1, col=1,
            )

    # ---- fibonacci ------------------------------------------------------
    # Drawn as real traces (not shapes) so the whole overlay collapses behind a
    # single "Fibonacci" legend entry - one click, no server round-trip. Labels
    # ride along as trace text so they hide with their lines.
    fib_ymin = fib_ymax = None
    if show_fib and fib is not None:
        x_right = _x(d.index[-1])
        y_lo, y_hi = float(d["low"].min()), float(d["high"].max())
        span = y_hi - y_lo

        leg_start = fib.start_idx if fib.start_idx >= d.index[0] else d.index[0]
        leg_end = fib.end_idx if fib.end_idx >= d.index[0] else d.index[0]
        x_leg = _x(leg_start)
        shown_fib = False

        def _fib_trace(y0v, y1v, text, colour, width, dash, size, hover):
            nonlocal shown_fib
            fig.add_trace(
                go.Scatter(
                    x=[x_leg, x_right], y=[y0v, y1v],
                    mode="lines+text", text=[None, text], textposition="top left",
                    textfont=dict(size=size, color=c["fib"]),
                    line=dict(color=colour, width=width, dash=dash),
                    name="Fibonacci", legendgroup="fib", showlegend=not shown_fib,
                    hovertemplate=hover + ": %{y:,.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
            shown_fib = True

        # the golden pocket band first so it sits behind the lines
        gp_lo = min(fib.levels.get("0.5", np.nan), fib.levels.get("0.618", np.nan)) * rate
        gp_hi = max(fib.levels.get("0.5", np.nan), fib.levels.get("0.618", np.nan)) * rate
        if np.isfinite(gp_lo) and np.isfinite(gp_hi):
            fig.add_trace(
                go.Scatter(x=[x_leg, x_right], y=[gp_lo, gp_lo], mode="lines",
                           line=dict(width=0), hoverinfo="skip", showlegend=False,
                           name="pocket", legendgroup="fib"),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=[x_leg, x_right], y=[gp_hi, gp_hi], mode="lines",
                           line=dict(width=0), fill="tonexty", fillcolor=c["pocket"],
                           hoverinfo="skip", showlegend=False,
                           name="pocket", legendgroup="fib"),
                row=1, col=1)

        # the measured swing itself
        if fib.end_idx >= d.index[0]:
            fig.add_trace(
                go.Scatter(
                    x=[_x(leg_start), _x(leg_end)],
                    y=[fib.start_price * rate, fib.end_price * rate],
                    mode="lines+markers", name="Fibonacci", legendgroup="fib",
                    showlegend=not shown_fib,
                    line=dict(color=c["fib"], width=3),
                    marker=dict(size=11, color=c["fib"],
                                line=dict(color=c["surface"], width=2)),
                    hovertemplate="impulse leg %{y:,.2f}<extra></extra>",
                ),
                row=1, col=1)
            shown_fib = True

        # 0.236 is deliberately absent - least-used level, and on a tight leg it
        # collides with the moving-average labels. It stays in the table.
        priority = ["0.618", "0.5", "1", "0", "0.786", "0.382"]
        key_levels = {"0.382", "0.5", "0.618", "0.786"}
        min_gap = span * 0.055
        placed: list[float] = []
        pocket_prices: list[float] = []
        drawn: list[float] = []

        for name in priority:
            native_price = fib.levels.get(name)
            if native_price is None or not np.isfinite(native_price):
                continue
            price = native_price * rate
            if not (y_lo - 0.05 * span) <= price <= (y_hi + 0.05 * span):
                continue
            crowded = any(abs(price - y) < min_gap for y in placed)
            if crowded and name not in ("0.5", "0.618"):
                continue
            if not crowded:
                placed.append(price)
            drawn.append(price)

            anchor_level = name in ("0", "1")
            colour = (c["fib"] if (anchor_level or name in key_levels)
                      else _rgba(c["fib"], 0.75))
            if name in ("0.5", "0.618"):
                pocket_prices.append(price)
                _fib_trace(price, price, None, colour, 1.8, "dash", 11, name)
                continue
            _fib_trace(price, price, f"<b>{name}</b>  {price:,.2f}",
                       colour, 2.5 if anchor_level else 1.8,
                       "solid" if anchor_level else "dash", 12, name)

        # the two pocket edges share one label so neither is ever dropped
        if pocket_prices:
            lo_p, hi_p = min(pocket_prices), max(pocket_prices)
            fig.add_trace(
                go.Scatter(
                    x=[x_right], y=[hi_p], mode="text",
                    text=[f"<b>0.5-0.618 pocket</b>  {lo_p:,.2f}-{hi_p:,.2f}"],
                    textposition="top left", textfont=dict(size=12, color=c["fib"]),
                    name="Fibonacci", legendgroup="fib", showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1)

        # extensions - the profit projections - only when actually on screen
        for name, native_price in fib.extensions.items():
            price = native_price * rate
            if not np.isfinite(price) or not y_lo <= price <= y_hi:
                continue
            if any(abs(price - y) < min_gap * 1.4 for y in placed):
                continue
            placed.append(price)
            drawn.append(price)
            _fib_trace(price, price, f"ext {name}  {price:,.2f}",
                       _rgba(c["fib"], 0.8), 1.5, "dot", 11, f"ext {name}")

        if drawn:
            fib_ymin, fib_ymax = min(drawn), max(drawn)

    # ---- the actual trade plan -------------------------------------------
    # Also traces, under their own legend entry, so the order overlay toggles
    # independently of the Fibonacci one.
    if plan is not None:
        x0, x1 = _x(d.index[0]), _x(d.index[-1])
        lines = [("Entry", plan.entry * rate, c["text_secondary"]),
                 ("Stop", plan.stop * rate, c["critical"])]
        lines += [(f"Target {i + 1}", t * rate, c["good"]) for i, t in enumerate(plan.targets)]
        shown_plan = False
        for label, price, colour in lines:
            if not np.isfinite(price):
                continue
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1], y=[price, price], mode="lines+text",
                    text=[f"  {label} {price:,.2f}", None], textposition="top right",
                    textfont=dict(size=10, color=colour),
                    line=dict(color=colour, width=2, dash="dot"),
                    name="Trade levels", legendgroup="plan", showlegend=not shown_plan,
                    hovertemplate=f"{label}: %{{y:,.2f}}<extra></extra>",
                ),
                row=1, col=1)
            shown_plan = True

    # ---- row 2: volume ---------------------------------------------------
    vol_colors = np.where(
        d["close"].to_numpy() >= d["open"].to_numpy(), _rgba(c["up"], 0.45), _rgba(c["down"], 0.45)
    )
    fig.add_trace(
        go.Bar(x=d.index, y=d["volume"], marker=dict(color=vol_colors, line=dict(width=0)),
               name="Volume", showlegend=False,
               hovertemplate="Vol: %{y:,.0f}<extra></extra>"),
        row=2, col=1,
    )

    # ---- row 3: RSI ------------------------------------------------------
    fig.add_trace(
        go.Scatter(x=d.index, y=d["rsi14"], name="RSI 14", mode="lines",
                   line=dict(color=c["accent"], width=2), showlegend=False,
                   hovertemplate="RSI: %{y:.1f}<extra></extra>"),
        row=3, col=1,
    )
    for lvl, colour in ((70, c["critical"]), (50, c["muted"]), (30, c["good"])):
        fig.add_hline(y=lvl, line=dict(color=_rgba(colour, 0.75), width=1,
                                       dash="dash" if lvl != 50 else "solid"),
                      row=3, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=3, col=1)

    # ---- row 4: MACD -----------------------------------------------------
    hist_colors = np.where(d["macd_hist"] >= 0, _rgba(c["up"], 0.45), _rgba(c["down"], 0.45))
    fig.add_trace(
        go.Bar(x=d.index, y=d["macd_hist"], name="Histogram",
               marker=dict(color=hist_colors, line=dict(width=0)), showlegend=False,
               hovertemplate="Hist: %{y:.4f}<extra></extra>"),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["macd"], name="MACD", mode="lines",
                   line=dict(color=c["macd"], width=2),
                   hovertemplate="MACD: %{y:.4f}<extra></extra>"),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["macd_signal"], name="Signal", mode="lines",
                   line=dict(color=c["signal"], width=2),
                   hovertemplate="Signal: %{y:.4f}<extra></extra>"),
        row=4, col=1,
    )

    # ---- chrome ----------------------------------------------------------
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        font=dict(color=c["text_secondary"], size=11,
                  family="Inter, -apple-system, Segoe UI, sans-serif"),
        title=dict(text=title, font=dict(color=c["text"], size=16), x=0.01, xanchor="left"),
        height=860, margin=dict(l=8, r=66, t=54, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1,
                    traceorder="normal",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        bargap=0.15,
        dragmode="pan",
    )
    # Every axis stays draggable: dragging directly on an axis rescales just
    # that axis (compressing the price scale the way a trading terminal does),
    # the wheel zooms inside the plot, and double-click restores autoscale.
    fig.update_xaxes(
        showgrid=True, gridcolor=c["grid"], gridwidth=1, griddash="solid",
        zeroline=False, showspikes=True, spikemode="across",
        spikecolor=c["muted"], spikethickness=1, spikedash="solid",
        linecolor=c["grid"], fixedrange=False,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=c["grid"], gridwidth=1, griddash="solid",
        zeroline=False, linecolor=c["grid"], side="right",
        fixedrange=False, showspikes=True, spikemode="across",
        spikecolor=c["muted"], spikethickness=1, spikedash="solid",
    )
    # Overlay lines are traces now, so they feed autorange. Bound the price
    # panel to the candles, letting overlays stretch it only so far - a far-off
    # target must not squash the price action into a ribbon.
    lo_c, hi_c = float(d["low"].min()), float(d["high"].max())
    candle_span = max(hi_c - lo_c, 1e-9)
    slack = candle_span * 0.35
    extras = []
    if fib_ymin is not None:
        extras += [fib_ymin, fib_ymax]
    if plan is not None:
        extras += [plan.entry * rate, plan.stop * rate]
        extras += [t * rate for t in plan.targets]
    extras = [v for v in extras if np.isfinite(v)]
    lo = min([lo_c] + [max(v, lo_c - slack) for v in extras])
    hi = max([hi_c] + [min(v, hi_c + slack) for v in extras])
    pad = (hi - lo) * 0.03
    fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    if currency:
        fig.update_yaxes(tickprefix=currency, row=1, col=1)
    if weekend_breaks:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    for ann in fig.layout.annotations[:4]:
        if ann.text in ("Volume", "RSI (14)", "MACD (12, 26, 9)"):
            ann.font = dict(size=11, color=c["muted"])
            ann.x, ann.xanchor = 0.0, "left"
    return fig


def equity_chart(metrics, theme: str = "dark", title: str = "") -> go.Figure:
    """Single-series equity curve for one strategy's backtest."""
    c = THEMES[theme]
    eq = metrics.equity
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=(eq - 1.0) * 100.0, mode="lines", name="Equity",
            line=dict(color=c["accent"], width=2), fill="tozeroy",
            fillcolor=_rgba(c["accent"], 0.12),
            hovertemplate="%{x|%d %b %Y}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=c["muted"], width=1))
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        font=dict(color=c["text_secondary"], size=11),
        title=dict(text=title, font=dict(color=c["text"], size=13), x=0.01, xanchor="left"),
        height=260, margin=dict(l=8, r=8, t=40, b=8), showlegend=False, hovermode="x",
    )
    fig.update_xaxes(showgrid=False, linecolor=c["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=c["grid"], zeroline=False,
                     ticksuffix="%", side="right", linecolor=c["grid"])
    return fig


def score_gauge(score: float, theme: str = "dark") -> go.Figure:
    """Composite -100..100 verdict as a single hero indicator."""
    c = THEMES[theme]
    if score >= 18:
        colour = c["good"]
    elif score <= -18:
        colour = c["critical"]
    else:
        colour = c["warning"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(float(score)),
            number=dict(font=dict(size=34, color=c["text"]), valueformat="+.0f"),
            gauge=dict(
                axis=dict(range=[-100, 100], tickwidth=1, tickcolor=c["grid"],
                          tickfont=dict(size=9, color=c["muted"])),
                bar=dict(color=colour, thickness=0.7),
                bgcolor=c["surface"], borderwidth=0,
                steps=[
                    dict(range=[-100, -18], color=_rgba(c["critical"], 0.12)),
                    dict(range=[-18, 18], color=_rgba(c["muted"], 0.12)),
                    dict(range=[18, 100], color=_rgba(c["good"], 0.12)),
                ],
                threshold=dict(line=dict(color=c["text"], width=2), thickness=0.8, value=score),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor=c["surface"], height=190, margin=dict(l=16, r=16, t=8, b=8),
        font=dict(color=c["text_secondary"]),
    )
    return fig


def beta_chart(bt, theme: str = "dark") -> go.Figure:
    """Rolling beta - shows whether the market sensitivity is stable or drifting."""
    c = THEMES[theme]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt.rolling.index, y=bt.rolling.to_numpy(), mode="lines", name="Rolling beta",
            line=dict(color=c["accent"], width=2),
            hovertemplate="%{x|%d %b %Y}<br>beta %{y:.2f}<extra></extra>",
        )
    )
    # 1.0 is the reference: moving exactly with the benchmark
    fig.add_hline(y=1.0, line=dict(color=c["muted"], width=1, dash="dash"),
                  annotation_text="market (1.0)", annotation_position="top left",
                  annotation_font=dict(size=10, color=c["muted"]))
    fig.add_hline(y=float(bt.beta), line=dict(color=c["good"], width=1.5),
                  annotation_text=f"full-period {bt.beta:.2f}",
                  annotation_position="bottom left",
                  annotation_font=dict(size=10, color=c["good"]))
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        font=dict(color=c["text_secondary"], size=11),
        title=dict(text=f"Rolling beta vs {bt.benchmark_name}",
                   font=dict(color=c["text"], size=13), x=0.01, xanchor="left"),
        height=240, margin=dict(l=8, r=8, t=40, b=8), showlegend=False, hovermode="x",
    )
    fig.update_xaxes(showgrid=False, linecolor=c["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=c["grid"], zeroline=False,
                     side="right", linecolor=c["grid"])
    return fig
