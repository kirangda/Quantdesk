"""Exchange-grade chart built on TradingView's Lightweight Charts.

Plotly is fine for a static figure but never feels like a trading terminal.
This renders the same data through the library TradingView open-sourced - the
one most exchange front-ends are built on - so panning, zooming and the
crosshair behave the way a trader expects: grab anywhere and drag, wheel to
zoom, drag an axis to scale it, and a magnet crosshair that reads out every
series at the hovered bar.

The library is vendored in `assets/lwc.js` and inlined into the page, so the
chart works with no CDN and no network at render time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .charting import THEMES, convert_frame

ASSET = Path(__file__).resolve().parent.parent / "assets" / "lwc.js"

# panes are stacked and share one time axis
PANE_HEIGHTS = {"price": 0.60, "rsi": 0.20, "macd": 0.20}


def _times(index: pd.DatetimeIndex, intraday: bool):
    """Lightweight Charts wants day strings for daily bars, epoch secs intraday."""
    if intraday:
        idx = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        return [int(t.timestamp()) for t in idx]
    return [t.strftime("%Y-%m-%d") for t in index]


def _line(times, values) -> list:
    out = []
    for t, v in zip(times, values):
        if v is None or not np.isfinite(v):
            continue
        out.append({"time": t, "value": round(float(v), 6)})
    return out


def _payload(f: pd.DataFrame, fib, plan, theme: str, rate: float, intraday: bool,
             bars: int, currency: str) -> dict:
    c = THEMES[theme]
    d = convert_frame(f, rate).tail(bars)
    t = _times(d.index, intraday)

    candles = [
        {"time": tt, "open": round(float(o), 6), "high": round(float(h), 6),
         "low": round(float(l), 6), "close": round(float(cl), 6)}
        for tt, o, h, l, cl in zip(t, d["open"], d["high"], d["low"], d["close"])
        if np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(cl)
    ]
    volume = [
        {"time": tt, "value": float(v),
         "color": (c["up"] + "55") if cl >= o else (c["down"] + "55")}
        for tt, v, o, cl in zip(t, d["volume"], d["open"], d["close"])
        if np.isfinite(v)
    ]
    hist = [
        {"time": tt, "value": round(float(v), 6),
         "color": (c["up"] + "aa") if v >= 0 else (c["down"] + "aa")}
        for tt, v in zip(t, d["macd_hist"]) if np.isfinite(v)
    ]

    levels = []
    if fib is not None:
        key = {"0.382", "0.5", "0.618", "0.786"}
        for name, price in fib.levels.items():
            levels.append({
                "group": "fib", "title": f"fib {name}", "price": float(price) * rate,
                "color": c["fib"], "width": 2 if name in ("0", "1") else 1,
                "style": 0 if name in ("0", "1") else 2,
                "bold": name in key,
            })
        for name, price in fib.extensions.items():
            levels.append({
                "group": "fib", "title": f"ext {name}", "price": float(price) * rate,
                "color": c["fib"], "width": 1, "style": 1, "bold": False,
            })
    if plan is not None:
        levels.append({"group": "plan", "title": "Entry", "price": float(plan.entry) * rate,
                       "color": c["text_secondary"], "width": 2, "style": 2, "bold": True})
        levels.append({"group": "plan", "title": "Stop", "price": float(plan.stop) * rate,
                       "color": c["critical"], "width": 2, "style": 2, "bold": True})
        for i, tgt in enumerate(plan.targets):
            levels.append({"group": "plan", "title": f"Target {i + 1}",
                           "price": float(tgt) * rate, "color": c["good"],
                           "width": 2, "style": 2, "bold": True})

    return {
        "candles": candles,
        "volume": volume,
        "ema20": _line(t, d["ema20"]), "ema50": _line(t, d["ema50"]),
        "ema200": _line(t, d["ema200"]),
        "bbUpper": _line(t, d["bb_upper"]), "bbLower": _line(t, d["bb_lower"]),
        "vwap": _line(t, d["vwap"]),
        "rsi": _line(t, d["rsi14"]),
        "macd": _line(t, d["macd"]), "signal": _line(t, d["macd_signal"]),
        "hist": hist,
        "levels": levels,
        "currency": currency,
        "precision": 2 if float(d["close"].iloc[-1]) >= 1 else 6,
    }


def render(f: pd.DataFrame, fib=None, plan=None, theme: str = "dark",
           rate: float = 1.0, currency: str = "", symbol: str = "",
           interval: str = "1d", bars: int = 300, height: int = 780,
           show_bb: bool = True, show_vwap: bool = False, show_fib: bool = True,
           show_plan: bool = True) -> str:
    """Return a self-contained HTML document for st.components.v1.html()."""
    c = THEMES[theme]
    intraday = interval not in ("1d", "1wk", "1mo")
    data = _payload(f, fib, plan, theme, rate, intraday, bars, currency)

    try:
        lib = ASSET.read_text(encoding="utf-8")
    except FileNotFoundError:  # pragma: no cover - asset ships with the repo
        return ("<p style='color:#d03b3b;font-family:sans-serif'>"
                "assets/lwc.js is missing - reinstall or switch the chart engine "
                "to Plotly in the sidebar.</p>")

    cfg = {
        "theme": theme, "colors": c, "symbol": symbol, "interval": interval,
        "intraday": intraday,
        "show": {"bb": show_bb, "vwap": show_vwap, "fib": show_fib, "plan": show_plan},
        "heights": PANE_HEIGHTS, "height": height,
    }

    return _TEMPLATE.replace("__LIB__", lib) \
                    .replace("__DATA__", json.dumps(data)) \
                    .replace("__CFG__", json.dumps(cfg))


_TEMPLATE = r"""
<!doctype html>
<meta charset="utf-8">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin:0; padding:0; background:transparent;
               font-family: Inter, -apple-system, "Segoe UI", sans-serif; }
  #wrap { position:relative; width:100%; }
  #bar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; padding:0 0 8px; }
  .tg { border:1px solid var(--grid); background:transparent; color:var(--muted);
        font-size:11px; font-weight:600; letter-spacing:.02em; padding:4px 10px;
        border-radius:6px; cursor:pointer; transition:all .12s; font-family:inherit; }
  .tg:hover { border-color:var(--txt2); color:var(--txt); }
  .tg.on { color:var(--txt); border-color:transparent; }
  .sp { flex:1; }
  .hint { color:var(--muted); font-size:11px; }
  .pane { position:relative; width:100%; }
  #legend { position:absolute; left:10px; top:8px; z-index:5; pointer-events:none;
            font-size:11.5px; line-height:1.65; color:var(--txt2);
            text-shadow:0 1px 3px var(--surface); }
  #legend b { color:var(--txt); font-size:13px; }
  .sw { display:inline-block; width:8px; height:8px; border-radius:2px;
        margin-right:5px; vertical-align:middle; }
  .val { font-variant-numeric: tabular-nums; }
  #fiblayer { position:absolute; inset:0; pointer-events:none; z-index:4; }
  .fiblab { position:absolute; left:8px; font-size:11px; font-weight:600;
            font-variant-numeric:tabular-nums; white-space:nowrap;
            padding:1px 5px; border-radius:3px; }
</style>
<div id="wrap">
  <div id="bar"></div>
  <div class="pane" id="price"><div id="legend"></div><div id="fiblayer"></div></div>
  <div class="pane" id="rsi"></div>
  <div class="pane" id="macd"></div>
</div>
<script>__LIB__</script>
<script>
const DATA = __DATA__, CFG = __CFG__;
const C = CFG.colors, LWC = LightweightCharts;
const R = document.documentElement.style;
R.setProperty('--grid', C.grid); R.setProperty('--muted', C.muted);
R.setProperty('--txt', C.text); R.setProperty('--txt2', C.text_secondary);
R.setProperty('--surface', C.surface);
document.documentElement.style.colorScheme = CFG.theme;

let H = CFG.height;
let hPrice = Math.round(H * CFG.heights.price);
let hRsi   = Math.round(H * CFG.heights.rsi);
let hMacd  = H - hPrice - hRsi;

const base = {
  layout: { background: { type: 'solid', color: C.surface }, textColor: C.text_secondary,
            fontSize: 11, fontFamily: 'Inter, -apple-system, Segoe UI, sans-serif' },
  grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
  rightPriceScale: { borderColor: C.grid, scaleMargins: { top: 0.08, bottom: 0.08 } },
  timeScale: { borderColor: C.grid, rightOffset: 6, barSpacing: 8,
               timeVisible: CFG.intraday, secondsVisible: false },
  crosshair: {
    mode: LWC.CrosshairMode.Normal,
    vertLine: { color: C.muted, width: 1, style: 3, labelBackgroundColor: C.accent },
    horzLine: { color: C.muted, width: 1, style: 3, labelBackgroundColor: C.accent },
  },
  // grab anywhere and drag; wheel zooms; axis drag scales that axis
  handleScroll: { mouseWheel: true, pressedMouseMove: true,
                  horzTouchDrag: true, vertTouchDrag: true },
  handleScale: { mouseWheel: true, pinch: true,
                 axisPressedMouseMove: { time: true, price: true },
                 axisDoubleClickReset: true },
};

const mk = (id, h, extra) => LWC.createChart(document.getElementById(id),
  Object.assign({}, base, { width: document.getElementById(id).clientWidth, height: h }, extra || {}));

const hidden = { timeScale: Object.assign({}, base.timeScale, { visible: false }) };
const cPrice = mk('price', hPrice, hidden);
const cRsi   = mk('rsi', hRsi, hidden);
const cMacd  = mk('macd', hMacd);   // only the bottom pane shows the date axis
const charts = [cPrice, cRsi, cMacd];

const pf = { type: 'price', precision: DATA.precision, minMove: Math.pow(10, -DATA.precision) };

// ---- price pane ----------------------------------------------------------
const candles = cPrice.addCandlestickSeries({
  upColor: C.up, downColor: C.down, borderUpColor: C.up, borderDownColor: C.down,
  wickUpColor: C.up, wickDownColor: C.down, priceFormat: pf,
});
candles.setData(DATA.candles);

const vol = cPrice.addHistogramSeries({
  priceScaleId: 'vol', priceFormat: { type: 'volume' }, priceLineVisible: false,
  lastValueVisible: false,
});
cPrice.priceScale('vol').applyOptions({ scaleMargins: { top: 0.86, bottom: 0 } });
vol.setData(DATA.volume);

const mkLine = (chart, data, color, width, title, style) => {
  const s = chart.addLineSeries({
    color: color, lineWidth: width || 2, title: title || '', priceFormat: pf,
    priceLineVisible: false, lastValueVisible: false, lineStyle: style || 0,
    crosshairMarkerVisible: false,
  });
  s.setData(data); return s;
};

const ema20  = mkLine(cPrice, DATA.ema20,  C.ema[0], 2);
const ema50  = mkLine(cPrice, DATA.ema50,  C.ema[1], 2);
const ema200 = mkLine(cPrice, DATA.ema200, C.ema[2], 2);
const bbU = mkLine(cPrice, DATA.bbUpper, C.muted, 1, '', 2);
const bbL = mkLine(cPrice, DATA.bbLower, C.muted, 1, '', 2);
const vwap = mkLine(cPrice, DATA.vwap, C.vwap, 2, '', 0);

// ---- indicator panes -----------------------------------------------------
const rsi = mkLine(cRsi, DATA.rsi, C.accent, 2);
[[70, C.critical], [50, C.muted], [30, C.good]].forEach(([v, col]) =>
  rsi.createPriceLine({ price: v, color: col, lineWidth: 1, lineStyle: 2,
                        axisLabelVisible: true, title: '' }));

const macdHist = cMacd.addHistogramSeries({ priceFormat: pf, priceLineVisible: false,
                                            lastValueVisible: false });
macdHist.setData(DATA.hist);
const macdL = mkLine(cMacd, DATA.macd,   C.macd, 2);
const sigL  = mkLine(cMacd, DATA.signal, C.signal, 2);

// ---- overlay state (declared before the draw helpers read it) ------------
const state = { ema: true, bb: CFG.show.bb, vwap: CFG.show.vwap,
                fib: CFG.show.fib, plan: CFG.show.plan, vol: true };

// ---- horizontal levels (fib + the order plan) ----------------------------
let priceLines = [];
function drawLevels() {
  priceLines.forEach(l => candles.removePriceLine(l));
  priceLines = [];
  DATA.levels.forEach(function (lv) {
    if (!state[lv.group]) return;
    priceLines.push(candles.createPriceLine({
      price: lv.price, color: lv.color, lineWidth: lv.width,
      lineStyle: lv.style, title: lv.title,
      // 13 stacked axis labels is unreadable - only the order levels get one
      axisLabelVisible: lv.group === 'plan',
    }));
  });
}

// ---- fib labels drawn in-chart (the price axis is reserved for the order) --
const legend = document.getElementById('legend');
const fibLayer = document.getElementById('fiblayer');
const fmtPx = v => v.toLocaleString(undefined,
  { minimumFractionDigits: DATA.precision, maximumFractionDigits: DATA.precision });

function placeFibLabels() {
  fibLayer.innerHTML = '';
  if (!state.fib) return;
  const placed = [];
  DATA.levels.filter(l => l.group === 'fib').forEach(function (lv) {
    const y = candles.priceToCoordinate(lv.price);
    if (y === null || y === undefined || y < 10 || y > hPrice - 10) return;
    if (placed.some(p => Math.abs(p - y) < 15)) return;   // never overlap
    placed.push(y);
    const el = document.createElement('div');
    el.className = 'fiblab';
    el.style.top = (y - 16) + 'px';
    // labels landing in the readout's band step right so the two never overlap
    el.style.left = (y < legend.offsetHeight + 10 ? legend.offsetWidth + 18 : 8) + 'px';
    el.style.color = C.fib;
    el.style.background = C.surface + 'cc';
    el.textContent = lv.title + '  ' + fmtPx(lv.price);
    fibLayer.appendChild(el);
  });
}

// there is no price-scale-change event, so re-place whenever the mapping moves
let lastProbe = null;
(function watch() {
  const probe = DATA.levels.length
    ? candles.priceToCoordinate(DATA.levels[0].price) : null;
  if (probe !== lastProbe) { lastProbe = probe; placeFibLabels(); }
  requestAnimationFrame(watch);
})();

// ---- crosshair readout ---------------------------------------------------
// the crosshair event only carries series from the chart that fired it, so the
// other panes are looked up by timestamp instead
const byTime = arr => { const m = new Map(); arr.forEach(p => m.set(String(p.time), p.value)); return m; };
const rsiMap = byTime(DATA.rsi), macdMap = byTime(DATA.macd), sigMap = byTime(DATA.signal);
const money = v => (v === undefined || v === null || isNaN(v)) ? '–'
  : DATA.currency + v.toLocaleString(undefined,
      { minimumFractionDigits: DATA.precision, maximumFractionDigits: DATA.precision });
const sw = col => '<span class="sw" style="background:' + col + '"></span>';

function paint(param) {
  const last = DATA.candles[DATA.candles.length - 1];
  let bar = last, t = last ? last.time : null;
  if (param && param.time && param.seriesData) {
    const b = param.seriesData.get(candles);
    if (b) { bar = b; t = param.time; }
  }
  if (!bar) { legend.innerHTML = ''; return; }
  const get = s => {
    if (param && param.seriesData) { const d = param.seriesData.get(s);
      if (d) return d.value !== undefined ? d.value : d.close; }
    return undefined;
  };
  const chg = bar.open ? ((bar.close - bar.open) / bar.open * 100) : 0;
  const col = bar.close >= bar.open ? C.up : C.down;
  const when = (typeof t === 'number')
    ? new Date(t * 1000).toISOString().slice(0, 16).replace('T', ' ')
    : t;
  let html = '<b>' + CFG.symbol + '</b> <span style="color:' + C.muted + '">'
    + CFG.interval + ' · ' + when + '</span><br>'
    + '<span class="val">O ' + money(bar.open) + '  H ' + money(bar.high)
    + '  L ' + money(bar.low) + '  C <span style="color:' + col + '">' + money(bar.close)
    + '</span>  <span style="color:' + col + '">' + (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%</span></span><br>';
  const parts = [];
  if (state.ema) {
    parts.push(sw(C.ema[0]) + 'EMA20 ' + money(get(ema20)));
    parts.push(sw(C.ema[1]) + 'EMA50 ' + money(get(ema50)));
    parts.push(sw(C.ema[2]) + 'EMA200 ' + money(get(ema200)));
  }
  if (state.vwap) parts.push(sw(C.vwap) + 'VWAP ' + money(get(vwap)));
  const key = String(t);
  const rv = rsiMap.get(key), mv = macdMap.get(key), sv = sigMap.get(key);
  if (rv !== undefined) parts.push(sw(C.accent) + 'RSI ' + rv.toFixed(1));
  if (mv !== undefined && sv !== undefined)
    parts.push(sw(C.macd) + 'MACD ' + mv.toFixed(3) + ' / ' + sv.toFixed(3));
  html += '<span class="val">' + parts.join('&nbsp;&nbsp;') + '</span>';
  legend.innerHTML = html;
}

// ---- sync time scale + crosshair across the three panes ------------------
let syncing = false;
charts.forEach(function (src) {
  src.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
    if (syncing || !range) return;
    syncing = true;
    charts.forEach(function (o) { if (o !== src) o.timeScale().setVisibleLogicalRange(range); });
    syncing = false;
  });
  src.subscribeCrosshairMove(function (param) {
    if (src === cPrice) paint(param);
    if (syncing) return;
    syncing = true;
    charts.forEach(function (o) {
      if (o === src) return;
      if (!param || !param.time) { o.clearCrosshairPosition(); return; }
      const target = o === cPrice ? candles : (o === cRsi ? rsi : macdL);
      const d = param.seriesData && param.seriesData.get(target);
      const px = d ? (d.value !== undefined ? d.value : d.close) : 0;
      o.setCrosshairPosition(px, param.time, target);
    });
    syncing = false;
  });
});

// ---- overlay toggles -----------------------------------------------------
function apply() {
  [ema20, ema50, ema200].forEach(s => s.applyOptions({ visible: state.ema }));
  [bbU, bbL].forEach(s => s.applyOptions({ visible: state.bb }));
  vwap.applyOptions({ visible: state.vwap });
  vol.applyOptions({ visible: state.vol });
  drawLevels();
  placeFibLabels();
  paint(null);
}

const BUTTONS = [
  ['ema', 'EMA 20/50/200', C.ema[0]], ['bb', 'Bollinger', C.muted],
  ['vwap', 'VWAP', C.vwap], ['vol', 'Volume', C.up],
  ['fib', 'Fibonacci', C.fib], ['plan', 'Trade levels', C.good],
];
const bar = document.getElementById('bar');  // toolbar host
BUTTONS.forEach(function (b) {
  const el = document.createElement('button');
  el.className = 'tg' + (state[b[0]] ? ' on' : '');
  el.innerHTML = sw(b[2]) + b[1];
  if (state[b[0]]) el.style.background = b[2] + '22';
  el.onclick = function () {
    state[b[0]] = !state[b[0]];
    el.classList.toggle('on', state[b[0]]);
    el.style.background = state[b[0]] ? b[2] + '22' : 'transparent';
    apply();
  };
  bar.appendChild(el);
});
const spacer = document.createElement('div'); spacer.className = 'sp'; bar.appendChild(spacer);
const hint = document.createElement('span'); hint.className = 'hint';
hint.textContent = 'drag to pan · wheel to zoom · drag an axis to scale · double-click axis to reset';
bar.appendChild(hint);

const reset = document.createElement('button');
reset.className = 'tg'; reset.textContent = 'Reset';
reset.onclick = () => charts.forEach(c => c.timeScale().fitContent());
bar.appendChild(reset);

const maxBtn = document.createElement('button');
maxBtn.className = 'tg';
maxBtn.innerHTML = sw(C.accent) + 'Maximise';
maxBtn.title = 'Fill the window (F, or Esc to exit)';
maxBtn.onclick = () => setMax(!maximized);
bar.appendChild(maxBtn);

apply();
// open showing the most recent stretch, not every loaded bar
const N = DATA.candles.length, WIN = Math.min(N, 160);
cPrice.timeScale().setVisibleLogicalRange({ from: N - WIN, to: N + 4 });

// ---- maximise ------------------------------------------------------------
const wrap = document.getElementById('wrap');
let maximized = false, savedCss = null;

function layout() {
  H = maximized
    ? Math.max(320, window.innerHeight - bar.offsetHeight - 10)
    : CFG.height;
  hPrice = Math.round(H * CFG.heights.price);
  hRsi   = Math.round(H * CFG.heights.rsi);
  hMacd  = H - hPrice - hRsi;
  const w = wrap.clientWidth;
  cPrice.applyOptions({ width: w, height: hPrice });
  cRsi.applyOptions({ width: w, height: hRsi });
  cMacd.applyOptions({ width: w, height: hMacd });
  placeFibLabels();
}

// The component is a same-origin srcdoc iframe, so it can promote its own
// frame to a full-viewport overlay - that beats the Fullscreen API here, which
// the embedding iframe is not always permitted to use.
function overlayOn() {
  const fe = window.frameElement;
  if (!fe) return false;
  savedCss = fe.getAttribute('style') || '';
  fe.setAttribute('style',
    'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:2147483647;' +
    'border:0;margin:0;background:' + C.surface);
  try { fe.ownerDocument.body.style.overflow = 'hidden'; } catch (e) {}
  return true;
}

function overlayOff() {
  const fe = window.frameElement;
  if (!fe) return;
  fe.setAttribute('style', savedCss);
  try { fe.ownerDocument.body.style.overflow = ''; } catch (e) {}
}

function setMax(on) {
  maximized = on;
  if (on) { if (!overlayOn()) { maximized = false; return; } } else { overlayOff(); }
  maxBtn.innerHTML = sw(C.accent) + (on ? 'Exit (Esc)' : 'Maximise');
  maxBtn.classList.toggle('on', on);
  maxBtn.style.background = on ? C.accent + '22' : 'transparent';
  requestAnimationFrame(layout);
}

document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && maximized) setMax(false);
  else if ((e.key === 'f' || e.key === 'F') && !e.ctrlKey && !e.metaKey) setMax(!maximized);
});
window.addEventListener('resize', function () { if (maximized) layout(); });

new ResizeObserver(function () { layout(); }).observe(wrap);
layout();
</script>
"""
