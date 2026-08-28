"""Terminal report - same engine as the app, no browser needed.

    python cli.py NVDA
    python cli.py BTC --style day --interval 15m
    python cli.py AAPL --account 25000 --risk 1.5 --chart out.html
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from core.charting import price_chart
from core.data import currency_symbol, fetch, fx_rate, native_currency, normalise_symbol
from core.engine import analyse, size_position

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW, CYAN = "\033[32m", "\033[31m", "\033[33m", "\033[36m"


def fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.2f}"
    if a >= 1:
        return f"{x:,.2f}"
    return f"{x:,.8f}".rstrip("0")


def main() -> int:
    ap = argparse.ArgumentParser(description="QuantDesk terminal report")
    ap.add_argument("symbol")
    ap.add_argument("--style", choices=["swing", "day"], default="swing")
    ap.add_argument("--interval", default=None, help="1d, 1h, 15m, 5m ...")
    ap.add_argument("--account", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=1.0, help="percent of account per trade")
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--currency", default=None,
                    help="display currency, e.g. EUR (default: the listing currency)")
    ap.add_argument("--chart", default=None, help="write an interactive HTML chart here")
    args = ap.parse_args()

    # Windows consoles default to cp1252 and choke on the box/bullet glyphs
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - older/redirected streams
        pass

    interval = args.interval or ("1d" if args.style == "swing" else "15m")

    inst = normalise_symbol(args.symbol)
    try:
        df = fetch(inst, interval)
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}{exc}{RESET}")
        return 1

    a = analyse(df, inst, interval, args.style,
                allow_short=not args.no_short, cost_bps=args.cost_bps)
    ctx, rec = a.context, a.recommendation

    native = native_currency(inst)
    display = (args.currency or native).upper()
    try:
        rate = fx_rate(native, display)
    except Exception as exc:  # noqa: BLE001 - stay in the native currency
        print(f"{YELLOW}{exc}{RESET}")
        rate, display = 1.0, native
    sym = currency_symbol(display)

    def money(x: float) -> str:
        return f"{sym}{fmt(x * rate)}"

    def shown(x: float) -> str:
        return f"{sym}{fmt(x)}"

    w = 96
    print()
    print(f"{BOLD}{inst.symbol}{RESET}  {money(ctx.price)}  ({ctx.change_pct:+.2f}%)   "
          f"{DIM}{interval} bars · {args.style} · {len(df):,} bars"
          + (f" · {display} @ {rate:,.4f}" if rate != 1.0 else "") + RESET)
    print(f"{DIM}{ctx.trend} · volatility {ctx.volatility} · RSI {ctx.rsi:.1f} · "
          f"ADX {ctx.adx:.1f} · ATR {ctx.atr_pct:.2f}%{RESET}")
    print("─" * w)

    colour = GREEN if "BUY" in rec.verdict else (RED if "SELL" in rec.verdict else YELLOW)
    print(f"{BOLD}{colour}{rec.verdict}{RESET}   score {rec.score:+.0f}/100   {DIM}{rec.agreement}{RESET}")
    for line in rec.reasoning:
        print(f"  {DIM}·{RESET} {line}")
    if rec.caution:
        print()
        print(f"  {YELLOW}! {rec.caution}{RESET}")

    if rec.best is not None:
        b = rec.best
        sz = size_position(args.account, args.risk, rec.entry * rate, rec.stop * rate)
        side = "BUY" if b.direction == "LONG" else "SELL/SHORT"
        print()
        print(f"{BOLD}ORDER{RESET} via {CYAN}{b.name}{RESET}  {DIM}({b.state}){RESET}")
        print(f"  {side:11s} at {BOLD}{money(rec.entry)}{RESET}")
        print(f"  {'Stop':11s} at {RED}{money(rec.stop)}{RESET}  "
              f"{DIM}({abs(rec.entry - rec.stop) / rec.entry * 100:.2f}% risk){RESET}")
        for i, (t, r) in enumerate(zip(rec.targets, b.targets_r)):
            print(f"  {'Target ' + str(i + 1):11s} at {GREEN}{money(t)}{RESET}  {DIM}({r:.1f}R){RESET}")
        print(f"  {'Size':11s}    {sz['units']:,.4f} units  {DIM}= {shown(sz['notional'])} notional, "
              f"risking {shown(sz['risk_amount'])}{RESET}")

    print()
    print("─" * w)
    print(f"{BOLD}{'STRATEGY':<32}{'SIGNAL':<11}{'STATE':<15}{'CONV':>5}{'EDGE':>6}"
          f"{'N':>5}{'WIN%':>6}{'EXP':>7}{'PF':>6}{RESET}")
    for p in a.plans:
        m = p.metrics
        sig_col = GREEN if p.direction == "LONG" else RED
        if p.state == "NO SETUP":
            sig_col = DIM
        exp = f"{m.expectancy_r:+.2f}" if np.isfinite(m.expectancy_r) else "  -  "
        pf = f"{m.profit_factor:.2f}" if np.isfinite(m.profit_factor) else "  -  "
        wr = f"{m.win_rate:.0f}" if np.isfinite(m.win_rate) else "-"
        print(f"{p.name:<32}{sig_col}{p.direction:<11}{RESET}{p.state:<15}"
              f"{p.conviction:>5.0f}{m.edge_score:>6.0f}{m.trades:>5}{wr:>6}{exp:>7}{pf:>6}")
        print(f"{DIM}    entry {money(p.entry):<15} stop {money(p.stop):<15} "
              f"targets {' / '.join(money(t) for t in p.targets)}{RESET}")

    if a.fib is not None:
        leg = a.fib
        print()
        print(f"{BOLD}FIBONACCI{RESET} {DIM}{leg.direction}-leg {money(leg.start_price)} -> "
              f"{money(leg.end_price)}, retraced {leg.retrace_pct(ctx.price) * 100:.1f}%{RESET}")
        cells = [f"{k}: {money(v)}" for k, v in leg.levels.items()]
        print("  " + "   ".join(cells))

    if args.chart:
        fig = price_chart(a.features, fib=a.fib, plan=rec.best, theme="dark",
                          title=f"{inst.symbol} · {interval}",
                          show_vwap=(args.style == "day"), rate=rate, currency=sym)
        fig.write_html(args.chart, include_plotlyjs="cdn")
        print(f"\n{DIM}chart written to {args.chart}{RESET}")

    print()
    print(f"{DIM}Technical analysis only - not investment advice. Backtested edge describes past "
          f"behaviour of this symbol on this timeframe.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
