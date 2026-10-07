"""
V3.14 Dual-Side Research Backtest
---------------------------------
Architecture-safe research layer. Does NOT modify btc_analyzer.py.

Purpose:
- Keep the existing V3.14 Long-only engine unchanged.
- Test an experimental SHORT side using the short-ready logic already present
  in trade_setup_engine().
- Compare Long-only vs Long+Short on the same historical 5m/15m/1h data.

Important:
- This is a research experiment, NOT a V3.14 architecture change.
- The existing short path is less complete than the long path:
  it currently uses downside-break + 15m/1h bearish confirmation.
  It does NOT add a new short retest/pullback architecture.
- Risk = 1% of current equity per trade.
- Entry = next 5m candle OPEN after signal.
- No overlapping positions.
- TP1 = 50%, TP2 = remaining 50%.
- Same-candle SL/TP ambiguity: SL first.
- Fees are charged on entry and each exit side when configured.
"""

import argparse
from dataclasses import dataclass

import pandas as pd

import btc_analyzer as v314
from backtest_v314 import (
    BASE_RESOLUTION,
    HIGHER_RESOLUTIONS,
    SL_VALUES,
    fetch_history,
    _get_mtf_results,
    signal_at as long_signal_at,
)


@dataclass
class Trade:
    side: str
    setup: str
    entry_time: int
    exit_time: int
    entry: float
    sl: float
    tp1: float
    tp2: float
    qty: float
    pnl: float
    r_multiple: float
    exit_reason: str
    risk_amount: float


def prepare_history(days):
    print(f"Fetching {days}-day history...")
    data = {BASE_RESOLUTION: fetch_history(BASE_RESOLUTION, days)}
    for tf in HIGHER_RESOLUTIONS:
        data[tf] = fetch_history(tf, days)
    for tf in data:
        data[tf] = v314.calculate_indicators(data[tf])
    return data


def short_signal_at(history, ts):
    """Use the analyzer's existing SHORT READY diagnostic path."""
    results = _get_mtf_results(history, ts)
    if results is None:
        return None

    try:
        plan = v314.trade_setup_engine(results)
        if plan.get("plan") != "SHORT READY":
            return None

        tf5 = results["5m"]
        return {
            "side": "SHORT",
            "setup": "Breakout",
            "signal_price": float(tf5["price"]),
            "short_tp1": float(tf5["short_tp1"]),
            "short_tp2": float(tf5["short_tp2"]),
            "trigger": plan.get("trigger", ""),
        }
    except Exception:
        return None


def dual_signal_at(history, ts):
    """Return a side only when one side is ready; skip ambiguous dual signals."""
    long_sig = long_signal_at(history, ts)
    short_sig = short_signal_at(history, ts)

    if long_sig and short_sig:
        return None
    if long_sig:
        return {
            "side": "LONG",
            "setup": long_sig["setup"],
            "signal_price": long_sig["entry_signal"],
            "tp1": long_sig["smart_tp1"],
            "tp2": long_sig["smart_tp2"],
        }
    if short_sig:
        return short_sig
    return None


def _position_size(equity, entry, sl):
    risk_amount = equity * 0.01
    distance = abs(entry - sl)
    if distance <= 0:
        return 0.0, 0.0
    return risk_amount / distance, risk_amount


def _fee(notional, fee_rate):
    return abs(notional) * fee_rate


def simulate_trade(signal, entry_candle, future, sl_pct, mode,
                    equity, fee_rate=0.0, slippage_pct=0.0):
    side = signal["side"]
    entry = float(entry_candle["open"])

    if side == "LONG":
        actual_entry = entry * (1 + slippage_pct)
        sl = actual_entry * (1 - sl_pct)

        if mode == "rr":
            risk_distance = actual_entry - sl
            tp1 = actual_entry + risk_distance
            tp2 = actual_entry + 2 * risk_distance
        else:
            tp1 = float(signal["tp1"])
            tp2 = float(signal["tp2"])
            if tp1 <= actual_entry:
                tp1 = actual_entry + (actual_entry - sl)
            if tp2 <= tp1:
                tp2 = actual_entry + 2 * (actual_entry - sl)

    else:
        actual_entry = entry * (1 - slippage_pct)
        sl = actual_entry * (1 + sl_pct)

        if mode == "rr":
            risk_distance = sl - actual_entry
            tp1 = actual_entry - risk_distance
            tp2 = actual_entry - 2 * risk_distance
        else:
            tp1 = float(signal["short_tp1"])
            tp2 = float(signal["short_tp2"])
            if tp1 >= actual_entry:
                tp1 = actual_entry - (sl - actual_entry)
            if tp2 >= tp1:
                tp2 = actual_entry - 2 * (sl - actual_entry)

    qty, risk_amount = _position_size(equity, actual_entry, sl)
    if qty <= 0:
        return None

    pnl = -_fee(actual_entry * qty, fee_rate)
    remaining = qty
    tp1_hit = False
    exit_time = None
    exit_reason = None

    for _, c in future.iterrows():
        hi = float(c["high"])
        lo = float(c["low"])
        ts = int(c["ts"])

        if side == "LONG":
            sl_hit = lo <= sl
            tp1_hit_now = hi >= tp1
            tp2_hit_now = hi >= tp2
        else:
            sl_hit = hi >= sl
            tp1_hit_now = lo <= tp1
            tp2_hit_now = lo <= tp2

        # Conservative: SL first if both are touched in the same candle.
        if not tp1_hit:
            if sl_hit:
                pnl += (sl - actual_entry) * remaining if side == "LONG" else (actual_entry - sl) * remaining
                pnl -= _fee(sl * remaining, fee_rate)
                exit_time = ts
                exit_reason = "SL"
                remaining = 0
                break

            if tp1_hit_now:
                half = qty * 0.50
                pnl += (tp1 - actual_entry) * half if side == "LONG" else (actual_entry - tp1) * half
                pnl -= _fee(tp1 * half, fee_rate)
                remaining -= half
                tp1_hit = True

                if tp2_hit_now:
                    pnl += (tp2 - actual_entry) * remaining if side == "LONG" else (actual_entry - tp2) * remaining
                    pnl -= _fee(tp2 * remaining, fee_rate)
                    remaining = 0
                    exit_time = ts
                    exit_reason = "TP2"
                    break

        else:
            if sl_hit:
                pnl += (sl - actual_entry) * remaining if side == "LONG" else (actual_entry - sl) * remaining
                pnl -= _fee(sl * remaining, fee_rate)
                remaining = 0
                exit_time = ts
                exit_reason = "SL_AFTER_TP1"
                break

            if tp2_hit_now:
                pnl += (tp2 - actual_entry) * remaining if side == "LONG" else (actual_entry - tp2) * remaining
                pnl -= _fee(tp2 * remaining, fee_rate)
                remaining = 0
                exit_time = ts
                exit_reason = "TP2"
                break

    if remaining > 0:
        last = future.iloc[-1]
        close = float(last["close"])
        pnl += (close - actual_entry) * remaining if side == "LONG" else (actual_entry - close) * remaining
        pnl -= _fee(close * remaining, fee_rate)
        exit_time = int(last["ts"])
        exit_reason = "TIMEOUT"

    r = pnl / risk_amount if risk_amount > 0 else 0.0

    return Trade(
        side=side,
        setup=signal["setup"],
        entry_time=int(entry_candle["ts"]),
        exit_time=exit_time,
        entry=actual_entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        qty=qty,
        pnl=pnl,
        r_multiple=r,
        exit_reason=exit_reason,
        risk_amount=risk_amount,
    )


def build_signals(history):
    """Evaluate V3.14 signals ONCE per 5m candle.

    The previous implementation recalculated the full MTF decision engine
    separately for every SL/mode and Long/Long+Short run. That multiplied
    the expensive signal logic 16x. Precomputing point-in-time signals once
    keeps the research architecture identical while removing that bottleneck.
    """
    base = history[BASE_RESOLUTION]
    signals = {}
    print("Precomputing Long/Short signals once...")
    for i in range(210, len(base) - 2):
        ts = int(base.iloc[i]["ts"])
        long_sig = long_signal_at(history, ts)
        short_sig = short_signal_at(history, ts)

        if long_sig and short_sig:
            signals[i] = None
        elif long_sig:
            signals[i] = {
                "side": "LONG",
                "setup": long_sig["setup"],
                "signal_price": long_sig["entry_signal"],
                "tp1": long_sig["smart_tp1"],
                "tp2": long_sig["smart_tp2"],
            }
        elif short_sig:
            signals[i] = short_sig
        else:
            signals[i] = None

    print(f"Signal scan complete: {len(signals):,} candles evaluated.")
    return signals


def run_mode(base, signals, sl_pct, mode, initial_equity,
             dual_side=True, fee_rate=0.0, slippage_pct=0.0):
    trades = []
    equity = float(initial_equity)

    i = 210
    while i < len(base) - 2:
        signal = signals.get(i)
        if signal is None:
            i += 1
            continue

        if not dual_side and signal["side"] != "LONG":
            i += 1
            continue

        entry_idx = i + 1
        entry_candle = base.iloc[entry_idx]
        future = base.iloc[entry_idx + 1:]

        trade = simulate_trade(
            signal, entry_candle, future, sl_pct, mode, equity,
            fee_rate=fee_rate, slippage_pct=slippage_pct
        )
        if trade is None:
            i += 1
            continue

        equity += trade.pnl
        trades.append(trade)

        exit_pos = base["ts"].searchsorted(trade.exit_time, side="right")
        i = max(i + 1, int(exit_pos))

    return trades


def summarize(trades, initial_equity):
    if not trades:
        return {
            "trades": 0, "win_rate": 0.0, "net_profit": 0.0,
            "return_pct": 0.0, "max_drawdown": 0.0,
            "profit_factor": 0.0, "avg_r": 0.0, "expectancy": 0.0
        }

    equity = float(initial_equity)
    peak = equity
    max_dd = 0.0
    pnls = []

    for t in trades:
        equity += t.pnl
        pnls.append(t.pnl)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "net_profit": sum(pnls),
        "return_pct": (equity / initial_equity - 1) * 100,
        "max_drawdown": max_dd,
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
        "avg_r": sum(t.r_multiple for t in trades) / len(trades),
        "expectancy": sum(pnls) / len(trades),
    }


def print_side_breakdown(trades, initial_equity):
    print("\nSIDE BREAKDOWN")
    print("-" * 95)
    for side in ("LONG", "SHORT"):
        subset = [t for t in trades if t.side == side]
        s = summarize(subset, initial_equity)
        print(
            f"{side:5} trades={s['trades']:4d} "
            f"win={s['win_rate']:6.2f}% "
            f"return={s['return_pct']:8.2f}% "
            f"DD={s['max_drawdown']:6.2f}% "
            f"PF={s['profit_factor']:6.2f} "
            f"avgR={s['avg_r']:7.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--initial-equity", type=float, default=1_000_000)
    parser.add_argument("--fee", type=float, default=0.0,
                        help="Fee rate charged on each executed entry/exit side.")
    parser.add_argument("--slippage", type=float, default=0.0,
                        help="Entry slippage fraction.")
    args = parser.parse_args()

    history = prepare_history(args.days)
    signals = build_signals(history)
    base = history[BASE_RESOLUTION]

    rows = []
    for mode in ("smart_tp", "rr"):
        for sl in SL_VALUES:
            print(f"Running {mode}, SL={sl*100:.1f}% ...")

            long_trades = run_mode(
                base, signals, sl, mode, args.initial_equity,
                dual_side=False, fee_rate=args.fee,
                slippage_pct=args.slippage
            )
            dual_trades = run_mode(
                base, signals, sl, mode, args.initial_equity,
                dual_side=True, fee_rate=args.fee,
                slippage_pct=args.slippage
            )

            for label, trades in (
                ("LONG_ONLY", long_trades),
                ("LONG_SHORT", dual_trades),
            ):
                s = summarize(trades, args.initial_equity)
                rows.append({
                    "TEST": label,
                    "MODE": mode,
                    "SL": f"{sl*100:.1f}%",
                    **s,
                })

            print_side_breakdown(dual_trades, args.initial_equity)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("V3.14 — LONG ONLY vs LONG + EXPERIMENTAL SHORT")
    print("=" * 120)
    if not df.empty:
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nNOTE")
    print("SHORT is experimental and uses the existing trade_setup_engine SHORT READY path.")
    print("It is NOT yet a full symmetric V3.14 Short architecture.")
    print("Do not merge this branch into the production analyzer based on this test alone.")


if __name__ == "__main__":
    main()
