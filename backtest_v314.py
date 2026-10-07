"""
V3.14 Backtest Engine
----------------------
Architecture-safe backtest layer. Does NOT modify btc_analyzer.py.

Tests:
1) SL-only: 1%, 1.5%, 2%, 2.5% with existing V3.14 Smart TP1/TP2.
2) Fixed R:R: TP1=1R, TP2=2R for the same four SL values.

Rules:
- Risk = 1% of current equity per trade.
- 50% position exits at TP1, 50% at TP2.
- Entry at next 5m candle OPEN after an ENTRY READY signal.
- No overlapping positions.
- Same-candle SL/TP ambiguity is resolved conservatively: SL first.
- Long setups only, matching the current V3.14 entry engine.
- Fees/slippage are configurable and default to zero.
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

import btc_analyzer as v314


SL_VALUES = [0.01, 0.015, 0.02, 0.025]
BASE_RESOLUTION = "5m"
HIGHER_RESOLUTIONS = ("15m", "1h")
CANDLES_PER_REQUEST = 500


@dataclass
class Trade:
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


def _ts(dt):
    return int(dt.timestamp())


def fetch_history(resolution, days=60, chunk=CANDLES_PER_REQUEST):
    """Fetch enough history in API-sized chunks, oldest to newest."""
    end = int(time.time())
    start = end - days * 86400
    step = chunk * v314.TIMEFRAMES[resolution]["seconds"]
    rows = []

    cursor = start
    while cursor < end:
        chunk_end = min(cursor + step, end)
        url = f"{v314.BASE_URL}/api/v1/mkt/candles/"
        params = {
            "symbol": v314.SYMBOL,
            "from": cursor,
            "to": chunk_end,
            "res": resolution,
        }

        r = v314.requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = data.get("results", data.get("data", []))

        for x in data:
            try:
                if isinstance(x, dict):
                    ts = x.get("timestamp", x.get("ts", x.get("time")))
                    op = x.get("open")
                    hi = x.get("high")
                    lo = x.get("low")
                    cl = x.get("close")
                    vol = x.get("volume", 0)
                else:
                    ts, op, hi, lo, cl, vol = x[:6]

                rows.append([int(ts), float(op), float(hi), float(lo),
                             float(cl), float(vol)])
            except Exception:
                continue

        cursor = chunk_end
        time.sleep(0.10)

    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    if df.empty:
        raise ValueError(f"No historical data returned for {resolution}")

    return (
        df.drop_duplicates("ts")
          .sort_values("ts")
          .reset_index(drop=True)
    )


def prepare_history(days):
    print(f"Fetching {days}-day history...")
    data = {BASE_RESOLUTION: fetch_history(BASE_RESOLUTION, days)}
    for tf in HIGHER_RESOLUTIONS:
        data[tf] = fetch_history(tf, days)

    # Warm-up is handled by the analyzer functions themselves.
    for tf in data:
        data[tf] = v314.calculate_indicators(data[tf])

    return data


def _slice_at_or_before(df, ts):
    """Return a bounded point-in-time window for V3.14 decision logic.

    Indicators are calculated once on the full historical series. The
    decision functions only need recent candles for levels/price action,
    while the precomputed indicator columns retain full-series warm-up.
    Limiting the slice to 210 candles avoids repeatedly copying and scanning
    the entire 60-day dataframe at every 5m signal timestamp.
    """
    pos = df["ts"].searchsorted(ts, side="right")
    if pos < 210:
        return df.iloc[:pos].copy()
    return df.iloc[pos - 210:pos].copy()


def _get_mtf_results(history, ts):
    results = {}
    for tf in ("1m", "5m", "15m", "1h"):
        if tf == "1m":
            # Current V3.14 entry engine does not require 1m.
            continue
        part = _slice_at_or_before(history[tf], ts)
        if len(part) < 210:
            return None
        results[tf] = v314.analyze_timeframe(part, tf)
    return results


def _setup_name(results):
    tf5 = results["5m"]
    if tf5.get("pullback") in ("POSSIBLE BULLISH PULLBACK",
                                "POSSIBLE BEARISH PULLBACK"):
        pull = v314.pullback_entry_engine(results)
        if pull.get("status") == "PULLBACK ENTRY READY":
            return "Pullback"

    if tf5.get("retest") == "RETEST CONFIRMED":
        return "Retest"

    if tf5.get("breakout") == "VALID BREAKOUT":
        return "Breakout"

    return "Other"


def signal_at(history, ts):
    results = _get_mtf_results(history, ts)
    if results is None:
        return None

    try:
        entry_state = v314.entry_state_engine(results)
        pullback = v314.pullback_entry_engine(results)

        if pullback.get("status") == "PULLBACK ENTRY READY":
            entry_state["state"] = "ENTRY READY"
            entry_state["trigger"] = "PULLBACK ENTRY CONFIRMED"

        if entry_state["state"] != "ENTRY READY":
            return None

        setup = _setup_name(results)
        if setup == "Other":
            return None

        return {
            "setup": setup,
            "entry_signal": float(results["5m"]["price"]),
            "smart_tp1": float(entry_state["long_tp1"]),
            "smart_tp2": float(entry_state["long_tp2"]),
        }
    except Exception:
        return None


def _position_size(equity, entry, sl):
    risk_amount = equity * 0.01
    distance = abs(entry - sl)
    if distance <= 0:
        return 0.0, 0.0
    return risk_amount / distance, risk_amount


def _apply_fee(value, fee_rate):
    return abs(value) * fee_rate


def simulate_trade(signal, entry_candle, future, sl_pct, mode,
                    equity, fee_rate=0.0, slippage_pct=0.0):
    entry = float(entry_candle["open"])
    sl = entry * (1 - sl_pct)

    if mode == "rr":
        risk_distance = entry - sl
        tp1 = entry + risk_distance       # 1R
        tp2 = entry + 2 * risk_distance   # 2R
    else:
        # Preserve V3.14 Smart TP logic, but anchor it to actual execution.
        tp1 = float(signal["smart_tp1"])
        tp2 = float(signal["smart_tp2"])
        if tp1 <= entry:
            tp1 = entry + (entry - sl)
        if tp2 <= tp1:
            tp2 = entry + 2 * (entry - sl)

    qty, risk_amount = _position_size(equity, entry, sl)
    if qty <= 0:
        return None

    # Conservative entry slippage for long: pay slightly more.
    actual_entry = entry * (1 + slippage_pct)
    qty, risk_amount = _position_size(equity, actual_entry, sl)

    remaining = qty
    pnl = 0.0
    tp1_hit = False
    exit_time = None
    exit_reason = None

    for _, c in future.iterrows():
        hi = float(c["high"])
        lo = float(c["low"])
        ts = int(c["ts"])

        # Conservative rule: if SL and TP are both touched in one candle,
        # assume SL occurred first.
        if not tp1_hit:
            if lo <= sl:
                pnl += (sl - actual_entry) * remaining
                exit_time = ts
                exit_reason = "SL"
                remaining = 0
                break

            if hi >= tp1:
                half = qty * 0.50
                pnl += (tp1 - actual_entry) * half
                remaining -= half
                tp1_hit = True

                # TP1 and TP2 can both be touched in the same candle.
                if hi >= tp2:
                    pnl += (tp2 - actual_entry) * remaining
                    remaining = 0
                    exit_time = ts
                    exit_reason = "TP2"
                    break
        else:
            if lo <= sl:
                pnl += (sl - actual_entry) * remaining
                remaining = 0
                exit_time = ts
                exit_reason = "SL_AFTER_TP1"
                break

            if hi >= tp2:
                pnl += (tp2 - actual_entry) * remaining
                remaining = 0
                exit_time = ts
                exit_reason = "TP2"
                break

    if remaining > 0:
        last = future.iloc[-1]
        close = float(last["close"])
        pnl += (close - actual_entry) * remaining
        exit_time = int(last["ts"])
        exit_reason = "TIMEOUT"

    # Approximate round-trip fee on executed notional.
    fee = _apply_fee(actual_entry * qty, fee_rate)
    pnl -= fee

    r = pnl / risk_amount if risk_amount > 0 else 0.0

    return Trade(
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


def run_backtest(history, sl_pct, mode, initial_equity,
                 fee_rate=0.0, slippage_pct=0.0):
    base = history[BASE_RESOLUTION]
    trades = []
    equity = float(initial_equity)
    equity_curve = [(int(base.iloc[0]["ts"]), equity)]

    i = 210
    while i < len(base) - 2:
        ts = int(base.iloc[i]["ts"])
        signal = signal_at(history, ts)

        if signal is None:
            i += 1
            continue

        # Execute on the next 5m candle open.
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
        equity_curve.append((trade.exit_time, equity))

        # No overlapping trades: resume after the exit candle.
        exit_pos = base["ts"].searchsorted(trade.exit_time, side="right")
        i = max(i + 1, int(exit_pos))

    return trades, equity_curve


def summarize(trades, initial_equity):
    if not trades:
        return {
            "trades": 0, "win_rate": 0.0, "net_profit": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0,
            "avg_r": 0.0, "expectancy": 0.0
        }

    pnls = [t.pnl for t in trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]

    equity = initial_equity
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss else float("inf")

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "net_profit": sum(pnls),
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "avg_r": sum(t.r_multiple for t in trades) / len(trades),
        "expectancy": sum(pnls) / len(trades),
    }


def print_report(all_results, initial_equity):
    rows = []
    for key, trades in all_results.items():
        s = summarize(trades, initial_equity)
        rows.append({
            "TEST": key[0],
            "SL": f"{key[1]*100:.1f}%",
            **s
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        print("\n" + "=" * 110)
        print("V3.14 — 60 DAY STOP-LOSS COMPARISON")
        print("=" * 110)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

        print("\nSETUP BREAKDOWN")
        for key, trades in all_results.items():
            if not trades:
                continue
            sl = key[1]
            mode = key[0]
            for setup in ("Breakout", "Retest", "Pullback"):
                subset = [t for t in trades if t.setup == setup]
                s = summarize(subset, initial_equity)
                print(
                    f"{mode:8} SL={sl*100:4.1f}% "
                    f"{setup:9} trades={s['trades']:4d} "
                    f"win={s['win_rate']:6.2f}% "
                    f"net={s['net_profit']:,.0f} "
                    f"PF={s['profit_factor']:.2f} "
                    f"avgR={s['avg_r']:.3f}"
                )

        return df

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--initial-equity", type=float, default=1_000_000)
    parser.add_argument("--fee", type=float, default=0.0,
                        help="Fee rate per trade side, e.g. 0.001")
    parser.add_argument("--slippage", type=float, default=0.0,
                        help="Entry slippage fraction, e.g. 0.0005")
    args = parser.parse_args()

    history = prepare_history(args.days)

    all_results = {}
    for mode in ("smart_tp", "rr"):
        for sl in SL_VALUES:
            print(f"Running {mode}, SL={sl*100:.1f}% ...")
            trades, _ = run_backtest(
                history, sl, mode, args.initial_equity,
                fee_rate=args.fee, slippage_pct=args.slippage
            )
            all_results[(mode, sl)] = trades

    print_report(all_results, args.initial_equity)


if __name__ == "__main__":
    main()
