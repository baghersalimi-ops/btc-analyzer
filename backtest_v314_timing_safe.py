"""
RESEARCH-ONLY V3.14 backtest with candle-close and executable-entry timing safeguards.
Do not use as production strategy until full validation, fees/slippage and OOS tests pass.
"""
import argparse
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

import btc_analyzer as analyzer


BASE_URL = analyzer.BASE_URL
SYMBOL = analyzer.SYMBOL
TIMEFRAMES = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def fetch_history(resolution, start_ts, end_ts, limit=120):
    step = TIMEFRAMES[resolution]
    rows = []
    cursor = start_ts

    while cursor < end_ts:
        window_end = min(end_ts, cursor + step * limit)
        url = f"{BASE_URL}/api/v1/mkt/candles/"
        params = {
            "symbol": SYMBOL,
            "from": cursor,
            "to": window_end,
            "res": resolution,
        }

        data = None
        for attempt in range(1, 6):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code in RETRYABLE_STATUS:
                    raise requests.HTTPError(
                        f"retryable HTTP {r.status_code}", response=r
                    )
                r.raise_for_status()
                data = r.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 5:
                    raise RuntimeError(
                        f"Bitpin {resolution} request failed after 5 attempts: {exc}"
                    ) from exc
                wait = min(8.0, 0.8 * (2 ** (attempt - 1)))
                print(
                    f"  RETRY {resolution} {attempt}/5 | "
                    f"HTTP={getattr(getattr(exc, 'response', None), 'status_code', 'N/A')} | "
                    f"wait={wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)

        if isinstance(data, dict):
            data = data.get("results", data.get("data", []))

        batch = []
        for x in data:
            if isinstance(x, dict):
                ts = x.get("timestamp", x.get("ts", x.get("time")))
                op = x.get("open")
                hi = x.get("high")
                lo = x.get("low")
                cl = x.get("close")
                vol = x.get("volume", 0)
            else:
                ts, op, hi, lo, cl, vol = x[:6]

            if ts is None:
                continue
            try:
                batch.append({
                    "timestamp": int(float(ts)),
                    "open": float(op),
                    "high": float(hi),
                    "low": float(lo),
                    "close": float(cl),
                    "volume": float(vol or 0),
                })
            except (TypeError, ValueError):
                continue

        rows.extend(batch)

        if not batch:
            cursor = window_end
        else:
            newest = max(x["timestamp"] for x in batch)
            next_cursor = newest + step
            if next_cursor <= cursor:
                next_cursor = window_end
            cursor = min(next_cursor, end_ts)

        time.sleep(0.20)

    if not rows:
        raise RuntimeError(f"No {resolution} data returned by Bitpin")

    df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    df = df[
        (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    ].reset_index(drop=True)
    return df


def make_results_at(t, frames, cache):
    """
    Build features using only candles whose full interval ended by decision time t.
    Candle timestamps from Bitpin behave like candle-start timestamps for 5m:
    a candle starting at t is not available to the signal until t + interval.
    """
    results = {}

    for tf, seconds in TIMEFRAMES.items():
        df = frames[tf]
        closed = df[(df["timestamp"] + seconds) <= t].tail(500).copy()
        if len(closed) < 220:
            return None

        latest_ts = int(closed["timestamp"].iloc[-1])
        if tf == "5m":
            closed["datetime"] = pd.to_datetime(closed["timestamp"], unit="s", utc=True)
            closed = analyzer.calculate_indicators(closed)
            results[tf] = analyzer.analyze_timeframe(closed, tf)
        else:
            cached = cache[tf]
            if cached["timestamp"] != latest_ts:
                closed["datetime"] = pd.to_datetime(closed["timestamp"], unit="s", utc=True)
                closed = analyzer.calculate_indicators(closed)
                cached["timestamp"] = latest_ts
                cached["result"] = analyzer.analyze_timeframe(closed, tf)
            results[tf] = cached["result"]

    return results


def build_route(results):
    entry_state = analyzer.entry_state_engine(results)
    pullback_entry = analyzer.pullback_entry_engine(results)
    retest_plan = analyzer.retest_trade_plan_engine(results)
    pullback_plan = analyzer.pullback_trade_plan_engine(results)

    route_decision = analyzer.route_priority_engine(
        results, entry_state, pullback_entry
    )

    route = route_decision["route"]

    if route == "BREAKOUT":
        plan = {
            "entry": entry_state.get("entry_price"),
            "stop_loss": entry_state.get("long_sl"),
            "tp1": entry_state.get("long_tp1"),
            "tp2": entry_state.get("long_tp2"),
            "rr_tp1": entry_state.get("long_rr_tp1"),
            "rr_tp2": entry_state.get("long_rr_tp2"),
        }
    elif route == "RETEST":
        plan = retest_plan
    elif route == "PULLBACK":
        plan = pullback_plan
    else:
        plan = None

    final = analyzer.final_route_engine(route_decision, plan)

    return route, final, plan


def evaluate_signal(signal, future_df):
    sl = signal["stop_loss"]
    tp1 = signal["tp1"]
    tp2 = signal["tp2"]

    for _, candle in future_df.iterrows():
        ts = int(candle["timestamp"])
        high = float(candle["high"])
        low = float(candle["low"])

        if high >= tp2 and low <= sl:
            return "AMBIGUOUS", ts
        if high >= tp2:
            return "TP2_HIT", ts
        if high >= tp1 and low <= sl:
            return "AMBIGUOUS", ts
        if low <= sl:
            return "SL", ts
        if high >= tp1:
            return "TP1_HIT_OPEN", ts

    return "OPEN", None



def validate_plan(route, plan):
    """Backtest-only safety gate; V3.14 analyzer logic is unchanged."""
    if not plan:
        return False, "NO PLAN"
    try:
        entry = float(plan["entry"])
        sl = float(plan["stop_loss"])
        tp1 = float(plan["tp1"])
        tp2 = float(plan["tp2"])
    except (KeyError, TypeError, ValueError):
        return False, "MISSING/INVALID LEVEL"

    if not (entry > sl > 0):
        return False, "INVALID ENTRY/SL"
    if tp1 <= entry:
        return False, "TP1 NOT ABOVE ENTRY"
    if tp2 <= tp1:
        return False, "TP2 NOT ABOVE TP1"

    risk = entry - sl
    rr1 = (tp1 - entry) / risk
    rr2 = (tp2 - entry) / risk

    if rr1 < 1.0:
        return False, f"TP1 R={rr1:.2f} < 1R"
    if rr2 <= rr1:
        return False, "TP2 R NOT ABOVE TP1 R"

    return True, "VALID"


def trade_fingerprint(signal):
    return (
        signal["route"],
        round(signal["entry"], -4),
        round(signal["stop_loss"], -4),
        round(signal["tp1"], -4),
        round(signal["tp2"], -4),
    )


def evaluate_signal(signal, future_df):
    """
    Conservative OHLC simulation. Entry occurs at the next 5m candle open.
    Gaps through a level are filled at the open, not at the better limit price.
    Same-candle SL/TP ambiguity is resolved conservatively as ambiguous.
    """
    entry = signal["entry"]
    sl = signal["stop_loss"]
    tp1 = signal["tp1"]
    tp2 = signal["tp2"]
    risk = entry - sl
    tp1_hit = False

    def result(outcome, ts, realized_r, management):
        return {
            "outcome": outcome,
            "outcome_time_utc": datetime.fromtimestamp(int(ts), timezone.utc).isoformat(),
            "realized_r": realized_r,
            "management": management,
        }

    for _, candle in future_df.iterrows():
        ts = int(candle["timestamp"])
        op = float(candle["open"])
        high = float(candle["high"])
        low = float(candle["low"])

        # Entry candle is entered at its open; reject impossible level ordering upstream.
        if not tp1_hit:
            if op <= sl:
                return result("SL_GAP", ts, (op - entry) / risk, "GAP_STOP")
            if op >= tp2:
                rr1 = (tp1 - entry) / risk
                rr2 = (tp2 - entry) / risk
                return result("TP2_GAP", ts, 0.5 * rr1 + 0.5 * rr2, "GAP_TP2")
            if op >= tp1:
                tp1_hit = True
                if high >= tp2:
                    rr1 = (tp1 - entry) / risk
                    rr2 = (tp2 - entry) / risk
                    return result("TP2_HIT", ts, 0.5 * rr1 + 0.5 * rr2, "TP1_50pct_THEN_TP2_50pct")
                if low <= entry:
                    return result("TP1_THEN_BE", ts, 0.5 * ((tp1 - entry) / risk), "TP1_50pct_THEN_BREAKEVEN")
                continue

            if high >= tp1 and low <= sl:
                return result("AMBIGUOUS", ts, 0.0, "AMBIGUOUS_BEFORE_TP1")
            if low <= sl:
                return result("SL", ts, -1.0, "FULL_STOP")
            if high >= tp1:
                tp1_hit = True
                if high >= tp2:
                    rr1 = (tp1 - entry) / risk
                    rr2 = (tp2 - entry) / risk
                    return result("TP2_HIT", ts, 0.5 * rr1 + 0.5 * rr2, "TP1_50pct_THEN_TP2_50pct")
                continue
        else:
            if op <= entry:
                return result("TP1_THEN_BE_GAP", ts, 0.5 * ((tp1 - entry) / risk), "TP1_50pct_THEN_BREAKEVEN")
            if op >= tp2:
                rr1 = (tp1 - entry) / risk
                rr2 = (tp2 - entry) / risk
                return result("TP2_GAP", ts, 0.5 * rr1 + 0.5 * rr2, "TP1_50pct_THEN_TP2_50pct")
            if high >= tp2:
                rr1 = (tp1 - entry) / risk
                rr2 = (tp2 - entry) / risk
                return result("TP2_HIT", ts, 0.5 * rr1 + 0.5 * rr2, "TP1_50pct_THEN_TP2_50pct")
            if low <= entry:
                rr1 = (tp1 - entry) / risk
                return result("TP1_THEN_BE", ts, 0.5 * rr1, "TP1_50pct_THEN_BREAKEVEN")

    if tp1_hit:
        return {"outcome": "OPEN_AFTER_TP1", "outcome_time_utc": "", "realized_r": None, "management": "OPEN_AFTER_TP1"}
    return {"outcome": "OPEN", "outcome_time_utc": "", "realized_r": None, "management": "OPEN_BEFORE_TP1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--output", default="data/v314_backtest_60d.csv")
    args = parser.parse_args()

    if analyzer.route_realistic_self_test() != "PASS":
        raise SystemExit("V3.14 route self-test failed")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print("V3.14 HISTORICAL BACKTEST V2")
    print("=" * 60)
    print(f"WINDOW       {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print("SOURCE       api.bitpin.org")
    print("SYMBOL       BTC_IRT")
    print("TIMEFRAMES   5m / 15m / 1h")
    print("MODEL        CLOSED-CANDLE SIGNALS -> NEXT 5m OPEN FILL; 50% TP1 -> BE -> 50% TP2")

    frames = {}
    for tf in TIMEFRAMES:
        print(f"DOWNLOADING  {tf} ...", flush=True)
        frames[tf] = fetch_history(tf, start_ts - 500 * TIMEFRAMES[tf], end_ts)
        print(f"  {tf}: {len(frames[tf])} candles", flush=True)

    base = frames["5m"]
    signals = []
    previous_route = "WAIT"
    active_until_ts = None
    seen_fingerprints = set()
    rejected = {}

    cache = {
        "15m": {"timestamp": None, "result": None},
        "1h": {"timestamp": None, "result": None},
    }

    for i, t in enumerate(base["timestamp"].tolist()):
        results = make_results_at(t, frames, cache)
        if results is None:
            continue

        route, final, plan = build_route(results)

        # Do not open a second trade while the previous simulated trade
        # is still active. This removes clustered/overlapping entries.
        trade_available = active_until_ts is None or t > active_until_ts

        if final["ready"] and route != "WAIT" and trade_available and plan:
            valid, reason = validate_plan(route, plan)

            if not valid:
                rejected[reason] = rejected.get(reason, 0) + 1
            else:
                # The signal is computed only from candles closed by t.
                # The candle stamped t is the first executable 5m candle;
                # simulate entry at its open rather than the plan's hindsight price.
                entry_candle = base[base["timestamp"] == t]
                if entry_candle.empty:
                    rejected["NO_EXECUTABLE_ENTRY_CANDLE"] = rejected.get("NO_EXECUTABLE_ENTRY_CANDLE", 0) + 1
                    previous_route = route
                    continue

                plan_entry = float(plan["entry"])
                entry = float(entry_candle.iloc[0]["open"])
                sl = float(plan["stop_loss"])
                tp1 = float(plan["tp1"])
                tp2 = float(plan["tp2"])

                if not (entry > sl > 0 and tp1 > entry and tp2 > tp1):
                    rejected["INVALID_LEVELS_AT_NEXT_OPEN"] = rejected.get("INVALID_LEVELS_AT_NEXT_OPEN", 0) + 1
                    previous_route = route
                    continue

                signal = {
                    "signal_time_utc": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                    "plan_entry": plan_entry,
                    "entry_fill_model": "NEXT_5M_OPEN",
                    "route": route,
                    "entry": entry,
                    "stop_loss": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rr_tp1": (tp1 - entry) / (entry - sl),
                    "rr_tp2": (tp2 - entry) / (entry - sl),
                }

                fingerprint = trade_fingerprint(signal)

                if fingerprint in seen_fingerprints:
                    rejected["DUPLICATE_FINGERPRINT"] = (
                        rejected.get("DUPLICATE_FINGERPRINT", 0) + 1
                    )
                else:
                    seen_fingerprints.add(fingerprint)

                    # Include the entry candle itself: entry is at its open.
                    future = base[base["timestamp"] >= t]
                    result = evaluate_signal(signal, future)

                    signal["outcome"] = result["outcome"]
                    signal["outcome_time_utc"] = result["outcome_time_utc"]
                    signal["realized_r"] = result["realized_r"]
                    signal["management"] = result["management"]

                    signals.append(signal)

                    if result["outcome_time_utc"]:
                        active_until_ts = int(
                            datetime.fromisoformat(
                                result["outcome_time_utc"]
                            ).timestamp()
                        )
                    else:
                        # Still-open trade blocks further entries through
                        # the end of the historical window.
                        active_until_ts = end_ts

                    print(
                        f"SIGNAL {signal['signal_time_utc']} | "
                        f"{route} | ENTRY={entry:,.0f} | "
                        f"R1={signal['rr_tp1']:.2f} | "
                        f"R2={signal['rr_tp2']:.2f} | "
                        f"OUTCOME={result['outcome']} | "
                        f"REALIZED_R={result['realized_r']}",
                        flush=True,
                    )

        previous_route = route

        if i % 1000 == 0:
            print(f"PROGRESS {i}/{len(base)}", flush=True)

    out = pd.DataFrame(signals)
    out.to_csv(args.output, index=False)

    weeks = args.days / 7
    total = len(out)

    print()
    print("V3.14 BACKTEST V2 SUMMARY")
    print("-" * 60)
    print(f"DAYS                 {args.days}")
    print(f"VALID TRADES         {total}")
    print(f"TRADES / WEEK        {total / weeks:.2f}")

    if total:
        print()
        print("ROUTES")
        print(out["route"].value_counts().to_string())

        print()
        print("OUTCOMES")
        print(out["outcome"].value_counts().to_string())

        realized = out["realized_r"].dropna()
        if len(realized):
            print()
            print(f"REALIZED R TOTAL     {realized.sum():.2f}R")
            print(f"REALIZED R MEAN      {realized.mean():.2f}R")
            print(f"WIN RATE (R>0)       {(realized > 0).mean() * 100:.1f}%")

            equity = realized.cumsum()
            drawdown = equity - equity.cummax()
            print(f"MAX DRAWDOWN         {drawdown.min():.2f}R")

    if rejected:
        print()
        print("REJECTED / FILTERED")
        for reason, count in sorted(
            rejected.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"{reason:28s} {count}")

    print(f"CSV                  {args.output}")




if __name__ == "__main__":
    main()
