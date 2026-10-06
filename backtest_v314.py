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
    results = {}

    # 5m changes on every backtest step.
    df5 = frames["5m"]
    sub5 = df5[df5["timestamp"] <= t].tail(500).copy()
    if len(sub5) < 220:
        return None

    sub5["datetime"] = pd.to_datetime(sub5["timestamp"], unit="s", utc=True)
    sub5 = analyzer.calculate_indicators(sub5)
    results["5m"] = analyzer.analyze_timeframe(sub5, "5m")

    # Higher timeframes only change when a new candle closes.
    for tf in ("15m", "1h"):
        df = frames[tf]
        sub = df[df["timestamp"] <= t].tail(500).copy()
        if len(sub) < 220:
            return None

        latest_ts = int(sub["timestamp"].iloc[-1])
        cached = cache[tf]
        if cached["timestamp"] != latest_ts:
            sub["datetime"] = pd.to_datetime(sub["timestamp"], unit="s", utc=True)
            sub = analyzer.calculate_indicators(sub)
            cached["timestamp"] = latest_ts
            cached["result"] = analyzer.analyze_timeframe(sub, tf)

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

    print("V3.14 HISTORICAL BACKTEST")
    print("=" * 60)
    print(f"WINDOW       {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print("SOURCE       api.bitpin.org")
    print("SYMBOL       BTC_IRT")
    print("TIMEFRAMES   5m / 15m / 1h")

    frames = {}
    for tf in TIMEFRAMES:
        print(f"DOWNLOADING  {tf} ...", flush=True)
        frames[tf] = fetch_history(tf, start_ts - 500 * TIMEFRAMES[tf], end_ts)
        print(f"  {tf}: {len(frames[tf])} candles", flush=True)

    base = frames["5m"]
    signals = []
    previous_route = "WAIT"
    cache = {
        "15m": {"timestamp": None, "result": None},
        "1h": {"timestamp": None, "result": None},
    }

    for i, t in enumerate(base["timestamp"].tolist()):
        results = make_results_at(t, frames, cache)
        if results is None:
            continue

        route, final, plan = build_route(results)

        if final["ready"] and route != "WAIT" and route != previous_route:
            if plan:
                entry = plan.get("entry")
                sl = plan.get("stop_loss")
                tp1 = plan.get("tp1")
                tp2 = plan.get("tp2")

                if all(x is not None for x in (entry, sl, tp1, tp2)):
                    signal = {
                        "signal_time_utc": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                        "route": route,
                        "entry": float(entry),
                        "stop_loss": float(sl),
                        "tp1": float(tp1),
                        "tp2": float(tp2),
                    }
                    future = base[base["timestamp"] > t]
                    outcome, outcome_ts = evaluate_signal(signal, future)
                    signal["outcome"] = outcome
                    signal["outcome_time_utc"] = (
                        datetime.fromtimestamp(outcome_ts, timezone.utc).isoformat()
                        if outcome_ts else ""
                    )
                    signals.append(signal)
                    print(
                        f"SIGNAL {signal['signal_time_utc']} | "
                        f"{route} | ENTRY={entry:,.0f} | OUTCOME={outcome}",
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
    print("V3.14 BACKTEST SUMMARY")
    print("-" * 60)
    print(f"DAYS                 {args.days}")
    print(f"SIGNALS              {total}")
    print(f"SIGNALS / WEEK       {total / weeks:.2f}")
    if total:
        print(out["route"].value_counts().to_string())
        print()
        print(out["outcome"].value_counts().to_string())
    print(f"CSV                  {args.output}")


if __name__ == "__main__":
    main()
