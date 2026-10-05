import argparse
import csv
import sys
import time
from datetime import datetime, timezone

import btc_analyzer as analyzer


def snapshot():
    results = {}
    for tf in analyzer.TIMEFRAMES:
        df = analyzer.get_candles(tf)
        df = analyzer.calculate_indicators(df)
        results[tf] = analyzer.analyze_timeframe(df, tf)

    entry_state = analyzer.entry_state_engine(results)
    pullback_entry = analyzer.pullback_entry_engine(results)
    retest_plan = analyzer.retest_trade_plan_engine(results)
    pullback_plan = analyzer.pullback_trade_plan_engine(results)
    route = analyzer.route_priority_engine(results, entry_state, pullback_entry)

    if route["route"] == "BREAKOUT":
        plan_state = entry_state["state"]
        plan = "BREAKOUT"
    elif route["route"] == "RETEST":
        plan_state = retest_plan["state"]
        plan = "RETEST"
    elif route["route"] == "PULLBACK":
        plan_state = pullback_plan["state"]
        plan = "PULLBACK"
    else:
        plan_state = "WAIT"
        plan = "WAIT"

    tf5 = results["5m"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "price": round(tf5["price"], 2),
        "market_regime": " / ".join([
            results["1h"]["trend"],
            results["15m"]["trend"],
        ]),
        "breakout_5m": tf5["breakout"],
        "retest_5m": tf5["retest"],
        "retest_momentum_5m": tf5["retest_momentum"],
        "pullback_15m": results["15m"]["pullback"],
        "entry_state": entry_state["state"],
        "retest_plan": retest_plan["state"],
        "pullback_plan": pullback_plan["state"],
        "final_route": route["route"],
        "final_action": "ENTRY READY" if route["route"] != "WAIT" and plan_state.endswith("ENTRY READY") else "WAIT",
        "selected_plan": plan_state,
        "reason": route["reason"],
        "rsi_5m": round(tf5["rsi"], 2),
        "resistance_5m": round(tf5["resistance"], 2),
    }


def main():
    parser = argparse.ArgumentParser(description="V3.14 public-market observer")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=int, default=0)
    parser.add_argument("--output", default="observer_results.csv")
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")

    # Deterministic tests: no API/network dependency.
    assert analyzer.route_realistic_self_test()["status"] == "PASS"

    rows = []
    for i in range(args.samples):
        row = snapshot()
        rows.append(row)
        print(
            f"[{i + 1}/{args.samples}] "
            f"{row['timestamp_utc']} | "
            f"PRICE={row['price']:,.0f} | "
            f"ROUTE={row['final_route']} | "
            f"ACTION={row['final_action']}"
        )
        if i + 1 < args.samples and args.interval > 0:
            time.sleep(args.interval)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["final_route"]] = counts.get(row["final_route"], 0) + 1

    print("\nV3.14 OBSERVER SUMMARY")
    print("-" * 60)
    print(f"SAMPLES        {len(rows)}")
    for route, count in sorted(counts.items()):
        print(f"{route:<15} {count}")
    print(f"CSV            {args.output}")


if __name__ == "__main__":
    main()
