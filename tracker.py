import argparse
import csv
import hashlib
import os
from datetime import datetime, timezone

import pandas as pd

import btc_analyzer as analyzer


FIELDS = [
    "signal_id",
    "route",
    "signal_time_utc",
    "entry",
    "stop_loss",
    "tp1",
    "tp2",
    "rr_tp1",
    "rr_tp2",
    "status",
    "tp1_hit_at",
    "tp2_hit_at",
    "sl_hit_at",
    "outcome_time",
    "last_checked_utc",
]

DEFAULT_DATA = "data/signal_outcomes.csv"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def analyze_market():
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
        plan = {
            "state": entry_state["state"],
            "entry": entry_state["entry_price"],
            "stop_loss": entry_state["long_sl"],
            "tp1": entry_state["long_tp1"],
            "tp2": entry_state["long_tp2"],
            "rr_tp1": entry_state["long_rr_tp1"],
            "rr_tp2": entry_state["long_rr_tp2"],
        }
    elif route["route"] == "RETEST":
        plan = retest_plan
    elif route["route"] == "PULLBACK":
        plan = pullback_plan
    else:
        plan = {
            "state": "WAIT",
            "entry": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
            "rr_tp1": 0,
            "rr_tp2": 0,
        }

    return results, route, plan


def make_signal_id(route, plan):
    raw = "|".join(
        [
            route,
            f"{float(plan['entry']):.2f}",
            f"{float(plan['stop_loss']):.2f}",
            f"{float(plan['tp1']):.2f}",
            f"{float(plan['tp2']):.2f}",
            utc_now(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_rows(path):
    if not os.path.exists(path):
        return []

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_time(value):
    if not value:
        return pd.NaT

    if str(value).isdigit():
        number = int(value)
        unit = "ms" if number > 10**11 else "s"
        return pd.to_datetime(number, unit=unit, utc=True)

    return pd.to_datetime(value, utc=True)


def candle_time_series(df):
    values = pd.to_numeric(df["ts"], errors="coerce")

    if values.notna().any():
        unit = "ms" if values.dropna().median() > 10**11 else "s"
        return pd.to_datetime(values, unit=unit, utc=True)

    return pd.to_datetime(df["ts"], utc=True, errors="coerce")


def new_signal(route, plan):
    now = utc_now()
    signal_id = make_signal_id(route, plan)

    return {
        "signal_id": signal_id,
        "route": route,
        "signal_time_utc": now,
        "entry": f"{float(plan['entry']):.2f}",
        "stop_loss": f"{float(plan['stop_loss']):.2f}",
        "tp1": f"{float(plan['tp1']):.2f}",
        "tp2": f"{float(plan['tp2']):.2f}",
        "rr_tp1": f"{float(plan.get('rr_tp1', 0)):.2f}",
        "rr_tp2": f"{float(plan.get('rr_tp2', 0)):.2f}",
        "status": "OPEN",
        "tp1_hit_at": "",
        "tp2_hit_at": "",
        "sl_hit_at": "",
        "outcome_time": "",
        "last_checked_utc": now,
    }


def evaluate_row(row, candles):
    if row["status"] in {"TP2_HIT", "SL_HIT", "SL_AFTER_TP1", "AMBIGUOUS", "TIMEOUT"}:
        return row

    signal_time = parse_time(row["signal_time_utc"])
    if pd.isna(signal_time):
        return row

    candle_times = candle_time_series(candles)
    future = candles.loc[candle_times >= signal_time].copy()
    future["_time"] = candle_times.loc[future.index]

    entry = float(row["entry"])
    sl = float(row["stop_loss"])
    tp1 = float(row["tp1"])
    tp2 = float(row["tp2"])

    tp1_hit = bool(row["tp1_hit_at"])
    tp2_hit = bool(row["tp2_hit_at"])
    sl_hit = bool(row["sl_hit_at"])

    for _, candle in future.iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        candle_time = pd.Timestamp(candle["_time"]).isoformat()

        hit_tp2 = high >= tp2
        hit_tp1 = high >= tp1
        hit_sl = low <= sl

        if not tp1_hit and not tp2_hit and not sl_hit:
            if hit_sl and hit_tp1:
                row["status"] = "AMBIGUOUS"
                row["outcome_time"] = candle_time
                row["last_checked_utc"] = utc_now()
                return row

            if hit_tp2:
                row["tp1_hit_at"] = candle_time
                row["tp2_hit_at"] = candle_time
                row["status"] = "TP2_HIT"
                row["outcome_time"] = candle_time
                tp1_hit = True
                tp2_hit = True
                break

            if hit_tp1:
                row["tp1_hit_at"] = candle_time
                row["status"] = "TP1_HIT_OPEN"
                tp1_hit = True
                continue

            if hit_sl:
                row["sl_hit_at"] = candle_time
                row["status"] = "SL_HIT"
                row["outcome_time"] = candle_time
                sl_hit = True
                break

        elif tp1_hit and not tp2_hit and not sl_hit:
            if hit_tp2 and hit_sl:
                row["status"] = "AMBIGUOUS"
                row["outcome_time"] = candle_time
                return row

            if hit_tp2:
                row["tp2_hit_at"] = candle_time
                row["status"] = "TP2_HIT"
                row["outcome_time"] = candle_time
                tp2_hit = True
                break

            if hit_sl:
                row["sl_hit_at"] = candle_time
                row["status"] = "SL_AFTER_TP1"
                row["outcome_time"] = candle_time
                sl_hit = True
                break

    latest_time = future["_time"].max() if not future.empty else signal_time
    age_hours = max(
        0.0,
        (latest_time - signal_time).total_seconds() / 3600,
    )

    if row["status"] == "TP1_HIT_OPEN" and age_hours >= 24:
        row["status"] = "TIMEOUT"
        row["outcome_time"] = pd.Timestamp(latest_time).isoformat()

    row["last_checked_utc"] = utc_now()
    return row


def summary(rows):
    counts = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1

    print("\nV3.14 SIGNAL OUTCOME SUMMARY")
    print("-" * 60)
    print(f"SIGNALS        {len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status:<20} {count}")


def main():
    parser = argparse.ArgumentParser(description="V3.14 signal outcome tracker")
    parser.add_argument("--output", default=DEFAULT_DATA)
    args = parser.parse_args()

    # Deterministic tests: no API/network dependency.
    assert analyzer.route_realistic_self_test() == "PASS"

    results, route, plan = analyze_market()
    rows = load_rows(args.output)

    ready = (
        route["route"] in {"BREAKOUT", "RETEST", "PULLBACK"}
        and plan["state"].endswith("ENTRY READY")
    )

    if ready:
        open_match = any(
            row["route"] == route["route"]
            and row["status"] in {
                "OPEN",
                "TP1_HIT_OPEN",
            }
            and abs(float(row["entry"]) - float(plan["entry"])) < 0.01
            and abs(float(row["stop_loss"]) - float(plan["stop_loss"])) < 0.01
            and abs(float(row["tp1"]) - float(plan["tp1"])) < 0.01
            and abs(float(row["tp2"]) - float(plan["tp2"])) < 0.01
            for row in rows
        )

        if not open_match:
            signal = new_signal(route["route"], plan)
            rows.append(signal)
            print(
                f"NEW SIGNAL | ROUTE={signal['route']} | "
                f"ENTRY={float(signal['entry']):,.0f} | "
                f"SL={float(signal['stop_loss']):,.0f} | "
                f"TP1={float(signal['tp1']):,.0f} | "
                f"TP2={float(signal['tp2']):,.0f}"
            )
        else:
            print(f"SIGNAL ALREADY OPEN | ROUTE={route['route']}")
    else:
        print(f"NO NEW SIGNAL | ROUTE={route['route']}")

    candles = analyzer.get_candles("5m")

    for index, row in enumerate(rows):
        rows[index] = evaluate_row(row, candles)

    save_rows(args.output, rows)
    summary(rows)


if __name__ == "__main__":
    main()
