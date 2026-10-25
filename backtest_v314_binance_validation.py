"""Research-only V3.14 benchmark backtest on Binance public BTCUSDT spot klines.

Uses the same strategy/analyzer and conservative execution simulation as the Bitpin
timing-safe backtest. This is a cross-market robustness check, not a direct
BTC_IRT-vs-BTCUSDT price comparison. No API key is required.
"""
import builtins
import sys
import time
from datetime import datetime, timezone
import pandas as pd
import requests
import backtest_v314_timing_safe as bt

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVALS = {"5m": 300, "15m": 900, "1h": 3600}
RETRYABLE = {429, 418, 500, 502, 503, 504}


def fetch_binance_history(resolution, start_ts, end_ts, limit=1000):
    step = INTERVALS[resolution]
    cursor = int(start_ts)
    rows = []
    session = requests.Session()
    while cursor < int(end_ts):
        params = {
            "symbol": SYMBOL,
            "interval": resolution,
            "startTime": cursor * 1000,
            "endTime": int(end_ts) * 1000,
            "limit": min(limit, 1000),
        }
        data = None
        for attempt in range(5):
            try:
                response = session.get(BINANCE_URL, params=params, timeout=30)
                if response.status_code in RETRYABLE:
                    if response.status_code == 418:
                        raise RuntimeError("Binance public API rate/IP restriction (HTTP 418)")
                    response.raise_for_status()
                response.raise_for_status()
                data = response.json()
                break
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if attempt == 4:
                    raise RuntimeError(f"Binance {resolution} download failed: {exc}") from exc
                wait = min(12, 1.0 * (2 ** attempt))
                print(f"RETRY Binance {resolution} {attempt + 1}/5 after {wait:.1f}s: {exc}", flush=True)
                time.sleep(wait)
        if not data:
            break
        for k in data:
            try:
                ts = int(k[0] // 1000)
                # Keep only fully closed candles; exclude the currently forming candle.
                if ts >= int(start_ts) and ts + step <= int(end_ts):
                    rows.append({
                        "timestamp": ts,
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
            except (TypeError, ValueError, IndexError):
                continue
        newest = int(data[-1][0] // 1000)
        next_cursor = newest + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.10)
    if not rows:
        raise RuntimeError(f"No closed Binance {resolution} candles returned")
    frame = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    frame = frame[(frame.timestamp >= int(start_ts)) & (frame.timestamp + step <= int(end_ts))]
    return frame.reset_index(drop=True)


def main():
    # Keep the exact same V3.14 route and execution logic, swapping only the public OHLCV source.
    bt.fetch_history = fetch_binance_history
    original_print = builtins.print

    def labeled_print(*args, **kwargs):
        if args == ("SOURCE       api.bitpin.org",):
            args = ("SOURCE       Binance public spot API",)
        elif args == ("SYMBOL       BTC_IRT",):
            args = ("SYMBOL       BTCUSDT",)
        original_print(*args, **kwargs)

    builtins.print = labeled_print
    try:
        bt.main()
    finally:
        builtins.print = original_print


if __name__ == "__main__":
    main()
