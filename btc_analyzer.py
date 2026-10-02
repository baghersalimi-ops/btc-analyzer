import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.bitpin.org"
SYMBOL = "BTC_IRT"

TIMEFRAMES = {
    "1m": {"seconds": 60, "weight": 1},
    "5m": {"seconds": 300, "weight": 2},
    "15m": {"seconds": 900, "weight": 3},
    "1h": {"seconds": 3600, "weight": 4},
}

def get_candles(resolution, limit=500):
    now = int(time.time())
    start = now - limit * TIMEFRAMES[resolution]["seconds"]

    url = f"{BASE_URL}/api/v1/mkt/candles/"
    params = {
        "symbol": SYMBOL,
        "from": start,
        "to": now,
        "res": resolution
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        data = data.get("results", data.get("data", []))

    rows = []

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

            rows.append([
                ts, float(op), float(hi), float(lo),
                float(cl), float(vol)
            ])
        except:
            continue

    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume"]
    )

    if df.empty:
        raise ValueError(f"No candle data for {resolution}")

    return (
        df.drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )


def calculate_indicators(df):

    df["ema20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200, adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    df.loc[
        (avg_gain > 0) & (avg_loss == 0), "rsi"
    ] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss > 0), "rsi"
    ] = 0

    df.loc[
        (avg_gain == 0) & (avg_loss == 0), "rsi"
    ] = 50

    # MACD
    ema12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    # ATR
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    # Volume
    df["volume_avg20"] = df["volume"].rolling(20).mean()

    return df


def market_levels(df):
    price = df.iloc[-1]["close"]

    # Exclude current candle
    recent = df.iloc[-51:-1]

    resistance = recent["high"].max()
    support = recent["low"].min()

    distance_resistance = ((resistance - price) / price) * 100
    distance_support = ((price - support) / price) * 100

    return {
        "support": support,
        "resistance": resistance,
        "distance_support": distance_support,
        "distance_resistance": distance_resistance
    }

def detect_price_action(df):
    price = df.iloc[-1]["close"]

    # Previous candles only — exclude current candle
    previous = df.iloc[-21:-1]

    previous_high = previous["high"].max()
    previous_low = previous["low"].min()

    # Real breakout detection
    breakout_up = price > previous_high
    breakout_down = price < previous_low

    # Distance from previous support/resistance
    distance_high = abs(price - previous_high) / price
    distance_low = abs(price - previous_low) / price

    near_resistance = distance_high < 0.003
    near_support = distance_low < 0.003

    if breakout_up:
        state = "BREAKOUT UP"
    elif breakout_down:
        state = "BREAKOUT DOWN"
    elif near_resistance:
        state = "NEAR RESISTANCE"
    elif near_support:
        state = "NEAR SUPPORT"
    else:
        state = "RANGE"

    return {
        "state": state,
        "previous_high": previous_high,
        "previous_low": previous_low,
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
        "near_support": near_support,
        "near_resistance": near_resistance
    }

def next_resistance(df, current_resistance):
    price = df.iloc[-1]["close"]
    candles = df.iloc[:-1]

    # Find structural swing highs above current price
    swing_highs = []

    for i in range(3, len(candles) - 3):
        h = candles.iloc[i]["high"]

        is_swing_high = (
            h > candles.iloc[i-1]["high"]
            and h > candles.iloc[i-2]["high"]
            and h > candles.iloc[i-3]["high"]
            and h >= candles.iloc[i+1]["high"]
            and h >= candles.iloc[i+2]["high"]
            and h >= candles.iloc[i+3]["high"]
        )

        threshold = max(price, current_resistance)
        if is_swing_high and h > threshold:
            swing_highs.append(h)

    if swing_highs:
        return min(swing_highs)

    return None


def structural_tp(price, atr, current_resistance, next_resistance, mode):
    atr_tp1 = price + 1.5 * atr
    atr_tp2 = price + 3.0 * atr

    if mode == "PRE-BREAKOUT":
        structural_resistance = current_resistance
    else:
        structural_resistance = next_resistance

    if structural_resistance is None:
        return {
            "tp1": atr_tp1,
            "tp2": atr_tp2,
            "tp1_source": "ATR",
            "tp2_source": "ATR"
        }

    structural_tp = structural_resistance * 0.997

    if structural_tp <= price:
        return {
            "tp1": price,
            "tp2": price,
            "tp1_source": "BLOCKED",
            "tp2_source": "BLOCKED"
        }

    tp1 = min(atr_tp1, structural_tp)
    tp2 = min(atr_tp2, structural_tp)

    return {
        "tp1": tp1,
        "tp2": tp2,
        "tp1_source": "STRUCTURAL" if tp1 != atr_tp1 else "ATR",
        "tp2_source": "STRUCTURAL" if tp2 != atr_tp2 else "ATR"
    }

def smart_tp_mode(breakout_status):
    if breakout_status == "VALID BREAKOUT":
        return "POST-BREAKOUT"
    return "PRE-BREAKOUT"


def analyze_timeframe(df, tf):

    row = df.iloc[-1]

    price = row["close"]
    ema20 = row["ema20"]
    ema50 = row["ema50"]
    ema200 = row["ema200"]
    rsi = row["rsi"]
    macd = row["macd"]
    macd_signal = row["macd_signal"]
    atr = row["atr"]

    long_score = 0
    short_score = 0

    if ema20 > ema50 > ema200:
        trend = "BULLISH"
        long_score += 2
    elif ema20 < ema50 < ema200:
        trend = "BEARISH"
        short_score += 2
    else:
        trend = "MIXED"

    if price > ema20:
        long_score += 1
    else:
        short_score += 1

    if pd.isna(rsi):
        rsi_state = "N/A"
    elif rsi > 70:
        rsi_state = "EXTENDED"
    elif rsi >= 55:
        rsi_state = "BULLISH"
        long_score += 1
    elif rsi < 45:
        rsi_state = "BEARISH"
        short_score += 1
    else:
        rsi_state = "NEUTRAL"

    if macd > macd_signal:
        long_score += 1
    else:
        short_score += 1

    atr_pct = atr / price * 100

    if atr_pct < 0.5:
        volatility = "LOW"
    elif atr_pct < 1.5:
        volatility = "NORMAL"
    else:
        volatility = "HIGH"

    levels = market_levels(df)
    next_res = next_resistance(df, levels["resistance"])
    action = detect_price_action(df)
    breakout = breakout_confirmation(df)
    momentum = retest_momentum(df)
    retest = breakout_retest(df)

    # Pullback detection
    pullback = "NO"

    if (
        price < ema20
        and trend == "BULLISH"
        and rsi < 50
    ):
        pullback = "POSSIBLE BULLISH PULLBACK"

    elif (
        price > ema20
        and trend == "BEARISH"
        and rsi > 50
    ):
        pullback = "POSSIBLE BEARISH PULLBACK"

    # Entry quality
    if (
        price > ema20
        and macd > macd_signal
        and 50 <= rsi <= 70
    ):
        entry_quality = "GOOD"
    elif price > ema20 and macd > macd_signal:
        entry_quality = "MEDIUM"
    else:
        entry_quality = "POOR"

    # ATR levels
    long_sl = price - 1.5 * atr
    long_tp1 = price + 1.5 * atr
    long_tp2 = price + 3.0 * atr

    short_sl = price + 1.5 * atr
    short_tp1 = price - 1.5 * atr
    short_tp2 = price - 3.0 * atr
    smart_tp_mode_value = smart_tp_mode(breakout["status"])
    smart_tp = structural_tp(price, atr, levels["resistance"], next_res, smart_tp_mode_value)
    smart_tp1 = smart_tp["tp1"]
    smart_tp2 = smart_tp["tp2"]
    smart_tp1_source = smart_tp["tp1_source"]
    smart_tp2_source = smart_tp["tp2_source"]

    tp1_resistance_distance = (levels["resistance"] - long_tp1) / long_tp1 * 100
    tp2_resistance_distance = (levels["resistance"] - long_tp2) / long_tp2 * 100
    next_resistance_distance = ((next_res - price) / price * 100) if next_res is not None else None

    long_tp1_status = "BLOCKED" if tp1_resistance_distance < 0 else ("NEAR RESISTANCE" if tp1_resistance_distance < 0.30 else "CLEAR")
    long_tp2_status = "BLOCKED" if tp2_resistance_distance < 0 else ("NEAR RESISTANCE" if tp2_resistance_distance < 0.30 else "CLEAR")
    next_tp1_status = "NO RESISTANCE" if next_res is None else ("BLOCKED" if next_res < long_tp1 else ("NEAR RESISTANCE" if ((next_res - long_tp1) / long_tp1 * 100) < 0.30 else "CLEAR"))
    next_tp2_status = "NO RESISTANCE" if next_res is None else ("BLOCKED" if next_res < long_tp2 else ("NEAR RESISTANCE" if ((next_res - long_tp2) / long_tp2 * 100) < 0.30 else "CLEAR"))
    return {
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "rsi_state": rsi_state,
        "macd": macd,
        "macd_signal": macd_signal,
        "atr": atr,
        "atr_pct": atr_pct,
        "trend": trend,
        "volatility": volatility,
        "breakout": breakout["status"],
        "retest_momentum": momentum["status"],
        "retest": retest["status"],
        "breakout_strength": breakout["strength"],
        "breakout_distance": breakout["breakout_distance"],
        "breakout_volume_ratio": breakout["volume_ratio"],
        "long_score": long_score,
        "short_score": short_score,
        "support": levels["support"],
        "resistance": levels["resistance"],
        "next_resistance": next_res,
        "distance_support": levels["distance_support"],
        "distance_resistance": levels["distance_resistance"],
        "price_action": action["state"],
        "pullback": pullback,
        "entry_quality": entry_quality,
        "long_sl": long_sl,
        "long_tp1": long_tp1,        
"long_tp2": long_tp2,
        "smart_tp_mode": smart_tp_mode_value,
        "smart_tp1": smart_tp1,
        "smart_tp2": smart_tp2,
        "smart_tp1_source": smart_tp1_source,
        "smart_tp2_source": smart_tp2_source,
        
        "long_rr_tp1": (smart_tp1 - price) / (price - long_sl) if price > long_sl and smart_tp1 > price else 0,
        "long_rr_tp2": (smart_tp2 - price) / (price - long_sl) if price > long_sl and smart_tp2 > price else 0,
        "short_sl": short_sl,
        "short_tp1": short_tp1,
        "short_tp2": short_tp2,
        "long_tp1_status": long_tp1_status,
        "long_tp2_status": long_tp2_status,
        "next_tp1_status": next_tp1_status,
        "next_tp2_status": next_tp2_status,
    }


def breakout_confirmation(df):
    if len(df) < 25:
        return {
            "status": "NO DATA",
            "strength": 0
        }

    current = df.iloc[-1]
    previous = df.iloc[-21:-1]

    previous_resistance = previous["high"].max()
    previous_volume = previous["volume"].iloc[-10:].mean()

    close = current["close"]
    high = current["high"]
    volume = current["volume"]

    breakout_distance = (
        (close - previous_resistance) / previous_resistance
    ) * 100

    volume_ratio = (
        volume / previous_volume
        if previous_volume > 0 else 0
    )

    # V2.9 confirmation rules
    close_above_resistance = close > previous_resistance
    distance_confirmed = breakout_distance >= 0.10
    volume_confirmed = volume_ratio >= 1.20

    if close_above_resistance and distance_confirmed and volume_confirmed:
        status = "VALID BREAKOUT"
        strength = 3

    elif close_above_resistance and distance_confirmed:
        status = "BREAKOUT - LOW VOLUME"
        strength = 1

    elif high > previous_resistance:
        status = "RESISTANCE TEST"
        strength = 0

    else:
        status = "NO BREAKOUT"
        strength = 0

    return {
        "status": status,
        "strength": strength,
        "resistance": previous_resistance,
        "breakout_distance": breakout_distance,
        "volume_ratio": volume_ratio
    }
def retest_momentum(df):
    if len(df) < 5:
        return {
            "status": "NO DATA",
            "histogram": None,
            "previous_histogram": None
        }

    current = df.iloc[-1]
    previous = df.iloc[-2]

    histogram = current["macd"] - current["macd_signal"]
    previous_histogram = previous["macd"] - previous["macd_signal"]

    # Momentum is bullish when histogram is recovering
    if histogram > 0 and histogram > previous_histogram:
        status = "MOMENTUM CONFIRMED"

    elif histogram <= 0 and histogram > previous_histogram:
        status = "MOMENTUM RECOVERING"

    elif histogram < previous_histogram:
        status = "MOMENTUM WEAKENING"

    else:
        status = "MOMENTUM FLAT"

    return {
        "status": status,
        "histogram": histogram,
        "previous_histogram": previous_histogram
    }

def breakout_retest(df):
    if len(df) < 30:
        return {
            "status": "NO DATA",
            "resistance": None,
            "distance": None
        }

    current = df.iloc[-1]
    candles = df.iloc[:-1]

    # Look for a recent valid breakout
    breakout_index = None
    breakout_resistance = None

    start = max(20, len(candles) - 15)

    for i in range(start, len(candles)):
        previous = candles.iloc[max(0, i-20):i]

        if len(previous) < 10:
            continue

        resistance = previous["high"].max()
        close = candles.iloc[i]["close"]
        volume = candles.iloc[i]["volume"]

        previous_volume = previous["volume"].iloc[-10:].mean()

        if previous_volume <= 0:
            continue

        breakout_distance = (
            (close - resistance) / resistance
        ) * 100

        volume_ratio = volume / previous_volume

        if (
            close > resistance
            and breakout_distance >= 0.10
            and volume_ratio >= 1.20
        ):
            breakout_index = i
            breakout_resistance = resistance

    if breakout_index is None:
        return {
            "status": "NO RECENT BREAKOUT",
            "resistance": None,
            "distance": None
        }

    # Only candles after the breakout are relevant
    after_breakout = candles.iloc[breakout_index + 1:]

    if len(after_breakout) == 0:
        return {
            "status": "WAITING FOR RETEST",
            "resistance": breakout_resistance,
            "distance": (
                (current["close"] - breakout_resistance)
                / breakout_resistance
            ) * 100
        }

    # Retest zone: within 0.30% of the old resistance
    distance = (
        (current["close"] - breakout_resistance)
        / breakout_resistance
    ) * 100

    touched = (
        after_breakout["low"].min()
        <= breakout_resistance * 1.003
    )

    held = (
        current["close"] >= breakout_resistance
    )

    failed = (
        current["close"] < breakout_resistance
    )

    if touched and held:
        status = "RETEST CONFIRMED"
    elif touched and failed:
        status = "RETEST FAILED"
    elif distance > 0.30:
        status = "WAITING FOR RETEST"
    else:
        status = "APPROACHING RETEST"

    return {
        "status": status,
        "resistance": breakout_resistance,
        "distance": distance
    }

def pullback_confirmation(results):
    tf5 = results["5m"]
    tf15 = results["15m"]
    tf1h = results["1h"]

    confirmations = 0
    reasons = []

    # Resistance filter
    near_resistance = (
        tf5["distance_resistance"] < 0.30
        and tf5["price_action"] != "BREAKOUT UP"
    )

    tp1_status = (
        tf5["next_tp1_status"]
        if tf5["breakout"] == "VALID BREAKOUT"
        else tf5["smart_tp1_source"]
    )

    tp1_blocked = (
        tp1_status == "BLOCKED"
    )

    if near_resistance:
        return {
            "status": "WAIT - NEAR RESISTANCE",
            "confirmations": 0,
            "reasons": ["5m NEAR RESISTANCE"]
        }

    if tp1_blocked:
        return {
            "status": "WAIT - TP1 BLOCKED",
            "confirmations": 0,
            "reasons": ["5m TP1 BLOCKED BY RESISTANCE"]
        }

    if tf1h["trend"] == "BULLISH":
        confirmations += 1
        reasons.append("1h BULLISH")

    if tf15["price_action"] == "NEAR SUPPORT":
        confirmations += 1
        reasons.append("15m NEAR SUPPORT")

    if tf15["pullback"] == "POSSIBLE BULLISH PULLBACK":
        confirmations += 1
        reasons.append("15m BULLISH PULLBACK")

    if tf5["rsi"] > 50:
        confirmations += 1
        reasons.append("5m RSI RECOVERY")

    if tf5["macd"] > tf5["macd_signal"]:
        confirmations += 1
        reasons.append("5m MACD BULLISH")

    if tf5["price"] > tf5["ema20"]:
        confirmations += 1
        reasons.append("5m ABOVE EMA20")

    if confirmations >= 5:
        status = "LONG CONFIRMED"
    elif confirmations >= 3:
        status = "WATCH BULLISH PULLBACK"
    else:
        status = "WAIT"

    return {
        "status": status,
        "confirmations": confirmations,
        "reasons": reasons
    }

def multi_tf_breakout_confirmation(results):
    tf5 = results["5m"]
    tf15 = results["15m"]
    tf1h = results["1h"]

    if tf5["breakout"] != "VALID BREAKOUT":
        return {
            "status": "NO BREAKOUT CONFIRMATION",
            "confirmations": 0,
            "reasons": ["5m NO VALID BREAKOUT"]
        }

    confirmations = 1
    reasons = ["5m VALID BREAKOUT"]

    if tf15["trend"] == "BULLISH":
        confirmations += 1
        reasons.append("15m BULLISH")

    if tf1h["trend"] == "BULLISH":
        confirmations += 1
        reasons.append("1h BULLISH")

    if confirmations == 3:
        status = "MULTI-TF BREAKOUT CONFIRMED"
    elif confirmations == 2:
        status = "BREAKOUT PARTIALLY CONFIRMED"
    else:
        status = "BREAKOUT - HIGHER TF NOT CONFIRMED"

    return {
        "status": status,
        "confirmations": confirmations,
        "reasons": reasons
    }

def final_decision(results):
    confirmation = pullback_confirmation(results)
    breakout_mtf = multi_tf_breakout_confirmation(results)

    force_wait = confirmation["status"] in ["WAIT - NEAR RESISTANCE", "WAIT - TP1 BLOCKED"]

    long_score = 0
    short_score = 0

    for tf, result in results.items():

        weight = TIMEFRAMES[tf]["weight"]

        long_score += result["long_score"] * weight
        short_score += result["short_score"] * weight

    max_score = 5 * sum(
        x["weight"] for x in TIMEFRAMES.values()
    )

    long_power = long_score / max_score * 100
    short_power = short_score / max_score * 100

    t1 = results["1h"]["trend"]
    t15 = results["15m"]["trend"]

    if t1 == "BULLISH" and t15 == "BULLISH":
        regime = "BULLISH"
    elif t1 == "BEARISH" and t15 == "BEARISH":
        regime = "BEARISH"
    else:
        regime = "RANGE / MIXED"

    tf5 = results["5m"]

    long_blocked_by_resistance = (
        tf5["distance_resistance"] < 0.30
        and tf5["price_action"] != "BREAKOUT UP"
    )

    short_blocked_by_support = (
        tf5["distance_support"] < 0.30
        and tf5["price_action"] != "BREAKOUT DOWN"
    )

    valid_breakout = (
        tf5["breakout"] == "VALID BREAKOUT"
    )

    if force_wait and not valid_breakout:
        signal = "WAIT"
    elif (
        regime == "BULLISH"
        and long_power >= 60
        and tf5["entry_quality"] in ["GOOD", "MEDIUM"]
        and not long_blocked_by_resistance
    ):
        signal = "LONG SETUP"

    elif (
        regime == "BEARISH"
        and short_power >= 60
        and not short_blocked_by_support
    ):
        signal = "SHORT SETUP"

    elif (
        regime == "BEARISH"
        and short_power >= 60
    ):
        signal = "SHORT SETUP"

    elif (
        regime == "BULLISH"
        and long_power >= 50
    ):
        signal = "WATCH PULLBACK"

    else:
        signal = "WAIT"

    return {
        "regime": regime,
        "long_score": long_score,
        "short_score": short_score,
        "long_power": long_power,
        "short_power": short_power,
        "signal": signal,
        "confirmation_status": confirmation["status"],
        "confirmations": confirmation["confirmations"],
        "breakout_mtf": breakout_mtf["status"],
        "breakout_mtf_confirmations": breakout_mtf["confirmations"]
    }



def trade_setup_engine(results):
    tf5 = results["5m"]
    tf15 = results["15m"]
    tf1h = results["1h"]

    long_checks = []
    short_checks = []
    long_ready = True
    short_ready = True

    if tf1h["trend"] != "BULLISH":
        long_ready = False
        long_checks.append("1h NOT BULLISH")
    else:
        long_checks.append("1h BULLISH")

    if tf15["trend"] != "BULLISH":
        long_ready = False
        long_checks.append("15m NOT BULLISH")
    else:
        long_checks.append("15m BULLISH")

    if tf5["breakout"] != "VALID BREAKOUT":
        long_ready = False
        long_checks.append("5m NO VALID BREAKOUT")
    else:
        long_checks.append("5m VALID BREAKOUT")

    if tf5["rsi"] >= 75:
        long_ready = False
        long_checks.append("5m RSI TOO HIGH")
    else:
        long_checks.append("5m RSI OK")

    if tf5["next_tp1_status"] == "BLOCKED":
        long_ready = False
        long_checks.append("LONG TP1 BLOCKED")
    else:
        long_checks.append("LONG STRUCTURAL SPACE OK")

    if tf5["entry_quality"] not in ["GOOD", "MEDIUM"]:
        long_ready = False
        long_checks.append("5m ENTRY QUALITY POOR")
    else:
        long_checks.append("5m ENTRY QUALITY OK")

    if tf1h["trend"] != "BEARISH":
        short_ready = False
        short_checks.append("1h NOT BEARISH")
    else:
        short_checks.append("1h BEARISH")

    if tf15["trend"] != "BEARISH":
        short_ready = False
        short_checks.append("15m NOT BEARISH")
    else:
        short_checks.append("15m BEARISH")

    if tf5["price_action"] != "BREAKOUT DOWN":
        short_ready = False
        short_checks.append("5m NO DOWNSIDE BREAK")
    else:
        short_checks.append("5m DOWNSIDE BREAK")

    if tf5["rsi"] >= 45:
        short_ready = False
        short_checks.append("5m RSI NOT WEAK ENOUGH")
    else:
        short_checks.append("5m RSI OK")

    if tf5["distance_support"] < 0.30 and tf5["price_action"] != "BREAKOUT DOWN":
        short_ready = False
        short_checks.append("SHORT SUPPORT TOO CLOSE")
    else:
        short_checks.append("SHORT STRUCTURAL SPACE OK")

    if long_ready:
        plan = "LONG READY"
        trigger = "5m VALID BREAKOUT + 15m/1h BULLISH"
    elif short_ready:
        plan = "SHORT READY"
        trigger = "5m DOWNSIDE BREAK + 15m/1h BEARISH"
    elif tf1h["trend"] == "BULLISH" and tf15["trend"] in ["BULLISH", "MIXED"]:
        plan = "LONG WATCH"
        trigger = "WAIT FOR 5m VALID BREAKOUT"
    elif tf1h["trend"] == "BEARISH" and tf15["trend"] in ["BEARISH", "MIXED"]:
        plan = "SHORT WATCH"
        trigger = "WAIT FOR 5m DOWNSIDE BREAK"
    else:
        plan = "NO SETUP"
        trigger = "WAIT FOR DIRECTION + ENTRY CONFIRMATION"

    return {
        "plan": plan,
        "trigger": trigger,
        "long_checks": long_checks,
        "short_checks": short_checks
    }


def smart_entry_engine(results):
    tf5 = results["5m"]
    tf15 = results["15m"]
    tf1h = results["1h"]

    reasons = []

    # 1. Higher timeframe trend
    if tf1h["trend"] != "BULLISH":
        reasons.append("1h NOT BULLISH")

    if tf15["trend"] != "BULLISH":
        reasons.append("15m NOT BULLISH")

    # 2. 5m valid breakout
    if tf5["breakout"] != "VALID BREAKOUT":
        reasons.append("5m NO VALID BREAKOUT")

    # 3. MTF confirmation
    mtf = multi_tf_breakout_confirmation(results)

    if mtf["status"] != "MULTI-TF BREAKOUT CONFIRMED":
        reasons.append("MTF BREAKOUT NOT CONFIRMED")

    # 4. RSI filter
    if tf5["rsi"] >= 75:
        reasons.append("5m RSI TOO HIGH")

    # 5. Structural resistance filter
    tp1_status = (
        tf5["next_tp1_status"]
        if tf5["breakout"] == "VALID BREAKOUT"
        else tf5["smart_tp1_source"]
    )

    if tp1_status == "BLOCKED":
        reasons.append("TP1 BLOCKED BY RESISTANCE")

    # 6. Entry quality
    if tf5["entry_quality"] not in ["GOOD", "MEDIUM"]:
        reasons.append("ENTRY QUALITY POOR")

    # Final decision
    if not reasons:
        status = "ENTRY READY"
    elif tf5["breakout"] == "VALID BREAKOUT":
        status = "BREAKOUT WATCH"
    else:
        status = "WAIT"

    return {
        "status": status,
        "reasons": reasons
    }



def entry_state_engine(results):
    tf5 = results["5m"]
    retest_confirmed = tf5["retest"] == "RETEST CONFIRMED"
    retest_momentum_confirmed = tf5["retest_momentum"] == "MOMENTUM CONFIRMED"
    tf15 = results["15m"]
    tf1h = results["1h"]

    # Core market conditions
    bullish_htf = (
        tf1h["trend"] == "BULLISH"
        and tf15["trend"] == "BULLISH"
    )

    valid_breakout = (
        tf5["breakout"] == "VALID BREAKOUT"
    )

    mtf_confirmed = (
        multi_tf_breakout_confirmation(results)["status"]
        == "MULTI-TF BREAKOUT CONFIRMED"
    )

    rsi_ok = tf5["rsi"] < 75

    tp1_status = (
        tf5["next_tp1_status"]
        if tf5["breakout"] == "VALID BREAKOUT"
        else tf5["smart_tp1_source"]
    )

    tp1_clear = (
        tp1_status != "BLOCKED"
    )

    entry_quality_ok = (
        tf5["entry_quality"] in ["GOOD", "MEDIUM"]
    )

    distance_resistance = tf5["distance_resistance"]
    # 1. ENTRY READY
    if (
        bullish_htf
        and (
            (
                valid_breakout
                and mtf_confirmed
            )
            or (
                retest_confirmed
                and retest_momentum_confirmed
            )
        )
        and rsi_ok
        and tp1_clear
        and entry_quality_ok
    ):
        state = "ENTRY READY"

        if retest_confirmed:
            trigger = "RETEST CONFIRMED + HIGHER TF BULLISH"
        else:
            trigger = "5m VALID BREAKOUT + MTF CONFIRMATION"

    # 2. BREAKOUT WATCH
    elif bullish_htf and valid_breakout:
        state = "BREAKOUT WATCH"
        trigger = "5m VALID BREAKOUT"


    # 3. APPROACHING BREAKOUT
    elif (
        bullish_htf
        and not valid_breakout
        and distance_resistance <= 1.00
        and rsi_ok
    ):
        state = "APPROACHING BREAKOUT"
        trigger = "WAIT FOR 5m CLOSE ABOVE RESISTANCE"

    # 4. WAIT
    else:
        state = "WAIT"
        trigger = "NO VALID ENTRY SETUP"

    return {
        "state": state,
        "trigger": trigger,
        "distance_resistance": distance_resistance
    }


def main():

    print("\n" + "=" * 60)
    print("BTC/IRT SMART ANALYZER V3.6")
    print("=" * 60)

    results = {}

    for tf in TIMEFRAMES:

        try:

            df = get_candles(tf)
            df = calculate_indicators(df)

            result = analyze_timeframe(df, tf)
            results[tf] = result

            print("\n" + "-" * 60)
            print(tf)

            print(f"Price          {result['price']:,.0f}")
            print(f"EMA20          {result['ema20']:,.0f}")
            print(f"EMA50          {result['ema50']:,.0f}")
            print(f"EMA200         {result['ema200']:,.0f}")

            print(
                f"RSI14          {result['rsi']:.2f}"
                f" [{result['rsi_state']}]"
            )

            print(f"MACD           {result['macd']:.2f}")
            print(f"MACD Signal    {result['macd_signal']:.2f}")
            print(f"ATR14          {result['atr']:,.0f}")
            print(f"ATR %          {result['atr_pct']:.2f}%")

            print(f"Trend          {result['trend']}")
            print(f"Volatility     {result['volatility']}")

            print(f"Support        {result['support']:,.0f}")
            print(f"Resistance     {result['resistance']:,.0f}")
            print(f"Next Resistance {result['next_resistance']:,.0f}" if result["next_resistance"] is not None else "Next Resistance NONE")

            print(
                f"Dist. Support  "
                f"{result['distance_support']:.2f}%"
            )

            print(
                f"Dist. Resist.  "
                f"{result['distance_resistance']:.2f}%"
            )

            print(f"Price Action   {result['price_action']}")
            print("Retest         " + result["retest"])
            print("Momentum       " + result["retest_momentum"])
            print(f"Breakout       {result['breakout']}")
            print(f"Breakout Dist. {result['breakout_distance']:.2f}%")
            print(f"Breakout Vol.  {result['breakout_volume_ratio']:.2f}x")
            print(f"Breakout Str.  {result['breakout_strength']}/3")
            print(f"Pullback       {result['pullback']}")
            print(f"Entry Quality  {result['entry_quality']}")

            print(f"LONG Score     {result['long_score']}/5")
            print(f"SHORT Score    {result['short_score']}/5")

            print("\nATR LEVELS:")

            print(
                f"Long SL        {result['long_sl']:,.0f}"
            )

            print(
                f"Long TP1       {result['long_tp1']:,.0f}"
            )

            print(
                f"Long TP2       {result['long_tp2']:,.0f}"
            )

            if result["smart_tp1_source"] == "BLOCKED":
                print("Smart TP1      BLOCKED BY RESISTANCE")
            else:
                print(
                    f"Smart TP1      {result['smart_tp1']:,.0f} [{result['smart_tp1_source']}]"
                )

            if result["smart_tp2_source"] == "BLOCKED":
                print("Smart TP2      BLOCKED BY RESISTANCE")
            else:
                print(
                    f"Smart TP2      {result['smart_tp2']:,.0f} [{result['smart_tp2_source']}]"
                )
            print(
                f"Smart TP Mode  {result['smart_tp_mode']}"
            )
            print(
                f"Short SL       {result['short_sl']:,.0f}"
            )

            print(
                f"Short TP1      {result['short_tp1']:,.0f}"
            )

            print(
                f"Short TP2      {result['short_tp2']:,.0f}"
            )
            print(f"Long R/R TP1    {"BLOCKED" if result["long_rr_tp1"] == 0 else f"{result["long_rr_tp1"]:.2f}"}")
            print(f"Long R/R TP2    {"BLOCKED" if result["long_rr_tp2"] == 0 else f"{result["long_rr_tp2"]:.2f}"}")
            print(f"TP1 vs Resistance {result['smart_tp1_source']}")
            print(f"TP2 vs Resistance {result['smart_tp2_source']}")
        except Exception as e:


            print(f"{tf}: ERROR -> {e}")
    if len(results) == 4:



        breakout_quality = breakout_quality_engine(results)
        final = final_decision(results)
        entry = smart_entry_engine(results)
        entry_state = entry_state_engine(results)
        trade_setup = trade_setup_engine(results)

        print("\n" + "=" * 60)
        print("FINAL")
        print("=" * 60)

        print(f"MARKET REGIME  {final['regime']}")
        print(f"LONG SCORE     {final['long_score']}")
        print(f"SHORT SCORE    {final['short_score']}")
        print(f"LONG POWER     {final['long_power']:.1f}%")
        print(f"SHORT POWER    {final['short_power']:.1f}%")
        print(f"SIGNAL         {final['signal']}")
        print(f"PULLBACK       {final['confirmation_status']}")
        print(f"CONFIRMATIONS  {final['confirmations']}/6")
        print(f"MTF BREAKOUT   {final['breakout_mtf']}")
        print(f"MTF CONFIRM    {final['breakout_mtf_confirmations']}/3")
        print(f"BREAKOUT {breakout_quality['state']} {breakout_quality['score']}/100 ({breakout_quality['quality']})")
        print(f"ENTRY ENGINE   {entry['status']}")
        print(f"ENTRY STATE    {entry_state['state']}")
        print(f"ENTRY TRIGGER  {entry_state['trigger']}")
        print("TRADE PLAN     " + trade_setup["plan"])
        print("PLAN TRIGGER   " + trade_setup["trigger"])
        print("LONG CHECK     " + " | ".join(trade_setup["long_checks"]))
        print("SHORT CHECK    " + " | ".join(trade_setup["short_checks"]))
        print(f"RESISTANCE DIST {entry_state['distance_resistance']:.2f}%")
        if entry["reasons"]:
            print("ENTRY REASONS  " + " | ".join(entry["reasons"]))

        print("=" * 60)


def breakout_quality_engine(results):
    tf5 = results["5m"]
    tf15 = results["15m"]
    tf1h = results["1h"]

    score = 0
    reasons = []

    distance = tf5["breakout_distance"]

    if distance >= 0.30:
        score += 25
        reasons.append("STRONG DISTANCE")
    elif distance >= 0.10:
        score += 15
        reasons.append("ACCEPTABLE DISTANCE")
    elif distance > 0:
        score += 5
        reasons.append("WEAK DISTANCE")
    else:
        reasons.append("NO BREAKOUT DISTANCE")

    volume_ratio = tf5["breakout_volume_ratio"]

    if volume_ratio >= 1.50:
        score += 25
        reasons.append("STRONG VOLUME")
    elif volume_ratio >= 1.20:
        score += 15
        reasons.append("ACCEPTABLE VOLUME")
    elif volume_ratio >= 1.00:
        score += 5
        reasons.append("WEAK VOLUME")
    else:
        reasons.append("LOW VOLUME")

    atr_pct = tf5["atr_pct"]

    if atr_pct >= 0.50:
        score += 15
        reasons.append("STRONG VOLATILITY")
    elif atr_pct >= 0.30:
        score += 10
        reasons.append("NORMAL VOLATILITY")
    else:
        score += 5
        reasons.append("LOW VOLATILITY")

    if tf15["trend"] == "BULLISH":
        score += 10
        reasons.append("15m BULLISH")

    if tf1h["trend"] == "BULLISH":
        score += 10
        reasons.append("1h BULLISH")

    resistance_distance = tf5["distance_resistance"]

    if resistance_distance >= 1.00:
        score += 15
        reasons.append("GOOD STRUCTURAL SPACE")
    elif resistance_distance >= 0.50:
        score += 10
        reasons.append("LIMITED STRUCTURAL SPACE")
    else:
        reasons.append("POOR STRUCTURAL SPACE")

    if score >= 80:
        quality = "EXCELLENT"
    elif score >= 65:
        quality = "GOOD"
    elif score >= 50:
        quality = "MEDIUM"
    else:
        quality = "POOR"

    return {
        "score": score,
        "quality": quality,
        "state": "QUALITY" if tf5["breakout"] == "VALID BREAKOUT" else "READINESS",
        "reasons": reasons
    }

if __name__ == "__main__":
    main()
