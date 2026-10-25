"""Post-process timing-safe BTC_IRT trades under explicit round-trip cost scenarios.

Costs are approximated as a fraction of entry notional and converted to R using
entry-to-stop distance. This is a sensitivity model, not an exchange fee quote.
"""
import argparse
import pandas as pd

SCENARIOS_BPS = [0, 5, 10, 20]


def metrics(values):
    values = pd.Series(values).dropna().astype(float)
    if values.empty:
        return {"trades": 0, "net_R": 0.0, "mean_R": 0.0, "win_rate_pct": 0.0,
                "profit_factor": 0.0, "max_drawdown_R": 0.0}
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()
    equity = pd.concat([pd.Series([0.0]), values.reset_index(drop=True).cumsum()], ignore_index=True)
    drawdown = equity - equity.cummax()
    return {
        "trades": int(len(values)),
        "net_R": float(values.sum()),
        "mean_R": float(values.mean()),
        "win_rate_pct": float((values > 0).mean() * 100),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0,
        "max_drawdown_R": float(drawdown.min()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--trades-output", default="data/v314_cost_sensitivity_trades.csv")
    p.add_argument("--summary-output", default="data/v314_cost_sensitivity_summary.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    required = {"entry", "stop_loss", "realized_r", "route"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df["realized_r"] = pd.to_numeric(df["realized_r"], errors="coerce")
    df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
    df["stop_loss"] = pd.to_numeric(df["stop_loss"], errors="coerce")
    df = df[df["realized_r"].notna() & (df["entry"] > df["stop_loss"])].copy()
    if df.empty:
        raise SystemExit("No closed trades with valid entry/stop levels")

    df["risk_price"] = df["entry"] - df["stop_loss"]
    summary = []
    for bps in SCENARIOS_BPS:
        # Combined round-trip fee + slippage as a fraction of entry notional.
        df[f"cost_R_{bps}bps"] = (bps / 10000.0) * df["entry"] / df["risk_price"]
        df[f"net_R_{bps}bps"] = df["realized_r"] - df[f"cost_R_{bps}bps"]
        overall = metrics(df[f"net_R_{bps}bps"])
        summary.append({"round_trip_cost_bps": bps, "gross_R": float(df["realized_r"].sum()),
                        "total_cost_R": float(df[f"cost_R_{bps}bps"].sum()), **overall})
        for route, grp in df.groupby("route"):
            route_m = metrics(grp[f"net_R_{bps}bps"])
            summary.append({"round_trip_cost_bps": bps, "route": route,
                            "gross_R": float(grp["realized_r"].sum()),
                            "total_cost_R": float(grp[f"cost_R_{bps}bps"].sum()), **route_m})

    df.to_csv(args.trades_output, index=False)
    pd.DataFrame(summary).to_csv(args.summary_output, index=False)

    print("COST SENSITIVITY (round-trip cost = combined fee + slippage)")
    print("Open trades without realized R are excluded from performance statistics.")
    for bps in SCENARIOS_BPS:
        m = metrics(df[f"net_R_{bps}bps"])
        cost = df[f"cost_R_{bps}bps"].sum()
        print(f"{bps:>2} bps | trades={m['trades']} | gross={df['realized_r'].sum():.2f}R "
              f"| costs={cost:.2f}R | net={m['net_R']:.2f}R | mean={m['mean_R']:.3f}R "
              f"| win={m['win_rate_pct']:.1f}% | PF={m['profit_factor']:.2f} "
              f"| maxDD={m['max_drawdown_R']:.2f}R")
    print(f"TRADE_DETAILS={args.trades_output}")
    print(f"SCENARIO_SUMMARY={args.summary_output}")


if __name__ == "__main__":
    main()
