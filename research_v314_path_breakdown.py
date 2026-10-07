"""
V3.14 Path-Level Research
Architecture-safe: research only, no production changes.

Reports Long/Short performance by Breakout/Retest/Pullback.
Important: current experimental SHORT generator only creates Breakout signals.
Therefore SHORT Retest/Pullback are reported as unsupported by the current
engine, not as zero-performance trading strategies.
"""
import argparse
import pandas as pd
from research_v314_dual_side import (
    prepare_history, build_signals, run_mode, summarize
)
from backtest_v314 import SL_VALUES, BASE_RESOLUTION

def path_counts(signals):
    rows=[]
    for side in ("LONG","SHORT"):
        for setup in ("Breakout","Retest","Pullback"):
            n=sum(1 for s in signals.values()
                  if s and s.get("side")==side and s.get("setup")==setup)
            rows.append((side,setup,n))
    return rows

def print_path_breakdown(trades, initial_equity, mode, sl):
    print(f"\nPATH BREAKDOWN | {mode} | SL={sl*100:.1f}%")
    print("-"*100)
    print(f"{'SIDE':<7}{'PATH':<11}{'TRADES':>8}{'WIN%':>9}{'RETURN%':>11}{'PF':>9}{'AVG R':>9}")
    for side in ("LONG","SHORT"):
        for setup in ("Breakout","Retest","Pullback"):
            subset=[t for t in trades if t.side==side and t.setup==setup]
            if subset:
                s=summarize(subset,initial_equity)
                pf="inf" if s["profit_factor"]==float("inf") else f"{s['profit_factor']:.2f}"
                print(f"{side:<7}{setup:<11}{s['trades']:>8}{s['win_rate']:>8.2f}%{s['return_pct']:>10.2f}%{pf:>9}{s['avg_r']:>9.3f}")
            else:
                print(f"{side:<7}{setup:<11}{0:>8}{'N/A':>9}{'N/A':>11}{'N/A':>9}{'N/A':>9}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--days",type=int,default=60)
    p.add_argument("--initial-equity",type=float,default=1_000_000)
    p.add_argument("--fee",type=float,default=0.0)
    p.add_argument("--slippage",type=float,default=0.0)
    a=p.parse_args()

    history=prepare_history(a.days)
    signals=build_signals(history)
    base=history[BASE_RESOLUTION]

    print("\nSIGNAL PATH COUNTS")
    print("-"*60)
    for side,setup,n in path_counts(signals):
        print(f"{side:<7} {setup:<11} signals={n}")

    print("\nENGINE LIMITATION")
    print("SHORT Retest/Pullback are NOT implemented in the current experimental SHORT generator.")
    print("They are therefore marked unsupported below; no artificial zero-performance conclusion is drawn.")

    for mode in ("smart_tp","rr"):
        for sl in SL_VALUES:
            trades=run_mode(base,signals,sl,mode,a.initial_equity,
                            dual_side=True,fee_rate=a.fee,
                            slippage_pct=a.slippage)
            print_path_breakdown(trades,a.initial_equity,mode,sl)

    print("\nDECISION GUIDE")
    print("1) Long path results are directly comparable across Breakout/Retest/Pullback.")
    print("2) Short Breakout is the only currently testable Short path.")
    print("3) Short Retest/Pullback require a separate symmetric Short research engine before activation.")

if __name__=="__main__":
    main()
