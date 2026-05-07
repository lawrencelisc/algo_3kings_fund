#!/usr/bin/env python3
"""Per-coin live trade analysis — run any time to verify EV alignment with backtest.

Usage:
    arch -x86_64 venv/bin/python daily_stats.py
    arch -x86_64 venv/bin/python daily_stats.py --since 2026-05-07

Kill criteria flagged automatically:
    1. n >= 30 trades AND win rate < 45%       — strategy mis-fit
    2. realized PnL < -5% × INITIAL_EQUITY_USDC — single-coin damage cap
    3. last 8 trades all losing                — extreme tail event
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hype_bot"))
from config import CFG  # noqa: E402

TRADES_CSV = ROOT / CFG.TRADES_CSV

KILL_MIN_TRADES_FOR_WR = 30
KILL_MIN_WR            = 0.45
KILL_PER_COIN_LOSS_PCT = 0.05    # of INITIAL_EQUITY_USDC
KILL_CONSEC_LOSSES     = 8


def load_trades(since: str | None) -> pd.DataFrame:
    if not TRADES_CSV.exists():
        sys.exit(f"trades.csv not found at {TRADES_CSV}")
    df = pd.read_csv(TRADES_CSV, parse_dates=["exit_ts_utc"])
    if since:
        df = df[df["exit_ts_utc"] >= pd.Timestamp(since, tz="UTC")]
    return df.reset_index(drop=True)


def per_coin_stats(df: pd.DataFrame, capital: float) -> pd.DataFrame:
    rows = []
    for coin, g in df.groupby("coin"):
        n         = len(g)
        wins      = (g["net_pnl"] > 0).sum()
        wr        = wins / n if n else 0.0
        cum_pnl   = g["net_pnl"].sum()
        avg_win   = g.loc[g["net_pnl"] > 0, "net_pnl"].mean() or 0.0
        avg_loss  = g.loc[g["net_pnl"] <= 0, "net_pnl"].mean() or 0.0
        wl_ratio  = abs(avg_win / avg_loss) if avg_loss else float("inf")
        # consecutive losses at the tail
        tail = g["net_pnl"].iloc[-KILL_CONSEC_LOSSES:] if n >= KILL_CONSEC_LOSSES else g["net_pnl"]
        consec_loss = (tail <= 0).all() and len(tail) == KILL_CONSEC_LOSSES

        flags = []
        if n >= KILL_MIN_TRADES_FOR_WR and wr < KILL_MIN_WR:
            flags.append(f"WR<{KILL_MIN_WR:.0%}")
        if cum_pnl < -KILL_PER_COIN_LOSS_PCT * capital:
            flags.append(f"PnL<-{KILL_PER_COIN_LOSS_PCT:.0%}cap")
        if consec_loss:
            flags.append(f"{KILL_CONSEC_LOSSES}LossStreak")

        rows.append({
            "coin": coin, "n": n, "wins": wins, "wr": wr,
            "avg_win": avg_win, "avg_loss": avg_loss, "wl": wl_ratio,
            "cum_pnl": cum_pnl, "fees": (g["entry_fee"]+g["exit_fee"]).sum(),
            "funding": g["funding_pnl"].sum(),
            "kill_flags": ",".join(flags) if flags else "OK",
        })
    return pd.DataFrame(rows).sort_values("cum_pnl", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="ISO date, e.g. 2026-05-07")
    args = ap.parse_args()

    df = load_trades(args.since)
    if df.empty:
        sys.exit("No trades in window.")

    capital = CFG.INITIAL_EQUITY_USDC
    stats = per_coin_stats(df, capital)

    print(f"\n=== Live Trade Stats — n={len(df)}  capital={capital:.0f} USDC ===")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n--- Aggregate ---")
    total_pnl   = df["net_pnl"].sum()
    total_fees  = (df["entry_fee"] + df["exit_fee"]).sum()
    total_fund  = df["funding_pnl"].sum()
    overall_wr  = (df["net_pnl"] > 0).mean()
    print(f"Total trades : {len(df)}")
    print(f"Win rate     : {overall_wr:.2%}")
    print(f"Net PnL      : {total_pnl:+.4f} USDC ({total_pnl/capital:+.2%} of capital)")
    print(f"Total fees   : {total_fees:.4f}")
    print(f"Total funding: {total_fund:+.4f}  (vs fees: {abs(total_fund/total_fees):.1%})")

    flagged = stats[stats["kill_flags"] != "OK"]
    if not flagged.empty:
        print("\n*** KILL FLAGS ***")
        for _, row in flagged.iterrows():
            print(f"  {row['coin']}: {row['kill_flags']} (n={row['n']}, "
                  f"wr={row['wr']:.0%}, cum={row['cum_pnl']:+.3f})")
    else:
        print("\nAll coins within tolerance.")


if __name__ == "__main__":
    main()
