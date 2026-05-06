"""
HYPE D1 Extreme Percentile — Rolling Walk-Forward Analysis
===========================================================
Option B: IS = 4 months (120d), OOS = 2 months (60d), Rolling (no IS overlap)

Window structure (requires ~18 months of HYPE data from Nov 2024):
  Window 1: IS [Nov 2024 → Mar 2025]  OOS [Mar 2025 → Apr 2025]
  Window 2: IS [Mar 2025 → Jun 2025]  OOS [Jun 2025 → Aug 2025]
  Window 3: IS [Jun 2025 → Oct 2025]  OOS [Oct 2025 → Dec 2025]
  Window 4: IS [Oct 2025 → Feb 2026]  OOS [Feb 2026 → Apr 2026]
  Tail:     Apr 2026 → present        (extra OOS, never used in any window)

Usage:
  pip install requests pandas numpy matplotlib scipy
  python hype_rolling_wfa.py

The script will:
  1. Download full HYPE history from Hyperliquid API (Nov 2024 → now)
  2. Run rolling WFA with IS optimisation over threshold grid
  3. Save:  hype_18m.csv              full aligned dataset
            wfa_rolling_trades.csv    every OOS trade
            wfa_rolling_summary.csv   per-window stats
            wfa_rolling_results.png   9-panel analysis chart
"""

import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────
COIN            = "HYPE"
API_URL         = "https://api.hyperliquid.xyz/info"

IS_DAYS         = 120          # 4 months IS
OOS_DAYS        = 60           # 2 months OOS
WFA_START       = datetime(2024, 11, 1, tzinfo=timezone.utc)

THRESHOLD_GRID  = [0.02, 0.03, 0.05, 0.07, 0.10]
WINSORIZE_LO    = 0.01
WINSORIZE_HI    = 0.99

FEE_MAKER       = 0.0135       # maker fee % each side
FEE_TAKER       = 0.035        # taker fee % each side (for comparison)

OUTPUT_CSV_DATA    = "hype_18m.csv"
OUTPUT_CSV_TRADES  = "wfa_rolling_trades.csv"
OUTPUT_CSV_SUMMARY = "wfa_rolling_summary.csv"
OUTPUT_PNG         = "wfa_rolling_results.png"
# ──────────────────────────────────────────────────────────────────────


# ── API helpers ───────────────────────────────────────────────────────

def hl_post(payload: dict, retries: int = 3) -> list | dict:
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    Retry {attempt+1}: {e}")
            time.sleep(2)


def fetch_funding_history(coin: str, start_ms: int) -> pd.DataFrame:
    print(f"  Downloading funding history from "
          f"{datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}...")
    all_records, cursor = [], start_ms
    while True:
        data = hl_post({"type": "fundingHistory", "coin": coin, "startTime": cursor})
        if not data:
            break
        all_records.extend(data)
        last_t = data[-1]["time"]
        if last_t <= cursor or len(data) < 2:
            break
        cursor = last_t + 1
        if last_t > int(datetime.now(timezone.utc).timestamp() * 1000):
            break
        time.sleep(0.2)
    df = pd.DataFrame(all_records)
    df["time"]        = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["premium"]     = df["premium"].astype(float)
    print(f"    {len(df)} funding records")
    return df.sort_values("time").reset_index(drop=True)


def fetch_candles(coin: str, start_ms: int) -> pd.DataFrame:
    print(f"  Downloading hourly candles from "
          f"{datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}...")
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Hyperliquid candle API returns max ~5000 bars; batch if needed
    all_candles = []
    cursor = start_ms
    while True:
        data = hl_post({
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": "1h",
                    "startTime": cursor, "endTime": end_ms}
        })
        if not data:
            break
        all_candles.extend(data)
        last_t = data[-1]["T"] if "T" in data[-1] else data[-1].get("t", 0)
        if last_t <= cursor or len(data) < 2:
            break
        cursor = last_t + 1
        if cursor >= end_ms:
            break
        time.sleep(0.2)

    df = pd.DataFrame(all_candles)
    df = df.rename(columns={"t": "time_open", "o": "open", "h": "high",
                             "l": "low",       "c": "close", "v": "volume"})
    df["time_open"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("time_open").drop_duplicates("time_open").reset_index(drop=True)
    print(f"    {len(df)} candles")
    return df


def build_aligned_df(funding_df: pd.DataFrame, candle_df: pd.DataFrame) -> pd.DataFrame:
    funding_df = funding_df.copy()
    candle_df  = candle_df.copy()
    funding_df["hour"] = funding_df["time"].dt.floor("h")
    candle_df["hour"]  = candle_df["time_open"].dt.floor("h")

    fund_hr = (funding_df.groupby("hour").last()
               .reset_index()[["hour", "fundingRate", "premium"]])

    candle_df["log_return_pct"] = (
        np.log(candle_df["close"] / candle_df["close"].shift(1)) * 100
    )
    merged = (candle_df[["hour", "close", "log_return_pct"]]
              .merge(fund_hr, on="hour", how="inner")
              .dropna(subset=["log_return_pct", "premium"])
              .reset_index(drop=True))
    merged.rename(columns={"fundingRate": "funding_rate"}, inplace=True)
    print(f"  Aligned rows: {len(merged)}")
    return merged


# ── Feature helpers ───────────────────────────────────────────────────

def winsorize(s: pd.Series) -> pd.Series:
    return s.clip(s.quantile(WINSORIZE_LO), s.quantile(WINSORIZE_HI))


def add_next_ret(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_ret"] = df["log_return_pct"].shift(-1)
    return df.dropna(subset=["next_ret"]).reset_index(drop=True)


# ── Backtest engine ───────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, threshold: float,
                 fee: float = FEE_MAKER) -> pd.DataFrame:
    """
    D1 Extreme Percentile strategy on df.
    Threshold is computed WITHIN df (IS period defines its own quantiles).
    """
    pw   = winsorize(df["premium"])
    q_hi = pw.quantile(1 - threshold)
    q_lo = pw.quantile(threshold)

    records = []
    for i in range(len(df)):
        p = pw.iloc[i]
        if   p >= q_hi: sig = -1
        elif p <= q_lo: sig =  1
        else: continue

        fund_pnl = -sig * df["funding_rate"].iloc[i] * 100  # convert to %
        raw_ret  =  sig * df["next_ret"].iloc[i]
        records.append({
            "hour":        df["hour"].iloc[i],
            "premium":     p,
            "signal":      sig,
            "raw_ret":     raw_ret,
            "fund_pnl":    fund_pnl,
            "net_ret":     raw_ret + fund_pnl - 2 * fee,
            "threshold":   threshold,
            "q_hi":        q_hi,
            "q_lo":        q_lo,
        })
    return pd.DataFrame(records)


def trade_stats(trades: pd.DataFrame, col: str = "net_ret") -> dict:
    if trades.empty:
        return dict(n=0, win_rate=np.nan, avg_ret=np.nan,
                    total_ret=np.nan, sharpe=np.nan, max_dd=np.nan,
                    gross_ret=np.nan, fund_pnl=np.nan)
    r   = trades[col]
    eq  = (r / 100).cumsum()
    dd  = eq - eq.cummax()
    sh  = r.mean() / r.std() * np.sqrt(24 * 365) if r.std() > 0 else 0
    return dict(
        n         = len(r),
        win_rate  = (r > 0).mean(),
        avg_ret   = r.mean(),
        total_ret = r.sum(),
        sharpe    = sh,
        max_dd    = dd.min(),
        gross_ret = trades["raw_ret"].sum(),
        fund_pnl  = trades["fund_pnl"].sum(),
    )


# ── IS optimisation ───────────────────────────────────────────────────

def optimise_is(is_df: pd.DataFrame) -> tuple[float, dict]:
    """Return (best_threshold, all_threshold_stats) by IS Sharpe."""
    best_th, best_sharpe = THRESHOLD_GRID[0], -np.inf
    all_stats = {}
    for th in THRESHOLD_GRID:
        t = run_backtest(is_df, th)
        s = trade_stats(t)
        all_stats[th] = s
        if s["sharpe"] > best_sharpe and s["n"] > 0:
            best_sharpe = s["sharpe"]
            best_th     = th
    return best_th, all_stats


# ── Rolling WFA ───────────────────────────────────────────────────────

def run_rolling_wfa(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rolling WFA: IS window slides forward with NO overlap.
    Each IS starts exactly where the previous IS ended.

    Returns:
        oos_trades  : all OOS trades concatenated
        summary     : per-window IS/OOS statistics
        tail_trades : trades in the leftover tail period
    """
    df = df[df["hour"] >= WFA_START].copy()
    df = add_next_ret(df)
    df = df.reset_index(drop=True)

    is_td  = timedelta(days=IS_DAYS)
    oos_td = timedelta(days=OOS_DAYS)
    today  = df["hour"].max()

    windows       = []
    all_oos       = []
    window_num    = 1
    is_start_dt   = WFA_START

    print(f"\n{'='*65}")
    print(f"Rolling WFA: IS={IS_DAYS}d  OOS={OOS_DAYS}d  No IS overlap")
    print(f"{'='*65}")

    while True:
        is_end_dt  = is_start_dt + is_td
        oos_end_dt = is_end_dt   + oos_td

        if oos_end_dt > today:
            break

        is_mask  = (df["hour"] >= is_start_dt)  & (df["hour"] < is_end_dt)
        oos_mask = (df["hour"] >= is_end_dt)    & (df["hour"] < oos_end_dt)

        is_df  = df[is_mask].copy().reset_index(drop=True)
        oos_df = df[oos_mask].copy().reset_index(drop=True)

        if len(is_df) < 500 or len(oos_df) < 200:
            print(f"  Window {window_num}: insufficient data, skipping")
            is_start_dt = is_end_dt
            window_num += 1
            continue

        # IS optimisation
        best_th, is_all_stats = optimise_is(is_df)
        is_trades  = run_backtest(is_df,  best_th)
        oos_trades = run_backtest(oos_df, best_th)

        is_s   = trade_stats(is_trades)
        oos_s  = trade_stats(oos_trades)

        print(f"\n  Window {window_num}")
        print(f"    IS : {is_start_dt.strftime('%Y-%m-%d')} → {is_end_dt.strftime('%Y-%m-%d')}"
              f"  ({len(is_df)} hrs)")
        print(f"    OOS: {is_end_dt.strftime('%Y-%m-%d')} → {oos_end_dt.strftime('%Y-%m-%d')}"
              f"  ({len(oos_df)} hrs)")
        print(f"    Best threshold (IS Sharpe): {best_th:.0%}")
        print(f"    IS  — trades:{is_s['n']:3d}  Sharpe:{is_s['sharpe']:+.2f}"
              f"  WinRt:{is_s['win_rate']:.1%}  Net:{is_s['total_ret']:+.2f}%")
        print(f"    OOS — trades:{oos_s['n']:3d}  Sharpe:{oos_s['sharpe']:+.2f}"
              f"  WinRt:{oos_s['win_rate']:.1%}  Net:{oos_s['total_ret']:+.2f}%")

        # IS threshold sensitivity
        print(f"    IS threshold grid:")
        for th in THRESHOLD_GRID:
            s = is_all_stats[th]
            marker = " ← best" if th == best_th else ""
            print(f"      {th:.0%}: Sharpe={s['sharpe']:+.2f}  "
                  f"trades={s['n']}  net={s['total_ret']:+.2f}%{marker}")

        oos_trades["window"] = window_num
        all_oos.append(oos_trades)

        # Per-window OOS by signal direction
        oos_long  = oos_trades[oos_trades["signal"] ==  1]
        oos_short = oos_trades[oos_trades["signal"] == -1]
        ls_long   = trade_stats(oos_long)
        ls_short  = trade_stats(oos_short)

        windows.append({
            "window":          window_num,
            "is_start":        is_start_dt,
            "is_end":          is_end_dt,
            "oos_start":       is_end_dt,
            "oos_end":         oos_end_dt,
            "best_threshold":  best_th,
            "is_n":            is_s["n"],
            "is_sharpe":       is_s["sharpe"],
            "is_win_rate":     is_s["win_rate"],
            "is_total_ret":    is_s["total_ret"],
            "oos_n":           oos_s["n"],
            "oos_sharpe":      oos_s["sharpe"],
            "oos_win_rate":    oos_s["win_rate"],
            "oos_total_ret":   oos_s["total_ret"],
            "oos_gross_ret":   oos_s["gross_ret"],
            "oos_fund_pnl":    oos_s["fund_pnl"],
            "oos_max_dd":      oos_s["max_dd"],
            "oos_long_net":    ls_long["total_ret"],
            "oos_short_net":   ls_short["total_ret"],
            "oos_long_sharpe": ls_long["sharpe"],
            "oos_short_sharpe":ls_short["sharpe"],
        })

        # Next window IS starts where this IS ended (no overlap)
        is_start_dt = is_end_dt
        window_num += 1

    # Tail period (after last OOS end)
    last_oos_end = windows[-1]["oos_end"] if windows else WFA_START
    tail_mask    = df["hour"] >= last_oos_end
    tail_df      = df[tail_mask].copy().reset_index(drop=True)
    tail_trades  = pd.DataFrame()
    if len(tail_df) > 100:
        # Use last window's best threshold for tail
        last_th    = windows[-1]["best_threshold"] if windows else 0.05
        tail_trades = run_backtest(tail_df, last_th)
        tail_stats  = trade_stats(tail_trades)
        print(f"\n  Tail (extra OOS, never used in any window):")
        print(f"    Period: {last_oos_end.strftime('%Y-%m-%d')} → {today.strftime('%Y-%m-%d')}")
        print(f"    trades:{tail_stats['n']}  Sharpe:{tail_stats['sharpe']:+.2f}"
              f"  Net:{tail_stats['total_ret']:+.2f}%")

    oos_all = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    summary = pd.DataFrame(windows)
    return oos_all, summary, tail_trades


# ── Plotting ──────────────────────────────────────────────────────────

def plot_results(df_full: pd.DataFrame,
                 oos_all: pd.DataFrame,
                 summary: pd.DataFrame,
                 tail_trades: pd.DataFrame) -> None:

    cmap   = plt.cm.tab10
    n_win  = len(summary)
    colors = [cmap(i % 10) for i in range(n_win)]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        f"HYPE D1 Extreme Percentile — Rolling WFA  "
        f"(IS={IS_DAYS}d / OOS={OOS_DAYS}d / {n_win} windows / No IS overlap)\n"
        f"Maker fee {FEE_MAKER}% each side | Includes funding P&L",
        fontsize=12, fontweight="bold", y=0.99
    )

    df_plot = df_full[df_full["hour"] >= WFA_START].copy()
    pw_plot = winsorize(df_plot["premium"])

    # ── 1. Timeline: IS/OOS shading on premium ───────────────────────
    ax1 = fig.add_subplot(3, 3, (1, 2))
    ax1.plot(df_plot["hour"], pw_plot, color="#bbb", linewidth=0.5, zorder=1)
    for i, row in summary.iterrows():
        ax1.axvspan(row["is_start"],  row["is_end"],
                    alpha=0.10, color=colors[i], zorder=0)
        ax1.axvspan(row["oos_start"], row["oos_end"],
                    alpha=0.28, color=colors[i], zorder=0,
                    label=f"W{int(row['window'])} OOS")
    if not tail_trades.empty:
        ax1.axvspan(summary.iloc[-1]["oos_end"], df_plot["hour"].max(),
                    alpha=0.12, color="gray", zorder=0, label="Tail")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax1.set_title("Premium Index — IS (light) / OOS (dark) / Tail (grey)", fontsize=10)
    ax1.set_ylabel("Premium (winsorized)", fontsize=8)
    ax1.legend(fontsize=7, loc="upper right", ncol=n_win+1)
    ax1.tick_params(labelsize=7); ax1.grid(True, alpha=0.15)

    # ── 2. Best threshold per window ─────────────────────────────────
    ax2 = fig.add_subplot(3, 3, 3)
    win_labels = [f"W{int(w)}" for w in summary["window"]]
    ax2.bar(win_labels, summary["best_threshold"] * 100,
            color=colors[:n_win], alpha=0.85, width=0.5)
    ax2.set_ylabel("Best IS threshold (%)", fontsize=8)
    ax2.set_title("Optimal Threshold per Window\n(selected by IS Sharpe)", fontsize=10)
    for i, v in enumerate(summary["best_threshold"]):
        ax2.text(i, v*100 + 0.1, f"{v:.0%}", ha="center", fontsize=9)
    ax2.tick_params(labelsize=8); ax2.grid(True, alpha=0.15, axis="y")

    # ── 3. Concatenated OOS equity curve ─────────────────────────────
    ax3 = fig.add_subplot(3, 3, (4, 5))
    cumulative = 0.0
    for i, row in summary.iterrows():
        wt = oos_all[oos_all["window"] == row["window"]].copy()
        if wt.empty:
            continue
        eq = (wt["net_ret"] / 100).cumsum() + cumulative
        ax3.plot(wt["hour"], eq, color=colors[i], linewidth=1.5,
                 label=f"W{int(row['window'])} ({row['oos_total_ret']:+.1f}%)")
        cumulative = eq.iloc[-1]
    if not tail_trades.empty:
        tail_eq = (tail_trades["net_ret"] / 100).cumsum() + cumulative
        ax3.plot(tail_trades["hour"], tail_eq, color="gray",
                 linewidth=1.2, linestyle="--", label=f"Tail")
    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax3.set_title("Concatenated OOS Equity Curve (net of fees + funding)", fontsize=10)
    ax3.set_ylabel("Cumulative log return", fontsize=8)
    ax3.legend(fontsize=8, loc="upper left"); ax3.tick_params(labelsize=7)
    ax3.grid(True, alpha=0.15)

    # ── 4. IS vs OOS Sharpe scatter ──────────────────────────────────
    ax4 = fig.add_subplot(3, 3, 6)
    ax4.scatter(summary["is_sharpe"], summary["oos_sharpe"],
                c=colors[:n_win], s=100, zorder=3, edgecolors="white", linewidths=1.5)
    for i, row in summary.iterrows():
        ax4.annotate(f"  W{int(row['window'])}",
                     (row["is_sharpe"], row["oos_sharpe"]), fontsize=8)
    mn = min(summary[["is_sharpe","oos_sharpe"]].min()) - 1
    mx = max(summary[["is_sharpe","oos_sharpe"]].max()) + 1
    ax4.plot([mn,mx], [mn,mx], "k--", linewidth=0.8, alpha=0.4, label="IS = OOS")
    ax4.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax4.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax4.set_xlabel("IS Sharpe", fontsize=8); ax4.set_ylabel("OOS Sharpe", fontsize=8)
    ax4.set_title("IS vs OOS Sharpe\n(above diagonal = OOS beats IS)", fontsize=10)
    ax4.legend(fontsize=7); ax4.tick_params(labelsize=7); ax4.grid(True, alpha=0.15)

    # ── 5. OOS metrics bar per window ────────────────────────────────
    ax5 = fig.add_subplot(3, 3, 7)
    x = np.arange(n_win); w = 0.25
    wr  = summary["oos_win_rate"] * 100
    sh  = summary["oos_sharpe"]
    tr  = summary["oos_total_ret"]
    ax5.bar(x - w, wr, w, color=colors[:n_win], alpha=0.85, label="Win rate %")
    ax5r = ax5.twinx()
    ax5r.bar(x,    tr, w, color=colors[:n_win], alpha=0.45, label="Total net %")
    ax5r.bar(x + w, sh, w, color=colors[:n_win], alpha=0.25, label="Ann. Sharpe",
             hatch="//")
    ax5.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax5.set_xticks(x); ax5.set_xticklabels(win_labels, fontsize=9)
    ax5.set_ylabel("Win rate (%)", fontsize=8)
    ax5r.set_ylabel("Net ret % / Sharpe", fontsize=8)
    ax5.set_title("OOS Metrics per Window", fontsize=10)
    ax5.set_ylim(0, 80)
    ax5.tick_params(labelsize=7); ax5r.tick_params(labelsize=7)
    ax5.grid(True, alpha=0.15, axis="y")
    lines = [plt.Rectangle((0,0),1,1,fc=c,alpha=0.7) for c in colors[:n_win]]
    ax5.legend(lines + [plt.Line2D([0],[0],linestyle="--",color="gray")],
               win_labels + ["50% line"], fontsize=7, loc="upper right")

    # ── 6. PnL decomposition per window ──────────────────────────────
    ax6 = fig.add_subplot(3, 3, 8)
    x = np.arange(n_win); w = 0.25
    ax6.bar(x - w, summary["oos_gross_ret"], w,
            color=colors[:n_win], alpha=0.85, label="Price gross")
    ax6.bar(x,     summary["oos_fund_pnl"],  w,
            color=colors[:n_win], alpha=0.55, label="Funding P&L", hatch="//")
    fee_drag = -(summary["oos_gross_ret"] + summary["oos_fund_pnl"] -
                 summary["oos_total_ret"])
    ax6.bar(x + w, -fee_drag, w,
            color=colors[:n_win], alpha=0.30, label="Fee drag (neg)", hatch="xx")
    ax6.axhline(0, color="black", linewidth=0.8)
    ax6.set_xticks(x); ax6.set_xticklabels(win_labels, fontsize=9)
    ax6.set_ylabel("Return (%)", fontsize=8)
    ax6.set_title("OOS P&L Decomposition per Window\n(Price + Funding - Fee)", fontsize=10)
    ax6.legend(fontsize=7); ax6.tick_params(labelsize=7)
    ax6.grid(True, alpha=0.15, axis="y")

    # ── 7. Summary table ─────────────────────────────────────────────
    ax7 = fig.add_subplot(3, 3, 9)
    ax7.axis("off")
    if not oos_all.empty:
        combined = trade_stats(oos_all)
        rows = [
            ["Metric",            "OOS Combined"],
            ["Total OOS trades",  f"{combined['n']}"],
            ["Win rate",          f"{combined['win_rate']:.1%}"],
            ["Avg net/trade",     f"{combined['avg_ret']:+.4f}%"],
            ["Price gross",       f"{combined['gross_ret']:+.2f}%"],
            ["Funding P&L",       f"{combined['fund_pnl']:+.2f}%"],
            ["Total net return",  f"{combined['total_ret']:+.2f}%"],
            ["Ann. Sharpe",       f"{combined['sharpe']:.2f}"],
            ["Max drawdown",      f"{combined['max_dd']:+.4f}"],
            ["WFA windows",       f"{n_win}"],
            ["IS / OOS",          f"{IS_DAYS}d / {OOS_DAYS}d"],
            ["Fee (each side)",   f"{FEE_MAKER}%"],
        ]
        tbl = ax7.table(cellText=rows[1:], colLabels=rows[0],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#ddd")
            if r == 0:
                cell.set_facecolor("#378ADD")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f5f5f5")
        ax7.set_title("OOS Combined Statistics", fontsize=10, pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nChart saved: {OUTPUT_PNG}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("HYPE Rolling WFA — Downloading 18-month data")
    print("=" * 65)

    start_ms = int(WFA_START.timestamp() * 1000)

    # Download
    funding_df = fetch_funding_history(COIN, start_ms)
    candle_df  = fetch_candles(COIN, start_ms)
    df         = build_aligned_df(funding_df, candle_df)

    # Save full dataset
    df.to_csv(OUTPUT_CSV_DATA, index=False)
    print(f"Saved: {OUTPUT_CSV_DATA}  ({len(df)} rows, "
          f"{df['hour'].min().strftime('%Y-%m-%d')} to "
          f"{df['hour'].max().strftime('%Y-%m-%d')})")

    # Run WFA
    oos_all, summary, tail_trades = run_rolling_wfa(df)

    # Print OOS combined summary
    if not oos_all.empty:
        combined = trade_stats(oos_all)
        print(f"\n{'='*65}")
        print("OOS COMBINED RESULTS (all windows concatenated)")
        print(f"{'='*65}")
        print(f"  Total trades  : {combined['n']}")
        print(f"  Win rate      : {combined['win_rate']:.1%}")
        print(f"  Avg net/trade : {combined['avg_ret']:+.4f}%")
        print(f"  Price gross   : {combined['gross_ret']:+.2f}%")
        print(f"  Funding P&L   : {combined['fund_pnl']:+.2f}%")
        print(f"  Total net     : {combined['total_ret']:+.2f}%")
        print(f"  Ann. Sharpe   : {combined['sharpe']:.2f}")
        print(f"  Max drawdown  : {combined['max_dd']:+.4f}")

        # Per-window table
        print(f"\n{'='*65}")
        print("PER-WINDOW SUMMARY")
        print(f"{'='*65}")
        cols = ["window","best_threshold","is_sharpe","oos_sharpe",
                "oos_win_rate","oos_total_ret","oos_gross_ret",
                "oos_fund_pnl","oos_max_dd"]
        print(summary[cols].to_string(index=False, float_format="{:+.3f}".format))

    # Save outputs
    if not oos_all.empty:
        oos_all.to_csv(OUTPUT_CSV_TRADES, index=False)
        print(f"\nSaved: {OUTPUT_CSV_TRADES}")

    summary.to_csv(OUTPUT_CSV_SUMMARY, index=False)
    print(f"Saved: {OUTPUT_CSV_SUMMARY}")

    if not tail_trades.empty:
        tail_trades.to_csv("wfa_rolling_tail.csv", index=False)
        print(f"Saved: wfa_rolling_tail.csv")

    # Plot
    print("\nGenerating charts...")
    plot_results(df, oos_all, summary, tail_trades)
    print("\nDone. Output files:")
    print(f"  {OUTPUT_CSV_DATA}")
    print(f"  {OUTPUT_CSV_TRADES}")
    print(f"  {OUTPUT_CSV_SUMMARY}")
    print(f"  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
