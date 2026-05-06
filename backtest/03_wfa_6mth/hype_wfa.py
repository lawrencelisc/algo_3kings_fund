"""
HYPE Premium Index — D1 Extreme Percentile Strategy
Walk-Forward Analysis (WFA)
====================================================

WFA 原理：
  把歷史數據分成多個連續的「窗口」，每個窗口分兩段：
    - IS  (In-Sample)  : 用嚟「訓練」／揀最佳 threshold
    - OOS (Out-of-Sample): 用揀好的 threshold 「盲測」

  目的：避免 in-sample overfitting，模擬真實實盤部署的表現。

窗口設計（Anchored WFA）：
  IS 固定起點（2024-11-01），逐步延長；OOS 固定長度（60日）

  Window 1: IS = Nov24-Feb25  OOS = Mar25-Apr25
  Window 2: IS = Nov24-Apr25  OOS = May25-Jun25
  ...
  最後一窗: IS = Nov24-Mar26  OOS = Apr26-May26

  優點：每次 IS 都用上全部已知歷史，IS 越來越長，模擬真實上線後滾動更新。

需要先行以下命令安裝依賴：
  pip install requests pandas numpy matplotlib scipy

若已有 hype_aligned.csv 可直接使用（--csv 模式）：
  python hype_wfa.py --csv hype_aligned.csv

若需要重新下載 2 年數據：
  python hype_wfa.py --download

輸出檔案：
  wfa_oos_trades.csv     每筆 OOS 交易記錄
  wfa_window_summary.csv 每個窗口的 IS/OOS 統計
  wfa_results.png        完整分析圖
"""

import argparse
import time
from datetime import datetime, timezone, timedelta
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats


# ── Config ────────────────────────────────────────────────────────────
COIN            = "HYPE"
API_URL         = "https://api.hyperliquid.xyz/info"

OOS_DAYS        = 60          # OOS window length (days)
IS_MIN_DAYS     = 90          # minimum IS length before first OOS

# Threshold candidates to optimize in IS (percentile each tail)
THRESHOLD_GRID  = [0.02, 0.03, 0.05, 0.07, 0.10]

HOLD_HOURS      = 1           # hold period (hours)
FEE_PCT         = 0.035       # taker fee % per side (Hyperliquid)
WINSORIZE_LO    = 0.01
WINSORIZE_HI    = 0.99

# WFA start date — change if you have more data
WFA_START       = datetime(2025, 11, 1, tzinfo=timezone.utc)
# ─────────────────────────────────────────────────────────────────────


# ── Data helpers ──────────────────────────────────────────────────────

def hl_post(payload: dict, retries: int = 3) -> list | dict:
    import requests
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2)


def fetch_funding_history(coin: str, start_ms: int) -> pd.DataFrame:
    print(f"Downloading {coin} funding history...")
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
        time.sleep(0.25)
    df = pd.DataFrame(all_records)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["premium"]     = df["premium"].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def fetch_candles(coin: str, start_ms: int) -> pd.DataFrame:
    print(f"Downloading {coin} hourly candles...")
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    data = hl_post({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": "1h",
                "startTime": start_ms, "endTime": end_ms}
    })
    df = pd.DataFrame(data).rename(columns={
        "t": "time_open", "o": "open", "h": "high",
        "l": "low",       "c": "close", "v": "volume"
    })
    df["time_open"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df.sort_values("time_open").reset_index(drop=True)


def build_aligned_df(funding_df: pd.DataFrame, candle_df: pd.DataFrame) -> pd.DataFrame:
    funding_df = funding_df.copy()
    candle_df  = candle_df.copy()
    funding_df["hour"] = funding_df["time"].dt.floor("h")
    candle_df["hour"]  = candle_df["time_open"].dt.floor("h")

    fund_hr = (funding_df.groupby("hour")
               .last().reset_index()[["hour", "fundingRate", "premium"]])
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


def download_data(start_dt: datetime) -> pd.DataFrame:
    start_ms = int(start_dt.timestamp() * 1000)
    f = fetch_funding_history(COIN, start_ms)
    c = fetch_candles(COIN, start_ms)
    return build_aligned_df(f, c)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    df = df.sort_values("hour").reset_index(drop=True)
    df.rename(columns={"funding_rate_pct": "funding_rate"}, errors="ignore", inplace=True)
    # funding_rate stored as %, convert to raw
    if df["funding_rate"].abs().max() < 0.1:
        pass  # already raw
    else:
        df["funding_rate"] = df["funding_rate"] / 100
    print(f"Loaded {len(df)} rows from {path}")
    return df


# ── Feature engineering ───────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_ret"] = df["log_return_pct"].shift(-1)
    return df.dropna(subset=["next_ret"]).reset_index(drop=True)


def winsorize(s: pd.Series) -> pd.Series:
    return s.clip(s.quantile(WINSORIZE_LO), s.quantile(WINSORIZE_HI))


# ── Single-window backtest ────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, threshold: float,
                 fee: float = FEE_PCT) -> pd.DataFrame:
    """
    Apply D1 Extreme Percentile strategy to df.
    threshold : tail percentile (e.g. 0.05 = top/bot 5%)
    Returns   : DataFrame with trade details.
    """
    pw = winsorize(df["premium"])
    q_hi = pw.quantile(1 - threshold)
    q_lo = pw.quantile(threshold)

    trades = []
    for _, row in df.iterrows():
        # Use label index: slices like df.iloc[a:b] keep original labels, so iloc[i] would be wrong.
        p = pw.loc[row.name]
        if p >= q_hi:
            signal = -1   # short
        elif p <= q_lo:
            signal = 1    # long
        else:
            continue

        raw_ret   = signal * row["next_ret"]          # % gross return
        net_ret   = raw_ret - 2 * fee                 # deduct both sides fee
        trades.append({
            "hour":       row["hour"],
            "premium":    p,
            "signal":     signal,
            "raw_ret":    raw_ret,
            "net_ret":    net_ret,
            "threshold":  threshold,
        })

    return pd.DataFrame(trades)


def trade_stats(trades: pd.DataFrame, col: str = "net_ret") -> dict:
    if trades.empty:
        return dict(n=0, win_rate=np.nan, avg_ret=np.nan,
                    total_ret=np.nan, sharpe=np.nan, max_dd=np.nan)
    r = trades[col]
    equity = (r / 100).cumsum()
    dd     = equity - equity.cummax()
    sharpe = (r.mean() / r.std() * np.sqrt(24 * 365)) if r.std() > 0 else 0
    return dict(
        n        = len(r),
        win_rate = (r > 0).mean(),
        avg_ret  = r.mean(),
        total_ret= r.sum(),
        sharpe   = sharpe,
        max_dd   = dd.min(),
    )


# ── IS optimisation ───────────────────────────────────────────────────

def optimise_is(is_df: pd.DataFrame) -> float:
    """Return best threshold by IS Sharpe ratio (net of fees)."""
    best_th, best_sharpe = THRESHOLD_GRID[0], -np.inf
    for th in THRESHOLD_GRID:
        trades = run_backtest(is_df, th)
        if trades.empty:
            continue
        s = trade_stats(trades)["sharpe"]
        if s > best_sharpe:
            best_sharpe, best_th = s, th
    return best_th


# ── Walk-Forward Analysis ─────────────────────────────────────────────

def run_wfa(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Anchored WFA: IS always starts from WFA_START, OOS window = OOS_DAYS.

    Returns:
        all_oos_trades  : every OOS trade across all windows
        window_summary  : per-window IS/OOS stats
    """
    df = df[df["hour"] >= WFA_START].reset_index(drop=True)
    df = prepare_features(df)

    total_hours = len(df)
    is_min      = IS_MIN_DAYS * 24
    oos_len     = OOS_DAYS * 24

    windows         = []
    all_oos_trades  = []

    is_end_idx = is_min  # first IS ends here

    window_num = 1
    while is_end_idx + oos_len <= total_hours:
        oos_end_idx = is_end_idx + oos_len

        is_df  = df.iloc[:is_end_idx].copy()
        oos_df = df.iloc[is_end_idx:oos_end_idx].copy()

        # IS optimisation
        best_th = optimise_is(is_df)
        is_stats_raw = {th: trade_stats(run_backtest(is_df, th))
                        for th in THRESHOLD_GRID}
        is_best  = trade_stats(run_backtest(is_df,  best_th))
        oos_trades = run_backtest(oos_df, best_th)
        oos_stats  = trade_stats(oos_trades)

        is_period  = (df["hour"].iloc[0],           df["hour"].iloc[is_end_idx-1])
        oos_period = (df["hour"].iloc[is_end_idx],  df["hour"].iloc[oos_end_idx-1])

        print(f"  Window {window_num:02d} | "
              f"IS: {is_period[0].strftime('%Y-%m-%d')} → {is_period[1].strftime('%Y-%m-%d')} | "
              f"OOS: {oos_period[0].strftime('%Y-%m-%d')} → {oos_period[1].strftime('%Y-%m-%d')} | "
              f"best_th={best_th:.0%} | "
              f"IS Sharpe={is_best['sharpe']:.2f}  OOS Sharpe={oos_stats['sharpe']:.2f}")

        oos_trades["window"] = window_num
        all_oos_trades.append(oos_trades)

        windows.append({
            "window":       window_num,
            "is_start":     is_period[0],
            "is_end":       is_period[1],
            "oos_start":    oos_period[0],
            "oos_end":      oos_period[1],
            "best_threshold": best_th,
            "is_n":         is_best["n"],
            "is_win_rate":  is_best["win_rate"],
            "is_avg_ret":   is_best["avg_ret"],
            "is_sharpe":    is_best["sharpe"],
            "is_total_ret": is_best["total_ret"],
            "oos_n":        oos_stats["n"],
            "oos_win_rate": oos_stats["win_rate"],
            "oos_avg_ret":  oos_stats["avg_ret"],
            "oos_sharpe":   oos_stats["sharpe"],
            "oos_total_ret":oos_stats["total_ret"],
            "oos_max_dd":   oos_stats["max_dd"],
        })

        is_end_idx += oos_len   # slide forward by one OOS block
        window_num += 1

    oos_all = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    summary = pd.DataFrame(windows)
    return oos_all, summary


# ── Plotting ──────────────────────────────────────────────────────────

def plot_wfa(df_full: pd.DataFrame,
             oos_all: pd.DataFrame,
             summary: pd.DataFrame,
             output: str = "wfa_results.png") -> None:

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"HYPE D1 Extreme Percentile — Walk-Forward Analysis\n"
        f"OOS window: {OOS_DAYS}d | Hold: {HOLD_HOURS}h | Fee: {FEE_PCT}% each side | "
        f"Windows: {len(summary)}",
        fontsize=13, fontweight="bold", y=0.99
    )

    cmap   = plt.cm.tab10
    n_win  = len(summary)
    colors = [cmap(i % 10) for i in range(n_win)]

    # ── 1. Full premium time series with IS/OOS shading ──────────────
    ax1 = fig.add_subplot(3, 3, (1, 2))
    df_plot = df_full[df_full["hour"] >= WFA_START].copy()
    pw_plot = winsorize(df_plot["premium"])
    ax1.plot(df_plot["hour"], pw_plot, color="#aaa", linewidth=0.6, zorder=1)
    for i, row in summary.iterrows():
        ax1.axvspan(row["is_start"],  row["is_end"],
                    alpha=0.08, color=colors[i], zorder=0)
        ax1.axvspan(row["oos_start"], row["oos_end"],
                    alpha=0.22, color=colors[i], zorder=0, label=f"W{row['window']} OOS")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax1.set_title("Premium Index — IS (light) / OOS (dark) Windows", fontsize=10)
    ax1.set_ylabel("Premium (winsorized)", fontsize=8)
    ax1.tick_params(labelsize=7); ax1.grid(True, alpha=0.15)

    # ── 2. Best threshold per window ─────────────────────────────────
    ax2 = fig.add_subplot(3, 3, 3)
    th_pct = [f"{t:.0%}" for t in summary["best_threshold"]]
    bar_c  = [colors[i] for i in range(n_win)]
    ax2.bar([f"W{w}" for w in summary["window"]], summary["best_threshold"]*100,
            color=bar_c, alpha=0.8)
    ax2.set_ylabel("Best IS threshold (%)", fontsize=8)
    ax2.set_title("Optimal Threshold per Window\n(selected by IS Sharpe)", fontsize=10)
    ax2.tick_params(labelsize=8); ax2.grid(True, alpha=0.15, axis="y")
    for i, v in enumerate(summary["best_threshold"]):
        ax2.text(i, v*100+0.1, f"{v:.0%}", ha="center", fontsize=8)

    # ── 3. OOS equity curve ───────────────────────────────────────────
    ax3 = fig.add_subplot(3, 3, (4, 5))
    cumulative = 0.0
    for i, row in summary.iterrows():
        w_trades = oos_all[oos_all["window"] == row["window"]].copy()
        if w_trades.empty:
            continue
        eq = (w_trades["net_ret"] / 100).cumsum() + cumulative
        ax3.plot(w_trades["hour"], eq, color=colors[i],
                 linewidth=1.4, label=f"W{int(row['window'])}")
        cumulative = eq.iloc[-1]

    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax3.set_title("Concatenated OOS Equity Curve (net of fees)", fontsize=10)
    ax3.set_ylabel("Cumulative log return", fontsize=8)
    ax3.legend(fontsize=7, loc="upper left", ncol=4)
    ax3.tick_params(labelsize=7); ax3.grid(True, alpha=0.15)

    # ── 4. IS vs OOS Sharpe scatter ──────────────────────────────────
    ax4 = fig.add_subplot(3, 3, 6)
    ax4.scatter(summary["is_sharpe"], summary["oos_sharpe"],
                c=colors[:n_win], s=80, zorder=3, edgecolors="white")
    for i, row in summary.iterrows():
        ax4.annotate(f"W{int(row['window'])}",
                     (row["is_sharpe"], row["oos_sharpe"]),
                     fontsize=7, ha="left", va="bottom")
    mn = min(summary["is_sharpe"].min(), summary["oos_sharpe"].min()) - 1
    mx = max(summary["is_sharpe"].max(), summary["oos_sharpe"].max()) + 1
    ax4.plot([mn, mx], [mn, mx], "k--", linewidth=0.8, alpha=0.4, label="IS = OOS line")
    ax4.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax4.set_xlabel("IS Sharpe", fontsize=8); ax4.set_ylabel("OOS Sharpe", fontsize=8)
    ax4.set_title("IS vs OOS Sharpe per Window\n(above diagonal = OOS beats IS)", fontsize=10)
    ax4.legend(fontsize=7); ax4.tick_params(labelsize=7); ax4.grid(True, alpha=0.15)

    # ── 5. Per-window OOS stats bar ───────────────────────────────────
    ax5 = fig.add_subplot(3, 3, 7)
    x  = np.arange(n_win)
    wr = summary["oos_win_rate"] * 100
    tr = summary["oos_total_ret"]
    b1 = ax5.bar(x - 0.2, wr, 0.35, color=colors[:n_win], alpha=0.8, label="OOS Win rate %")
    ax5r = ax5.twinx()
    b2 = ax5r.bar(x + 0.2, tr, 0.35, color=colors[:n_win], alpha=0.4, label="OOS Total ret %")
    ax5.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax5.set_xticks(x); ax5.set_xticklabels([f"W{w}" for w in summary["window"]], fontsize=8)
    ax5.set_ylabel("Win rate (%)", fontsize=8)
    ax5r.set_ylabel("Total OOS return (%)", fontsize=8)
    ax5.set_title("OOS Win Rate & Total Return per Window", fontsize=10)
    ax5.set_ylim(0, 80)
    ax5.tick_params(labelsize=7); ax5r.tick_params(labelsize=7)
    ax5.grid(True, alpha=0.15, axis="y")

    # ── 6. OOS drawdown ───────────────────────────────────────────────
    ax6 = fig.add_subplot(3, 3, 8)
    if not oos_all.empty:
        oos_eq = (oos_all.sort_values("hour")["net_ret"] / 100).cumsum()
        oos_dd = oos_eq - oos_eq.cummax()
        ax6.fill_between(range(len(oos_dd)), oos_dd, 0,
                         color="#D85A30", alpha=0.6)
        ax6.axhline(0, color="black", linewidth=0.5)
        ax6.set_title(f"Concatenated OOS Drawdown\n(max: {oos_dd.min():+.4f})", fontsize=10)
        ax6.set_ylabel("Drawdown", fontsize=8)
        ax6.set_xlabel("OOS trade index", fontsize=8)
        ax6.tick_params(labelsize=7); ax6.grid(True, alpha=0.15)

    # ── 7. Summary table ──────────────────────────────────────────────
    ax7 = fig.add_subplot(3, 3, 9)
    ax7.axis("off")
    if not oos_all.empty:
        oos_total = trade_stats(oos_all)
        rows = [
            ["Metric",              "OOS Combined"],
            ["Total OOS trades",    f"{oos_total['n']}"],
            ["Win rate",            f"{oos_total['win_rate']:.1%}"],
            ["Avg net ret/trade",   f"{oos_total['avg_ret']:+.4f}%"],
            ["Total net return",    f"{oos_total['total_ret']:+.2f}%"],
            ["Ann. Sharpe",         f"{oos_total['sharpe']:.2f}"],
            ["Max drawdown",        f"{oos_total['max_dd']:+.4f}"],
            ["WFA windows",         f"{n_win}"],
            ["OOS period each",     f"{OOS_DAYS} days"],
            ["Hold period",         f"{HOLD_HOURS} hour"],
            ["Fee (each side)",     f"{FEE_PCT}%"],
        ]
        t = ax7.table(cellText=rows[1:], colLabels=rows[0],
                      loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.55)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#ddd")
            if r == 0:
                cell.set_facecolor("#378ADD")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f5f5f5")
        ax7.set_title("OOS Combined Statistics", fontsize=10, pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {output}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # ── ▼▼▼ EDIT HERE to switch modes ▼▼▼ ────────────────────────────
    #
    # Option A — use existing CSV (put the full path if not in same folder):
    CSV_PATH = "hype_aligned.csv"   # ← change to your file path
    #
    # Option B — download fresh 2-year data from Hyperliquid:
    DOWNLOAD = False                 # ← set True to download instead
    #
    # ── ▲▲▲ ─────────────────────────────────────────────────────────

    # command-line override (optional, works fine if run from terminal too)
    import sys
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description="HYPE WFA — D1 Extreme Percentile")
        group  = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--csv",      type=str)
        group.add_argument("--download", action="store_true")
        args = parser.parse_args()
        DOWNLOAD = args.download
        CSV_PATH = args.csv if args.csv else CSV_PATH

    # ── Load data ─────────────────────────────────────────────────────
    if DOWNLOAD:
        start_dt = datetime.now(timezone.utc) - timedelta(days=2*365)
        df = download_data(start_dt)
        df.to_csv("hype_aligned_2y.csv", index=False)
        print("Saved: hype_aligned_2y.csv")
    else:
        df = load_csv(CSV_PATH)

    # Ensure premium column exists
    if "premium" not in df.columns:
        raise ValueError("CSV must have 'premium' column. Re-run with --download.")

    print(f"\nData range: {df['hour'].min()} → {df['hour'].max()}")
    print(f"Total rows: {len(df)}")

    # ── Run WFA ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Running Anchored Walk-Forward Analysis")
    print(f"  IS start   : {WFA_START.strftime('%Y-%m-%d')}")
    print(f"  OOS length : {OOS_DAYS} days")
    print(f"  IS min     : {IS_MIN_DAYS} days")
    print(f"  Thresholds : {THRESHOLD_GRID}")
    print(f"{'='*60}\n")

    oos_all, summary = run_wfa(df)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WFA WINDOW SUMMARY")
    print(f"{'='*60}")
    print(summary[[
        "window","best_threshold",
        "is_sharpe","is_total_ret",
        "oos_sharpe","oos_total_ret","oos_win_rate","oos_max_dd"
    ]].to_string(index=False, float_format="{:+.3f}".format))

    if not oos_all.empty:
        oos_combined = trade_stats(oos_all)
        print(f"\n{'='*60}")
        print("OOS COMBINED RESULTS (all windows concatenated)")
        print(f"{'='*60}")
        print(f"  Total trades : {oos_combined['n']}")
        print(f"  Win rate     : {oos_combined['win_rate']:.1%}")
        print(f"  Avg net ret  : {oos_combined['avg_ret']:+.4f}% per trade")
        print(f"  Total return : {oos_combined['total_ret']:+.2f}%")
        print(f"  Ann. Sharpe  : {oos_combined['sharpe']:.2f}")
        print(f"  Max drawdown : {oos_combined['max_dd']:+.4f}")

    # ── Save outputs ──────────────────────────────────────────────────
    if not oos_all.empty:
        oos_all.to_csv("wfa_oos_trades.csv", index=False)
        print("\nSaved: wfa_oos_trades.csv")

    summary.to_csv("wfa_window_summary.csv", index=False)
    print("Saved: wfa_window_summary.csv")

    plot_wfa(df, oos_all, summary, "wfa_results.png")
    print("Saved: wfa_results.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
