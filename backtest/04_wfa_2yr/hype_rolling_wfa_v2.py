"""
HYPE D1 Extreme Percentile — Rolling WFA (Fixed Historical Download)
=====================================================================
修正點：
  - candleSnapshot 每次最多 5000 條，用 endTime 向後批次拉取，確保拎齊完整歷史
  - fundingHistory 同樣分批拉取
  - 下載完成後儲存 hype_18m.csv，下次可直接讀取唔需重下載

Option B Rolling WFA:
  IS = 120d (4 個月), OOS = 60d (2 個月), Rolling (IS 唔重複)
  目標 windows: 4 個

Usage:
  pip install requests pandas numpy matplotlib scipy
  
  # 首次運行：下載數據 + 跑 WFA
  python hype_rolling_wfa_v2.py --download

  # 已有 CSV 直接跑 WFA：
  python hype_rolling_wfa_v2.py --csv hype_18m.csv

Output files:
  hype_18m.csv               完整對齊數據
  wfa_rolling_trades.csv     每筆 OOS 交易
  wfa_rolling_summary.csv    每個 window 統計
  wfa_rolling_tail.csv       尾段交易
  wfa_rolling_results.png    9 格分析圖
"""

import sys
import time
import argparse
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────
COIN          = "HYPE"
API_URL       = "https://api.hyperliquid.xyz/info"
WFA_START     = datetime(2024, 11, 1, tzinfo=timezone.utc)  # HYPE listing date

IS_DAYS       = 120    # 4 months
OOS_DAYS      = 60     # 2 months

THRESHOLD_GRID = [0.02, 0.03, 0.05, 0.07, 0.10]
WINSORIZE_LO   = 0.01
WINSORIZE_HI   = 0.99
FEE_MAKER      = 0.0135
FEE_TAKER      = 0.035

CANDLE_BATCH   = 5000   # max candles per API request
SLEEP          = 0.25   # seconds between requests
# ───────────────────────────────────────────────────────────────────────


# ── API helpers ─────────────────────────────────────────────────────────

def hl_post(payload: dict, retries: int = 3) -> list | dict:
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    [retry {attempt+1}] {e}")
            time.sleep(2)


def fetch_candles_full(coin: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch ALL hourly candles between start_dt and end_dt.
    Uses forward batching: each request uses the last returned candle's
    time + 1ms as the next startTime, until end_dt is covered.
    
    This is the correct approach — do NOT rely on endTime batching
    because Hyperliquid returns up to 5000 bars from startTime forward.
    """
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    print(f"  Downloading candles: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}")

    all_candles = []
    cursor      = start_ms
    batch_num   = 0

    while cursor < end_ms:
        batch_num += 1
        data = hl_post({
            "type": "candleSnapshot",
            "req": {
                "coin":      coin,
                "interval":  "1h",
                "startTime": cursor,
                "endTime":   end_ms,
            }
        })

        if not data:
            print(f"    Batch {batch_num}: empty — stopping")
            break

        # Filter to only include candles within our range
        data = [c for c in data if c["t"] >= cursor and c["t"] <= end_ms]
        if not data:
            break

        all_candles.extend(data)
        last_t = max(c["t"] for c in data)

        pct = (last_t - start_ms) / (end_ms - start_ms) * 100
        dt  = datetime.fromtimestamp(last_t / 1000, tz=timezone.utc)
        print(f"    Batch {batch_num}: {len(data):4d} candles "
              f"up to {dt.strftime('%Y-%m-%d %H:%M')}  ({pct:.1f}%)")

        # If we got fewer than batch size, we've reached the end
        if len(data) < CANDLE_BATCH:
            break

        # Advance cursor past the last returned candle (1 hour + 1ms)
        cursor = last_t + 3_600_001
        time.sleep(SLEEP)

    if not all_candles:
        raise ValueError(f"No candle data returned for {coin}")

    df = pd.DataFrame(all_candles)
    df = df.rename(columns={
        "t": "time_open", "o": "open", "h": "high",
        "l": "low",       "c": "close", "v": "volume"
    })
    df["time_open"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    df = (df.sort_values("time_open")
          .drop_duplicates("time_open")
          .reset_index(drop=True))

    print(f"  Total candles: {len(df)}  "
          f"({df['time_open'].min().strftime('%Y-%m-%d')} → "
          f"{df['time_open'].max().strftime('%Y-%m-%d')})")
    return df


def fetch_funding_full(coin: str, start_dt: datetime) -> pd.DataFrame:
    """
    Fetch ALL funding history records from start_dt to now.
    fundingHistory returns records from startTime forward; batch by
    advancing cursor past the last returned record's time.
    """
    start_ms = int(start_dt.timestamp() * 1000)
    print(f"  Downloading funding history from {start_dt.strftime('%Y-%m-%d')}...")

    all_records = []
    cursor      = start_ms
    batch_num   = 0

    while True:
        batch_num += 1
        data = hl_post({
            "type":      "fundingHistory",
            "coin":      coin,
            "startTime": cursor,
        })

        if not data:
            break

        new = [r for r in data if r["time"] >= cursor]
        if not new:
            break

        all_records.extend(new)
        last_t = max(r["time"] for r in new)

        print(f"    Batch {batch_num}: {len(new):4d} records  "
              f"up to {datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")

        if len(new) < 500:   # funding API typically returns ~500 per batch
            break

        cursor = last_t + 1
        time.sleep(SLEEP)

    df = pd.DataFrame(all_records)
    df["time"]        = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["premium"]     = df["premium"].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  Total funding records: {len(df)}  "
          f"({df['time'].min().strftime('%Y-%m-%d')} → "
          f"{df['time'].max().strftime('%Y-%m-%d')})")
    return df


def build_aligned_df(candle_df: pd.DataFrame,
                     funding_df: pd.DataFrame) -> pd.DataFrame:
    candle_df  = candle_df.copy()
    funding_df = funding_df.copy()

    candle_df["hour"]  = candle_df["time_open"].dt.floor("h")
    funding_df["hour"] = funding_df["time"].dt.floor("h")

    fund_hr = (funding_df.groupby("hour").last()
               .reset_index()[["hour", "fundingRate", "premium"]])
    fund_hr.rename(columns={"fundingRate": "funding_rate"}, inplace=True)

    candle_df["log_return_pct"] = (
        np.log(candle_df["close"] / candle_df["close"].shift(1)) * 100
    )

    merged = (candle_df[["hour", "close", "log_return_pct"]]
              .merge(fund_hr, on="hour", how="inner")
              .dropna(subset=["log_return_pct", "premium"])
              .reset_index(drop=True))

    print(f"\n  Aligned rows: {len(merged)}  "
          f"({merged['hour'].min().strftime('%Y-%m-%d')} → "
          f"{merged['hour'].max().strftime('%Y-%m-%d')})")
    return merged


def download_and_save(csv_path: str = "hype_18m.csv") -> pd.DataFrame:
    print("=" * 65)
    print(f"Downloading HYPE history: {WFA_START.strftime('%Y-%m-%d')} → now")
    print("=" * 65)

    end_dt = datetime.now(timezone.utc)

    print("\n[1/2] Candle data")
    candles = fetch_candles_full(COIN, WFA_START, end_dt)

    print("\n[2/2] Funding history")
    funding = fetch_funding_full(COIN, WFA_START)

    print("\n[3/3] Aligning...")
    df = build_aligned_df(candles, funding)
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return df


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)

    # Normalise column names from older CSV versions
    if "funding_rate_pct" in df.columns and "funding_rate" not in df.columns:
        df.rename(columns={"funding_rate_pct": "funding_rate"}, inplace=True)
    if "funding_rate" in df.columns:
        # Ensure funding_rate is in raw decimal form (not %)
        if df["funding_rate"].abs().median() > 0.01:
            df["funding_rate"] = df["funding_rate"] / 100

    df = df.sort_values("hour").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {csv_path}  "
          f"({df['hour'].min().strftime('%Y-%m-%d')} → "
          f"{df['hour'].max().strftime('%Y-%m-%d')})")
    return df


# ── Strategy helpers ────────────────────────────────────────────────────

def winsorize(s: pd.Series) -> pd.Series:
    return s.clip(s.quantile(WINSORIZE_LO), s.quantile(WINSORIZE_HI))


def add_next_ret(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_ret"] = df["log_return_pct"].shift(-1)
    return df.dropna(subset=["next_ret"]).reset_index(drop=True)


def run_backtest(df: pd.DataFrame, threshold: float,
                 fee: float = FEE_MAKER) -> pd.DataFrame:
    pw   = winsorize(df["premium"])
    q_hi = pw.quantile(1 - threshold)
    q_lo = pw.quantile(threshold)

    records = []
    for i in range(len(df)):
        p = pw.iloc[i]
        if   p >= q_hi: sig = -1
        elif p <= q_lo: sig =  1
        else: continue

        # funding_rate is raw decimal (e.g. 0.000125)
        fund_pnl = -sig * df["funding_rate"].iloc[i] * 100  # to %
        raw_ret  =  sig * df["next_ret"].iloc[i]
        records.append({
            "hour":      df["hour"].iloc[i],
            "premium":   p,
            "signal":    sig,
            "raw_ret":   raw_ret,
            "fund_pnl":  fund_pnl,
            "net_ret":   raw_ret + fund_pnl - 2 * fee,
            "threshold": threshold,
        })
    return pd.DataFrame(records)


def trade_stats(t: pd.DataFrame, col: str = "net_ret") -> dict:
    if t is None or t.empty:
        return dict(n=0, win_rate=np.nan, avg_ret=np.nan,
                    total_ret=np.nan, sharpe=np.nan, max_dd=np.nan,
                    gross_ret=np.nan, fund_pnl=np.nan)
    r  = t[col]
    eq = (r / 100).cumsum()
    dd = eq - eq.cummax()
    sh = r.mean() / r.std() * np.sqrt(24 * 365) if r.std() > 0 else 0
    return dict(
        n         = len(r),
        win_rate  = (r > 0).mean(),
        avg_ret   = r.mean(),
        total_ret = r.sum(),
        sharpe    = sh,
        max_dd    = dd.min(),
        gross_ret = t["raw_ret"].sum() if "raw_ret" in t else np.nan,
        fund_pnl  = t["fund_pnl"].sum() if "fund_pnl" in t else np.nan,
    )


def optimise_is(is_df: pd.DataFrame) -> tuple[float, dict]:
    best_th, best_sh = THRESHOLD_GRID[0], -np.inf
    all_stats = {}
    for th in THRESHOLD_GRID:
        t = run_backtest(is_df, th)
        s = trade_stats(t)
        all_stats[th] = s
        if s["n"] > 0 and s["sharpe"] > best_sh:
            best_sh = s["sharpe"]
            best_th = th
    return best_th, all_stats


# ── Rolling WFA ─────────────────────────────────────────────────────────

def run_rolling_wfa(df: pd.DataFrame):
    df = df[df["hour"] >= WFA_START].copy()
    df = add_next_ret(df)
    df = df.reset_index(drop=True)

    today  = df["hour"].max()
    is_td  = timedelta(days=IS_DAYS)
    oos_td = timedelta(days=OOS_DAYS)

    total_days  = (today - WFA_START).days
    n_possible  = (total_days - IS_DAYS) // OOS_DAYS
    print(f"\n{'='*65}")
    print(f"Rolling WFA  IS={IS_DAYS}d  OOS={OOS_DAYS}d  (no IS overlap)")
    print(f"Data: {total_days}d  |  Expected windows: {n_possible}")
    print(f"{'='*65}")

    windows, all_oos = [], []
    is_start = WFA_START
    w = 1

    while True:
        is_end  = is_start + is_td
        oos_end = is_end   + oos_td
        if oos_end > today:
            break

        is_df  = df[(df["hour"] >= is_start) & (df["hour"] < is_end)].copy().reset_index(drop=True)
        oos_df = df[(df["hour"] >= is_end)   & (df["hour"] < oos_end)].copy().reset_index(drop=True)

        if len(is_df) < 500 or len(oos_df) < 200:
            print(f"  W{w}: skipped (IS={len(is_df)}h OOS={len(oos_df)}h — insufficient)")
            is_start = is_end
            w += 1
            continue

        best_th, is_grid = optimise_is(is_df)
        is_t  = run_backtest(is_df,  best_th)
        oos_t = run_backtest(oos_df, best_th)
        is_s  = trade_stats(is_t)
        oos_s = trade_stats(oos_t)

        oos_long  = oos_t[oos_t["signal"] ==  1]
        oos_short = oos_t[oos_t["signal"] == -1]
        ls_l = trade_stats(oos_long)
        ls_s = trade_stats(oos_short)

        print(f"\n  Window {w}")
        print(f"    IS : {is_start.strftime('%Y-%m-%d')} → {is_end.strftime('%Y-%m-%d')}  ({len(is_df)}h)")
        print(f"    OOS: {is_end.strftime('%Y-%m-%d')} → {oos_end.strftime('%Y-%m-%d')}  ({len(oos_df)}h)")
        print(f"    IS  threshold grid:")
        for th in THRESHOLD_GRID:
            g = is_grid[th]
            mk = " ← best" if th == best_th else ""
            print(f"      {th:.0%}  Sharpe={g['sharpe']:+6.2f}  "
                  f"trades={g['n']:3d}  net={g['total_ret']:+6.2f}%{mk}")
        print(f"    IS  best={best_th:.0%}  Sharpe={is_s['sharpe']:+.2f}  "
              f"WR={is_s['win_rate']:.1%}  Net={is_s['total_ret']:+.2f}%")
        print(f"    OOS           Sharpe={oos_s['sharpe']:+.2f}  "
              f"WR={oos_s['win_rate']:.1%}  Net={oos_s['total_ret']:+.2f}%  "
              f"Trades={oos_s['n']}")
        print(f"    OOS Long  net={ls_l['total_ret']:+.2f}%  Sharpe={ls_l['sharpe']:+.2f}")
        print(f"    OOS Short net={ls_s['total_ret']:+.2f}%  Sharpe={ls_s['sharpe']:+.2f}")

        oos_t["window"] = w
        all_oos.append(oos_t)

        windows.append({
            "window":           w,
            "is_start":         is_start,
            "is_end":           is_end,
            "oos_start":        is_end,
            "oos_end":          oos_end,
            "best_threshold":   best_th,
            "is_n":             is_s["n"],
            "is_sharpe":        is_s["sharpe"],
            "is_win_rate":      is_s["win_rate"],
            "is_total_ret":     is_s["total_ret"],
            "oos_n":            oos_s["n"],
            "oos_sharpe":       oos_s["sharpe"],
            "oos_win_rate":     oos_s["win_rate"],
            "oos_total_ret":    oos_s["total_ret"],
            "oos_gross_ret":    oos_s["gross_ret"],
            "oos_fund_pnl":     oos_s["fund_pnl"],
            "oos_max_dd":       oos_s["max_dd"],
            "oos_long_net":     ls_l["total_ret"],
            "oos_short_net":    ls_s["total_ret"],
            "oos_long_sharpe":  ls_l["sharpe"],
            "oos_short_sharpe": ls_s["sharpe"],
        })

        is_start = is_end   # Rolling: next IS starts here
        w += 1

    # Tail (after last OOS)
    tail_df = pd.DataFrame()
    if windows:
        tail_start = windows[-1]["oos_end"]
        tail_df    = df[df["hour"] >= tail_start].copy().reset_index(drop=True)
        if len(tail_df) > 50:
            last_th      = windows[-1]["best_threshold"]
            tail_trades  = run_backtest(tail_df, last_th)
            tail_s       = trade_stats(tail_trades)
            tail_trades["window"] = 0   # 0 = tail
            print(f"\n  Tail (never used in any window)")
            print(f"    {tail_start.strftime('%Y-%m-%d')} → {today.strftime('%Y-%m-%d')}"
                  f"  threshold={last_th:.0%}")
            print(f"    trades={tail_s['n']}  Sharpe={tail_s['sharpe']:+.2f}  "
                  f"Net={tail_s['total_ret']:+.2f}%")
            tail_df = tail_trades
        else:
            tail_df = pd.DataFrame()

    oos_all = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    summary = pd.DataFrame(windows)
    return oos_all, summary, tail_df


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_results(df_full, oos_all, summary, tail_df):
    cmap   = plt.cm.tab10
    n_win  = len(summary)
    colors = [cmap(i % 10) for i in range(n_win)]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        f"HYPE D1 Extreme Percentile — Rolling WFA  "
        f"(IS={IS_DAYS}d / OOS={OOS_DAYS}d / {n_win} windows / No IS overlap)\n"
        f"Maker fee {FEE_MAKER}% each side  |  Includes funding P&L",
        fontsize=12, fontweight="bold", y=0.99
    )

    df_wfa = df_full[df_full["hour"] >= WFA_START].copy()
    pw_all = winsorize(df_wfa["premium"])

    # ── 1. Timeline ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(3, 3, (1, 2))
    ax1.plot(df_wfa["hour"], pw_all, color="#ccc", linewidth=0.5, zorder=1)
    for i, row in summary.iterrows():
        ax1.axvspan(row["is_start"],  row["is_end"],
                    alpha=0.08, color=colors[i], zorder=0)
        ax1.axvspan(row["oos_start"], row["oos_end"],
                    alpha=0.25, color=colors[i], zorder=0,
                    label=f"W{int(row['window'])} OOS")
    if not tail_df.empty:
        ax1.axvspan(summary.iloc[-1]["oos_end"], df_wfa["hour"].max(),
                    alpha=0.10, color="gray", zorder=0, label="Tail")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax1.set_title("Premium Index — IS (light) / OOS (dark) / Tail (grey)", fontsize=10)
    ax1.set_ylabel("Premium (winsorized)", fontsize=8)
    ax1.legend(fontsize=7, ncol=n_win + 1, loc="upper right")
    ax1.tick_params(labelsize=7); ax1.grid(True, alpha=0.15)

    # ── 2. Optimal threshold per window ──────────────────────────────
    ax2 = fig.add_subplot(3, 3, 3)
    wlabels = [f"W{int(w)}" for w in summary["window"]]
    ax2.bar(wlabels, summary["best_threshold"] * 100,
            color=colors[:n_win], alpha=0.85, width=0.5)
    ax2.set_ylabel("Best IS threshold (%)", fontsize=8)
    ax2.set_title("Optimal Threshold per Window\n(by IS Sharpe)", fontsize=10)
    for i, v in enumerate(summary["best_threshold"]):
        ax2.text(i, v * 100 + 0.05, f"{v:.0%}", ha="center", fontsize=9)
    ax2.tick_params(labelsize=8); ax2.grid(True, alpha=0.15, axis="y")

    # ── 3. OOS equity curve ───────────────────────────────────────────
    ax3 = fig.add_subplot(3, 3, (4, 5))
    cum = 0.0
    for i, row in summary.iterrows():
        wt = oos_all[oos_all["window"] == row["window"]]
        if wt.empty: continue
        eq = (wt["net_ret"] / 100).cumsum() + cum
        ax3.plot(wt["hour"], eq, color=colors[i], linewidth=1.5,
                 label=f"W{int(row['window'])} ({row['oos_total_ret']:+.1f}%)")
        cum = eq.iloc[-1]
    if not tail_df.empty:
        teq = (tail_df["net_ret"] / 100).cumsum() + cum
        ax3.plot(tail_df["hour"], teq, color="gray",
                 linewidth=1.2, linestyle="--",
                 label=f"Tail ({trade_stats(tail_df)['total_ret']:+.1f}%)")
    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax3.set_title("Concatenated OOS Equity Curve (net = price + funding - fee)", fontsize=10)
    ax3.set_ylabel("Cumulative log return", fontsize=8)
    ax3.legend(fontsize=8, loc="upper left", ncol=2)
    ax3.tick_params(labelsize=7); ax3.grid(True, alpha=0.15)

    # ── 4. IS vs OOS Sharpe scatter ──────────────────────────────────
    ax4 = fig.add_subplot(3, 3, 6)
    ax4.scatter(summary["is_sharpe"], summary["oos_sharpe"],
                c=colors[:n_win], s=120, zorder=3,
                edgecolors="white", linewidths=1.5)
    for i, row in summary.iterrows():
        ax4.annotate(f"  W{int(row['window'])}",
                     (row["is_sharpe"], row["oos_sharpe"]), fontsize=8)
    pad = 2
    mn = min(summary[["is_sharpe","oos_sharpe"]].min()) - pad
    mx = max(summary[["is_sharpe","oos_sharpe"]].max()) + pad
    ax4.plot([mn,mx],[mn,mx],"k--",linewidth=0.8,alpha=0.4,label="IS = OOS")
    ax4.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax4.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax4.set_xlabel("IS Sharpe", fontsize=8)
    ax4.set_ylabel("OOS Sharpe", fontsize=8)
    ax4.set_title("IS vs OOS Sharpe per Window\n(above diagonal = OOS beats IS)", fontsize=10)
    ax4.legend(fontsize=7); ax4.tick_params(labelsize=7); ax4.grid(True, alpha=0.15)

    # ── 5. OOS metrics per window ────────────────────────────────────
    ax5 = fig.add_subplot(3, 3, 7)
    x = np.arange(n_win); w = 0.25
    ax5.bar(x - w, summary["oos_win_rate"] * 100,
            w, color=colors[:n_win], alpha=0.85, label="Win rate %")
    ax5r = ax5.twinx()
    ax5r.bar(x,     summary["oos_total_ret"], w,
             color=colors[:n_win], alpha=0.50, label="Net ret %")
    ax5r.bar(x + w, summary["oos_sharpe"],   w,
             color=colors[:n_win], alpha=0.25, label="Sharpe", hatch="//")
    ax5.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax5.set_xticks(x); ax5.set_xticklabels(wlabels, fontsize=9)
    ax5.set_ylabel("Win rate (%)", fontsize=8)
    ax5r.set_ylabel("Net ret % / Sharpe", fontsize=8)
    ax5.set_title("OOS Metrics per Window", fontsize=10)
    ax5.set_ylim(0, 80)
    ax5.tick_params(labelsize=7); ax5r.tick_params(labelsize=7)
    ax5.grid(True, alpha=0.15, axis="y")

    # ── 6. OOS Long vs Short ─────────────────────────────────────────
    ax6 = fig.add_subplot(3, 3, 8)
    x = np.arange(n_win); w = 0.3
    long_c  = ["#378ADD" if v >= 0 else "#D85A30"
               for v in summary["oos_long_net"]]
    short_c = ["#639922" if v >= 0 else "#E8601C"
               for v in summary["oos_short_net"]]
    ax6.bar(x - w/2, summary["oos_long_net"],  w,
            color=long_c,  alpha=0.85, label="Long net")
    ax6.bar(x + w/2, summary["oos_short_net"], w,
            color=short_c, alpha=0.85, label="Short net")
    ax6.axhline(0, color="black", linewidth=0.8)
    ax6.set_xticks(x); ax6.set_xticklabels(wlabels, fontsize=9)
    ax6.set_ylabel("Net return (%)", fontsize=8)
    ax6.set_title("OOS Long vs Short Net Return per Window", fontsize=10)
    ax6.legend(fontsize=8); ax6.tick_params(labelsize=7)
    ax6.grid(True, alpha=0.15, axis="y")

    # ── 7. Summary table ─────────────────────────────────────────────
    ax7 = fig.add_subplot(3, 3, 9)
    ax7.axis("off")
    if not oos_all.empty:
        cs = trade_stats(oos_all)
        rows = [
            ["Metric",           "OOS Combined"],
            ["Total OOS trades", f"{cs['n']}"],
            ["Win rate",         f"{cs['win_rate']:.1%}"],
            ["Avg net/trade",    f"{cs['avg_ret']:+.4f}%"],
            ["Price gross",      f"{cs['gross_ret']:+.2f}%"],
            ["Funding P&L",      f"{cs['fund_pnl']:+.2f}%"],
            ["Total net",        f"{cs['total_ret']:+.2f}%"],
            ["Ann. Sharpe",      f"{cs['sharpe']:.2f}"],
            ["Max drawdown",     f"{cs['max_dd']:+.4f}"],
            ["WFA windows",      f"{n_win}"],
            ["IS / OOS",         f"{IS_DAYS}d / {OOS_DAYS}d"],
            ["Fee (each side)",  f"{FEE_MAKER}%"],
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
    plt.savefig("wfa_rolling_results.png", dpi=150, bbox_inches="tight")
    print("Chart saved: wfa_rolling_results.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # ── PyCharm: edit these two lines directly ──────────────────────
    DOWNLOAD = True              # True = download fresh data
    CSV_PATH = "hype_18m.csv"    # used when DOWNLOAD = False
    # ────────────────────────────────────────────────────────────────

    # Command-line override (optional)
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        grp = parser.add_mutually_exclusive_group(required=True)
        grp.add_argument("--download", action="store_true")
        grp.add_argument("--csv", type=str)
        args = parser.parse_args()
        DOWNLOAD = args.download
        if args.csv:
            CSV_PATH = args.csv

    # Load / download data
    if DOWNLOAD:
        df = download_and_save(CSV_PATH)
    else:
        df = load_csv(CSV_PATH)

    days = (df["hour"].max() - df["hour"].min()).days
    print(f"\nData span: {days} days  ({days/30:.1f} months)")
    exp_windows = max(0, (days - IS_DAYS) // OOS_DAYS)
    print(f"Expected WFA windows: {exp_windows}")
    if exp_windows < 3:
        print(f"WARNING: only {exp_windows} window(s) possible. "
              f"Need >= {IS_DAYS + 3*OOS_DAYS} days ({(IS_DAYS+3*OOS_DAYS)/30:.0f}m) "
              f"for 3 windows.")

    # Run WFA
    oos_all, summary, tail_df = run_rolling_wfa(df)

    # Combined OOS summary
    if not oos_all.empty:
        cs = trade_stats(oos_all)
        print(f"\n{'='*65}")
        print("OOS COMBINED RESULTS")
        print(f"{'='*65}")
        print(f"  Trades       : {cs['n']}")
        print(f"  Win rate     : {cs['win_rate']:.1%}")
        print(f"  Avg net/trade: {cs['avg_ret']:+.4f}%")
        print(f"  Price gross  : {cs['gross_ret']:+.2f}%")
        print(f"  Funding P&L  : {cs['fund_pnl']:+.2f}%")
        print(f"  Total net    : {cs['total_ret']:+.2f}%")
        print(f"  Ann. Sharpe  : {cs['sharpe']:.2f}")
        print(f"  Max drawdown : {cs['max_dd']:+.4f}")

        print(f"\n{'='*65}")
        print("PER-WINDOW TABLE")
        print(f"{'='*65}")
        cols = ["window", "best_threshold", "is_sharpe", "oos_sharpe",
                "oos_win_rate", "oos_total_ret", "oos_gross_ret",
                "oos_fund_pnl", "oos_max_dd",
                "oos_long_net", "oos_short_net"]
        print(summary[cols].to_string(index=False,
              float_format=lambda x: f"{x:+.3f}"))

    # Save files
    if not oos_all.empty:
        oos_all.to_csv("wfa_rolling_trades.csv", index=False)
    summary.to_csv("wfa_rolling_summary.csv", index=False)
    if not tail_df.empty:
        tail_df.to_csv("wfa_rolling_tail.csv", index=False)

    print("\nGenerating chart...")
    plot_results(df, oos_all, summary, tail_df)

    print("\nDone. Files saved:")
    print("  hype_18m.csv")
    print("  wfa_rolling_trades.csv")
    print("  wfa_rolling_summary.csv")
    print("  wfa_rolling_tail.csv")
    print("  wfa_rolling_results.png")


if __name__ == "__main__":
    main()
