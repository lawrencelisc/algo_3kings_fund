"""
Multi-Coin D1 Extreme Percentile — Rolling WFA v3
==================================================
全部幣種改用 CCXT 下載，包括 HYPE：
  BTC  : Binance USDT perp  BTC/USDT:USDT
  ETH  : Binance USDT perp  ETH/USDT:USDT
  SOL  : Binance USDT perp  SOL/USDT:USDT
  HYPE : Hyperliquid perp   HYPE/USDC:USDC  ← CCXT 直接支持

Premium index:
  BTC/ETH/SOL : funding_rate proxy (Binance 8h → 除以 8 得 1h)
  HYPE        : Hyperliquid 原生 premium（fundingHistory API）

WFA: IS=120d, OOS=60d, Rolling (no IS overlap)

Install:
  pip install ccxt requests pandas numpy matplotlib scipy

Run:
  python multi_coin_wfa.py                  # download + WFA
  python multi_coin_wfa.py --skip-download  # WFA only (reuse CSVs)
"""

import sys, os, time, argparse, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from scipy import stats

os.makedirs("data",    exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
COINS = ["BTC", "ETH", "SOL", "HYPE"]

CCXT_SYMBOLS = {
    "BTC":  ("binanceusdm", "BTC/USDT:USDT"),
    "ETH":  ("binanceusdm", "ETH/USDT:USDT"),
    "SOL":  ("binanceusdm", "SOL/USDT:USDT"),
    "HYPE": ("hyperliquid", "HYPE/USDC:USDC"),   # ← CCXT native
}

WFA_STARTS = {
    "BTC":  datetime(2022,  1,  1, tzinfo=timezone.utc),
    "ETH":  datetime(2022,  1,  1, tzinfo=timezone.utc),
    "SOL":  datetime(2022,  1,  1, tzinfo=timezone.utc),
    "HYPE": datetime(2024, 11,  1, tzinfo=timezone.utc),
}

IS_DAYS        = 120
OOS_DAYS       = 60
THRESHOLD_GRID = [0.02, 0.03, 0.05, 0.07, 0.10]
WINSORIZE_LO   = 0.01
WINSORIZE_HI   = 0.99
FEE_MAKER      = 0.0135

HL_API = "https://api.hyperliquid.xyz/info"
SLEEP  = 0.3
# ───────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA DOWNLOAD
# ════════════════════════════════════════════════════════════════════════

def ccxt_fetch_ohlcv(exchange_id: str, symbol: str,
                     since_ms: int) -> pd.DataFrame:
    """Fetch full hourly OHLCV via CCXT, batched 1000 bars."""
    import ccxt
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    all_bars, cursor = [], since_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    batch  = 0

    print(f"  [{symbol}] OHLCV from "
          f"{datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}...")

    while cursor < now_ms:
        batch += 1
        data = exchange.fetch_ohlcv(symbol, "1h", since=cursor, limit=1000)
        if not data:
            break
        all_bars.extend(data)
        last_t = data[-1][0]
        pct    = min((last_t - since_ms) / max(now_ms - since_ms, 1) * 100, 100)
        print(f"    Batch {batch:3d}: {len(data):4d} bars  "
              f"up to {datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
              f"  ({pct:.0f}%)")
        if len(data) < 1000:
            break
        cursor = last_t + 3_600_001
        time.sleep(SLEEP)

    df = pd.DataFrame(all_bars,
                      columns=["ts","open","high","low","close","volume"])
    df["hour"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("h")
    df = df.drop_duplicates("hour").sort_values("hour").reset_index(drop=True)
    print(f"    Total: {len(df)} bars  "
          f"({df['hour'].min().strftime('%Y-%m-%d')} → "
          f"{df['hour'].max().strftime('%Y-%m-%d')})")
    return df


def ccxt_fetch_funding_binance(symbol: str, since_ms: int) -> pd.DataFrame:
    """
    Fetch Binance funding history, resample to 1h.
    Premium proxy = funding_rate (8h equivalent).
    """
    import ccxt
    exchange = ccxt.binanceusdm({"enableRateLimit": True})

    all_rates, cursor = [], since_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    batch  = 0

    print(f"  [{symbol}] Funding history...")

    while cursor < now_ms:
        batch += 1
        data = exchange.fetch_funding_rate_history(
            symbol, since=cursor, limit=1000)
        if not data:
            break
        all_rates.extend(data)
        last_t = data[-1]["timestamp"]
        print(f"    Batch {batch:3d}: {len(data):4d} records  "
              f"up to {datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
        if len(data) < 1000:
            break
        cursor = last_t + 1
        time.sleep(SLEEP)

    df = pd.DataFrame(all_rates)
    df["hour"]            = pd.to_datetime(df["timestamp"],
                                           unit="ms", utc=True).dt.floor("h")
    df["funding_rate_1h"] = df["fundingRate"].astype(float) / 8
    df["premium"]         = df["fundingRate"].astype(float)   # 8h proxy

    # Forward-fill every hour
    full = pd.DataFrame({"hour": pd.date_range(
        df["hour"].min(), df["hour"].max(), freq="1h", tz="UTC")})
    df = (full.merge(df[["hour","funding_rate_1h","premium"]],
                     on="hour", how="left")
          .ffill()
          .drop_duplicates("hour")
          .reset_index(drop=True))
    print(f"    Total (1h resampled): {len(df)} records")
    return df


def hl_post(payload: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.post(HL_API, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)


def hl_fetch_funding_hype(since_ms: int) -> pd.DataFrame:
    """
    Fetch HYPE native premium + funding from Hyperliquid.
    No batch limit — retrieves full 18-month history.
    """
    all_r, cursor, batch = [], since_ms, 0
    print("  [HYPE] Native funding + premium...")

    while True:
        batch += 1
        data = hl_post({"type": "fundingHistory",
                        "coin": "HYPE", "startTime": cursor})
        if not data:
            break
        new = [r for r in data if r["time"] >= cursor]
        if not new:
            break
        all_r.extend(new)
        last_t = max(r["time"] for r in new)
        print(f"    Batch {batch:3d}: {len(new):4d} records  "
              f"up to {datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
        if len(new) < 500:
            break
        cursor = last_t + 1
        time.sleep(SLEEP)

    df = pd.DataFrame(all_r)
    df["hour"]            = pd.to_datetime(df["time"],
                                           unit="ms", utc=True).dt.floor("h")
    df["funding_rate_1h"] = df["fundingRate"].astype(float)
    df["premium"]         = df["premium"].astype(float)
    df = (df[["hour","funding_rate_1h","premium"]]
          .drop_duplicates("hour")
          .sort_values("hour")
          .reset_index(drop=True))
    print(f"    Total: {len(df)} records  "
          f"({df['hour'].min().strftime('%Y-%m-%d')} → "
          f"{df['hour'].max().strftime('%Y-%m-%d')})")
    return df


def build_aligned(price_df: pd.DataFrame,
                  fund_df:  pd.DataFrame) -> pd.DataFrame:
    price_df = price_df.copy()
    fund_hr  = fund_df.drop_duplicates("hour").copy()
    price_df["log_return_pct"] = (
        np.log(price_df["close"] / price_df["close"].shift(1)) * 100)
    merged = (price_df[["hour","close","log_return_pct"]]
              .merge(fund_hr[["hour","funding_rate_1h","premium"]],
                     on="hour", how="inner")
              .rename(columns={"funding_rate_1h": "funding_rate"})
              .dropna(subset=["log_return_pct","premium"])
              .reset_index(drop=True))
    print(f"  Aligned: {len(merged)} rows  "
          f"({merged['hour'].min().strftime('%Y-%m-%d')} → "
          f"{merged['hour'].max().strftime('%Y-%m-%d')})")
    return merged


def download_coin(coin: str) -> pd.DataFrame:
    since_ms        = int(WFA_STARTS[coin].timestamp() * 1000)
    exc_id, sym     = CCXT_SYMBOLS[coin]
    csv_path        = f"data/{coin}_aligned.csv"

    print(f"\n{'─'*55}")
    print(f"Downloading {coin}  ({exc_id} / {sym})")
    print(f"{'─'*55}")

    price_df = ccxt_fetch_ohlcv(exc_id, sym, since_ms)

    if coin == "HYPE":
        fund_df = hl_fetch_funding_hype(since_ms)
    else:
        fund_df = ccxt_fetch_funding_binance(sym, since_ms)

    df = build_aligned(price_df, fund_df)
    df.to_csv(csv_path, index=False)
    days = (df["hour"].max() - df["hour"].min()).days
    print(f"  Saved: {csv_path}  ({len(df)} rows, {days}d)\n")
    return df


def load_coin(coin: str) -> pd.DataFrame:
    csv_path = f"data/{coin}_aligned.csv"
    df = pd.read_csv(csv_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    for old in ["funding_rate_pct","funding_rate_1h"]:
        if old in df.columns and "funding_rate" not in df.columns:
            df.rename(columns={old: "funding_rate"}, inplace=True)
    if "funding_rate" in df.columns and df["funding_rate"].abs().median() > 0.01:
        df["funding_rate"] = df["funding_rate"] / 100
    df = df.sort_values("hour").reset_index(drop=True)
    days = (df["hour"].max() - df["hour"].min()).days
    print(f"  Loaded {csv_path}  ({len(df)} rows, {days}d  "
          f"{df['hour'].min().strftime('%Y-%m-%d')} → "
          f"{df['hour'].max().strftime('%Y-%m-%d')})")
    return df


# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — STRATEGY
# ════════════════════════════════════════════════════════════════════════

def winsorize(s: pd.Series) -> pd.Series:
    return s.clip(s.quantile(WINSORIZE_LO), s.quantile(WINSORIZE_HI))


def run_backtest(df: pd.DataFrame, threshold: float,
                 fee: float = FEE_MAKER) -> pd.DataFrame:
    df = df.copy()
    df["next_ret"] = df["log_return_pct"].shift(-1)
    df = df.dropna(subset=["next_ret","premium"]).reset_index(drop=True)
    pw   = winsorize(df["premium"])
    q_hi = pw.quantile(1 - threshold)
    q_lo = pw.quantile(threshold)
    rows = []
    for i in range(len(df)):
        p = pw.iloc[i]
        if   p >= q_hi: sig = -1
        elif p <= q_lo: sig =  1
        else: continue
        fund_pnl = -sig * df["funding_rate"].iloc[i] * 100
        raw_ret  =  sig * df["next_ret"].iloc[i]
        rows.append({"hour": df["hour"].iloc[i],
                     "premium": p, "signal": sig,
                     "raw_ret": raw_ret, "fund_pnl": fund_pnl,
                     "net_ret": raw_ret + fund_pnl - 2*fee,
                     "threshold": threshold})
    return pd.DataFrame(rows)


def trade_stats(t: pd.DataFrame, col: str = "net_ret") -> dict:
    if t is None or t.empty:
        return dict(n=0, win_rate=np.nan, avg_ret=np.nan, total_ret=np.nan,
                    sharpe=np.nan, max_dd=np.nan, gross_ret=np.nan,
                    fund_pnl=np.nan, p_value=np.nan)
    r  = t[col]
    eq = (r/100).cumsum()
    dd = eq - eq.cummax()
    sh = r.mean()/r.std()*np.sqrt(24*365) if r.std() > 0 else 0
    _, pv = stats.ttest_1samp(r, 0)
    return dict(n=len(r), win_rate=(r>0).mean(), avg_ret=r.mean(),
                total_ret=r.sum(), sharpe=sh, max_dd=dd.min(),
                gross_ret=t["raw_ret"].sum(), fund_pnl=t["fund_pnl"].sum(),
                p_value=pv)


def optimise_is(is_df: pd.DataFrame) -> tuple[float, dict]:
    best_th, best_sh = THRESHOLD_GRID[0], -np.inf
    all_stats = {}
    for th in THRESHOLD_GRID:
        t = run_backtest(is_df, th)
        s = trade_stats(t)
        all_stats[th] = s
        if s["n"] > 0 and s["sharpe"] > best_sh:
            best_sh, best_th = s["sharpe"], th
    return best_th, all_stats


# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — ROLLING WFA
# ════════════════════════════════════════════════════════════════════════

def run_rolling_wfa(df: pd.DataFrame, coin: str):
    wfa_start = WFA_STARTS[coin]
    df    = df[df["hour"] >= wfa_start].copy().reset_index(drop=True)
    today = df["hour"].max()
    days  = (today - wfa_start).days
    is_td = timedelta(days=IS_DAYS)
    oos_td= timedelta(days=OOS_DAYS)

    print(f"\n  {'─'*55}")
    print(f"  {coin}  IS={IS_DAYS}d  OOS={OOS_DAYS}d  "
          f"data={days}d  exp_windows={max(0,(days-IS_DAYS)//OOS_DAYS)}")
    print(f"  {'─'*55}")

    windows, all_oos = [], []
    is_start = wfa_start
    w = 1

    while True:
        is_end  = is_start + is_td
        oos_end = is_end   + oos_td
        if oos_end > today:
            break
        is_df  = df[(df["hour"] >= is_start) &
                    (df["hour"] < is_end)].copy().reset_index(drop=True)
        oos_df = df[(df["hour"] >= is_end)   &
                    (df["hour"] < oos_end)].copy().reset_index(drop=True)
        if len(is_df) < 500 or len(oos_df) < 200:
            is_start = is_end; w += 1; continue

        best_th, grid = optimise_is(is_df)
        is_t  = run_backtest(is_df,  best_th)
        oos_t = run_backtest(oos_df, best_th)
        is_s  = trade_stats(is_t)
        oos_s = trade_stats(oos_t)
        ls_l  = trade_stats(oos_t[oos_t["signal"] ==  1])
        ls_s  = trade_stats(oos_t[oos_t["signal"] == -1])

        print(f"  W{w:02d}  "
              f"IS:{is_start.strftime('%Y-%m-%d')}→{is_end.strftime('%Y-%m-%d')}  "
              f"OOS:{is_end.strftime('%Y-%m-%d')}→{oos_end.strftime('%Y-%m-%d')}")
        print(f"       best={best_th:.0%}  "
              f"IS Sh={is_s['sharpe']:+.2f}  "
              f"OOS Sh={oos_s['sharpe']:+.2f}  "
              f"OOS Net={oos_s['total_ret']:+.2f}%  "
              f"p={oos_s['p_value']:.3f}  n={oos_s['n']}")
        print(f"       grid: " +
              "  ".join(f"{th:.0%}→{grid[th]['sharpe']:+.1f}"
                        for th in THRESHOLD_GRID))

        oos_t["window"] = w
        all_oos.append(oos_t)
        windows.append({
            "coin": coin, "window": w,
            "is_start": is_start, "is_end": is_end,
            "oos_start": is_end,  "oos_end": oos_end,
            "best_threshold":    best_th,
            "is_n":              is_s["n"],
            "is_sharpe":         is_s["sharpe"],
            "is_win_rate":       is_s["win_rate"],
            "is_total_ret":      is_s["total_ret"],
            "oos_n":             oos_s["n"],
            "oos_sharpe":        oos_s["sharpe"],
            "oos_win_rate":      oos_s["win_rate"],
            "oos_total_ret":     oos_s["total_ret"],
            "oos_gross_ret":     oos_s["gross_ret"],
            "oos_fund_pnl":      oos_s["fund_pnl"],
            "oos_max_dd":        oos_s["max_dd"],
            "oos_p_value":       oos_s["p_value"],
            "oos_long_net":      ls_l["total_ret"],
            "oos_short_net":     ls_s["total_ret"],
            "oos_long_sharpe":   ls_l["sharpe"],
            "oos_short_sharpe":  ls_s["sharpe"],
        })
        is_start = is_end
        w += 1

    # Tail
    tail_df = pd.DataFrame()
    if windows:
        tail_start = windows[-1]["oos_end"]
        raw_tail   = df[df["hour"] >= tail_start].copy().reset_index(drop=True)
        if len(raw_tail) > 50:
            tail_df = run_backtest(raw_tail, windows[-1]["best_threshold"])
            ts = trade_stats(tail_df)
            print(f"  Tail {tail_start.strftime('%Y-%m-%d')}→{today.strftime('%Y-%m-%d')}"
                  f"  n={ts['n']}  Net={ts['total_ret']:+.2f}%"
                  f"  Sh={ts['sharpe']:+.2f}")

    oos_all = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    summary = pd.DataFrame(windows)

    if not oos_all.empty:
        cs = trade_stats(oos_all)
        print(f"\n  {coin} OOS COMBINED"
              f"  n={cs['n']}  WR={cs['win_rate']:.1%}"
              f"  Net={cs['total_ret']:+.2f}%  Sh={cs['sharpe']:+.2f}"
              f"  p={cs['p_value']:.4f}  MaxDD={cs['max_dd']:+.4f}")

    return oos_all, summary, tail_df


# ════════════════════════════════════════════════════════════════════════
# SECTION 4 — PLOTTING
# ════════════════════════════════════════════════════════════════════════

def plot_single_coin(coin, df_full, oos_all, summary, tail_df):
    if summary.empty:
        return
    cmap   = plt.cm.tab10
    n_win  = len(summary)
    colors = [cmap(i % 10) for i in range(n_win)]
    wfa_s  = WFA_STARTS[coin]
    fig    = plt.figure(figsize=(20, 13))
    fig.suptitle(
        f"{coin} D1 Extreme Percentile — Rolling WFA"
        f"  (IS={IS_DAYS}d / OOS={OOS_DAYS}d / {n_win} windows)\n"
        f"Maker fee {FEE_MAKER}%  |  Includes funding P&L  |  "
        f"{'Native premium' if coin=='HYPE' else 'Funding proxy'}",
        fontsize=11, fontweight="bold", y=0.99)

    df_p   = df_full[df_full["hour"] >= wfa_s].copy()
    pw_all = winsorize(df_p["premium"])

    ax1 = fig.add_subplot(3, 3, (1, 2))
    ax1.plot(df_p["hour"], pw_all, color="#ccc", linewidth=0.5, zorder=1)
    for i, row in summary.iterrows():
        ax1.axvspan(row["is_start"],  row["is_end"],
                    alpha=0.08, color=colors[i], zorder=0)
        ax1.axvspan(row["oos_start"], row["oos_end"],
                    alpha=0.25, color=colors[i], zorder=0,
                    label=f"W{int(row['window'])}")
    if not tail_df.empty:
        ax1.axvspan(summary.iloc[-1]["oos_end"], df_p["hour"].max(),
                    alpha=0.10, color="gray", zorder=0, label="Tail")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax1.set_title("Premium Index — IS (light) / OOS (dark)", fontsize=10)
    ax1.set_ylabel("Premium (winsorized)", fontsize=8)
    ax1.legend(fontsize=6, ncol=min(n_win+1,10), loc="upper right")
    ax1.tick_params(labelsize=7); ax1.grid(True, alpha=0.15)

    ax2 = fig.add_subplot(3, 3, 3)
    wlbls = [f"W{int(w)}" for w in summary["window"]]
    ax2.bar(wlbls, summary["best_threshold"]*100,
            color=colors[:n_win], alpha=0.85, width=0.6)
    ax2.set_ylabel("Best IS threshold (%)", fontsize=8)
    ax2.set_title("Optimal Threshold per Window", fontsize=10)
    for i, v in enumerate(summary["best_threshold"]):
        ax2.text(i, v*100+0.05, f"{v:.0%}", ha="center", fontsize=8)
    ax2.tick_params(labelsize=7); ax2.grid(True, alpha=0.15, axis="y")

    ax3 = fig.add_subplot(3, 3, (4, 5))
    cum = 0.0
    for i, row in summary.iterrows():
        wt = oos_all[oos_all["window"] == row["window"]]
        if wt.empty: continue
        eq = (wt["net_ret"]/100).cumsum() + cum
        ax3.plot(wt["hour"], eq, color=colors[i], linewidth=1.3,
                 label=f"W{int(row['window'])} ({row['oos_total_ret']:+.1f}%)")
        cum = eq.iloc[-1]
    if not tail_df.empty:
        ts  = trade_stats(tail_df)
        teq = (tail_df["net_ret"]/100).cumsum() + cum
        ax3.plot(tail_df["hour"], teq, color="gray", linewidth=1.2,
                 linestyle="--", label=f"Tail ({ts['total_ret']:+.1f}%)")
    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax3.set_title("Concatenated OOS Equity Curve (net)", fontsize=10)
    ax3.set_ylabel("Cumulative log return", fontsize=8)
    ax3.legend(fontsize=7, loc="upper left", ncol=3)
    ax3.tick_params(labelsize=7); ax3.grid(True, alpha=0.15)

    ax4 = fig.add_subplot(3, 3, 6)
    ax4.scatter(summary["is_sharpe"], summary["oos_sharpe"],
                c=colors[:n_win], s=120, zorder=3,
                edgecolors="white", linewidths=1.5)
    for i, row in summary.iterrows():
        ax4.annotate(f" W{int(row['window'])}",
                     (row["is_sharpe"], row["oos_sharpe"]), fontsize=8)
    pad = 2
    mn = min(summary[["is_sharpe","oos_sharpe"]].min()) - pad
    mx = max(summary[["is_sharpe","oos_sharpe"]].max()) + pad
    ax4.plot([mn,mx],[mn,mx],"k--",linewidth=0.8,alpha=0.4,label="IS=OOS")
    ax4.axhline(0,color="gray",linewidth=0.5,linestyle=":")
    ax4.axvline(0,color="gray",linewidth=0.5,linestyle=":")
    ax4.set_xlabel("IS Sharpe", fontsize=8)
    ax4.set_ylabel("OOS Sharpe", fontsize=8)
    ax4.set_title("IS vs OOS Sharpe", fontsize=10)
    ax4.legend(fontsize=7); ax4.tick_params(labelsize=7); ax4.grid(True, alpha=0.15)

    ax5 = fig.add_subplot(3, 3, 7)
    x = np.arange(n_win); w = 0.25
    ax5.bar(x-w, summary["oos_win_rate"]*100, w,
            color=colors[:n_win], alpha=0.85)
    ax5r = ax5.twinx()
    ax5r.bar(x,   summary["oos_total_ret"], w,
             color=colors[:n_win], alpha=0.50)
    ax5r.bar(x+w, summary["oos_sharpe"],   w,
             color=colors[:n_win], alpha=0.25, hatch="//")
    ax5.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax5.set_xticks(x); ax5.set_xticklabels(wlbls, fontsize=7)
    ax5.set_ylabel("Win rate (%)", fontsize=8)
    ax5r.set_ylabel("Net ret% / Sharpe", fontsize=8)
    ax5.set_title("OOS per Window\n(bar=WR%, mid=Net%, hatch=Sharpe)", fontsize=10)
    ax5.set_ylim(0, 80); ax5.tick_params(labelsize=7); ax5r.tick_params(labelsize=7)
    ax5.grid(True, alpha=0.15, axis="y")

    ax6 = fig.add_subplot(3, 3, 8)
    x = np.arange(n_win); w = 0.3
    lc = ["#378ADD" if v>=0 else "#D85A30" for v in summary["oos_long_net"]]
    sc = ["#639922" if v>=0 else "#E8601C" for v in summary["oos_short_net"]]
    ax6.bar(x-w/2, summary["oos_long_net"],  w, color=lc,  alpha=0.85, label="Long")
    ax6.bar(x+w/2, summary["oos_short_net"], w, color=sc,  alpha=0.85, label="Short")
    ax6.axhline(0, color="black", linewidth=0.8)
    ax6.set_xticks(x); ax6.set_xticklabels(wlbls, fontsize=7)
    ax6.set_ylabel("Net return (%)", fontsize=8)
    ax6.set_title("OOS Long vs Short Net Return", fontsize=10)
    ax6.legend(fontsize=8); ax6.tick_params(labelsize=7)
    ax6.grid(True, alpha=0.15, axis="y")

    ax7 = fig.add_subplot(3, 3, 9)
    ax7.axis("off")
    if not oos_all.empty:
        cs = trade_stats(oos_all)
        rows = [
            ["Metric",        "OOS Combined"],
            ["Trades",        f"{cs['n']}"],
            ["Win rate",      f"{cs['win_rate']:.1%}"],
            ["Avg net/trade", f"{cs['avg_ret']:+.4f}%"],
            ["Price gross",   f"{cs['gross_ret']:+.2f}%"],
            ["Funding P&L",   f"{cs['fund_pnl']:+.2f}%"],
            ["Total net",     f"{cs['total_ret']:+.2f}%"],
            ["Ann. Sharpe",   f"{cs['sharpe']:.2f}"],
            ["p-value",       f"{cs['p_value']:.4f}"],
            ["Max drawdown",  f"{cs['max_dd']:+.4f}"],
            ["Windows",       f"{n_win}"],
            ["IS / OOS",      f"{IS_DAYS}d / {OOS_DAYS}d"],
        ]
        tbl = ax7.table(cellText=rows[1:], colLabels=rows[0],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.45)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#ddd")
            if r == 0:
                cell.set_facecolor("#378ADD")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f5f5f5")
            if r == 9:
                pv = cs["p_value"]
                cell.set_facecolor("#d4edda" if pv < 0.05 else
                                   "#fff3cd" if pv < 0.10 else "#f8d7da")
        ax7.set_title("OOS Combined Statistics", fontsize=10, pad=10)

    plt.tight_layout(rect=[0,0,1,0.97])
    out = f"results/{coin}_wfa_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {out}")


def plot_multi_coin_comparison(all_summaries, all_oos):
    coins  = [c for c in COINS if c in all_summaries
              and not all_summaries[c].empty]
    colors = {"BTC":"#F7931A","ETH":"#627EEA",
               "SOL":"#9945FF","HYPE":"#00B4D8"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Multi-Coin D1 Premium Mean Reversion — WFA Comparison\n"
        "BTC / ETH / SOL (Binance funding proxy) | HYPE (Hyperliquid native premium)\n"
        f"IS={IS_DAYS}d  OOS={OOS_DAYS}d  Rolling  |  Maker fee {FEE_MAKER}%",
        fontsize=11, fontweight="bold", y=0.99)

    ax = axes[0,0]
    for coin in coins:
        t  = all_oos[coin]
        cs = trade_stats(t)
        if t.empty: continue
        ax.plot(t["hour"], (t["net_ret"]/100).cumsum(),
                color=colors.get(coin,"#555"), linewidth=1.5,
                label=f"{coin} ({cs['total_ret']:+.1f}%  Sh={cs['sharpe']:.1f})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_title("Concatenated OOS Equity Curves", fontsize=10)
    ax.set_ylabel("Cumulative log return", fontsize=8)
    ax.legend(fontsize=8); ax.tick_params(labelsize=7); ax.grid(True, alpha=0.15)

    ax = axes[0,1]
    for coin in coins:
        s = all_summaries[coin]
        ax.plot(range(1, len(s)+1), s["oos_sharpe"],
                "o-", color=colors.get(coin,"#555"),
                linewidth=1.5, markersize=7, label=coin)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(2, color="green", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_xlabel("Window number", fontsize=8)
    ax.set_ylabel("OOS Sharpe", fontsize=8)
    ax.set_title("OOS Sharpe per Window\n(green = Sharpe 2 target)", fontsize=10)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8); ax.grid(True, alpha=0.15)

    ax = axes[1,0]
    for coin in coins:
        s = all_summaries[coin]
        ax.scatter(s["is_sharpe"], s["oos_sharpe"],
                   color=colors.get(coin,"#555"), s=90,
                   edgecolors="white", linewidths=1.2,
                   zorder=3, label=coin)
        for _, row in s.iterrows():
            ax.annotate(f" {coin[0]}{int(row['window'])}",
                        (row["is_sharpe"], row["oos_sharpe"]), fontsize=7)
    mn = min(all_summaries[c]["is_sharpe"].min() for c in coins) - 1
    mx = max(all_summaries[c]["is_sharpe"].max() for c in coins) + 1
    ax.plot([mn,mx],[mn,mx],"k--",linewidth=0.8,alpha=0.4,label="IS=OOS")
    ax.axhline(0,color="gray",linewidth=0.5,linestyle=":")
    ax.set_xlabel("IS Sharpe", fontsize=8); ax.set_ylabel("OOS Sharpe", fontsize=8)
    ax.set_title("IS vs OOS Sharpe — All Coins All Windows", fontsize=10)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7); ax.grid(True, alpha=0.15)

    ax = axes[1,1]
    ax.axis("off")
    hdr  = ["Coin","Win","Trades","WinRate","Net","Sharpe",
            "p-val","MaxDD","Premium"]
    rows = [hdr]
    for coin in coins:
        cs = trade_stats(all_oos[coin])
        pm = "Native" if coin == "HYPE" else "Funding proxy"
        rows.append([coin,
                     str(len(all_summaries[coin])),
                     str(cs["n"]),
                     f"{cs['win_rate']:.1%}",
                     f"{cs['total_ret']:+.2f}%",
                     f"{cs['sharpe']:.2f}",
                     f"{cs['p_value']:.3f}",
                     f"{cs['max_dd']:+.3f}",
                     pm])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd")
        if r == 0:
            cell.set_facecolor("#333")
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0:
            base = colors.get(rows[r][0], "#888")
            cell.set_facecolor(base + "22")
        if r > 0 and c == 6:
            try:
                pv = float(rows[r][6])
                cell.set_facecolor("#d4edda" if pv < 0.05 else
                                   "#fff3cd" if pv < 0.10 else "#f8d7da")
            except: pass
    ax.set_title("Cross-Coin OOS Summary", fontsize=10, pad=12)

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig("results/multi_coin_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved: results/multi_coin_comparison.png")


# ════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    # ── PyCharm: edit here ────────────────────────────────────────
    SKIP_DOWNLOAD = False   # True = reuse existing CSVs
    # ─────────────────────────────────────────────────────────────

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip-download", action="store_true")
        args = parser.parse_args()
        # SKIP_DOWNLOAD = args.skip_download
        SKIP_DOWNLOAD = True

    print("=" * 60)
    print(f"Multi-Coin D1 Rolling WFA  v3")
    print(f"Coins : {', '.join(COINS)}")
    print(f"IS={IS_DAYS}d  OOS={OOS_DAYS}d  Fee={FEE_MAKER}%")
    print(f"HYPE  : CCXT hyperliquid  HYPE/USDC:USDC + HL native premium")
    print(f"Others: CCXT binanceusdm  + funding proxy premium")
    print("=" * 60)

    coin_data = {}
    for coin in COINS:
        csv = f"data/{coin}_aligned.csv"
        use_cache = SKIP_DOWNLOAD or os.path.exists(csv)
        print(f"\n[{coin}] {'Loading' if use_cache else 'Downloading'}...")
        try:
            coin_data[coin] = load_coin(coin) if use_cache else download_coin(coin)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    all_summaries, all_oos_trades = {}, {}
    for coin, df in coin_data.items():
        print(f"\n{'='*60}\nRunning WFA: {coin}\n{'='*60}")
        try:
            oos_all, summary, tail_df = run_rolling_wfa(df, coin)
            all_summaries[coin]  = summary
            all_oos_trades[coin] = oos_all
            if not oos_all.empty:
                oos_all.to_csv(f"results/{coin}_wfa_trades.csv",  index=False)
            if not summary.empty:
                summary.to_csv(f"results/{coin}_wfa_summary.csv", index=False)
            if not tail_df.empty:
                tail_df.to_csv(f"results/{coin}_wfa_tail.csv",    index=False)
            plot_single_coin(coin, df, oos_all, summary, tail_df)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    valid = {c for c in all_summaries if not all_summaries[c].empty}
    if len(valid) > 1:
        print(f"\n{'='*60}\nMulti-coin comparison chart...\n{'='*60}")
        plot_multi_coin_comparison(
            {c: all_summaries[c]  for c in valid},
            {c: all_oos_trades[c] for c in valid})

    print(f"\n{'='*60}")
    print("Done. Output files:")
    print("  data/<coin>_aligned.csv")
    print("  results/<coin>_wfa_summary.csv")
    print("  results/<coin>_wfa_trades.csv")
    print("  results/<coin>_wfa_results.png")
    print("  results/multi_coin_comparison.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
