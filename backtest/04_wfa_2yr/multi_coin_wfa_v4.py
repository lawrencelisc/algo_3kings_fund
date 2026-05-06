"""
Multi-Coin D1 Extreme Percentile — Rolling WFA v4
==================================================
42 coins: BTC ETH SOL HYPE TON ETC ALGO FET WLD WIF SNX GMX PENDLE IMX
          SAND PYTH GALA FIL XLM TRUMP ONDO AVAX ADA DOT LINK UNI INJ
          ATOM CRV LDO ARB OP NEAR APT TIA SEI SUI JUP DOGE LTC AXS XRP

Download strategy (anti-ban):
  - Conservative sleep between every API call
  - Jitter (random delay) on each request
  - Exponential backoff on 429 / errors
  - Resume: skip coins whose CSV already exists
  - Per-coin download log so partial runs are recoverable
  - Respects Binance X-MBX-USED-WEIGHT header

WFA: IS=120d, OOS=60d, Rolling (no IS overlap)

Install:
    pip install ccxt requests pandas numpy matplotlib scipy

Run (first time — downloads everything):
    python multi_coin_wfa_v4.py

Run (resume / skip already-downloaded coins):
    python multi_coin_wfa_v4.py --skip-download

Run (WFA only on specific coins):
    python multi_coin_wfa_v4.py --skip-download --coins BTC ETH SOL
"""

import sys, os, time, argparse, random, requests
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

# ── Coin list ──────────────────────────────────────────────────────────
ALL_COINS = [
    "BTC","ETH","SOL","HYPE",
    "TON","ETC","ALGO","FET","WLD","WIF","SNX","GMX","PENDLE","IMX",
    "SAND","PYTH","GALA","FIL","XLM","TRUMP","ONDO","AVAX","ADA","DOT",
    "LINK","UNI","INJ","ATOM","CRV","LDO","ARB","OP","NEAR","APT","TIA",
    "SEI","SUI","JUP","DOGE","LTC","AXS","XRP",
]

# Binance USDT-M perp symbol map (override where symbol differs)
BINANCE_SYMBOL_OVERRIDES = {
    "TRUMP": "TRUMPUSDT",   # verify listing
}

# WFA start dates (older coins get 3yr history)
WFA_STARTS = {
    "HYPE": datetime(2024, 11,  1, tzinfo=timezone.utc),
    "PYTH": datetime(2023, 11,  1, tzinfo=timezone.utc),
    "JUP":  datetime(2024,  1,  1, tzinfo=timezone.utc),
    "TIA":  datetime(2023, 10,  1, tzinfo=timezone.utc),
    "SEI":  datetime(2023,  8,  1, tzinfo=timezone.utc),
    "WLD":  datetime(2023,  7,  1, tzinfo=timezone.utc),
    "TRUMP":datetime(2025,  1,  1, tzinfo=timezone.utc),
    "ONDO": datetime(2024,  1,  1, tzinfo=timezone.utc),
    "WIF":  datetime(2024,  1,  1, tzinfo=timezone.utc),
    # Default for all others: 3 years back
}
DEFAULT_WFA_START = datetime(2022, 1, 1, tzinfo=timezone.utc)

# ── Rate limiting config ───────────────────────────────────────────────
SLEEP_BETWEEN_CALLS   = 1.2    # base sleep seconds between API calls
SLEEP_JITTER          = 0.6    # random jitter 0..JITTER added each call
SLEEP_BETWEEN_COINS   = 5.0    # extra pause between coins
SLEEP_ON_429          = 60.0   # sleep on rate limit error
MAX_RETRIES           = 5      # retry attempts per batch
BACKOFF_FACTOR        = 2.0    # exponential backoff multiplier

IS_DAYS        = 120
OOS_DAYS       = 60
THRESHOLD_GRID = [0.02, 0.03, 0.05, 0.07, 0.10]
WINSORIZE_LO   = 0.01
WINSORIZE_HI   = 0.99
FEE_MAKER      = 0.0135

HL_API = "https://api.hyperliquid.xyz/info"
# ──────────────────────────────────────────────────────────────────────


def safe_sleep(base: float, jitter: float = SLEEP_JITTER):
    """Sleep base + random jitter to avoid pattern detection."""
    t = base + random.uniform(0, jitter)
    time.sleep(t)


def binance_symbol(coin: str) -> str:
    if coin in BINANCE_SYMBOL_OVERRIDES:
        return BINANCE_SYMBOL_OVERRIDES[coin]
    return f"{coin}USDT"


def get_wfa_start(coin: str) -> datetime:
    return WFA_STARTS.get(coin, DEFAULT_WFA_START)


# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — SAFE DOWNLOAD HELPERS
# ════════════════════════════════════════════════════════════════════════

def binance_get(url: str, params: dict, coin: str = "") -> list | dict | None:
    """
    GET request to Binance with exponential backoff and weight monitoring.
    Returns parsed JSON or None on permanent failure.
    """
    sleep_t = SLEEP_BETWEEN_CALLS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            safe_sleep(sleep_t)
            r = requests.get(url, params=params, timeout=15)

            # Monitor rate limit weight
            used_weight = int(r.headers.get("X-MBX-USED-WEIGHT-1M", 0))
            if used_weight > 900:   # >75% of 1200 limit → slow down
                extra = 10.0
                print(f"    ⚠ Weight {used_weight}/1200 — pausing {extra}s")
                time.sleep(extra)

            if r.status_code == 429:
                wait = SLEEP_ON_429 * attempt
                print(f"    ⚠ Rate limited (429) [{coin}] — sleeping {wait}s")
                time.sleep(wait)
                continue

            if r.status_code == 418:   # IP banned
                print(f"    ✗ IP banned (418) — sleeping 5 min")
                time.sleep(300)
                continue

            if r.status_code != 200:
                print(f"    ✗ HTTP {r.status_code} [{coin}] attempt {attempt}")
                sleep_t *= BACKOFF_FACTOR
                continue

            return r.json()

        except requests.exceptions.Timeout:
            print(f"    Timeout [{coin}] attempt {attempt}")
            sleep_t *= BACKOFF_FACTOR
        except Exception as e:
            print(f"    Error [{coin}] attempt {attempt}: {e}")
            sleep_t *= BACKOFF_FACTOR

    print(f"    ✗ Failed after {MAX_RETRIES} attempts [{coin}]")
    return None


def fetch_ohlcv_binance(coin: str, since_ms: int) -> pd.DataFrame | None:
    """Fetch hourly OHLCV from Binance futures (safe, batched)."""
    sym    = binance_symbol(coin)
    url    = "https://fapi.binance.com/fapi/v1/klines"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # First check the symbol exists
    check = binance_get(
        "https://fapi.binance.com/fapi/v1/exchangeInfo", {}, coin)
    if check is None:
        print(f"    ✗ Cannot reach Binance")
        return None
    valid_syms = {s["symbol"] for s in check.get("symbols", [])}
    if sym not in valid_syms:
        print(f"    ✗ {sym} not listed on Binance futures — skipping")
        return None

    all_bars, cursor, batch = [], since_ms, 0
    print(f"  [{coin}] OHLCV from "
          f"{datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}...")

    while cursor < now_ms:
        batch += 1
        data = binance_get(url, {
            "symbol":    sym,
            "interval":  "1h",
            "startTime": cursor,
            "limit":     1000,
        }, coin)
        if not data:
            break
        all_bars.extend(data)
        last_t = data[-1][0]
        pct    = min((last_t - since_ms) / max(now_ms - since_ms, 1) * 100, 100)
        print(f"    Batch {batch:3d}: {len(data):4d} bars  "
              f"{datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
              f"  ({pct:.0f}%)")
        if len(data) < 1000:
            break
        cursor = last_t + 3_600_001

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars)
    df["hour"]  = pd.to_datetime(df[0], unit="ms", utc=True).dt.floor("h")
    df["open"]  = df[1].astype(float)
    df["high"]  = df[2].astype(float)
    df["low"]   = df[3].astype(float)
    df["close"] = df[4].astype(float)
    df = df[["hour","open","high","low","close"]].drop_duplicates("hour")
    df = df.sort_values("hour").reset_index(drop=True)
    print(f"    Total: {len(df)} bars  "
          f"({df['hour'].min().strftime('%Y-%m-%d')} → "
          f"{df['hour'].max().strftime('%Y-%m-%d')})")
    return df


def fetch_funding_binance(coin: str, since_ms: int) -> pd.DataFrame | None:
    """Fetch Binance funding history, resample to 1h."""
    sym    = binance_symbol(coin)
    url    = "https://fapi.binance.com/fapi/v1/fundingRate"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rates, cursor, batch = [], since_ms, 0
    print(f"  [{coin}] Funding history...")

    while cursor < now_ms:
        batch += 1
        data = binance_get(url, {
            "symbol":    sym,
            "startTime": cursor,
            "limit":     1000,
        }, coin)
        if not data:
            break
        all_rates.extend(data)
        last_t = data[-1]["fundingTime"]
        print(f"    Batch {batch:3d}: {len(data):4d} records  "
              f"{datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
        if len(data) < 1000:
            break
        cursor = last_t + 1

    if not all_rates:
        return None

    df = pd.DataFrame(all_rates)
    df["hour"]            = pd.to_datetime(df["fundingTime"],
                                           unit="ms", utc=True).dt.floor("h")
    df["funding_rate_1h"] = df["fundingRate"].astype(float) / 8
    df["premium"]         = df["fundingRate"].astype(float)

    # Forward-fill every hour
    full = pd.DataFrame({"hour": pd.date_range(
        df["hour"].min(), df["hour"].max(), freq="1h", tz="UTC")})
    df = (full.merge(df[["hour","funding_rate_1h","premium"]],
                     on="hour", how="left")
          .ffill()
          .drop_duplicates("hour")
          .reset_index(drop=True))
    print(f"    Total (1h): {len(df)} records")
    return df


def hl_post(payload: dict, retries: int = 5) -> list | dict:
    """POST to Hyperliquid with backoff."""
    sleep_t = SLEEP_BETWEEN_CALLS
    for attempt in range(1, retries + 1):
        try:
            safe_sleep(sleep_t)
            r = requests.post(HL_API, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries:
                raise
            sleep_t *= BACKOFF_FACTOR
            print(f"    HL retry {attempt}: {e}")


def fetch_hype_data(since_ms: int) -> pd.DataFrame | None:
    """Fetch HYPE via CCXT (candles) + HL native API (premium/funding)."""
    try:
        import ccxt
        exchange = ccxt.hyperliquid({"enableRateLimit": True})
        all_bars, cursor = [], since_ms
        now_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
        batch   = 0
        print("  [HYPE] OHLCV via CCXT hyperliquid...")

        while cursor < now_ms:
            batch += 1
            safe_sleep(SLEEP_BETWEEN_CALLS)
            data = exchange.fetch_ohlcv(
                "HYPE/USDC:USDC", "1h", since=cursor, limit=1000)
            if not data:
                break
            all_bars.extend(data)
            last_t = data[-1][0]
            pct    = min((last_t - since_ms) / max(now_ms - since_ms, 1) * 100, 100)
            print(f"    Batch {batch:3d}: {len(data):4d} bars  "
                  f"{datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
                  f"  ({pct:.0f}%)")
            if len(data) < 1000:
                break
            cursor = last_t + 3_600_001

        price_df = pd.DataFrame(all_bars,
                                columns=["ts","open","high","low","close","vol"])
        price_df["hour"] = pd.to_datetime(
            price_df["ts"], unit="ms", utc=True).dt.floor("h")
        price_df = price_df.drop_duplicates("hour").sort_values("hour")

    except Exception as e:
        print(f"    CCXT HYPE failed: {e} — falling back to HL REST")
        price_df = None

    # HL native funding + premium (no 5000 limit)
    print("  [HYPE] Native funding + premium...")
    all_r, cursor, batch = [], since_ms, 0
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
              f"{datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
        if len(new) < 500:
            break
        cursor = last_t + 1

    fund_df = pd.DataFrame(all_r)
    fund_df["hour"]         = pd.to_datetime(fund_df["time"],
                                             unit="ms", utc=True).dt.floor("h")
    fund_df["funding_rate"] = fund_df["fundingRate"].astype(float)
    fund_df["premium"]      = fund_df["premium"].astype(float)
    fund_df = fund_df[["hour","funding_rate","premium"]].drop_duplicates("hour")

    if price_df is None:
        print("    No HYPE price data available")
        return None

    return build_aligned(price_df, fund_df, "funding_rate")


def build_aligned(price_df: pd.DataFrame,
                  fund_df:  pd.DataFrame,
                  fund_col: str = "funding_rate_1h") -> pd.DataFrame:
    price_df = price_df.copy()
    fund_hr  = fund_df.drop_duplicates("hour").copy()

    price_df["log_return_pct"] = (
        np.log(price_df["close"] / price_df["close"].shift(1)) * 100)

    merged = (price_df[["hour","close","log_return_pct"]]
              .merge(fund_hr[["hour", fund_col, "premium"]],
                     on="hour", how="inner")
              .rename(columns={fund_col: "funding_rate"})
              .dropna(subset=["log_return_pct","premium"])
              .reset_index(drop=True))
    print(f"    Aligned: {len(merged)} rows  "
          f"({merged['hour'].min().strftime('%Y-%m-%d')} → "
          f"{merged['hour'].max().strftime('%Y-%m-%d')})")
    return merged


def download_coin(coin: str) -> pd.DataFrame | None:
    """Download one coin. Returns DataFrame or None if unavailable."""
    since_ms = int(get_wfa_start(coin).timestamp() * 1000)
    csv_path = f"data/{coin}_aligned.csv"

    print(f"\n{'─'*55}")
    print(f"Downloading {coin}")
    print(f"{'─'*55}")

    if coin == "HYPE":
        df = fetch_hype_data(since_ms)
    else:
        price_df = fetch_ohlcv_binance(coin, since_ms)
        if price_df is None:
            print(f"  ✗ {coin}: no price data — skipping")
            return None
        fund_df = fetch_funding_binance(coin, since_ms)
        if fund_df is None:
            print(f"  ✗ {coin}: no funding data — skipping")
            return None
        df = build_aligned(price_df, fund_df, "funding_rate_1h")

    if df is None or df.empty:
        print(f"  ✗ {coin}: empty data — skipping")
        return None

    df.to_csv(csv_path, index=False)
    days = (df["hour"].max() - df["hour"].min()).days
    print(f"  ✓ Saved {csv_path}  ({len(df)} rows, {days}d)")
    return df


def load_coin(coin: str) -> pd.DataFrame | None:
    csv_path = f"data/{coin}_aligned.csv"
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df["hour"] = pd.to_datetime(df["hour"], utc=True)
        for old in ["funding_rate_pct","funding_rate_1h"]:
            if old in df.columns and "funding_rate" not in df.columns:
                df.rename(columns={old: "funding_rate"}, inplace=True)
        if "funding_rate" in df.columns and df["funding_rate"].abs().median() > 0.01:
            df["funding_rate"] = df["funding_rate"] / 100
        df = df.sort_values("hour").reset_index(drop=True)
        days = (df["hour"].max() - df["hour"].min()).days
        print(f"  ✓ Loaded {csv_path}  ({len(df)} rows, {days}d  "
              f"{df['hour'].min().strftime('%Y-%m-%d')} → "
              f"{df['hour'].max().strftime('%Y-%m-%d')})")
        return df
    except Exception as e:
        print(f"  ✗ Error loading {csv_path}: {e}")
        return None


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
    wfa_start = get_wfa_start(coin)
    df    = df[df["hour"] >= wfa_start].copy().reset_index(drop=True)
    today = df["hour"].max()
    days  = (today - wfa_start).days
    is_td = timedelta(days=IS_DAYS)
    oos_td= timedelta(days=OOS_DAYS)
    exp   = max(0, (days - IS_DAYS) // OOS_DAYS)

    print(f"\n  {coin}  IS={IS_DAYS}d OOS={OOS_DAYS}d data={days}d  "
          f"expected_windows={exp}")

    windows, all_oos = [], []
    is_start = wfa_start
    w = 1

    while True:
        is_end  = is_start + is_td
        oos_end = is_end   + oos_td
        if oos_end > today:
            break
        is_df  = df[(df["hour"] >= is_start) & (df["hour"] < is_end)].copy().reset_index(drop=True)
        oos_df = df[(df["hour"] >= is_end)   & (df["hour"] < oos_end)].copy().reset_index(drop=True)
        if len(is_df) < 500 or len(oos_df) < 200:
            is_start = is_end; w += 1; continue

        best_th, grid = optimise_is(is_df)
        is_t  = run_backtest(is_df,  best_th)
        oos_t = run_backtest(oos_df, best_th)
        is_s  = trade_stats(is_t)
        oos_s = trade_stats(oos_t)
        ls_l  = trade_stats(oos_t[oos_t["signal"] ==  1])
        ls_s  = trade_stats(oos_t[oos_t["signal"] == -1])

        print(f"  W{w:02d} IS:{is_start.strftime('%Y-%m-%d')}→{is_end.strftime('%Y-%m-%d')}"
              f"  OOS:{is_end.strftime('%Y-%m-%d')}→{oos_end.strftime('%Y-%m-%d')}"
              f"  best={best_th:.0%}  IS_Sh={is_s['sharpe']:+.1f}"
              f"  OOS_Sh={oos_s['sharpe']:+.1f}  Net={oos_s['total_ret']:+.1f}%"
              f"  p={oos_s['p_value']:.3f}")

        oos_t["window"] = w
        all_oos.append(oos_t)
        windows.append({
            "coin": coin, "window": w,
            "is_start": is_start, "is_end": is_end,
            "oos_start": is_end,  "oos_end": oos_end,
            "best_threshold": best_th,
            "is_sharpe": is_s["sharpe"],    "is_total_ret": is_s["total_ret"],
            "is_win_rate": is_s["win_rate"],"is_n": is_s["n"],
            "oos_n": oos_s["n"],            "oos_sharpe": oos_s["sharpe"],
            "oos_win_rate": oos_s["win_rate"], "oos_total_ret": oos_s["total_ret"],
            "oos_gross_ret": oos_s["gross_ret"], "oos_fund_pnl": oos_s["fund_pnl"],
            "oos_max_dd": oos_s["max_dd"],  "oos_p_value": oos_s["p_value"],
            "oos_long_net": ls_l["total_ret"], "oos_short_net": ls_s["total_ret"],
            "oos_long_sharpe": ls_l["sharpe"], "oos_short_sharpe": ls_s["sharpe"],
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
                  f"  n={ts['n']}  Net={ts['total_ret']:+.2f}%")

    oos_all = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    summary = pd.DataFrame(windows)

    if not oos_all.empty:
        cs = trade_stats(oos_all)
        print(f"  {coin} OOS COMBINED  n={cs['n']}  WR={cs['win_rate']:.1%}"
              f"  Net={cs['total_ret']:+.2f}%  Sh={cs['sharpe']:+.2f}"
              f"  p={cs['p_value']:.4f}  MaxDD={cs['max_dd']:+.4f}")

    return oos_all, summary, tail_df


# ════════════════════════════════════════════════════════════════════════
# SECTION 4 — PLOTTING
# ════════════════════════════════════════════════════════════════════════

def plot_multi_coin_comparison(all_summaries: dict, all_oos: dict):
    """Top-level comparison: equity curves + ranking table + scatter."""
    coins = sorted(all_summaries.keys(),
                   key=lambda c: trade_stats(all_oos.get(c, pd.DataFrame()))["sharpe"],
                   reverse=True)
    if not coins:
        return

    # Build colour map
    cmap   = plt.cm.tab20
    colors = {c: cmap(i / max(len(coins)-1,1)) for i, c in enumerate(coins)}

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(
        f"Multi-Coin D1 Premium Mean Reversion — WFA Comparison  "
        f"({len(coins)} coins)\n"
        f"IS={IS_DAYS}d  OOS={OOS_DAYS}d  Rolling  |  "
        f"Maker fee {FEE_MAKER}%  |  Includes funding P&L",
        fontsize=12, fontweight="bold", y=0.99)

    # 1. OOS equity curves
    ax1 = fig.add_subplot(2, 3, (1, 2))
    for coin in coins:
        t  = all_oos.get(coin, pd.DataFrame())
        cs = trade_stats(t)
        if t.empty: continue
        ax1.plot(t["hour"], (t["net_ret"]/100).cumsum(),
                 color=colors[coin], linewidth=1.2, alpha=0.8,
                 label=f"{coin} ({cs['total_ret']:+.0f}%  Sh={cs['sharpe']:.1f})")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax1.set_title("Concatenated OOS Equity Curves (all coins)", fontsize=10)
    ax1.set_ylabel("Cumulative log return", fontsize=8)
    ncol = max(1, len(coins)//6)
    ax1.legend(fontsize=6, ncol=ncol, loc="upper left")
    ax1.tick_params(labelsize=7); ax1.grid(True, alpha=0.12)

    # 2. IS vs OOS Sharpe scatter
    ax2 = fig.add_subplot(2, 3, 3)
    for coin in coins:
        s = all_summaries[coin]
        if s.empty: continue
        ax2.scatter(s["is_sharpe"], s["oos_sharpe"],
                    color=colors[coin], s=60, alpha=0.8,
                    edgecolors="white", linewidths=0.8, zorder=3)
        for _, row in s.iterrows():
            ax2.annotate(f" {coin[:3]}{int(row['window'])}",
                         (row["is_sharpe"], row["oos_sharpe"]),
                         fontsize=5.5, alpha=0.7)
    mn = min(all_summaries[c]["is_sharpe"].min()  for c in coins if not all_summaries[c].empty) - 1
    mx = max(all_summaries[c]["is_sharpe"].max()  for c in coins if not all_summaries[c].empty) + 1
    ax2.plot([mn,mx],[mn,mx],"k--",linewidth=0.8,alpha=0.4,label="IS=OOS")
    ax2.axhline(0,color="gray",linewidth=0.5,linestyle=":")
    ax2.set_xlabel("IS Sharpe",fontsize=8); ax2.set_ylabel("OOS Sharpe",fontsize=8)
    ax2.set_title("IS vs OOS Sharpe — All Coins All Windows",fontsize=10)
    ax2.legend(fontsize=7); ax2.tick_params(labelsize=7); ax2.grid(True,alpha=0.12)

    # 3. OOS Sharpe ranking bar
    ax3 = fig.add_subplot(2, 3, 4)
    sharpes = [(c, trade_stats(all_oos.get(c,pd.DataFrame()))["sharpe"])
               for c in coins if not all_oos.get(c,pd.DataFrame()).empty]
    sharpes.sort(key=lambda x: x[1], reverse=True)
    names_s = [x[0] for x in sharpes]
    vals_s  = [x[1] for x in sharpes]
    bar_c   = ["#378ADD" if v>=0 else "#D85A30" for v in vals_s]
    ax3.barh(names_s, vals_s, color=bar_c, alpha=0.8)
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.axvline(2, color="green", linewidth=0.6, linestyle=":", alpha=0.6)
    ax3.set_xlabel("OOS Ann. Sharpe", fontsize=8)
    ax3.set_title("OOS Sharpe Ranking\n(green = 2.0 target)", fontsize=10)
    ax3.tick_params(labelsize=7); ax3.grid(True, alpha=0.12, axis="x")

    # 4. p-value ranking (significance)
    ax4 = fig.add_subplot(2, 3, 5)
    pvals = [(c, trade_stats(all_oos.get(c,pd.DataFrame()))["p_value"])
             for c in coins if not all_oos.get(c,pd.DataFrame()).empty]
    pvals.sort(key=lambda x: x[1])
    names_p = [x[0] for x in pvals]
    vals_p  = [x[1] for x in pvals]
    bar_p   = ["#d4edda" if v<0.05 else "#fff3cd" if v<0.10 else "#f8d7da"
               for v in vals_p]
    ax4.barh(names_p, vals_p, color=bar_p, alpha=0.9, edgecolor="#aaa", linewidth=0.5)
    ax4.axvline(0.05, color="green",  linewidth=1.0, linestyle="--",
                label="p=0.05 (5% sig)")
    ax4.axvline(0.10, color="orange", linewidth=0.8, linestyle=":",
                label="p=0.10 (10% sig)")
    ax4.set_xlabel("p-value (t-test)", fontsize=8)
    ax4.set_title("OOS Statistical Significance\n(green<0.05, yellow<0.10, red>0.10)", fontsize=10)
    ax4.legend(fontsize=7); ax4.tick_params(labelsize=7)
    ax4.grid(True, alpha=0.12, axis="x")

    # 5. Summary table (top 20 by Sharpe)
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.axis("off")
    top20  = sharpes[:20]
    hdr    = ["Coin","Win","Trades","WR","Net","Sharpe","p-val","MaxDD"]
    rows   = [hdr]
    for coin, _ in top20:
        cs = trade_stats(all_oos.get(coin, pd.DataFrame()))
        sw = len(all_summaries.get(coin, pd.DataFrame()))
        rows.append([coin, str(sw), str(cs["n"]),
                     f"{cs['win_rate']:.0%}",
                     f"{cs['total_ret']:+.0f}%",
                     f"{cs['sharpe']:.2f}",
                     f"{cs['p_value']:.3f}",
                     f"{cs['max_dd']:+.2f}"])
    tbl = ax5.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd")
        if r == 0:
            cell.set_facecolor("#333")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f9f9f9")
        if r > 0 and c == 6:
            try:
                pv = float(rows[r][6])
                cell.set_facecolor("#d4edda" if pv<0.05 else
                                   "#fff3cd" if pv<0.10 else "#f8d7da")
            except: pass
    ax5.set_title("Top 20 Coins by OOS Sharpe", fontsize=10, pad=10)

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig("results/multi_coin_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved: results/multi_coin_comparison.png")


def plot_single_coin(coin, df_full, oos_all, summary, tail_df):
    if summary.empty: return
    n_win  = len(summary)
    colors = [plt.cm.tab10(i % 10) for i in range(n_win)]
    wfa_s  = get_wfa_start(coin)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"{coin}  Rolling WFA  (IS={IS_DAYS}d / OOS={OOS_DAYS}d / {n_win} windows)",
                 fontsize=11, fontweight="bold", y=0.99)

    df_p   = df_full[df_full["hour"] >= wfa_s].copy()
    pw_all = winsorize(df_p["premium"])

    ax1 = fig.add_subplot(2, 3, (1, 2))
    ax1.plot(df_p["hour"], pw_all, color="#ccc", linewidth=0.4, zorder=1)
    for i, row in summary.iterrows():
        ax1.axvspan(row["is_start"],  row["is_end"],  alpha=0.07, color=colors[i])
        ax1.axvspan(row["oos_start"], row["oos_end"], alpha=0.22, color=colors[i],
                    label=f"W{int(row['window'])}")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax1.set_title("Premium Index — IS / OOS Windows", fontsize=9)
    ax1.legend(fontsize=6, ncol=min(n_win+1,8), loc="upper right")
    ax1.tick_params(labelsize=7); ax1.grid(True, alpha=0.12)

    ax2 = fig.add_subplot(2, 3, 3)
    cum = 0.0
    for i, row in summary.iterrows():
        wt = oos_all[oos_all["window"] == row["window"]]
        if wt.empty: continue
        eq = (wt["net_ret"]/100).cumsum() + cum
        ax2.plot(wt["hour"], eq, color=colors[i], linewidth=1.2,
                 label=f"W{int(row['window'])} ({row['oos_total_ret']:+.1f}%)")
        cum = eq.iloc[-1]
    if not tail_df.empty:
        ts  = trade_stats(tail_df)
        teq = (tail_df["net_ret"]/100).cumsum() + cum
        ax2.plot(tail_df["hour"], teq, color="gray", linewidth=1.0,
                 linestyle="--", label=f"Tail ({ts['total_ret']:+.1f}%)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax2.set_title("OOS Equity Curve", fontsize=9)
    ax2.legend(fontsize=6, loc="upper left", ncol=2)
    ax2.tick_params(labelsize=7); ax2.grid(True, alpha=0.12)

    ax3 = fig.add_subplot(2, 3, 4)
    ax3.scatter(summary["is_sharpe"], summary["oos_sharpe"],
                c=colors[:n_win], s=80, zorder=3, edgecolors="white")
    for i, row in summary.iterrows():
        ax3.annotate(f" W{int(row['window'])}",
                     (row["is_sharpe"], row["oos_sharpe"]), fontsize=7)
    mn = min(summary[["is_sharpe","oos_sharpe"]].min()) - 1
    mx = max(summary[["is_sharpe","oos_sharpe"]].max()) + 1
    ax3.plot([mn,mx],[mn,mx],"k--",linewidth=0.8,alpha=0.4)
    ax3.axhline(0,color="gray",linewidth=0.5,linestyle=":")
    ax3.set_xlabel("IS Sharpe",fontsize=8); ax3.set_ylabel("OOS Sharpe",fontsize=8)
    ax3.set_title("IS vs OOS Sharpe",fontsize=9)
    ax3.tick_params(labelsize=7); ax3.grid(True,alpha=0.12)

    ax4 = fig.add_subplot(2, 3, 5)
    x = np.arange(n_win); w = 0.3
    lc = ["#378ADD" if v>=0 else "#D85A30" for v in summary["oos_long_net"]]
    sc = ["#639922" if v>=0 else "#E8601C" for v in summary["oos_short_net"]]
    ax4.bar(x-w/2, summary["oos_long_net"],  w, color=lc,  alpha=0.85, label="Long")
    ax4.bar(x+w/2, summary["oos_short_net"], w, color=sc,  alpha=0.85, label="Short")
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"W{int(w)}" for w in summary["window"]], fontsize=7)
    ax4.set_ylabel("Net return (%)", fontsize=8)
    ax4.set_title("OOS Long vs Short", fontsize=9)
    ax4.legend(fontsize=7); ax4.tick_params(labelsize=7)
    ax4.grid(True, alpha=0.12, axis="y")

    ax5 = fig.add_subplot(2, 3, 6)
    ax5.axis("off")
    if not oos_all.empty:
        cs = trade_stats(oos_all)
        rows = [["Metric","OOS Combined"],
                ["Trades",       f"{cs['n']}"],
                ["Win rate",     f"{cs['win_rate']:.1%}"],
                ["Price gross",  f"{cs['gross_ret']:+.2f}%"],
                ["Funding P&L",  f"{cs['fund_pnl']:+.2f}%"],
                ["Total net",    f"{cs['total_ret']:+.2f}%"],
                ["Ann. Sharpe",  f"{cs['sharpe']:.2f}"],
                ["p-value",      f"{cs['p_value']:.4f}"],
                ["Max DD",       f"{cs['max_dd']:+.4f}"],
                ["Windows",      f"{n_win}"]]
        tbl = ax5.table(cellText=rows[1:], colLabels=rows[0],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#ddd")
            if r == 0:
                cell.set_facecolor("#378ADD")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f5f5f5")
            if r == 8:
                pv = cs["p_value"]
                cell.set_facecolor("#d4edda" if pv<0.05 else
                                   "#fff3cd" if pv<0.10 else "#f8d7da")
    plt.tight_layout(rect=[0,0,1,0.97])
    out = f"results/{coin}_wfa_results.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {out}")


# ════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    # ── PyCharm: edit here ────────────────────────────────────────
    SKIP_DOWNLOAD = False       # True = reuse existing CSVs
    COINS         = ALL_COINS   # or e.g. ["BTC","ETH"] for subset
    # ─────────────────────────────────────────────────────────────

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip-download", action="store_true")
        parser.add_argument("--coins", nargs="+", default=None)
        args = parser.parse_args()
        SKIP_DOWNLOAD = args.skip_download
        if args.coins:
            COINS = [c.upper() for c in args.coins]

    print("=" * 60)
    print(f"Multi-Coin D1 Rolling WFA  v4")
    print(f"Coins ({len(COINS)}): {', '.join(COINS)}")
    print(f"IS={IS_DAYS}d  OOS={OOS_DAYS}d  Fee={FEE_MAKER}%")
    print(f"Rate limit: {SLEEP_BETWEEN_CALLS}s + {SLEEP_JITTER}s jitter per call")
    print(f"Inter-coin pause: {SLEEP_BETWEEN_COINS}s")
    print("=" * 60)

    # ── Download phase ─────────────────────────────────────────
    coin_data = {}
    for idx, coin in enumerate(COINS):
        csv = f"data/{coin}_aligned.csv"
        if SKIP_DOWNLOAD or os.path.exists(csv):
            print(f"\n[{idx+1}/{len(COINS)}] {coin} — loading from CSV")
            df = load_coin(coin)
        else:
            print(f"\n[{idx+1}/{len(COINS)}] {coin} — downloading"
                  f"  ({len(COINS)-idx-1} coins remaining)")
            try:
                df = download_coin(coin)
            except Exception as e:
                print(f"  ✗ ERROR downloading {coin}: {e}")
                import traceback; traceback.print_exc()
                df = None

            # Pause between coins (except last)
            if idx < len(COINS) - 1 and not SKIP_DOWNLOAD:
                pause = SLEEP_BETWEEN_COINS + random.uniform(0, 3)
                print(f"  Pausing {pause:.1f}s before next coin...")
                time.sleep(pause)

        if df is not None and not df.empty:
            coin_data[coin] = df
        else:
            print(f"  ✗ {coin}: no data — will skip WFA")

    print(f"\nSuccessfully loaded: {len(coin_data)}/{len(COINS)} coins")
    skipped = [c for c in COINS if c not in coin_data]
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")

    # ── WFA phase ──────────────────────────────────────────────
    all_summaries, all_oos_trades = {}, {}
    for coin, df in coin_data.items():
        print(f"\n{'='*55}\nWFA: {coin}\n{'='*55}")
        try:
            oos_all, summary, tail_df = run_rolling_wfa(df, coin)
            if summary.empty:
                print(f"  No windows for {coin} — skipping")
                continue
            all_summaries[coin]  = summary
            all_oos_trades[coin] = oos_all

            if not oos_all.empty:
                oos_all.to_csv(f"results/{coin}_wfa_trades.csv",  index=False)
            summary.to_csv(f"results/{coin}_wfa_summary.csv", index=False)
            if not tail_df.empty:
                tail_df.to_csv(f"results/{coin}_wfa_tail.csv", index=False)

            plot_single_coin(coin, df, oos_all, summary, tail_df)
        except Exception as e:
            print(f"  ERROR in WFA for {coin}: {e}")
            import traceback; traceback.print_exc()

    # ── Comparison chart ───────────────────────────────────────
    valid = {c for c in all_summaries if not all_summaries[c].empty}
    if len(valid) >= 2:
        print(f"\n{'='*55}")
        print(f"Multi-coin comparison chart ({len(valid)} coins)...")
        plot_multi_coin_comparison(
            {c: all_summaries[c]  for c in valid},
            {c: all_oos_trades[c] for c in valid})

    # ── Final summary ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("FINAL SUMMARY (sorted by OOS Sharpe)")
    print(f"{'─'*55}")
    results = []
    for coin in valid:
        cs = trade_stats(all_oos_trades[coin])
        results.append((coin, cs["sharpe"], cs["total_ret"],
                        cs["p_value"], cs["n"], len(all_summaries[coin])))
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Coin':<8} {'Sharpe':>7} {'Net%':>8} {'p-val':>7} "
          f"{'Trades':>7} {'Windows':>8}")
    print("─" * 55)
    for coin, sh, net, pv, n, wins in results:
        sig = "✅" if pv < 0.05 else "🟡" if pv < 0.10 else "❌"
        print(f"{coin:<8} {sh:>+7.2f} {net:>+7.1f}% {pv:>7.3f} "
              f"{n:>7} {wins:>8}  {sig}")

    print(f"\n{'='*55}")
    print("Output files:")
    print("  data/<coin>_aligned.csv          ← raw data")
    print("  results/<coin>_wfa_summary.csv   ← per-window stats")
    print("  results/<coin>_wfa_trades.csv    ← OOS trades")
    print("  results/<coin>_wfa_results.png   ← per-coin chart")
    print("  results/multi_coin_comparison.png ← full comparison")


if __name__ == "__main__":
    main()
