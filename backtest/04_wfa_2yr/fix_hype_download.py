"""
HYPE Historical Data Downloader — Backward Batching Fix
========================================================
問題：Hyperliquid candleSnapshot API 只返最近 5000 bars
解決：由今日向過去逐批拉取，每批用 endTime 指向更早時段

Run:
    pip install requests pandas
    python fix_hype_download.py

Output:
    data/HYPE_aligned.csv   (完整 18 個月數據)
"""

import os, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

os.makedirs("data", exist_ok=True)

HL_API    = "https://api.hyperliquid.xyz/info"
WFA_START = datetime(2024, 11, 1, tzinfo=timezone.utc)
SLEEP     = 0.3


def hl_post(payload, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(HL_API, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(2)


def fetch_candles_backward(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch hourly candles by walking BACKWARD from end_dt to start_dt.
    Each batch requests up to 5000 bars ending at 'cursor'.
    Stops when we've covered start_dt.
    """
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)
    cursor   = end_ms      # start from now, go backward
    all_c    = []
    batch    = 0

    print(f"Downloading HYPE candles backward: "
          f"{start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}")

    while cursor > start_ms:
        batch += 1
        data = hl_post({
            "type": "candleSnapshot",
            "req":  {"coin": "HYPE", "interval": "1h",
                     "startTime": start_ms,    # always request from WFA_START
                     "endTime":   cursor}       # but cap at cursor
        })
        if not data:
            print(f"  Batch {batch}: empty — stopping")
            break

        # Filter to only bars within our range
        data = [c for c in data if start_ms <= c["t"] <= cursor]
        if not data:
            break

        all_c.extend(data)
        oldest_t = min(c["t"] for c in data)
        newest_t = max(c["t"] for c in data)
        pct = (end_ms - oldest_t) / max(end_ms - start_ms, 1) * 100

        print(f"  Batch {batch:3d}: {len(data):4d} bars  "
              f"{datetime.fromtimestamp(oldest_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
              f" → "
              f"{datetime.fromtimestamp(newest_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
              f"  ({100-pct:.0f}% covered from start)")

        # Move cursor to just before the oldest bar in this batch
        cursor = oldest_t - 3_600_001   # go back 1 hour + 1ms

        if oldest_t <= start_ms:
            print(f"  Reached start date — done")
            break

        time.sleep(SLEEP)

    # Deduplicate and sort
    df = pd.DataFrame(all_c).rename(
        columns={"t":"ts","o":"open","h":"high","l":"low",
                 "c":"close","v":"volume"})
    df["hour"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("h")
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates("hour").sort_values("hour").reset_index(drop=True)

    print(f"\nTotal candles: {len(df)}  "
          f"({df['hour'].min().strftime('%Y-%m-%d')} → "
          f"{df['hour'].max().strftime('%Y-%m-%d')})")
    return df


def fetch_funding_full(start_dt: datetime) -> pd.DataFrame:
    """Fetch all funding history from start_dt (no 5000 limit)."""
    start_ms = int(start_dt.timestamp() * 1000)
    all_r, cursor, batch = [], start_ms, 0
    print("\nDownloading HYPE funding + native premium...")

    while True:
        batch += 1
        data = hl_post({"type": "fundingHistory",
                        "coin": "HYPE", "startTime": cursor})
        if not data: break
        new = [r for r in data if r["time"] >= cursor]
        if not new: break
        all_r.extend(new)
        last_t = max(r["time"] for r in new)
        print(f"  Batch {batch:3d}: {len(new):4d} records  "
              f"up to {datetime.fromtimestamp(last_t/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
        if len(new) < 500: break
        cursor = last_t + 1
        time.sleep(SLEEP)

    df = pd.DataFrame(all_r)
    df["hour"]         = pd.to_datetime(df["time"], unit="ms", utc=True).dt.floor("h")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["premium"]      = df["premium"].astype(float)
    df = (df[["hour","funding_rate","premium"]]
          .drop_duplicates("hour").sort_values("hour").reset_index(drop=True))
    print(f"Total funding: {len(df)} records  "
          f"({df['hour'].min().strftime('%Y-%m-%d')} → "
          f"{df['hour'].max().strftime('%Y-%m-%d')})")
    return df


def main():
    end_dt = datetime.now(timezone.utc)

    # Download
    candles = fetch_candles_backward(WFA_START, end_dt)
    funding = fetch_funding_full(WFA_START)

    # Align
    candles["log_return_pct"] = (
        np.log(candles["close"] / candles["close"].shift(1)) * 100)
    merged = (candles[["hour","close","log_return_pct"]]
              .merge(funding, on="hour", how="inner")
              .dropna(subset=["log_return_pct","premium"])
              .reset_index(drop=True))

    days = (merged["hour"].max() - merged["hour"].min()).days
    print(f"\nAligned: {len(merged)} rows  {days} days")

    # Expected windows
    from datetime import timedelta
    IS, OOS = 120, 60
    exp = max(0, (days - IS) // OOS)
    print(f"Expected WFA windows: {exp}")

    is_start = WFA_START
    for w in range(1, exp+2):
        is_end  = is_start + timedelta(days=IS)
        oos_end = is_end   + timedelta(days=OOS)
        if oos_end > end_dt: break
        print(f"  W{w}: IS [{is_start.strftime('%Y-%m-%d')}→"
              f"{is_end.strftime('%Y-%m-%d')}]  "
              f"OOS [{is_end.strftime('%Y-%m-%d')}→"
              f"{oos_end.strftime('%Y-%m-%d')}]")
        is_start = is_end

    # Save
    out = "data/HYPE_aligned.csv"
    merged.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("Now run multi_coin_wfa.py with SKIP_DOWNLOAD=True")


if __name__ == "__main__":
    main()
