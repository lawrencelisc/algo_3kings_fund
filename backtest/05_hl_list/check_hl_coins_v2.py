"""
Find earliest available candle data for each coin on Hyperliquid
Fixed version: searches from coin listing dates more accurately

Run:
    python check_hl_coins_v2.py
"""

import requests
import time
from datetime import datetime, timezone, timedelta

API = "https://api.hyperliquid.xyz/info"

TARGET_COINS = [
    "GALA","ATOM","IMX","OP","TIA","PENDLE","SOL","LTC","INJ",
    "DOGE","XRP","AVAX","ADA","DOT","LINK","ARB","NEAR","APT",
    "SUI","BTC","ETH","HYPE",
]

def hl_candle(coin, start_ms, end_ms):
    try:
        r = requests.post(API, json={
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": "1h",
                    "startTime": start_ms, "endTime": end_ms}
        }, timeout=15)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

def find_earliest(coin: str) -> datetime | None:
    """
    Try progressively older start dates.
    HL candle API returns up to 5000 bars from startTime.
    Strategy: try each year boundary, find the earliest that returns data.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Try from each quarter going back
    test_starts = [
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        datetime(2022, 1, 1, tzinfo=timezone.utc),
        datetime(2022, 6, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 6, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 6, 1, tzinfo=timezone.utc),
        datetime(2024, 11, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    ]

    earliest_found = None

    for dt in test_starts:
        start_ms = int(dt.timestamp() * 1000)
        # Request 30 days window
        end_ms   = start_ms + 30 * 24 * 3600 * 1000
        data = hl_candle(coin, start_ms, end_ms)
        if data:
            # Found data in this window — record it and keep trying earlier
            ts = data[0]["t"]
            found = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            if earliest_found is None or found < earliest_found:
                earliest_found = found
        time.sleep(0.25)

    return earliest_found


def main():
    print("=" * 65)
    print("Hyperliquid — Earliest Data & WFA Windows Estimate")
    print("=" * 65)

    today    = datetime.now(timezone.utc)
    IS, OOS  = 120, 60

    results  = []

    for coin in TARGET_COINS:
        earliest = find_earliest(coin)
        if earliest:
            days    = (today - earliest).days
            windows = max(0, (days - IS) // OOS)
        else:
            days, windows = 0, 0

        results.append((coin, earliest, days, windows))
        status = (f"{earliest.strftime('%Y-%m-%d'):<14} "
                  f"{days:>4}d  ~{windows:>2} windows"
                  if earliest else "  no data found")
        print(f"  {coin:<10} {status}")
        time.sleep(0.3)

    # Summary table
    print("\n" + "=" * 65)
    print("SUMMARY — sorted by days available")
    print(f"{'Coin':<10} {'Earliest':<14} {'Days':>6} {'WFA Windows':>12}  Viable?")
    print("─" * 55)
    results.sort(key=lambda x: x[2], reverse=True)
    for coin, earliest, days, windows in results:
        viable  = "✅" if windows >= 3 else "⚠️" if windows >= 1 else "❌"
        date_s  = earliest.strftime("%Y-%m-%d") if earliest else "N/A"
        print(f"  {coin:<10} {date_s:<14} {days:>6} {windows:>12}  {viable}")

    print()
    print("Target: ≥3 windows for meaningful WFA")
    good = [(c, d, w) for c, e, d, w in results if w >= 3]
    print(f"Coins with ≥3 windows: {len(good)}")
    for c, d, w in good:
        print(f"  {c:<10} {d}d  {w} windows")


if __name__ == "__main__":
    main()
