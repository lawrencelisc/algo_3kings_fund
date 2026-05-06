"""
Check which coins are available on Hyperliquid as perpetuals
+ their earliest available candle data (listing date proxy)

Run:
    python check_hl_coins.py
"""

import requests
import time
from datetime import datetime, timezone

API = "https://api.hyperliquid.xyz/info"

TARGET_COINS = [
    # Tier 1 (Sharpe > 1.0 on Binance WFA)
    "GALA", "ATOM", "IMX", "OP",
    # Tier 2
    "TIA", "PENDLE", "SOL", "LTC", "INJ",
    # Others worth checking
    "DOGE", "XRP", "AVAX", "ADA", "DOT", "LINK",
    "ARB", "NEAR", "APT", "SUI", "BTC", "ETH",
]


def get_all_perps() -> list[str]:
    r = requests.post(API, json={"type": "meta"}, timeout=15)
    r.raise_for_status()
    return [u["name"] for u in r.json()["universe"]]


def get_earliest_candle(coin: str) -> datetime | None:
    """Binary search for earliest available hourly candle."""
    # Try from very early date
    earliest_ms = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    r = requests.post(API, json={
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": "1h",
                "startTime": earliest_ms,
                "endTime": earliest_ms + 30 * 24 * 3600 * 1000}
    }, timeout=15)
    if r.status_code != 200:
        return None
    data = r.json()
    if data:
        ts = data[0]["t"]
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    # Try 2024
    earliest_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    r = requests.post(API, json={
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": "1h",
                "startTime": earliest_ms,
                "endTime": earliest_ms + 60 * 24 * 3600 * 1000}
    }, timeout=15)
    if r.status_code == 200 and r.json():
        ts = r.json()[0]["t"]
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    return None


def main():
    print("=" * 60)
    print("Hyperliquid Perpetuals — Coin Availability Check")
    print("=" * 60)

    print("\nFetching all HL perps...")
    all_coins = get_all_perps()
    print(f"Total perps on Hyperliquid: {len(all_coins)}")

    # Show all coins
    print(f"\nFull list ({len(all_coins)} coins):")
    for i, c in enumerate(sorted(all_coins)):
        print(f"  {c:<10}", end="\n" if (i+1) % 7 == 0 else "")
    print()

    # Check targets
    print("\n" + "─" * 60)
    print("Target coins check:")
    print(f"{'Coin':<10} {'On HL?':<10} {'Earliest data':<20} {'Days available'}")
    print("─" * 60)

    today = datetime.now(timezone.utc)
    results = []

    for coin in TARGET_COINS:
        on_hl = coin in all_coins
        if on_hl:
            time.sleep(0.3)
            earliest = get_earliest_candle(coin)
            if earliest:
                days = (today - earliest).days
                results.append((coin, True, earliest, days))
                print(f"  {coin:<10} {'✅':<10} "
                      f"{earliest.strftime('%Y-%m-%d'):<20} {days}d")
            else:
                results.append((coin, True, None, 0))
                print(f"  {coin:<10} {'✅':<10} {'unknown':<20}")
        else:
            results.append((coin, False, None, 0))
            print(f"  {coin:<10} {'❌ not listed'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Available on HL with >365 days data:")
    print("=" * 60)
    good = [(c, d, days) for c, on, d, days in results
            if on and days >= 365]
    good.sort(key=lambda x: x[2], reverse=True)
    for coin, earliest, days in good:
        wfa_windows = max(0, (days - 120) // 60)
        print(f"  {coin:<10} since {earliest.strftime('%Y-%m-%d')}"
              f"  ({days}d)  → ~{wfa_windows} WFA windows")

    print("\nNot listed on HL:")
    missing = [c for c, on, _, _ in results if not on]
    print(f"  {', '.join(missing)}")


if __name__ == "__main__":
    main()
