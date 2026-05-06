"""
Hyperliquid Funding History Downloader
下載頭10幣（BTC/ETH/SOL/ZEC/XRP/DOGE/ASTER/AAVE/FARTCOIN/AVAX）
昨日至今的 hourly funding rate，存成 CSV

使用方法：
    pip install requests pandas
    python download_hl_funding.py
"""

import requests
import pandas as pd
import time
from datetime import datetime, timezone, timedelta

# ── 設定 ──────────────────────────────────────────────
COINS = ["BTC", "ETH", "SOL", "ZEC", "XRP", "DOGE", "ASTER", "AAVE", "FARTCOIN", "AVAX"]
OUTPUT_FILE = "hl_funding_history_top10.csv"

# 昨日 00:00:00 UTC
yesterday = datetime.now(timezone.utc).replace(
    hour=0, minute=0, second=0, microsecond=0
) - timedelta(days=1)
START_MS = int(yesterday.timestamp() * 1000)
# ─────────────────────────────────────────────────────


def get_funding_history(coin: str, start_ms: int) -> list[dict]:
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "fundingHistory", "coin": coin, "startTime": start_ms}
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"下載範圍：{yesterday.strftime('%Y-%m-%d 00:00 UTC')} → 現在\n")

    all_records = []

    for coin in COINS:
        try:
            data = get_funding_history(coin, START_MS)
            for item in data:
                ts = pd.to_datetime(item["time"], unit="ms", utc=True)
                funding = float(item["fundingRate"])
                premium = float(item["premium"])
                all_records.append({
                    "coin":             coin,
                    "time_utc":         ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "funding_rate":     funding,
                    "funding_rate_pct": round(funding * 100, 6),   # % 表示
                    "funding_8h_pct":   round(funding * 8 * 100, 6),  # 換算 8h
                    "funding_ann_pct":  round(funding * 24 * 365 * 100, 2),  # 年化
                    "premium":          premium,
                })
            print(f"  ✅ {coin:12s} {len(data):4d} 條記錄")
            time.sleep(0.3)   # 避免 rate limit
        except Exception as e:
            print(f"  ❌ {coin:12s} 失敗：{e}")

    if not all_records:
        print("\n沒有數據，請檢查網絡連線。")
        return

    df = pd.DataFrame(all_records)
    df = df.sort_values(["coin", "time_utc"]).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")  # utf-8-sig 方便 Excel 開

    print(f"\n✅ 已存檔：{OUTPUT_FILE}")
    print(f"   總記錄數：{len(df)}")
    print(f"   欄位：{list(df.columns)}\n")
    print("── 預覽（前 15 行）──────────────────────────────")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
