"""
Test: Can CCXT download HYPE data from Hyperliquid?
====================================================
Run this in PyCharm to check what's available.

pip install ccxt
python test_ccxt_hype.py
"""

import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta

print("=" * 60)
print(f"CCXT version: {ccxt.__version__}")
print("=" * 60)

# ── 1. Check if Hyperliquid is supported ──────────────────────
hl_exchanges = [e for e in ccxt.exchanges if "hyper" in e.lower()]
print(f"\nHyperliquid-related exchanges in CCXT:")
for e in hl_exchanges:
    print(f"  {e}")

if not hl_exchanges:
    print("  (none found — CCXT version may be too old)")
    print("  Try: pip install --upgrade ccxt")

# ── 2. Try connecting ─────────────────────────────────────────
print("\n" + "─" * 60)
print("Trying hyperliquid exchange...")

try:
    exchange = ccxt.hyperliquid({
        "enableRateLimit": True,
    })

    # Load markets
    print("Loading markets...")
    markets = exchange.load_markets()
    hype_symbols = [s for s in markets if "HYPE" in s]
    print(f"HYPE symbols found: {hype_symbols[:10]}")

    # Try fetching OHLCV
    if hype_symbols:
        sym = hype_symbols[0]
        print(f"\nFetching 5 hourly bars for {sym}...")
        since = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
        ohlcv = exchange.fetch_ohlcv(sym, "1h", since=since, limit=5)
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        print(df[["time","close","vol"]].to_string(index=False))
        print("\nCCXT → Hyperliquid OHLCV: SUCCESS")
    else:
        print("No HYPE symbol found in markets")

except AttributeError:
    print("ccxt.hyperliquid does not exist in this version")
    print("Try: pip install --upgrade ccxt")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# ── 3. Try hyperliquidfutures or similar ─────────────────────
print("\n" + "─" * 60)
print("Trying hyperliquidfutures / other variants...")
for name in ["hyperliquidfutures", "hyperliquid_futures"]:
    if name in ccxt.exchanges:
        print(f"Found: {name}")
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            m = ex.load_markets()
            hype = [s for s in m if "HYPE" in s]
            print(f"  HYPE symbols: {hype[:5]}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"Not found: {name}")

# ── 4. Summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
If CCXT supports Hyperliquid:
  → We can fetch unlimited HYPE history using batched OHLCV calls
  → No 5000-bar limit (that was the direct REST API limit)
  → Same script structure as BTC/ETH/SOL

If CCXT does NOT support Hyperliquid:
  → Must use Hyperliquid native REST API (5000 bar limit)
  → Only ~7 months of OHLCV data available via candles
  → However, fundingHistory has no limit → full 18m funding data
  → Workaround: reconstruct price from funding records or
    use a 3rd party data provider (e.g. Tardis.dev)
""")
