"""
One-click run for PyCharm
=========================
1. Put this file (run.py) in the SAME folder as multi_coin_wfa.py
2. In PyCharm: right-click run.py → Run 'run'
   OR set as default: Run → Edit Configurations → Script = run.py

First time setup (run once in PyCharm Terminal):
    pip install ccxt requests pandas numpy matplotlib scipy
"""

# ── Edit these settings before running ────────────────────────────────

COINS          = ["BTC", "ETH", "SOL", "HYPE"]   # remove any you don't want
SKIP_DOWNLOAD  = False   # True  = use existing CSVs (faster re-run)
                         # False = download fresh data from APIs

# ──────────────────────────────────────────────────────────────────────

import sys
import os

# Make sure we can find multi_coin_wfa.py in the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multi_coin_wfa_v4 as wfa

# Override settings
wfa.ALL_COINS = COINS

# Run
if __name__ == "__main__":
    if SKIP_DOWNLOAD:
        sys.argv = ["run.py", "--skip-download"]
    else:
        sys.argv = ["run.py"]

    wfa.main()
