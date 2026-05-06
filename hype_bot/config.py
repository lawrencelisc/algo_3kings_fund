"""Configuration for Multi-Coin Premium Index Trading Bot."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class CoinSpec:
    """Per-coin configuration. Strategy params can be tuned independently."""
    coin: str
    settlement_minute: int      # minute within UTC hour when funding settles
    lookback_hours: int = 168   # rolling window for quantile (7 days)
    min_history_hours: int = 72
    q_lo: float = 0.05          # LONG threshold (bottom 5%)
    q_hi: float = 0.95          # SHORT threshold (top 5%)
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99


@dataclass
class Config:
    # ===== Trade mode =====
    # "paper" : simulate fills locally (no real orders)
    # "live"  : place real orders on Hyperliquid
    TRADE_MODE: str = "paper"

    # ===== Live credentials (required when TRADE_MODE="live") =====
    # Set via environment variable — do NOT commit private key.
    WALLET_ADDRESS: str = ""    # 0x...
    PRIVATE_KEY: str = ""       # 0x...  (EVM private key)

    # ===== Coins =====
    # settlement_minute = the minute past each UTC hour when funding settles.
    # Tier 1 original: HYPE :00, ATOM :06, GALA :11, IMX :16
    # Tier 2 added:    SOL :21, LTC :26, OP :31, TIA :36, PENDLE :41, INJ :46
    COIN_SPECS: List[CoinSpec] = field(default_factory=lambda: [
        CoinSpec("HYPE",   settlement_minute=0),
        CoinSpec("ATOM",   settlement_minute=6),
        CoinSpec("GALA",   settlement_minute=11),
        CoinSpec("IMX",    settlement_minute=16),
        CoinSpec("SOL",    settlement_minute=21),
        CoinSpec("LTC",    settlement_minute=26),
        CoinSpec("OP",     settlement_minute=31),
        CoinSpec("TIA",    settlement_minute=36),
        CoinSpec("PENDLE", settlement_minute=41),
        CoinSpec("INJ",    settlement_minute=46),
    ])

    # ===== Strategy (shared defaults) =====
    HOLD_HOURS: int = 1         # exit after N settlement periods

    # ===== Capital & sizing =====
    INITIAL_EQUITY_USDC: float = 300.0   # 10 coins × 5% = 15 USDC/trade @ 1X, above min-order floor
    POSITION_PCT: float = 0.05  # margin per trade as % of equity

    # ===== Leverage =====
    # 1 = no leverage (safe default).  Upgrade to 3X / 5X after validation.
    # Paper mode: scales notional for realistic PnL simulation.
    # Live mode: also calls update_leverage on exchange before each order.
    LEVERAGE: int = 1

    # ===== Fees (Hyperliquid VIP0 rates) =====
    MAKER_FEE: float = 0.000142     # 0.0142% — always try maker to save cost
    TAKER_FEE: float = 0.000427     # 0.0427% — fallback
    USE_MAKER: bool = True
    SLIPPAGE_BPS: float = 0.5       # 0.5 bp conservative paper slippage

    # ===== Risk =====
    MAX_COMBINED_DD: float = 0.15   # halt if portfolio equity DD > 15%
    MAX_SINGLE_POS_DD: float = 0.08 # emergency close single pos at 8% loss

    # ===== Infra =====
    POLL_INTERVAL_SEC: int = 30
    API_BASE: str = "https://api.hyperliquid.xyz"
    STATE_FILE: str = "bot_state.json"
    LOG_FILE: str = "bot.log"
    TRADES_CSV: str = "trades.csv"
    EQUITY_CSV: str = "equity.csv"

    # ===== Alerting (optional) =====
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""


import os as _os
from pathlib import Path as _Path

def _cfg_from_env() -> "Config":
    """Override key settings from environment variables if present."""
    # Auto-load .env from the project root (two levels up from this file).
    try:
        from dotenv import load_dotenv
        _dotenv = _Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(_dotenv)
    except ImportError:
        pass  # dotenv optional; fall back to shell environment

    c = Config()
    c.TRADE_MODE     = _os.getenv("TRADE_MODE", c.TRADE_MODE)
    c.WALLET_ADDRESS = _os.getenv("HL_WALLET_ADDRESS", c.WALLET_ADDRESS)
    c.PRIVATE_KEY    = _os.getenv("HL_PRIVATE_KEY", c.PRIVATE_KEY)
    if _os.getenv("LEVERAGE"):
        c.LEVERAGE = int(_os.getenv("LEVERAGE"))
    if _os.getenv("INITIAL_EQUITY"):
        c.INITIAL_EQUITY_USDC = float(_os.getenv("INITIAL_EQUITY"))
    return c

CFG = _cfg_from_env()
