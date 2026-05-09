"""Configuration for Multi-Coin Premium Index Trading Bot."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class CoinSpec:
    """Per-coin configuration. Strategy params can be tuned independently.

    Defaults aligned with backtest IS=120d (2880h) lookback to reproduce the
    statistical object that walk-forward analysis validated.
    """
    coin: str
    settlement_minute: int = 0   # all coins settle on the hour boundary
    lookback_hours: int = 2880   # 120 days (matches WFA IS window)
    min_history_hours: int = 500 # matches WFA minimum IS window (500 data points)
    q_lo: float = 0.05           # LONG threshold (override per-coin from WFA)
    q_hi: float = 0.95           # SHORT threshold
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    size_multiplier: float = 1.0 # scale margin: 0.5 = half size for unproven coins


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
    # Selected from 2-year WFA OOS results (last 4 windows, ~8 months OOS).
    # q_lo / q_hi values come from per-coin best_threshold optimised in IS.
    # All coins settle on the UTC hour boundary; strategy doesn't depend on
    # funding-payment timing (funding pnl is < 5% of fees in real data).
    COIN_SPECS: List[CoinSpec] = field(default_factory=lambda: [
        # Core (5 windows+ OOS validated, Sharpe ≥ +1.0)
        CoinSpec("IMX",    q_lo=0.05, q_hi=0.95),               # OOS Sharpe +1.67
        CoinSpec("XLM",    q_lo=0.05, q_hi=0.95),               # OOS Sharpe +1.69
        CoinSpec("XRP",    q_lo=0.10, q_hi=0.90),               # OOS Sharpe +1.68
        CoinSpec("TIA",    q_lo=0.02, q_hi=0.98),               # OOS Sharpe +1.45
        CoinSpec("PENDLE", q_lo=0.02, q_hi=0.98),               # OOS Sharpe +1.07
        # Watch (limited samples — half size until 30 trades collected)
        CoinSpec("HYPE",   q_lo=0.02, q_hi=0.98,
                 size_multiplier=0.5),                          # 1 OOS window only
    ])

    # ===== Strategy (shared defaults) =====
    HOLD_HOURS: int = 1         # exit after N settlement periods

    # ===== Capital & sizing =====
    INITIAL_EQUITY_USDC: float = 690.0
    POSITION_PCT: float = 0.08  # margin per trade as % of equity
                                # 5 full + 0.5 HYPE = 5.5 slots × 8% = 44% utilisation

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
    MAX_ROLLS: int = 6              # max times a position can be rolled (cap holding at N+1 hours)

    # ===== Trailing stop =====
    # Activates once unrealized PnL reaches TRAIL_ACTIVATE_PCT of notional.
    # After activation, exits if PnL falls back TRAIL_STOP_PCT below the peak.
    # Set both to 0.0 to disable.
    # Example: activate=0.003 (0.3%), trail=0.005 (0.5%)
    #   → arms when position is up $0.165 on $55 notional
    #   → exits if it gives back 0.5% ($0.275) from the peak
    TRAIL_ACTIVATE_PCT: float = 0.003   # 0.3% of notional to arm the trail
    TRAIL_STOP_PCT: float = 0.005       # 0.5% pullback from peak to trigger exit

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
