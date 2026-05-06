"""Execution layer: paper order simulation, live execution, and risk control."""
import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ===========================================================================
# Data models
# ===========================================================================

@dataclass
class Fill:
    ts_ms: int
    side: str               # BUY / SELL
    price: float
    size: float             # in coin
    notional: float         # USDC
    fee_rate: float
    fee_usdc: float
    order_type: str         # MAKER / TAKER
    tag: str = ""


@dataclass
class Position:
    bot_id: str             # "LONG_BOT" or "SHORT_BOT"
    side: str               # LONG or SHORT
    entry_price: float
    size: float             # coin quantity (always positive)
    notional: float         # USDC notional at entry
    entry_ts_ms: int
    entry_fee: float
    funding_pnl: float = 0.0
    planned_exit_ts_ms: int = 0

    def unrealized_pnl(self, mark_px: float) -> float:
        direction = 1.0 if self.side == "LONG" else -1.0
        return direction * (mark_px - self.entry_price) * self.size

    def to_dict(self):
        return asdict(self)


# ===========================================================================
# Paper executor
# ===========================================================================

class PaperExecutor:
    """
    Simulates order fills.  Maker orders are assumed to fill at the reference
    price with slippage_bps of slippage against us.  This is conservative.

    With leverage > 1:
      margin   = equity * position_pct          (capital at risk)
      notional = margin * leverage              (actual exposure)
    PnL and fees are calculated on notional; margin is purely accounting.
    """
    def __init__(self, maker_fee: float, taker_fee: float,
                 use_maker: bool, slippage_bps: float, leverage: int = 1):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.use_maker = use_maker
        self.slippage_bps = slippage_bps
        self.leverage = max(1, leverage)
        self.fills: List[Fill] = []

    def _apply_slippage(self, px: float, side: str) -> float:
        slip = self.slippage_bps / 10_000.0
        return px * (1 + slip) if side == "BUY" else px * (1 - slip)

    def open_position(self, bot_id: str, side: str, ref_price: float,
                      equity: float, position_pct: float,
                      hold_hours: int, ts_ms: int) -> Optional[Position]:
        margin   = equity * position_pct
        notional = margin * self.leverage        # leverage-adjusted exposure
        if notional <= 0 or ref_price <= 0:
            return None
        trade_side = "BUY" if side == "LONG" else "SELL"
        fill_px = self._apply_slippage(ref_price, trade_side)
        size = notional / fill_px
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        fee = notional * fee_rate

        self.fills.append(Fill(
            ts_ms=ts_ms, side=trade_side, price=fill_px, size=size,
            notional=notional, fee_rate=fee_rate, fee_usdc=fee,
            order_type="MAKER" if self.use_maker else "TAKER",
            tag=f"OPEN-{bot_id}",
        ))

        pos = Position(
            bot_id=bot_id, side=side, entry_price=fill_px, size=size,
            notional=notional, entry_ts_ms=ts_ms, entry_fee=fee,
            planned_exit_ts_ms=ts_ms + hold_hours * 3_600_000,
        )
        log.info("[PAPER] OPEN %s %s px=%.6f size=%.6f margin=%.2f notional=%.2f lev=%dX fee=%.4f",
                 bot_id, side, fill_px, size, margin, notional, self.leverage, fee)
        return pos

    def close_position(self, pos: Position, ref_price: float, ts_ms: int,
                       reason: str = "TIME") -> Dict:
        trade_side = "SELL" if pos.side == "LONG" else "BUY"
        fill_px = self._apply_slippage(ref_price, trade_side)
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        exit_notional = fill_px * pos.size
        exit_fee = exit_notional * fee_rate

        direction = 1.0 if pos.side == "LONG" else -1.0
        gross_pnl = direction * (fill_px - pos.entry_price) * pos.size
        net_pnl = gross_pnl - pos.entry_fee - exit_fee + pos.funding_pnl

        self.fills.append(Fill(
            ts_ms=ts_ms, side=trade_side, price=fill_px, size=pos.size,
            notional=exit_notional, fee_rate=fee_rate, fee_usdc=exit_fee,
            order_type="MAKER" if self.use_maker else "TAKER",
            tag=f"CLOSE-{pos.bot_id}-{reason}",
        ))

        log.info("[PAPER] CLOSE %s px=%.6f gross=%.4f fees=%.4f funding=%.4f NET=%.4f (%s)",
                 pos.bot_id, fill_px, gross_pnl,
                 pos.entry_fee + exit_fee, pos.funding_pnl, net_pnl, reason)
        return {
            "bot_id": pos.bot_id,
            "side": pos.side,
            "entry_ts_ms": pos.entry_ts_ms,
            "exit_ts_ms": ts_ms,
            "entry_px": pos.entry_price,
            "exit_px": fill_px,
            "size": pos.size,
            "notional": pos.notional,
            "gross_pnl": gross_pnl,
            "entry_fee": pos.entry_fee,
            "exit_fee": exit_fee,
            "funding_pnl": pos.funding_pnl,
            "net_pnl": net_pnl,
            "reason": reason,
            "return_pct": net_pnl / pos.notional,
        }


# ===========================================================================
# Live executor (real orders via Hyperliquid SDK)
# ===========================================================================

class LiveExecutor:
    """
    Places real market orders on Hyperliquid via hyperliquid-python-sdk.
    Mirrors the PaperExecutor interface so MasterBot needs zero changes.
    """

    def __init__(self, wallet_address: str, private_key: str,
                 maker_fee: float, taker_fee: float,
                 leverage: int = 1,
                 base_url: str = "https://api.hyperliquid.xyz"):
        from hyperliquid.exchange import Exchange
        import eth_account

        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.leverage = max(1, leverage)
        self.wallet_address = wallet_address
        self.fills: List[Fill] = []

        account = eth_account.Account.from_key(private_key)
        self.exchange = Exchange(account, base_url)
        self._sz_decimals: Dict[str, int] = {}   # coin → szDecimals cache
        log.info("[LIVE] Executor ready — wallet=%s leverage=%dX",
                 wallet_address[:10] + "…", self.leverage)

    def _get_sz_decimals(self, coin: str, info) -> int:
        """Return szDecimals for coin (cached after first fetch)."""
        if coin not in self._sz_decimals:
            try:
                meta = info.meta()
                for asset in meta.get("universe", []):
                    self._sz_decimals[asset["name"]] = int(asset.get("szDecimals", 6))
            except Exception as e:
                log.warning("[LIVE] Could not fetch szDecimals: %s — defaulting to 4", e)
                self._sz_decimals[coin] = 4
        return self._sz_decimals.get(coin, 4)

    @staticmethod
    def _round_px(px: float) -> float:
        """Round price to Hyperliquid's tick size (5 significant figures, max 6 dp)."""
        if px <= 0:
            return px
        magnitude = math.floor(math.log10(px))
        decimals = min(6, max(0, 4 - magnitude))
        return round(px, decimals)

    @staticmethod
    def _ob_vwap(levels: list, n: int = 5) -> Optional[float]:
        """VWAP of top-n order book levels. levels = [{"px":str,"sz":str}, ...]"""
        total_val = total_sz = 0.0
        for lvl in levels[:n]:
            px = float(lvl["px"])
            sz = float(lvl["sz"])
            total_val += px * sz
            total_sz  += sz
        return total_val / total_sz if total_sz > 0 else None

    def _market_order(self, coin: str, is_buy: bool, sz: float) -> Dict:
        """IOC order priced at 5-level order book VWAP + small safety buffer."""
        order_type = {"limit": {"tif": "Ioc"}}
        ob_safety  = 0.0005   # 0.05 % buffer beyond VWAP to ensure IOC fills
        mid_safety = 0.02     # fallback: 2 % from mid if OB fetch fails

        from hyperliquid.info import Info
        info = Info(self.exchange.base_url, skip_ws=True)

        # ── size rounding ──────────────────────────────────────────────────
        sz_dec = self._get_sz_decimals(coin, info)
        sz = math.floor(sz * 10 ** sz_dec) / 10 ** sz_dec
        if sz <= 0:
            raise RuntimeError(f"Computed size rounds to zero for {coin} (szDecimals={sz_dec})")

        # ── price from order book ──────────────────────────────────────────
        try:
            ob = info.l2_snapshot(coin)
            # levels[0] = bids (descending), levels[1] = asks (ascending)
            bids, asks = ob["levels"][0], ob["levels"][1]
            side_levels = asks if is_buy else bids
            vwap = self._ob_vwap(side_levels, n=5)
            if vwap and vwap > 0:
                raw_px = vwap * (1 + ob_safety) if is_buy else vwap * (1 - ob_safety)
                px_src = "OB-VWAP"
            else:
                raise ValueError("empty book")
        except Exception as e:
            log.warning("[LIVE] OB fetch failed (%s), falling back to mid±%.0f%%", e, mid_safety*100)
            mids = info.all_mids()
            mid  = float(mids.get(coin, 0))
            if mid <= 0:
                raise RuntimeError(f"Cannot get mid price for {coin}")
            raw_px = mid * (1 + mid_safety) if is_buy else mid * (1 - mid_safety)
            px_src = "MID"

        px = self._round_px(raw_px)
        log.info("[LIVE] placing order coin=%s is_buy=%s sz=%s px=%s src=%s szDecimals=%d",
                 coin, is_buy, sz, px, px_src, sz_dec)
        result = self.exchange.order(coin, is_buy, sz, px, order_type)
        log.info("[LIVE] order result: %s", result)
        if result.get("status") != "ok":
            raise RuntimeError(f"Order failed: {result}")
        return result

    def _set_leverage(self, coin: str):
        """Set cross-margin leverage on the exchange before opening."""
        if self.leverage == 1:
            return
        try:
            result = self.exchange.update_leverage(self.leverage, coin, is_cross=True)
            log.info("[LIVE] Set leverage %dX for %s: %s", self.leverage, coin, result)
        except Exception as e:
            log.warning("[LIVE] update_leverage failed (non-fatal): %s", e)

    def open_position(self, bot_id: str, side: str, ref_price: float,
                      equity: float, position_pct: float,
                      hold_hours: int, ts_ms: int) -> Optional[Position]:
        margin   = equity * position_pct
        notional = margin * self.leverage
        if notional <= 0 or ref_price <= 0:
            return None
        is_buy = (side == "LONG")
        coin = bot_id.split("_")[0] if "_" in bot_id else "HYPE"
        sz = notional / ref_price

        self._set_leverage(coin)
        try:
            result = self._market_order(coin, is_buy, sz)
        except Exception as e:
            log.error("[LIVE] open_position failed: %s", e)
            return None

        # Validate that the IOC order was actually filled
        fills = result.get("response", {}).get("data", {}).get("statuses", [{}])
        filled_status = next((f["filled"] for f in fills if "filled" in f), None)
        if filled_status is None:
            # Order was rejected or not immediately filled (IOC cancelled)
            errors = [f.get("error", f) for f in fills]
            log.error("[LIVE] open_position NOT filled for %s — statuses: %s", bot_id, errors)
            return None
        fill_px = float(filled_status.get("avgPx", ref_price))
        sz      = float(filled_status.get("totalSz", sz))

        fee_rate = self.maker_fee
        fee = notional * fee_rate
        self.fills.append(Fill(
            ts_ms=ts_ms, side="BUY" if is_buy else "SELL",
            price=fill_px, size=sz, notional=notional,
            fee_rate=fee_rate, fee_usdc=fee,
            order_type="MARKET", tag=f"OPEN-{bot_id}",
        ))
        pos = Position(
            bot_id=bot_id, side=side, entry_price=fill_px, size=sz,
            notional=notional, entry_ts_ms=ts_ms, entry_fee=fee,
            planned_exit_ts_ms=ts_ms + hold_hours * 3_600_000,
        )
        log.info("[LIVE] OPEN %s %s fill_px=%.6f size=%.6f margin=%.2f notional=%.2f lev=%dX fee=%.4f",
                 bot_id, side, fill_px, sz, margin, notional, self.leverage, fee)
        return pos

    def close_position(self, pos: Position, ref_price: float, ts_ms: int,
                       reason: str = "TIME") -> Dict:
        is_buy = (pos.side == "SHORT")   # closing long = sell; closing short = buy
        coin = pos.bot_id.split("_")[0] if "_" in pos.bot_id else "HYPE"

        try:
            result = self._market_order(coin, is_buy, pos.size)
        except Exception as e:
            log.error("[LIVE] close_position failed: %s", e)
            # Fall back to ref_price for accounting even if order errored
            result = {}

        fills = result.get("response", {}).get("data", {}).get("statuses", [{}])
        filled_status = next((f["filled"] for f in fills if "filled" in f), None)
        if filled_status is None:
            errors = [f.get("error", f) for f in fills]
            log.error("[LIVE] close_position NOT filled for %s — statuses: %s", pos.bot_id, errors)
        fill_px = float(filled_status.get("avgPx", ref_price)) if filled_status else ref_price

        fee_rate = self.maker_fee
        exit_notional = fill_px * pos.size
        exit_fee = exit_notional * fee_rate
        direction = 1.0 if pos.side == "LONG" else -1.0
        gross_pnl = direction * (fill_px - pos.entry_price) * pos.size
        net_pnl = gross_pnl - pos.entry_fee - exit_fee + pos.funding_pnl

        self.fills.append(Fill(
            ts_ms=ts_ms, side="SELL" if not is_buy else "BUY",
            price=fill_px, size=pos.size, notional=exit_notional,
            fee_rate=fee_rate, fee_usdc=exit_fee,
            order_type="MARKET", tag=f"CLOSE-{pos.bot_id}-{reason}",
        ))
        log.info("[LIVE] CLOSE %s fill_px=%.6f gross=%.4f fees=%.4f funding=%.4f NET=%.4f (%s)",
                 pos.bot_id, fill_px, gross_pnl,
                 pos.entry_fee + exit_fee, pos.funding_pnl, net_pnl, reason)
        return {
            "bot_id": pos.bot_id, "side": pos.side,
            "entry_ts_ms": pos.entry_ts_ms, "exit_ts_ms": ts_ms,
            "entry_px": pos.entry_price, "exit_px": fill_px,
            "size": pos.size, "notional": pos.notional,
            "gross_pnl": gross_pnl, "entry_fee": pos.entry_fee,
            "exit_fee": exit_fee, "funding_pnl": pos.funding_pnl,
            "net_pnl": net_pnl, "reason": reason,
            "return_pct": net_pnl / pos.notional,
        }


# ===========================================================================
# Risk manager
# ===========================================================================

@dataclass
class RiskState:
    halted: bool = False
    halt_reason: str = ""
    peak_equity: float = 0.0
    current_dd: float = 0.0


class RiskManager:
    def __init__(self, max_combined_dd: float, max_single_pos_dd: float,
                 initial_equity: float):
        self.max_combined_dd = max_combined_dd
        self.max_single_pos_dd = max_single_pos_dd
        self.state = RiskState(peak_equity=initial_equity)

    def update_equity(self, equity: float):
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        if self.state.peak_equity > 0:
            self.state.current_dd = (equity - self.state.peak_equity) / self.state.peak_equity
        if self.state.current_dd <= -self.max_combined_dd and not self.state.halted:
            self.state.halted = True
            self.state.halt_reason = (
                f"Combined DD {self.state.current_dd:.2%} <= -{self.max_combined_dd:.0%}")
            log.error("🛑 RISK HALT: %s", self.state.halt_reason)

    def should_emergency_close(self, pos: Position, mark_px: float) -> bool:
        if pos is None or pos.notional <= 0:
            return False
        pos_dd = pos.unrealized_pnl(mark_px) / pos.notional
        if pos_dd <= -self.max_single_pos_dd:
            log.error("🛑 EMERGENCY CLOSE %s: dd=%.2f%% <= -%.0f%%",
                      pos.bot_id, pos_dd * 100, self.max_single_pos_dd * 100)
            return True
        return False

    def can_open_new(self) -> bool:
        return not self.state.halted

    def snapshot(self) -> dict:
        return {
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "peak_equity": self.state.peak_equity,
            "current_dd": self.state.current_dd,
        }
