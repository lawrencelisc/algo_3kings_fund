"""Master bot: orchestrates multi-coin data fetch, signal, execution, risk."""
import csv
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from config import CFG, CoinSpec
from client import HyperliquidRESTClient, StateManager
from signal import PremiumHistory, ThresholdEngine, SignalGenerator, LONG, SHORT, FLAT
from execution import PaperExecutor, LiveExecutor, Position, RiskManager

log = logging.getLogger(__name__)


# ===========================================================================
# Sub-bots
# ===========================================================================

class SubBot:
    def __init__(self, bot_id: str, side: str, executor):
        assert side in ("LONG", "SHORT")
        self.bot_id = bot_id
        self.side = side
        self.executor = executor
        self.position: Optional[Position] = None
        self.closed_trades: List[Dict] = []
        self.total_net_pnl: float = 0.0
        self.total_funding: float = 0.0
        self.total_fees: float = 0.0
        self.wins: int = 0
        self.losses: int = 0

    @property
    def has_position(self) -> bool:
        return self.position is not None

    def try_open(self, ref_price: float, equity: float, position_pct: float,
                 hold_hours: int, ts_ms: int,
                 size_multiplier: float = 1.0) -> bool:
        if self.has_position:
            log.debug("[%s] already has position, skip open", self.bot_id)
            return False
        pos = self.executor.open_position(
            bot_id=self.bot_id, side=self.side, ref_price=ref_price,
            equity=equity, position_pct=position_pct,
            hold_hours=hold_hours, ts_ms=ts_ms,
            size_multiplier=size_multiplier)
        if pos:
            self.position = pos
            return True
        return False

    def should_exit(self, now_ms: int) -> bool:
        if not self.has_position:
            return False
        return now_ms >= self.position.planned_exit_ts_ms

    def force_close(self, ref_price: float, ts_ms: int, reason: str) -> Optional[Dict]:
        if not self.has_position:
            return None
        rec = self.executor.close_position(self.position, ref_price, ts_ms, reason)
        self._record_trade(rec)
        self.position = None
        return rec

    def check_trail_stop(self, mark_px: float,
                         activate_pct: float, trail_pct: float) -> bool:
        """Update high-watermark and return True if trailing stop should fire."""
        if not self.has_position or activate_pct <= 0 or trail_pct <= 0:
            return False
        upnl_pct = self.position.unrealized_pct(mark_px)
        # Update high watermark
        if upnl_pct > self.position.trail_hwm_pct:
            self.position.trail_hwm_pct = upnl_pct
        # Only fire once armed (peak has reached activation level)
        if self.position.trail_hwm_pct < activate_pct:
            return False
        # Fire if current PnL falls more than trail_pct below the peak
        if upnl_pct <= self.position.trail_hwm_pct - trail_pct:
            log.info("[%s] TRAIL STOP armed at hwm=%.3f%% current=%.3f%% trail=%.3f%%",
                     self.bot_id,
                     self.position.trail_hwm_pct * 100,
                     upnl_pct * 100,
                     trail_pct * 100)
            return True
        return False

    def apply_funding(self, funding_rate: float, mark_px: float, ts_ms: int):
        """Funding credited/debited to open position at settlement."""
        if not self.has_position:
            return
        notional_now = self.position.size * mark_px
        direction = -1.0 if self.position.side == "LONG" else 1.0
        delta = direction * funding_rate * notional_now
        self.position.funding_pnl += delta
        self.total_funding += delta
        log.info("[%s] FUNDING rate=%.6f%% notional=%.2f delta=%.4f (cum=%.4f)",
                 self.bot_id, funding_rate * 100, notional_now, delta,
                 self.position.funding_pnl)

    def _record_trade(self, rec: Dict):
        self.closed_trades.append(rec)
        self.total_net_pnl += rec["net_pnl"]
        self.total_fees += rec["entry_fee"] + rec["exit_fee"]
        if rec["net_pnl"] > 0:
            self.wins += 1
        else:
            self.losses += 1

    def stats(self) -> Dict:
        n = len(self.closed_trades)
        return {
            "bot_id": self.bot_id,
            "trades": n,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": (self.wins / n) if n else 0.0,
            "net_pnl": self.total_net_pnl,
            "funding_pnl": self.total_funding,
            "fees": self.total_fees,
            "open_position": self.position.to_dict() if self.position else None,
        }


class LongSubBot(SubBot):
    def __init__(self, executor, coin: str = ""):
        super().__init__(f"{coin}_LONG" if coin else "LONG_BOT", "LONG", executor)


class ShortSubBot(SubBot):
    def __init__(self, executor, coin: str = ""):
        super().__init__(f"{coin}_SHORT" if coin else "SHORT_BOT", "SHORT", executor)


# ===========================================================================
# Per-coin runner
# ===========================================================================

class CoinRunner:
    """Manages signal + position lifecycle for one coin."""

    def __init__(self, spec: CoinSpec, executor, cfg):
        self.spec = spec
        self.cfg = cfg
        self.history = PremiumHistory(maxlen=spec.lookback_hours)
        self.engine = ThresholdEngine(
            spec.winsorize_lower, spec.winsorize_upper, spec.q_lo, spec.q_hi)
        self.sig_gen = SignalGenerator(self.history, self.engine, spec.min_history_hours)
        self.long_bot = LongSubBot(executor, coin=spec.coin)
        self.short_bot = ShortSubBot(executor, coin=spec.coin)
        self.last_settlement_ts: Optional[int] = None
        self.last_mark_px: float = 0.0
        self._last_sig: Optional[Dict] = None

    @property
    def coin(self) -> str:
        return self.spec.coin

    def settlement_ts_ms(self, hour_floor: datetime) -> int:
        """Unix ms timestamp when this coin settles within hour_floor's hour."""
        dt = hour_floor.replace(minute=self.spec.settlement_minute, second=0, microsecond=0)
        return int(dt.timestamp() * 1000)

    def should_settle(self, now: datetime, hour_floor: datetime) -> bool:
        settle_ts = self.settlement_ts_ms(hour_floor)
        return int(now.timestamp() * 1000) >= settle_ts and self.last_settlement_ts != settle_ts

    def warmup(self, client: HyperliquidRESTClient) -> None:
        """Load settled premium history for this coin."""
        log.info("[%s] Warming up premium history (%d h)...", self.coin, self.spec.lookback_hours)
        now = datetime.now(tz=timezone.utc)
        hour_floor = now.replace(minute=0, second=0, microsecond=0)
        # Align end to the most recent fully-settled point for this coin
        if now.minute >= self.spec.settlement_minute:
            end_dt = hour_floor.replace(minute=self.spec.settlement_minute)
        else:
            end_dt = (hour_floor - timedelta(hours=1)).replace(minute=self.spec.settlement_minute)
        end_ms   = int(end_dt.timestamp() * 1000)
        start_ms = end_ms - self.spec.lookback_hours * 3_600_000

        fh = client.funding_history(self.coin, start_ms, end_ms)
        added = 0
        for item in fh:
            try:
                self.history.push(int(item["time"]), float(item.get("premium", 0.0)))
                added += 1
            except (KeyError, TypeError, ValueError):
                continue
        log.info("[%s] Warmup loaded %d observations.", self.coin, added)

    def on_new_settlement(self, settle_ts_ms: int, mark_px: float,
                          premium: float, funding: float,
                          risk_halted: bool = False) -> List[Dict]:
        """Process one settlement. Returns list of closed trade records."""
        self.last_mark_px = mark_px
        trade_records = []

        # 1. Apply funding BEFORE closing/opening
        self.long_bot.apply_funding(funding, mark_px, settle_ts_ms)
        self.short_bot.apply_funding(funding, mark_px, settle_ts_ms)

        # 2. Push settled premium (overwrites stale warmup value if same ts)
        self.history.push(settle_ts_ms, premium)

        # 3. Generate signal FIRST so we can decide roll vs close
        sig = self.sig_gen.generate(premium)
        self._last_sig = sig
        signal = sig["signal"]

        # 4. Roll or close expired positions
        roll_secs = self.cfg.HOLD_HOURS * 3_600_000
        for bot in (self.long_bot, self.short_bot):
            if not bot.should_exit(settle_ts_ms):
                continue
            # Roll: if signal still agrees with position direction and no risk halt
            if not risk_halted and signal == bot.side:
                bot.position.planned_exit_ts_ms = settle_ts_ms + roll_secs
                log.info("[%s] ROLL %s — signal still %s, extending hold (saved 2× fee)",
                         self.coin, bot.bot_id, signal)
            else:
                rec = bot.force_close(mark_px, settle_ts_ms, "TIME")
                if rec:
                    trade_records.append(rec)

        log.info("[%s] SETTLE %s | px=%.4f prem=%.6f q_lo=%s q_hi=%s sig=%s (%s)",
                 self.coin,
                 datetime.fromtimestamp(settle_ts_ms / 1000, tz=timezone.utc)
                         .strftime("%Y-%m-%d %H:%M"),
                 mark_px, premium,
                 f"{sig['q_lo']:.6f}" if sig['q_lo'] is not None else "N/A",
                 f"{sig['q_hi']:.6f}" if sig['q_hi'] is not None else "N/A",
                 signal, sig["reason"])

        return trade_records

    def try_open_from_signal(self, equity: float, risk: RiskManager,
                             settle_ts_ms: int) -> None:
        sig = self._last_sig
        if sig is None or not risk.can_open_new():
            return
        sz_mult = self.spec.size_multiplier
        if sig["signal"] == LONG and not self.long_bot.has_position:
            self.long_bot.try_open(self.last_mark_px, equity, self.cfg.POSITION_PCT,
                                   self.cfg.HOLD_HOURS, settle_ts_ms,
                                   size_multiplier=sz_mult)
        elif sig["signal"] == SHORT and not self.short_bot.has_position:
            self.short_bot.try_open(self.last_mark_px, equity, self.cfg.POSITION_PCT,
                                    self.cfg.HOLD_HOURS, settle_ts_ms,
                                    size_multiplier=sz_mult)

    def unrealized_pnl(self) -> float:
        mark = self.last_mark_px
        upnl = 0.0
        if self.long_bot.has_position:
            upnl += self.long_bot.position.unrealized_pnl(mark)
            upnl += self.long_bot.position.funding_pnl
        if self.short_bot.has_position:
            upnl += self.short_bot.position.unrealized_pnl(mark)
            upnl += self.short_bot.position.funding_pnl
        return upnl

    def realized_pnl(self) -> float:
        return self.long_bot.total_net_pnl + self.short_bot.total_net_pnl

    def stats(self) -> Dict:
        return {
            "coin": self.coin,
            "last_settlement_ts": self.last_settlement_ts,
            "long_bot": self.long_bot.stats(),
            "short_bot": self.short_bot.stats(),
        }


# ===========================================================================
# Master bot
# ===========================================================================

class MasterBot:
    def __init__(self):
        self.cfg = CFG
        self.client = HyperliquidRESTClient(self.cfg.API_BASE)

        if self.cfg.TRADE_MODE == "live":
            if not self.cfg.WALLET_ADDRESS or not self.cfg.PRIVATE_KEY:
                raise ValueError("TRADE_MODE=live requires WALLET_ADDRESS and PRIVATE_KEY")
            self.executor = LiveExecutor(
                wallet_address=self.cfg.WALLET_ADDRESS,
                private_key=self.cfg.PRIVATE_KEY,
                maker_fee=self.cfg.MAKER_FEE,
                taker_fee=self.cfg.TAKER_FEE,
                leverage=self.cfg.LEVERAGE,
                base_url=self.cfg.API_BASE)
            log.info("==== LIVE trading mode | leverage=%dX ====", self.cfg.LEVERAGE)
        else:
            self.executor = PaperExecutor(
                maker_fee=self.cfg.MAKER_FEE, taker_fee=self.cfg.TAKER_FEE,
                use_maker=self.cfg.USE_MAKER, slippage_bps=self.cfg.SLIPPAGE_BPS,
                leverage=self.cfg.LEVERAGE)
            log.info("==== PAPER trading mode | leverage=%dX ====", self.cfg.LEVERAGE)

        self.risk = RiskManager(self.cfg.MAX_COMBINED_DD,
                                self.cfg.INITIAL_EQUITY_USDC)

        self.runners: List[CoinRunner] = [
            CoinRunner(spec, self.executor, self.cfg)
            for spec in self.cfg.COIN_SPECS
        ]
        self.runner_by_coin: Dict[str, CoinRunner] = {r.coin: r for r in self.runners}

        self.state_mgr = StateManager(self.cfg.STATE_FILE)
        self.equity = self.cfg.INITIAL_EQUITY_USDC

        self._init_csv_files()
        self._restore_state()

    # ------------------------------------------------------------------
    def _init_csv_files(self):
        if not os.path.exists(self.cfg.TRADES_CSV):
            with open(self.cfg.TRADES_CSV, "w", newline="") as f:
                csv.writer(f).writerow([
                    "exit_ts_utc", "coin", "bot_id", "side",
                    "entry_px", "exit_px", "notional",
                    "gross_pnl", "entry_fee", "exit_fee",
                    "funding_pnl", "net_pnl", "return_pct", "reason"])
        if not os.path.exists(self.cfg.EQUITY_CSV):
            with open(self.cfg.EQUITY_CSV, "w", newline="") as f:
                csv.writer(f).writerow(["ts_utc", "equity", "peak", "dd", "positions"])

    def _append_trade(self, rec: Dict):
        with open(self.cfg.TRADES_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.fromtimestamp(rec["exit_ts_ms"] / 1000, tz=timezone.utc).isoformat(),
                rec.get("coin", ""),
                rec["bot_id"], rec["side"],
                f"{rec['entry_px']:.6f}", f"{rec['exit_px']:.6f}",
                f"{rec['notional']:.4f}", f"{rec['gross_pnl']:.4f}",
                f"{rec['entry_fee']:.4f}", f"{rec['exit_fee']:.4f}",
                f"{rec['funding_pnl']:.4f}", f"{rec['net_pnl']:.4f}",
                f"{rec['return_pct']:.6f}", rec["reason"],
            ])

    def _append_equity(self):
        positions = " | ".join(
            f"{r.coin}:"
            f"{'L' if r.long_bot.has_position else '-'}"
            f"{'S' if r.short_bot.has_position else '-'}"
            for r in self.runners
        )
        with open(self.cfg.EQUITY_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(tz=timezone.utc).isoformat(),
                f"{self.equity:.4f}",
                f"{self.risk.state.peak_equity:.4f}",
                f"{self.risk.state.current_dd:.6f}",
                positions,
            ])

    # ------------------------------------------------------------------
    def _restore_state(self):
        """Restore equity, risk, and open positions from the last saved state."""
        saved = self.state_mgr.load()

        if saved:
            self.equity = float(saved.get("equity", self.cfg.INITIAL_EQUITY_USDC))

            risk_snap = saved.get("risk", {})
            self.risk.state.halted      = risk_snap.get("halted", False)
            self.risk.state.halt_reason = risk_snap.get("halt_reason", "")
            self.risk.state.peak_equity = float(risk_snap.get("peak_equity", self.equity))
            self.risk.state.current_dd  = float(risk_snap.get("current_dd", 0.0))

            saved_runners = saved.get("runners", {})

            # Legacy: single-coin state (long_bot / short_bot at top level → HYPE)
            if not saved_runners and "long_bot" in saved:
                saved_runners = {"HYPE": {
                    "long_bot": saved["long_bot"],
                    "short_bot": saved["short_bot"],
                    "last_settlement_ts": saved.get("last_hour_processed"),
                }}

            for runner in self.runners:
                rs = saved_runners.get(runner.coin, {})
                runner.last_settlement_ts = rs.get("last_settlement_ts")
                for bot, key in ((runner.long_bot, "long_bot"),
                                 (runner.short_bot, "short_bot")):
                    bd = rs.get(key, {})
                    op = bd.get("open_position")
                    if op:
                        runner.last_mark_px = float(op.get("entry_price", 0))
                        bot.position = Position(
                            bot_id=op["bot_id"],
                            side=op["side"],
                            entry_price=float(op["entry_price"]),
                            size=float(op["size"]),
                            notional=float(op["notional"]),
                            entry_ts_ms=int(op["entry_ts_ms"]),
                            entry_fee=float(op["entry_fee"]),
                            funding_pnl=float(op.get("funding_pnl", 0.0)),
                            planned_exit_ts_ms=int(op["planned_exit_ts_ms"]),
                        )
                        log.info("Restored [%s] %s: %s entry=%.4f size=%.6f",
                                 runner.coin, key, bot.position.side,
                                 bot.position.entry_price, bot.position.size)
                    bot.total_net_pnl = float(bd.get("net_pnl", 0.0))
                    bot.total_funding = float(bd.get("funding_pnl", 0.0))
                    bot.total_fees    = float(bd.get("fees", 0.0))
                    bot.wins          = int(bd.get("wins", 0))
                    bot.losses        = int(bd.get("losses", 0))

            log.info("State restored — equity=%.2f peak=%.2f dd=%.2f%% halted=%s",
                     self.equity, self.risk.state.peak_equity,
                     self.risk.state.current_dd * 100, self.risk.state.halted)
        else:
            log.info("No saved state — starting fresh.")

        self._reconcile_risk_after_restore(saved)

        if self.cfg.TRADE_MODE == "live" and self.cfg.WALLET_ADDRESS:
            self._sync_live_positions()

    def _reset_risk_baseline(self, reason: str):
        log.warning("Risk baseline reset: %s — equity=%.2f", reason, self.equity)
        self.risk.state.peak_equity = self.equity
        self.risk.state.current_dd  = 0.0
        self.risk.state.halted      = False
        self.risk.state.halt_reason = ""

    def _reconcile_risk_after_restore(self, saved: Optional[Dict]):
        """Prevent spurious halt when capital tier changes vs stale bot_state.json."""
        if self.risk.state.peak_equity < self.equity:
            self.risk.state.peak_equity = self.equity

        saved_init = saved.get("initial_equity_usdc") if saved else None
        if saved_init is not None:
            if abs(float(saved_init) - self.cfg.INITIAL_EQUITY_USDC) > 1e-6:
                self._reset_risk_baseline(
                    f"capital changed {saved_init} → {self.cfg.INITIAL_EQUITY_USDC}")
        elif saved:
            peak = self.risk.state.peak_equity
            cap  = self.cfg.INITIAL_EQUITY_USDC
            if peak > cap * 2 and self.equity <= cap and peak > self.equity * 3:
                self._reset_risk_baseline(f"legacy peak={peak:.0f} vs capital={cap:.0f}")

        if self.risk.state.peak_equity > 0:
            self.risk.state.current_dd = (
                (self.equity - self.risk.state.peak_equity) / self.risk.state.peak_equity)

    def _sync_live_positions(self):
        """Cross-check local state with exchange; reconstruct or clear as needed."""
        log.info("Syncing positions from exchange…")
        exchange_positions = self.client.get_open_positions(self.cfg.WALLET_ADDRESS)
        exch_by_coin = {p["coin"]: p for p in exchange_positions}

        for runner in self.runners:
            exch = exch_by_coin.get(runner.coin)
            for bot in (runner.long_bot, runner.short_bot):
                if exch and bot.position is None and exch["side"] == bot.side:
                    log.warning("Exchange has %s %s — reconstructing",
                                runner.coin, exch["side"])
                    bot.position = Position(
                        bot_id=bot.bot_id, side=exch["side"],
                        entry_price=exch["entry_price"], size=exch["size"],
                        notional=exch["size"] * exch["entry_price"],
                        entry_ts_ms=int(time.time() * 1000), entry_fee=0.0,
                        funding_pnl=exch.get("unrealized_pnl", 0.0),
                        planned_exit_ts_ms=0)
                elif bot.position and (not exch or exch["side"] != bot.side):
                    log.warning("[%s] %s pos missing on exchange — clearing",
                                runner.coin, bot.bot_id)
                    bot.position = None

        log.info("Exchange positions: %s",
                 [(p["coin"], p["side"], p["size"]) for p in exchange_positions] or "none")

    # ------------------------------------------------------------------
    def current_equity(self) -> float:
        realized = sum(r.realized_pnl() for r in self.runners)
        upnl     = sum(r.unrealized_pnl() for r in self.runners)
        return self.cfg.INITIAL_EQUITY_USDC + realized + upnl

    # ------------------------------------------------------------------
    def warmup_history(self):
        for runner in self.runners:
            runner.warmup(self.client)

    # ------------------------------------------------------------------
    def save_state(self):
        runners_snap = {
            r.coin: {
                "last_settlement_ts": r.last_settlement_ts,
                "long_bot":  r.long_bot.stats(),
                "short_bot": r.short_bot.stats(),
            }
            for r in self.runners
        }
        self.state_mgr.save({
            "equity":               self.equity,
            "initial_equity_usdc":  self.cfg.INITIAL_EQUITY_USDC,
            "runners":              runners_snap,
            "risk":                 self.risk.snapshot(),
            "saved_at":             datetime.now(tz=timezone.utc).isoformat(),
        })

    # ------------------------------------------------------------------
    def run(self):
        coins = [r.coin for r in self.runners]
        log.info("==== Multi-Coin Premium Bot | mode=%s | coins=%s ====",
                 self.cfg.TRADE_MODE.upper(), coins)
        log.info("Config: capital=%.0f lev=%dX pos_pct=%.1f%% "
                 "maker_fee=%.4f%% hold=%dh dd_lim=%.0f%%",
                 self.cfg.INITIAL_EQUITY_USDC, self.cfg.LEVERAGE,
                 self.cfg.POSITION_PCT * 100, self.cfg.MAKER_FEE * 100,
                 self.cfg.HOLD_HOURS, self.cfg.MAX_COMBINED_DD * 100)

        self.warmup_history()

        while True:
            try:
                all_ctxs = self.client.get_all_coin_ctxs(coins)
                if not all_ctxs:
                    log.warning("No market data returned, retry in %ds",
                                self.cfg.POLL_INTERVAL_SEC)
                    time.sleep(self.cfg.POLL_INTERVAL_SEC)
                    continue

                now        = datetime.now(tz=timezone.utc)
                hour_floor = now.replace(minute=0, second=0, microsecond=0)
                ts_ms      = int(time.time() * 1000)

                # Update mark prices + trailing stop check every poll
                trail_closed_any = False
                for runner in self.runners:
                    ctx = all_ctxs.get(runner.coin)
                    if not ctx:
                        continue
                    runner.last_mark_px = ctx["mark_px"]
                    if self.cfg.TRAIL_ACTIVATE_PCT > 0:
                        for bot in (runner.long_bot, runner.short_bot):
                            if bot.check_trail_stop(
                                    ctx["mark_px"],
                                    self.cfg.TRAIL_ACTIVATE_PCT,
                                    self.cfg.TRAIL_STOP_PCT):
                                rec = bot.force_close(ctx["mark_px"], ts_ms, "TRAIL_STOP")
                                if rec:
                                    rec["coin"] = runner.coin
                                    self._append_trade(rec)
                                    trail_closed_any = True

                if trail_closed_any:
                    self.equity = self.current_equity()
                    self.risk.update_equity(self.equity)
                    self._append_equity()
                    self.save_state()

                # Settlement trigger — each coin fires at its own minute
                settled_any = False
                for runner in self.runners:
                    if not runner.should_settle(now, hour_floor):
                        continue

                    ctx = all_ctxs.get(runner.coin)
                    if not ctx:
                        log.warning("[%s] missing ctx at settlement", runner.coin)
                        continue

                    settle_ts_ms = runner.settlement_ts_ms(hour_floor)
                    settled_rec  = self.client.get_settled_hour(runner.coin, settle_ts_ms)

                    if settled_rec:
                        premium = float(settled_rec.get("premium", 0.0))
                        funding = float(settled_rec.get("fundingRate", 0.0))
                    else:
                        log.warning("[%s] no settled record — using ctx snapshot", runner.coin)
                        premium = ctx["premium"]
                        funding = ctx["funding"]

                    # Process this coin's settlement
                    trade_recs = runner.on_new_settlement(
                        settle_ts_ms, ctx["mark_px"], premium, funding,
                        risk_halted=self.risk.state.halted)
                    for rec in trade_recs:
                        rec["coin"] = runner.coin
                        self._append_trade(rec)

                    # Portfolio equity & risk update
                    self.equity = self.current_equity()
                    self.risk.update_equity(self.equity)
                    log.info("Portfolio | equity=%.2f dd=%.2f%% halted=%s",
                             self.equity, self.risk.state.current_dd * 100,
                             self.risk.state.halted)

                    # Open new position for this coin if allowed
                    runner.try_open_from_signal(self.equity, self.risk, settle_ts_ms)
                    runner.last_settlement_ts = settle_ts_ms
                    settled_any = True

                if settled_any:
                    self._append_equity()
                    self.save_state()

                time.sleep(self.cfg.POLL_INTERVAL_SEC)

            except KeyboardInterrupt:
                log.info("Interrupted — saving state & exiting.")
                self.save_state()
                self.print_summary()
                break
            except Exception as e:
                log.exception("Main loop error: %s", e)
                time.sleep(self.cfg.POLL_INTERVAL_SEC)

    # ------------------------------------------------------------------
    def print_summary(self):
        print("\n" + "=" * 60)
        print(" Multi-Coin Premium Bot — Summary")
        print("=" * 60)
        print(f" Initial equity : {self.cfg.INITIAL_EQUITY_USDC:,.2f} USDC")
        print(f" Final equity   : {self.equity:,.2f} USDC")
        total_net = sum(r.realized_pnl() for r in self.runners)
        ret = total_net / self.cfg.INITIAL_EQUITY_USDC
        print(f" Net P&L        : {total_net:+,.2f} USDC ({ret:+.2%})")
        for runner in self.runners:
            ls = runner.long_bot.stats()
            ss = runner.short_bot.stats()
            print(f"  [{runner.coin:>4}] trades={ls['trades']+ss['trades']:>3} "
                  f"net={ls['net_pnl']+ss['net_pnl']:+.4f}  "
                  f"funding={ls['funding_pnl']+ss['funding_pnl']:+.4f}")
        print(f" Peak equity    : {self.risk.state.peak_equity:,.2f}")
        print(f" Current DD     : {self.risk.state.current_dd:.2%}")
        print("=" * 60)
