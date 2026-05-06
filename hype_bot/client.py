"""I/O layer: Hyperliquid REST API client + disk state persistence."""
import json
import logging
import os
import time
import requests
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ===========================================================================
# REST client
# ===========================================================================

class HyperliquidRESTClient:
    def __init__(self, base_url: str = "https://api.hyperliquid.xyz", timeout: int = 10):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _post_info(self, payload: Dict, retries: int = 3) -> Optional[Dict]:
        url = f"{self.base}/info"
        last_err = None
        for i in range(retries):
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                log.warning("POST /info failed (%s/%s): %s", i + 1, retries, e)
                time.sleep(1.5 ** i)
        log.error("POST /info gave up: %s", last_err)
        return None

    # --------------------------------------------------------------
    def candle_snapshot(self, coin: str, interval: str,
                        start_ms: int, end_ms: int) -> List[Dict]:
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval,
                    "startTime": start_ms, "endTime": end_ms},
        }
        out = self._post_info(payload)
        return out or []

    def funding_history(self, coin: str, start_ms: int,
                        end_ms: Optional[int] = None) -> List[Dict]:
        payload = {"type": "fundingHistory", "coin": coin, "startTime": start_ms}
        if end_ms is not None:
            payload["endTime"] = end_ms
        out = self._post_info(payload)
        return out or []

    def get_settled_hour(self, coin: str, hour_ts_ms: int) -> Optional[Dict]:
        """Return the funding_history record that settled at hour_ts_ms.

        Hyperliquid publishes funding records once per hour; the record's
        ``time`` field equals the settlement timestamp.  We add a 60-second
        buffer on the upper bound to tolerate API lag.  Falls back to the
        most recent record *before* hour_ts_ms if the new one is not yet
        available.
        """
        start_ms = hour_ts_ms - 3_600_000   # one hour back
        end_ms   = hour_ts_ms + 60_000       # 60-second API-lag buffer
        records  = self.funding_history(coin, start_ms, end_ms)
        if not records:
            return None
        # Prefer exact match at the settlement boundary
        for r in records:
            if int(r["time"]) == hour_ts_ms:
                return r
        # Fallback: latest record that arrived before the boundary
        valid = [r for r in records if int(r["time"]) < hour_ts_ms]
        return max(valid, key=lambda r: int(r["time"])) if valid else None

    def meta_and_asset_ctxs(self) -> Optional[List]:
        """Returns [meta, assetCtxs].  assetCtxs contains premium, funding, mid, OI..."""
        return self._post_info({"type": "metaAndAssetCtxs"})

    def get_coin_ctx(self, coin: str) -> Optional[Dict]:
        data = self.meta_and_asset_ctxs()
        if not data or len(data) != 2:
            return None
        meta, ctxs = data
        universe = meta.get("universe", [])
        for i, u in enumerate(universe):
            if u.get("name") == coin:
                if i < len(ctxs):
                    ctx = dict(ctxs[i])
                    ctx["name"] = coin
                    return ctx
        return None

    def get_all_coin_ctxs(self, coins: List[str]) -> Dict[str, Dict]:
        """Fetch market context for multiple coins in one API call.

        Returns dict keyed by coin name with the same fields as
        get_latest_mark_and_premium.
        """
        coins_set = set(coins)
        data = self.meta_and_asset_ctxs()
        if not data or len(data) != 2:
            return {}
        meta, ctxs = data
        universe = meta.get("universe", [])
        result = {}
        ts_ms = int(time.time() * 1000)
        for i, u in enumerate(universe):
            name = u.get("name")
            if name not in coins_set or i >= len(ctxs):
                continue
            ctx = ctxs[i]
            try:
                result[name] = {
                    "coin":          name,
                    "mark_px":       float(ctx["markPx"]),
                    "mid_px":        float(ctx.get("midPx") or ctx["markPx"]),
                    "oracle_px":     float(ctx["oraclePx"]),
                    "premium":       float(ctx.get("premium", 0.0)),
                    "funding":       float(ctx.get("funding", 0.0)),
                    "open_interest": float(ctx.get("openInterest", 0.0)),
                    "day_volume":    float(ctx.get("dayNtlVlm", 0.0)),
                    "ts_ms":         ts_ms,
                }
            except (KeyError, TypeError, ValueError) as e:
                log.error("get_all_coin_ctxs parse error [%s]: %s", name, e)
        return result

    def get_latest_mark_and_premium(self, coin: str) -> Optional[Dict]:
        ctx = self.get_coin_ctx(coin)
        if ctx is None:
            return None
        try:
            return {
                "coin": coin,
                "mark_px": float(ctx["markPx"]),
                "mid_px": float(ctx.get("midPx") or ctx["markPx"]),
                "oracle_px": float(ctx["oraclePx"]),
                "premium": float(ctx.get("premium", 0.0)),
                "funding": float(ctx.get("funding", 0.0)),
                "open_interest": float(ctx.get("openInterest", 0.0)),
                "day_volume": float(ctx.get("dayNtlVlm", 0.0)),
                "ts_ms": int(time.time() * 1000),
            }
        except (KeyError, TypeError, ValueError) as e:
            log.error("get_latest_mark_and_premium parse error: %s | ctx=%s", e, ctx)
            return None

    def get_open_positions(self, wallet_address: str) -> List[Dict]:
        """Query the exchange for all open perp positions of a wallet.

        Returns a list of dicts with keys:
            coin, side (LONG/SHORT), size, entry_price, unrealized_pnl,
            liquidation_px, margin_used, leverage
        """
        out = self._post_info({"type": "clearinghouseState", "user": wallet_address})
        if not out:
            return []
        positions = []
        for ap in out.get("assetPositions", []):
            pos = ap.get("position", {})
            szi = float(pos.get("szi", 0))
            if szi == 0:
                continue
            try:
                positions.append({
                    "coin":           pos["coin"],
                    "side":           "LONG" if szi > 0 else "SHORT",
                    "size":           abs(szi),
                    "entry_price":    float(pos.get("entryPx") or 0),
                    "unrealized_pnl": float(pos.get("unrealizedPnl") or 0),
                    "liquidation_px": float(pos.get("liquidationPx") or 0),
                    "margin_used":    float(pos.get("marginUsed") or 0),
                    "leverage":       pos.get("leverage", {}),
                })
            except (KeyError, TypeError, ValueError) as e:
                log.warning("get_open_positions parse error: %s | pos=%s", e, pos)
        return positions


# ===========================================================================
# State persistence
# ===========================================================================

class StateManager:
    def __init__(self, path: str):
        self.path = path

    def save(self, state: Dict):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, self.path)
        except Exception as e:
            log.error("Failed to save state: %s", e)

    def load(self) -> Optional[Dict]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception as e:
            log.error("Failed to load state: %s", e)
            return None
