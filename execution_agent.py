"""
execution_agent.py

Agentic AI Forex Trading System — Execution Agent
------------------------------------------------------
Talks to a locally running MT5 terminal via the official `MetaTrader5`
Python package. Only accepts orders that have already been sized/approved
by RiskManagerAgent (a SizedOrder) — it does not make sizing decisions
itself.

REQUIREMENTS:
  - Windows OS (MT5 Python API is Windows-only)
  - MetaTrader 5 terminal installed and logged into your broker account
  - pip install MetaTrader5

SAFETY:
  - `dry_run=True` (default) logs the exact order request that would be
    sent and returns a fake success result, without calling order_send.
    Flip to False only once you've verified behavior against live ticks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from risk_manager_agent import SizedOrder

log = logging.getLogger("execution_agent")

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


@dataclass
class MT5Credentials:
    login: int
    password: str
    server: str


@dataclass
class ExecutionResult:
    success: bool
    ticket: Optional[int] = None
    message: str = ""
    price: Optional[float] = None     # actual/assumed fill price
    sl: Optional[float] = None
    tp: Optional[float] = None
    volume: Optional[float] = None


@dataclass
class OpenPosition:
    ticket: int
    symbol: str
    direction: str        # "buy" or "sell"
    volume: float
    price_open: float
    sl: float             # 0.0 means no stop-loss set
    tp: float


class ExecutionAgent:
    def __init__(
        self,
        credentials: MT5Credentials,
        dry_run: bool = True,
        magic_number: int = 990011,
        deviation_points: int = 20,
        max_price_divergence_pct: float = 1.0,
        allow_live: bool = False,
    ):
        if mt5 is None and not dry_run:
            raise ImportError(
                "MetaTrader5 package not installed / not on Windows. "
                "Run with dry_run=True, or install it with: pip install MetaTrader5"
            )
        self.credentials = credentials
        self.dry_run = dry_run
        self.magic_number = magic_number
        self.deviation_points = deviation_points
        # Refuse to send orders to a REAL-money account unless this is explicitly True.
        self.allow_live = allow_live
        self.mode: Optional[str] = "dry-run" if dry_run else None   # "dry-run", "demo" or "real" once connected
        self._dry_run_tickets = 0
        # Safety net: reject if the broker's price is further than this % from the
        # signal's price (catches a wrong symbol mapping or a stale/bad feed).
        self.max_price_divergence_pct = max_price_divergence_pct
        self._connected = False

    # -- connection -----------------------------------------------------
    def connect(self) -> None:
        if self.dry_run:
            log.info("[DRY RUN] Orders will be logged, not sent.")
            self._connected = True
            return

        c = self.credentials
        if c.login and c.password and c.server:
            ok = mt5.initialize(login=c.login, password=c.password, server=c.server)
        else:
            ok = mt5.initialize()   # attach to the account the open terminal is logged into
        if not ok:
            raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")
        info = mt5.account_info()
        is_demo = info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        kind = "DEMO" if is_demo else ("CONTEST" if info.trade_mode == mt5.ACCOUNT_TRADE_MODE_CONTEST else "REAL")
        log.info("Connected to MT5 (%s account). Account=%s Balance=%.2f Equity=%.2f Server=%s",
                 kind, info.login, info.balance, info.equity, info.server)
        if kind == "REAL" and not self.allow_live:
            mt5.shutdown()
            raise PermissionError(
                f"Account {info.login} on {info.server} is a REAL-money account, and live trading "
                "is not enabled. Log the MT5 terminal into a demo account, or set ALLOW_LIVE=true "
                "in .env if you really mean to trade real money.")
        self.mode = kind.lower()
        self._connected = True

    def shutdown(self) -> None:
        if not self.dry_run and mt5 is not None:
            mt5.shutdown()
        self._connected = False

    # -- account / symbol info (feeds the risk manager) -------------------
    def _mt5_ready(self) -> bool:
        """True if the MT5 terminal is connected in this process (the data feed connects it)."""
        if mt5 is None:
            return False
        try:
            return mt5.terminal_info() is not None
        except Exception:
            return False

    def _use_real_data(self) -> bool:
        # Live mode always uses broker data. Dry-run uses it too when MT5 is connected,
        # so sizing in the logs matches what a live order would get; otherwise placeholders.
        return (not self.dry_run) or self._mt5_ready()

    def get_equity(self) -> float:
        if not self._use_real_data():
            return 10_000.0  # placeholder when MT5 isn't connected
        return mt5.account_info().equity

    def get_open_position_count(self, symbol: Optional[str] = None) -> int:
        if not self._use_real_data():
            return 0
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return len(positions) if positions is not None else 0

    def get_symbol_risk_params(self, symbol: str) -> tuple[float, float, float]:
        """Returns (point, trade_tick_value, trade_tick_size) for sizing math."""
        if not self._use_real_data():
            return 0.0001, 1.0, 0.00001  # rough EURUSD-like placeholders
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol {symbol} not found / not visible in Market Watch")
        if not info.visible:
            mt5.symbol_select(symbol, True)
        return info.point, info.trade_tick_value, info.trade_tick_size

    def get_current_price(self, symbol: str, direction: str) -> float:
        if not self._use_real_data():
            return 1.10000
        tick = mt5.symbol_info_tick(symbol)
        return tick.ask if direction == "buy" else tick.bid

    # -- position sync (what is actually open at the broker) ---------------
    def get_open_positions(self) -> Optional[list[OpenPosition]]:
        """
        Positions opened by THIS agent (matched by magic number).
        Returns None when it can't tell (dry run, or MT5 query failed) so callers
        keep their current state instead of wrongly assuming everything closed.
        """
        if self.dry_run:
            return None
        positions = mt5.positions_get()
        if positions is None:
            if mt5.last_error()[0] != 1:   # 1 == RES_S_OK, i.e. simply no positions
                log.warning("positions_get failed: %s", mt5.last_error())
                return None
            positions = ()
        return [
            OpenPosition(
                ticket=p.ticket, symbol=p.symbol,
                direction="buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                volume=p.volume, price_open=p.price_open, sl=p.sl, tp=p.tp,
            )
            for p in positions if p.magic == self.magic_number
        ]

    def get_close_info(self, ticket: int) -> Optional[dict]:
        """How a position ended: reason (stop-loss / take-profit / ...), net P&L, exit price."""
        if self.dry_run:
            return None
        deals = mt5.history_deals_get(position=ticket)
        if not deals:
            return None
        exits = [d for d in deals if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)]
        if not exits:
            return None
        reasons = {
            mt5.DEAL_REASON_SL: "stop-loss",
            mt5.DEAL_REASON_TP: "take-profit",
            mt5.DEAL_REASON_SO: "stop-out (margin)",
        }
        net = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0) for d in deals)
        last = exits[-1]
        return {"reason": reasons.get(last.reason, "manual/other"), "profit": net, "price": last.price}

    def estimate_position_risk(self, pos: OpenPosition) -> float:
        """Account-currency loss if the position's stop-loss is hit (0.0 if it has no SL)."""
        if self.dry_run or not pos.sl:
            return 0.0
        info = mt5.symbol_info(pos.symbol)
        if info is None or not info.trade_tick_size:
            return 0.0
        return abs(pos.price_open - pos.sl) / info.trade_tick_size * info.trade_tick_value * pos.volume

    @staticmethod
    def _filling_mode(info) -> int:
        """Pick a filling mode the broker actually supports for this symbol."""
        flags = info.filling_mode          # bit 1 = FOK allowed, bit 2 = IOC allowed
        if flags & 2:
            return mt5.ORDER_FILLING_IOC
        if flags & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def place_order(self, order: SizedOrder) -> ExecutionResult:
        """
        Sends a market order. The strategy's signal price (from the data feed) is
        NOT used as the order price. Instead we read the broker's live ask/bid at
        send time and re-anchor the stop-loss / take-profit to it, keeping the
        same distances the strategy and risk manager sized the trade with.
        """
        if not self._connected:
            raise RuntimeError("ExecutionAgent.connect() must be called first")

        is_buy = order.direction == "buy"
        order_type_str = "BUY" if is_buy else "SELL"
        sl_distance = abs(order.entry_price - order.stop_loss)
        tp_distance = (abs(order.take_profit - order.entry_price)
                       if order.take_profit is not None else None)

        if self.dry_run:
            request = {
                "symbol": order.symbol, "volume": order.volume, "type": order_type_str,
                "price": order.entry_price, "sl": order.stop_loss, "tp": order.take_profit,
                "deviation": self.deviation_points, "magic": self.magic_number,
                "comment": "agentic-fx-agent",
            }
            log.info("[DRY RUN] Would send order (live mode re-anchors price/SL/TP to the "
                     "broker's ask/bid): %s", request)
            self._dry_run_tickets += 1       # unique fake ticket per simulated trade: -1, -2, ...
            return ExecutionResult(success=True, ticket=-self._dry_run_tickets, message="dry_run",
                                   price=order.entry_price, sl=order.stop_loss,
                                   tp=order.take_profit, volume=order.volume)

        # --- live path ------------------------------------------------------
        info = mt5.symbol_info(order.symbol)
        if info is None:
            return ExecutionResult(False, message=f"Symbol {order.symbol} not found at broker")
        if not info.visible and not mt5.symbol_select(order.symbol, True):
            return ExecutionResult(False, message=f"Could not enable {order.symbol} in Market Watch")

        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None or tick.ask <= 0 or tick.bid <= 0:
            return ExecutionResult(False, message=f"No live quote for {order.symbol}")
        price = tick.ask if is_buy else tick.bid

        divergence_pct = abs(price - order.entry_price) / order.entry_price * 100
        if divergence_pct > self.max_price_divergence_pct:
            msg = (f"Broker price {price} differs {divergence_pct:.2f}% from signal price "
                   f"{order.entry_price} (limit {self.max_price_divergence_pct}%) - order rejected")
            log.error(msg)
            return ExecutionResult(False, message=msg)
        log.info("%s %s: signal price %.5f -> broker price %.5f (diff %.4f%%)",
                 order_type_str, order.symbol, order.entry_price, price, divergence_pct)

        # Re-anchor SL/TP to the broker price, same distances as the sized trade.
        sign = 1 if is_buy else -1
        sl = round(price - sign * sl_distance, info.digits)
        tp = round(price + sign * tp_distance, info.digits) if tp_distance else 0.0  # 0.0 = no TP

        min_dist = info.trade_stops_level * info.point
        if abs(price - sl) < min_dist or (tp and abs(tp - price) < min_dist):
            msg = (f"SL/TP closer than the broker's minimum stop distance "
                   f"({min_dist}) for {order.symbol} - order rejected")
            log.error(msg)
            return ExecutionResult(False, message=msg)

        # Snap volume DOWN to the broker's step (never up - that would exceed the risk budget).
        step = info.volume_step or 0.01
        volume = round(math.floor(order.volume / step + 1e-9) * step, 8)
        volume = min(volume, info.volume_max)
        if volume < info.volume_min:
            msg = (f"Sized volume {order.volume} is below the broker minimum "
                   f"{info.volume_min} for {order.symbol} - order rejected (would exceed risk)")
            log.error(msg)
            return ExecutionResult(False, message=msg)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation_points,
            "magic": self.magic_number,
            "comment": "agentic-fx-agent",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(info),
        }
        result = mt5.order_send(request)
        if result is None:
            return ExecutionResult(False, message=f"order_send returned None: {mt5.last_error()}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("Order FAILED for %s: retcode=%s comment=%s",
                      order.symbol, result.retcode, result.comment)
            return ExecutionResult(False, message=f"{result.retcode}: {result.comment}")

        # result.order is the ORDER ticket. The POSITION ticket (what positions_get and
        # close_position use) is read from the resulting deal; fall back to the order ticket.
        position_ticket = result.order
        deal = getattr(result, "deal", 0)
        if deal:
            deals = mt5.history_deals_get(ticket=deal)
            if deals:
                position_ticket = deals[0].position_id or position_ticket

        log.info("Order FILLED: %s %s vol=%.2f @ %.5f SL=%.5f TP=%.5f position=%s",
                 order_type_str, order.symbol, volume, price, sl, tp, position_ticket)
        fill_price = getattr(result, "price", 0) or price
        return ExecutionResult(success=True, ticket=position_ticket, message="filled",
                               price=fill_price, sl=sl, tp=tp or None, volume=volume)

    def close_position(self, ticket: int) -> ExecutionResult:
        if self.dry_run:
            log.info("[DRY RUN] Would close position ticket=%s", ticket)
            return ExecutionResult(success=True, ticket=ticket, message="dry_run_close")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return ExecutionResult(success=False, message=f"Position {ticket} not found")
        pos = position[0]
        direction = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if direction == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": direction,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "agentic-fx-agent-close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        success = result.retcode == mt5.TRADE_RETCODE_DONE
        return ExecutionResult(success=success, ticket=ticket,
                                message="closed" if success else str(result.comment))
