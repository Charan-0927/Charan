"""
risk_manager_agent.py

Agentic AI Forex Trading System — Risk Manager Agent
-------------------------------------------------------
Sits between the strategy agent and the execution agent. Every proposed
trade passes through here first. This agent can VETO a trade — the
execution agent should never place an order that hasn't been approved here.

Responsibilities:
  - position sizing from account risk %, not fixed lot sizes
  - per-trade risk cap
  - max concurrent exposure (total risk across open positions)
  - daily loss kill-switch (stops all new trades once hit)
  - basic sanity checks (SL/TP present and on the correct side of price)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

log = logging.getLogger("risk_manager_agent")


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0        # % of equity risked per trade
    max_concurrent_risk_pct: float = 5.0   # sum of open-trade risk, as % of equity
    max_daily_loss_pct: float = 3.0        # kill switch: stop new trades for the day
    max_open_positions: int = 5
    min_stop_distance_points: float = 5.0  # reject trades with a near-zero SL distance
    max_volume_lots: float = 1.0           # hard cap per trade; a tight stop can otherwise size a huge position
    fixed_volume_lots: Optional[float] = None   # e.g. 0.01 = always trade this size, ignoring risk-% sizing


@dataclass
class TradeProposal:
    symbol: str
    direction: str          # "buy" or "sell"
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    context: dict = field(default_factory=dict)   # why the strategy fired (EMAs, recent closes...); for the journal only


@dataclass
class SizedOrder:
    symbol: str
    direction: str
    volume: float            # lots
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    risk_amount: float       # account currency at risk on this trade


class RiskManagerAgent:
    def __init__(self, config: RiskConfig):
        self.config = config
        self._daily_start_equity: Optional[float] = None
        self._daily_date: Optional[date] = None
        self._open_risk: Dict[str, float] = {}   # ticket/id -> risk amount
        self._kill_switch_active = False

    # -- daily tracking -----------------------------------------------------
    def _refresh_daily_baseline(self, current_equity: float) -> None:
        today = date.today()
        if self._daily_date != today:
            self._daily_date = today
            self._daily_start_equity = current_equity
            self._kill_switch_active = False
            log.info("New trading day. Baseline equity set to %.2f", current_equity)

    def _check_kill_switch(self, current_equity: float) -> bool:
        if self._daily_start_equity is None:
            return False
        drawdown_pct = (self._daily_start_equity - current_equity) / self._daily_start_equity * 100
        if drawdown_pct >= self.config.max_daily_loss_pct:
            if not self._kill_switch_active:
                log.error(
                    "KILL SWITCH TRIGGERED: daily drawdown %.2f%% >= limit %.2f%%. "
                    "No new trades will be approved today.",
                    drawdown_pct, self.config.max_daily_loss_pct,
                )
            self._kill_switch_active = True
        return self._kill_switch_active

    # -- sizing ---------------------------------------------------------------
    def _position_size(
        self, equity: float, entry_price: float, stop_loss: float,
        symbol_point: float, tick_value_per_lot: float, tick_size: float,
    ) -> tuple[float, float]:
        """
        Returns (volume_in_lots, risk_amount_in_account_currency).
        tick_value_per_lot / tick_size come from the broker's symbol info
        (in MT5: mt5.symbol_info(symbol).trade_tick_value / trade_tick_size).
        """
        risk_amount = equity * (self.config.risk_per_trade_pct / 100)
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return 0.0, 0.0
        value_per_price_unit_per_lot = tick_value_per_lot / tick_size
        loss_per_lot = stop_distance * value_per_price_unit_per_lot
        if loss_per_lot <= 0:
            return 0.0, 0.0
        if self.config.fixed_volume_lots:
            # Fixed size: risk is whatever that volume loses if the stop-loss is hit.
            fixed = self.config.fixed_volume_lots
            return round(fixed, 2), fixed * loss_per_lot
        volume = risk_amount / loss_per_lot
        cap = self.config.max_volume_lots
        if cap and volume > cap:
            capped_risk = cap * loss_per_lot
            log.warning(
                "Sized volume %.2f lots exceeds the %.2f-lot cap; capping it "
                "(actual risk %.2f instead of the %.2f budget)",
                volume, cap, capped_risk, risk_amount)
            volume, risk_amount = cap, capped_risk
        return round(volume, 2), risk_amount

    # -- main entry point -------------------------------------------------
    def evaluate(
        self,
        proposal: TradeProposal,
        equity: float,
        symbol_point: float,
        tick_value_per_lot: float,
        tick_size: float,
        open_position_count: int,
    ) -> Optional[SizedOrder]:
        """Returns a SizedOrder if approved, or None if rejected/vetoed."""
        self._refresh_daily_baseline(equity)

        if self._check_kill_switch(equity):
            log.warning("Trade for %s rejected — daily loss kill switch is active", proposal.symbol)
            return None

        if open_position_count >= self.config.max_open_positions:
            log.warning("Trade for %s rejected — max open positions (%d) reached",
                        proposal.symbol, self.config.max_open_positions)
            return None

        stop_distance_points = abs(proposal.entry_price - proposal.stop_loss) / symbol_point
        if stop_distance_points < self.config.min_stop_distance_points:
            log.warning("Trade for %s rejected — stop distance %.1f points too tight",
                        proposal.symbol, stop_distance_points)
            return None

        if proposal.direction == "buy" and proposal.stop_loss >= proposal.entry_price:
            log.warning("Trade for %s rejected — buy stop-loss is above entry", proposal.symbol)
            return None
        if proposal.direction == "sell" and proposal.stop_loss <= proposal.entry_price:
            log.warning("Trade for %s rejected — sell stop-loss is below entry", proposal.symbol)
            return None

        volume, risk_amount = self._position_size(
            equity, proposal.entry_price, proposal.stop_loss,
            symbol_point, tick_value_per_lot, tick_size,
        )
        if volume <= 0:
            log.warning("Trade for %s rejected — computed volume is zero", proposal.symbol)
            return None

        current_open_risk_pct = sum(self._open_risk.values()) / equity * 100 if equity else 0
        this_trade_risk_pct = risk_amount / equity * 100 if equity else 0
        if current_open_risk_pct + this_trade_risk_pct > self.config.max_concurrent_risk_pct:
            log.warning(
                "Trade for %s rejected — would push concurrent risk to %.2f%% (limit %.2f%%)",
                proposal.symbol, current_open_risk_pct + this_trade_risk_pct,
                self.config.max_concurrent_risk_pct,
            )
            return None

        log.info(
            "Trade APPROVED: %s %s vol=%.2f risk=%.2f (%.3f%% of equity)",
            proposal.direction.upper(), proposal.symbol, volume,
            risk_amount, risk_amount / equity * 100,
        )
        return SizedOrder(
            symbol=proposal.symbol,
            direction=proposal.direction,
            volume=volume,
            entry_price=proposal.entry_price,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            risk_amount=risk_amount,
        )

    def register_open_risk(self, position_id: str, risk_amount: float) -> None:
        self._open_risk[position_id] = risk_amount

    def release_risk(self, position_id: str) -> None:
        self._open_risk.pop(position_id, None)
