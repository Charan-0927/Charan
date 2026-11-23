"""
strategy_agent.py

Agentic AI Forex Trading System — Strategy Agent
-----------------------------------------------------
EMA(21)/EMA(45) crossover with an EMA(100) trend filter:
  - EMA21 crosses ABOVE EMA45 AND price is above EMA100  -> go long
  - EMA21 crosses BELOW EMA45 AND price is below EMA100  -> go short
  - A cross that fails the EMA100 filter is ignored (logged, no trade)
  - Stop-loss: fixed distance from entry, in PRICE units
  - Take-profit: fixed distance from entry, in PRICE units

Distances are plain price units: for BTCUSD, 150 means a $150 move
(entry 67,000 -> stop 66,850 for a long). They are NOT MT5 points.

The strategy keeps the current position direction per symbol, but only as told
by the orchestrator through set_position(); generate() never changes it.
It does NOT place orders: it returns a TradeProposal for the RiskManagerAgent,
plus an optional close instruction for a position it is reversing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from risk_manager_agent import TradeProposal

log = logging.getLogger("strategy_agent")


@dataclass
class EmaCrossConfig:
    fast_period: int = 21
    slow_period: int = 45
    trend_period: int = 100          # EMA used as the trend filter
    stop_loss_distance: float = 150.0    # price units from entry (BTCUSD: dollars)
    take_profit_distance: float = 350.0  # price units from entry


class EmaCrossStrategy:
    def __init__(self, config: EmaCrossConfig):
        self.config = config
        # symbol -> "long" | "short" | None
        self._position: Dict[str, Optional[str]] = {}

    def describe(self, point: float) -> str:
        c = self.config
        return (f"EMA{c.fast_period}/EMA{c.slow_period} cross, filtered by EMA{c.trend_period}. "
                f"Stop-loss {c.stop_loss_distance:g} in price (= {c.stop_loss_distance / point:g} MT5 points); "
                f"take-profit {c.take_profit_distance:g} in price (= {c.take_profit_distance / point:g} MT5 points); "
                f"reward:risk {c.take_profit_distance / c.stop_loss_distance:.2f}")

    def set_position(self, symbol: str, direction: Optional[str]) -> None:
        """The orchestrator reports what is REALLY open ("long", "short" or None)."""
        self._position[symbol] = direction

    def generate(
        self, symbol: str, history: pd.DataFrame
    ) -> Tuple[Optional[TradeProposal], Optional[str]]:
        """
        Returns (proposal_or_None, close_action_or_None).
        close_action is "close_long" or "close_short" when this signal
        requires reversing an existing position before the new one opens.
        """
        c = self.config
        min_bars = max(c.slow_period, c.trend_period) + 2
        if len(history) < min_bars:
            log.debug("Not enough bars yet for %s (%d/%d)", symbol, len(history), min_bars)
            return None, None

        closes = history["close"]
        ema_fast = closes.ewm(span=c.fast_period, adjust=False).mean()
        ema_slow = closes.ewm(span=c.slow_period, adjust=False).mean()
        ema_trend = closes.ewm(span=c.trend_period, adjust=False).mean()

        prev_fast, prev_slow = ema_fast.iloc[-2], ema_slow.iloc[-2]
        curr_fast, curr_slow = ema_fast.iloc[-1], ema_slow.iloc[-1]

        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow
        if not (bullish_cross or bearish_cross):
            return None, None

        price = float(closes.iloc[-1])
        trend = float(ema_trend.iloc[-1])
        context = {
            "signal": "bullish_cross" if bullish_cross else "bearish_cross",
            "price": round(price, 5),
            f"ema{c.fast_period}": round(float(curr_fast), 5),
            f"ema{c.slow_period}": round(float(curr_slow), 5),
            f"ema{c.trend_period}": round(trend, 5),
            f"prev_ema{c.fast_period}": round(float(prev_fast), 5),
            f"prev_ema{c.slow_period}": round(float(prev_slow), 5),
            "price_minus_trend_ema": round(price - trend, 5),
            "recent_closes_oldest_first": [round(float(x), 5) for x in closes.iloc[-40:]],
        }
        stop_distance = c.stop_loss_distance
        tp_distance = c.take_profit_distance
        current_position = self._position.get(symbol)

        if bullish_cross:
            if price <= trend:
                log.info("Bullish cross on %s ignored: price %.5f is not above EMA%d (%.5f)",
                         symbol, price, c.trend_period, trend)
                return None, None
            if current_position == "long":
                return None, None
            log.info("BULLISH CROSS on %s @ %.5f (EMA%d=%.5f > EMA%d=%.5f, price above EMA%d=%.5f)",
                     symbol, price, c.fast_period, curr_fast, c.slow_period, curr_slow,
                     c.trend_period, trend)
            return TradeProposal(
                symbol=symbol, direction="buy", entry_price=price,
                stop_loss=price - stop_distance, take_profit=price + tp_distance,
                context=context,
            ), ("close_short" if current_position == "short" else None)

        # bearish cross
        if price >= trend:
            log.info("Bearish cross on %s ignored: price %.5f is not below EMA%d (%.5f)",
                     symbol, price, c.trend_period, trend)
            return None, None
        if current_position == "short":
            return None, None
        log.info("BEARISH CROSS on %s @ %.5f (EMA%d=%.5f < EMA%d=%.5f, price below EMA%d=%.5f)",
                 symbol, price, c.fast_period, curr_fast, c.slow_period, curr_slow,
                 c.trend_period, trend)
        return TradeProposal(
            symbol=symbol, direction="sell", entry_price=price,
            stop_loss=price + stop_distance, take_profit=price - tp_distance,
            context=context,
        ), ("close_long" if current_position == "long" else None)
