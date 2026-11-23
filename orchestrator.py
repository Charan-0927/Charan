"""
orchestrator.py

Wires the pipeline together:
  DataCollectorAgent -> StrategyAgent -> RiskManagerAgent -> ExecutionAgent

and, on the side, records every trade in the TradeJournal and (optionally) has the
TradeReviewerAgent (an LLM) write a post-trade review. The journal and reviewer only
observe: nothing they do can place, change or block an order, and any failure in them is
logged and ignored so trading is never affected.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from data_collector_agent import AgentConfig, DataCollectorAgent, MT5Connector, SymbolConfig
from risk_manager_agent import RiskConfig, RiskManagerAgent, TradeProposal
from execution_agent import ExecutionAgent, MT5Credentials
from strategy_agent import EmaCrossConfig, EmaCrossStrategy
from journal import TradeJournal
from trade_reviewer_agent import TradeReviewerAgent

log = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


class Orchestrator:
    def __init__(
        self,
        data_agent: DataCollectorAgent,
        risk_agent: RiskManagerAgent,
        execution_agent: ExecutionAgent,
        strategy: EmaCrossStrategy,
        sync_interval_sec: float = 5.0,
        journal: Optional[TradeJournal] = None,
        reviewer: Optional[TradeReviewerAgent] = None,
    ):
        self.data_agent = data_agent
        self.risk_agent = risk_agent
        self.execution_agent = execution_agent
        self.strategy = strategy
        self.sync_interval_sec = sync_interval_sec
        self._open_tickets: dict[str, int] = {}     # symbol -> ticket of its open position
        self._opened_at: dict[int, float] = {}      # ticket -> time.monotonic() when we opened it
        self._announced: set[str] = set()           # symbols whose strategy rules were logged
        self.journal = journal
        self.reviewer = reviewer
        self._review_tasks: set = set()             # keeps background review tasks alive
        self._reviewing: set[str] = set()           # trade_ids with a review in flight

    # ------------------------------------------------------------------
    # Journal + LLM review (observers only: failures never affect trading)
    # ------------------------------------------------------------------
    def _safe(self, what: str, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            log.exception("Journal/review step failed (%s); trading is unaffected", what)
            return None

    def _journal_open(self, symbol, proposal, sized_order, result, equity, point, history) -> None:
        self.journal.open_trade(
            symbol=symbol, direction=sized_order.direction, ticket=result.ticket,
            volume=result.volume or sized_order.volume,
            signal_price=proposal.entry_price, fill_price=result.price,
            stop_loss=result.sl or sized_order.stop_loss,
            take_profit=result.tp or sized_order.take_profit,
            risk_amount=sized_order.risk_amount, equity=equity,
            context=proposal.context, strategy_rules=self.strategy.describe(point),
            mode=self.execution_agent.mode or "unknown",
            bar_time=str(history["datetime"].iloc[-1]) if "datetime" in history else None,
        )
        self.journal.write_markdown()

    def _journal_close(self, symbol: str, ticket: int, reason: str,
                       exit_price: Optional[float], net_pnl: Optional[float]) -> None:
        if self.journal is None:
            return

        def work():
            rec = self.journal.close_trade(
                ticket, close_reason=reason, exit_price=exit_price, net_pnl=net_pnl,
                history=self.data_agent.history(symbol))
            if rec is None:          # not a journaled trade (e.g. opened before the journal existed)
                return
            self.journal.write_markdown()
            self._schedule_review(rec)

        self._safe("journal close", work)

    def _schedule_review(self, record: dict) -> None:
        if self.reviewer is None or self.journal is None:
            return
        tid = record["trade_id"]
        if tid in self._reviewing:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:         # not inside the event loop (e.g. a unit test): skip
            return
        self._reviewing.add(tid)
        task = loop.create_task(self._review_trade(record))
        self._review_tasks.add(task)
        task.add_done_callback(self._review_tasks.discard)

    async def _review_trade(self, record: dict) -> None:
        tid = record["trade_id"]
        try:
            # The API call is slow, so it runs in a worker thread and never blocks trading.
            review = await asyncio.to_thread(self.reviewer.review, record)
            self.journal.attach_review(tid, review, self.reviewer.model)
            self.journal.write_markdown()
            log.info("LLM review of %s: %s | tags: %s", tid,
                     review.get("outcome_summary", "(no summary)"), review.get("tags"))
        except Exception as exc:
            log.warning("LLM review of %s failed (%s); it will be retried the next time the agent starts.",
                        tid, exc)
        finally:
            self._reviewing.discard(tid)

    def reconcile_journal(self) -> None:
        """On startup: tidy trades left 'open' in the journal by a previous run."""
        if self.journal is None:
            return
        live = self.execution_agent.get_open_positions()          # None in dry-run
        live_tickets = {p.ticket for p in live} if live is not None else None
        for rec in self.journal.open_trades():
            if rec["mode"] == "dry-run":
                self._safe("abandon", self.journal.abandon, rec["trade_id"],
                           "agent restarted before this simulated trade closed")
            elif live_tickets is not None and rec["ticket"] not in live_tickets:
                info = self.execution_agent.get_close_info(rec["ticket"])
                reason = (f"{info['reason']} (while agent was offline)" if info
                          else "closed while agent was offline (reason unknown)")
                self._journal_close(rec["symbol"], rec["ticket"], reason,
                                    info["price"] if info else None, info["profit"] if info else None)
        self._safe("journal markdown", self.journal.write_markdown)

    def _backfill_reviews(self) -> None:
        """Review closed trades that never got a review (API was down, or the agent was stopped)."""
        if self.reviewer is None or self.journal is None:
            return
        for rec in (self._safe("list unreviewed", self.journal.unreviewed_closed) or [])[-20:]:
            self._schedule_review(rec)

    # ------------------------------------------------------------------
    # Position sync: make our books match what is really open at the broker
    # ------------------------------------------------------------------
    def adopt_existing_positions(self) -> None:
        """On startup, pick up positions this agent opened in a previous run."""
        live = self.execution_agent.get_open_positions()
        if not live:
            return
        known_symbols = {s.symbol for s in self.data_agent.config.symbols}
        for pos in live:
            if pos.symbol not in known_symbols:
                log.warning(
                    "Open position %s on %s was opened by this agent but %s is not a configured "
                    "data symbol, so it will NOT be managed (its broker-side SL/TP still apply).",
                    pos.ticket, pos.symbol, pos.symbol)
                continue
            if pos.symbol in self._open_tickets:
                log.warning("Second open position %s on %s ignored (already managing %s).",
                            pos.ticket, pos.symbol, self._open_tickets[pos.symbol])
                continue
            risk = self.execution_agent.estimate_position_risk(pos)
            if risk == 0.0:
                log.warning("Position %s has no stop-loss; its risk is counted as 0.", pos.ticket)
            self._open_tickets[pos.symbol] = pos.ticket
            self._opened_at[pos.ticket] = float("-inf")
            self.risk_agent.register_open_risk(str(pos.ticket), risk)
            self.strategy.set_position(pos.symbol, "long" if pos.direction == "buy" else "short")
            log.info("Adopted existing %s position %s on %s (vol=%.2f, risk=%.2f)",
                     pos.direction, pos.ticket, pos.symbol, pos.volume, risk)

    def sync_positions(self) -> None:
        """Detect positions the broker closed (stop-loss, take-profit, manual) and update state."""
        live = self.execution_agent.get_open_positions()
        if live is None:          # dry run or query failed: keep current state
            return
        live_tickets = {p.ticket for p in live}
        for symbol, ticket in list(self._open_tickets.items()):
            if ticket in live_tickets:
                continue
            if time.monotonic() - self._opened_at.get(ticket, float("-inf")) < 3.0:
                continue          # just opened; give the broker a moment to list it
            self._forget_position(symbol, ticket)
            info = self.execution_agent.get_close_info(ticket)
            if info:
                log.info("Position %s on %s closed by %s at %.5f | net P&L %.2f",
                         ticket, symbol, info["reason"], info["price"], info["profit"])
                self._journal_close(symbol, ticket, info["reason"], info["price"], info["profit"])
            else:
                log.info("Position %s on %s is no longer open at the broker", ticket, symbol)
                self._journal_close(symbol, ticket, "closed at broker (reason unknown)", None, None)

    def _forget_position(self, symbol: str, ticket: int) -> None:
        self._open_tickets.pop(symbol, None)
        self._opened_at.pop(ticket, None)
        self.risk_agent.release_risk(str(ticket))
        self.strategy.set_position(symbol, None)

    async def _sync_loop(self) -> None:
        while True:
            await asyncio.sleep(self.sync_interval_sec)
            try:
                self.sync_positions()
            except Exception:
                log.exception("Position sync failed; will retry")

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------
    def on_new_bar(self, symbol: str, bar: pd.Series) -> None:
        history = self.data_agent.history(symbol)
        if history is None:
            return

        point, tick_value, tick_size = self.execution_agent.get_symbol_risk_params(symbol)
        if symbol not in self._announced:
            self._announced.add(symbol)
            log.info("Strategy for %s: %s", symbol, self.strategy.describe(point))

        proposal, close_action = self.strategy.generate(symbol, history)
        if proposal is None:
            return

        # Reverse: close the existing opposite position before opening the new one
        ticket = self._open_tickets.get(symbol)
        if close_action and ticket is not None:
            result = self.execution_agent.close_position(ticket)
            if result.success:
                self._forget_position(symbol, ticket)
                log.info("Closed %s position (ticket=%s) for reversal on %s",
                         close_action.replace("close_", ""), ticket, symbol)
                info = self.execution_agent.get_close_info(ticket)     # None in dry-run
                self._journal_close(
                    symbol, ticket, "reversal (closed by strategy)",
                    info["price"] if info else float(history["close"].iloc[-1]),
                    info["profit"] if info else None)
            else:
                # Maybe the stop-loss/take-profit closed it a moment ago; re-check the broker.
                self.sync_positions()
                if symbol in self._open_tickets:
                    log.error("Could not close ticket=%s (%s); NOT opening the new trade so "
                              "positions don't stack.", ticket, result.message)
                    return

        equity = self.execution_agent.get_equity()
        open_count = self.execution_agent.get_open_position_count()

        sized_order = self.risk_agent.evaluate(
            proposal=proposal,
            equity=equity,
            symbol_point=point,
            tick_value_per_lot=tick_value,
            tick_size=tick_size,
            open_position_count=open_count,
        )
        if sized_order is None:
            return  # rejected by risk manager, already logged

        result = self.execution_agent.place_order(sized_order)
        if result.success and result.ticket:
            self.risk_agent.register_open_risk(str(result.ticket), sized_order.risk_amount)
            self._open_tickets[symbol] = result.ticket
            self._opened_at[result.ticket] = time.monotonic()
            self.strategy.set_position(symbol, "long" if sized_order.direction == "buy" else "short")
            if self.journal is not None:
                self._safe("journal open", self._journal_open,
                           symbol, proposal, sized_order, result, equity, point, history)
        else:
            log.error("Order for %s was not placed: %s", symbol, result.message)

    async def run(self) -> None:
        self.data_agent.bus.subscribe(self.on_new_bar)
        self.execution_agent.connect()
        self.adopt_existing_positions()
        self.reconcile_journal()
        self._backfill_reviews()
        await asyncio.gather(self.data_agent.run(), self._sync_loop())


async def main() -> None:
    data_config = AgentConfig(
        symbols=[
            SymbolConfig(symbol="BTCUSD", interval="1", n_bars=500),   # exact broker symbol name
        ],
        poll_interval_sec=1.0,   # MT5 is a local connection, so polling fast is cheap
    )
    risk_config = RiskConfig(
        risk_per_trade_pct=1.0,
        max_concurrent_risk_pct=5.0,
        max_daily_loss_pct=3.0,
        max_open_positions=5,
        fixed_volume_lots=None,   # None = size each trade from risk_per_trade_pct (set e.g. 0.01 for a fixed size)
    )
    strategy_config = EmaCrossConfig(
        fast_period=21,
        slow_period=45,
        trend_period=100,        # price must be above/below this EMA for long/short
        stop_loss_distance=150.0,    # price units from entry: $150 on BTCUSD
        take_profit_distance=350.0,  # price units from entry: $350 on BTCUSD
    )
    load_dotenv()  # reads the .env file in the project root into os.environ
    login_raw = (os.getenv("MT5_LOGIN") or "").strip()
    mt5_credentials = MT5Credentials(
        login=int(login_raw) if login_raw else 0,
        password=os.getenv("MT5_PASSWORD") or "",
        server=os.getenv("MT5_SERVER") or "",
    )
    # If the MT5 values are blank, both agents attach to the account the open terminal is logged into.
    data_connector = MT5Connector(
        mt5_credentials.login or None, mt5_credentials.password or None, mt5_credentials.server or None
    )

    data_agent = DataCollectorAgent(data_config, connector=data_connector)
    risk_agent = RiskManagerAgent(risk_config)
    strategy = EmaCrossStrategy(strategy_config)
    # Safe by default: DRY_RUN unless .env says DRY_RUN=false; real-money accounts are refused
    # unless .env also says ALLOW_LIVE=true.
    dry_run = (os.getenv("DRY_RUN") or "true").strip().lower() != "false"
    allow_live = (os.getenv("ALLOW_LIVE") or "false").strip().lower() == "true"
    log.info("Mode: %s%s", "DRY RUN (orders are only logged)" if dry_run else "SENDING ORDERS",
             "" if dry_run else (" to demo or real accounts" if allow_live else " (demo accounts only)"))
    execution_agent = ExecutionAgent(mt5_credentials, dry_run=dry_run, allow_live=allow_live)

    journal = TradeJournal(os.getenv("JOURNAL_DIR") or "journal")
    reviewer = None
    if (os.getenv("ANTHROPIC_API_KEY") or "").strip():
        try:
            reviewer = TradeReviewerAgent(model=(os.getenv("REVIEW_MODEL") or "claude-sonnet-5-5").strip())
            log.info("LLM trade reviews ON (model %s). Journal: %s", reviewer.model, journal.dir / "journal.md")
        except Exception as exc:
            log.warning("LLM trade reviews disabled: %s", exc)
    else:
        log.info("LLM trade reviews OFF (no ANTHROPIC_API_KEY in .env). Trades are still journaled: %s",
                 journal.dir / "journal.md")

    orchestrator = Orchestrator(data_agent, risk_agent, execution_agent, strategy,
                                journal=journal, reviewer=reviewer)
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down.")
