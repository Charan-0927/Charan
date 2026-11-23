"""
data_collector_agent.py

Agentic AI Forex Trading System — Data Collection Agent
---------------------------------------------------------
Responsibility: continuously perceive the market (poll the data source -
MT5 or TradingView - for the latest OHLCV bars on one or more forex pairs) and publish new bars onto an
event bus so downstream agents (signal/strategy, risk, execution) can react
to them without any coupling to how the data was fetched.

IMPORTANT — read before running:
TradingView does not offer an official public API for third-party real-time
data access. This module uses `tvDatafeed`, an open-source community
wrapper around the same internal feed TradingView's own charts use. It is
widely used for research/education but:
  - it is not officially sanctioned by TradingView (ToS grey area)
  - it can break if TradingView changes their internal protocol
  - "real-time" here means fast polling of the latest bar, not true
    tick-by-tick push streaming
For a production/live-money system, swap this connector for an official
data/broker API (OANDA, Twelve Data, Polygon.io, Finnhub, Interactive
Brokers, etc.) — the Agent/EventBus interface below is designed so that
swap only touches the `TradingViewConnector` class.

Install:
    pip install pandas python-dotenv
    pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:  # pragma: no cover
    TvDatafeed = None
    Interval = None

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("data_collector_agent")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class SymbolConfig:
    symbol: str          # e.g. "EURUSD"
    exchange: str = "FX_IDC"   # TradingView's forex exchange code
    interval: str = "1"        # "1","5","15","60","240","1D" (mapped below)
    n_bars: int = 200          # bars fetched per poll (needs enough history
                                # for indicators downstream)


@dataclass
class AgentConfig:
    symbols: List[SymbolConfig]
    poll_interval_sec: float = 5.0
    tv_username: Optional[str] = field(default_factory=lambda: os.getenv("TV_USERNAME"))
    tv_password: Optional[str] = field(default_factory=lambda: os.getenv("TV_PASSWORD"))
    max_backoff_sec: float = 60.0
    csv_dir: Optional[str] = None   # set to a path to persist bars to disk


_INTERVAL_MAP = {
    "1": "in_1_minute",
    "3": "in_3_minute",
    "5": "in_5_minute",
    "15": "in_15_minute",
    "30": "in_30_minute",
    "45": "in_45_minute",
    "60": "in_1_hour",
    "120": "in_2_hour",
    "180": "in_3_hour",
    "240": "in_4_hour",
    "1D": "in_daily",
    "1W": "in_weekly",
    "1M": "in_monthly",
}


# --------------------------------------------------------------------------
# Connector — the only piece that talks to TradingView.
# Swap this class out to point at a different data source.
# --------------------------------------------------------------------------
class TradingViewConnector:
    offload_to_thread = True    # blocking network calls: run in a worker thread

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        if TvDatafeed is None:
            raise ImportError(
                "tvDatafeed is not installed. Run:\n"
                "  pip install --upgrade --no-cache-dir "
                "git+https://github.com/rongardF/tvdatafeed.git"
            )
        self._username = username
        self._password = password
        self._tv: Optional[TvDatafeed] = None

    def connect(self) -> None:
        if self._username and self._password:
            log.info("Logging into TradingView as %s", self._username)
            self._tv = TvDatafeed(self._username, self._password)
        else:
            log.info("No TradingView credentials supplied — using anonymous session "
                      "(fewer symbols / lower rate limits)")
            self._tv = TvDatafeed()

    def fetch(self, cfg: SymbolConfig) -> pd.DataFrame:
        if self._tv is None:
            raise RuntimeError("Connector not connected — call connect() first")
        interval_name = _INTERVAL_MAP.get(cfg.interval)
        if interval_name is None:
            raise ValueError(f"Unsupported interval '{cfg.interval}'")
        interval = getattr(Interval, interval_name)
        df = self._tv.get_hist(
            symbol=cfg.symbol,
            exchange=cfg.exchange,
            interval=interval,
            n_bars=cfg.n_bars,
        )
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {cfg.symbol}@{cfg.exchange}")
        df = df.reset_index().rename(columns={"index": "datetime"})
        return df


# --------------------------------------------------------------------------
# MT5 connector — candles straight from your broker's terminal, so the
# signals are built on the same symbol and prices your orders execute at.
# --------------------------------------------------------------------------
class MT5Connector:
    # MT5 calls are fast local IPC. Keep ALL of them on the event-loop thread
    # (the position sync and order code use it too) instead of a worker thread.
    offload_to_thread = False

    _TIMEFRAMES = {
        "1": "TIMEFRAME_M1", "3": "TIMEFRAME_M3", "5": "TIMEFRAME_M5",
        "15": "TIMEFRAME_M15", "30": "TIMEFRAME_M30", "60": "TIMEFRAME_H1",
        "120": "TIMEFRAME_H2", "180": "TIMEFRAME_H3", "240": "TIMEFRAME_H4",
        "1D": "TIMEFRAME_D1", "1W": "TIMEFRAME_W1", "1M": "TIMEFRAME_MN1",
    }

    def __init__(self, login: Optional[int] = None, password: Optional[str] = None,
                 server: Optional[str] = None):
        if mt5 is None:
            raise ImportError("MetaTrader5 is not installed (Windows only): pip install MetaTrader5")
        self._login, self._password, self._server = login, password, server

    def connect(self) -> None:
        if self._login and self._password and self._server:
            ok = mt5.initialize(login=self._login, password=self._password, server=self._server)
        else:
            ok = mt5.initialize()    # attach to whichever account the open terminal is logged into
        if not ok:
            raise ConnectionError(
                f"MT5 initialize() failed: {mt5.last_error()}. "
                "Is the MT5 terminal open and logged in?")
        log.info("Connected to MT5 for market data")

    def fetch(self, cfg: SymbolConfig) -> pd.DataFrame:
        tf_name = self._TIMEFRAMES.get(cfg.interval)
        if tf_name is None:
            raise ValueError(f"Unsupported interval '{cfg.interval}'")
        if not mt5.symbol_select(cfg.symbol, True):
            raise RuntimeError(f"Symbol {cfg.symbol} is not available at your broker "
                               "(check the exact name in MT5 Market Watch)")
        # start position 0 = the current, still-forming candle (same behavior as before)
        rates = mt5.copy_rates_from_pos(cfg.symbol, getattr(mt5, tf_name), 0, cfg.n_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No candles returned for {cfg.symbol}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")   # broker server time
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["datetime", "open", "high", "low", "close", "volume"]]


# --------------------------------------------------------------------------
# Event bus — decouples this agent from whatever consumes the data
# --------------------------------------------------------------------------
class EventBus:
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._callbacks: List[Callable[[str, pd.Series], None]] = []

    def subscribe(self, callback: Callable[[str, pd.Series], None]) -> None:
        """Register a plain function callback: callback(symbol, bar_row)"""
        self._callbacks.append(callback)

    async def publish(self, symbol: str, bar: pd.Series) -> None:
        await self._queue.put((symbol, bar))
        for cb in self._callbacks:
            try:
                cb(symbol, bar)
            except Exception:
                log.exception("Subscriber callback raised an exception")

    async def get(self):
        """Await the next (symbol, bar) event — for consumers using the queue directly."""
        return await self._queue.get()


# --------------------------------------------------------------------------
# The Agent itself
# --------------------------------------------------------------------------
class DataCollectorAgent:
    """
    Polls TradingView for each configured symbol on its own async task,
    keeps a rolling in-memory DataFrame per symbol, and publishes any new
    bar (by timestamp) to the EventBus.
    """

    def __init__(self, config: AgentConfig, connector=None):
        self.config = config
        # Any object with connect() and fetch(SymbolConfig) -> DataFrame works.
        self.connector = connector or TradingViewConnector(config.tv_username, config.tv_password)
        self.bus = EventBus()
        self.store: Dict[str, pd.DataFrame] = {}
        self._last_seen_ts: Dict[str, pd.Timestamp] = {}
        self._running = False

        if config.csv_dir:
            os.makedirs(config.csv_dir, exist_ok=True)

    def start(self) -> None:
        self.connector.connect()
        self._running = True

    def stop(self) -> None:
        self._running = False

    async def _poll_symbol(self, sym_cfg: SymbolConfig) -> None:
        backoff = self.config.poll_interval_sec
        while self._running:
            try:
                if getattr(self.connector, "offload_to_thread", True):
                    df = await asyncio.to_thread(self.connector.fetch, sym_cfg)
                else:
                    df = self.connector.fetch(sym_cfg)
                backoff = self.config.poll_interval_sec  # reset on success

                self.store[sym_cfg.symbol] = df
                last_ts = df["datetime"].iloc[-1]

                if self._last_seen_ts.get(sym_cfg.symbol) != last_ts:
                    self._last_seen_ts[sym_cfg.symbol] = last_ts
                    latest_bar = df.iloc[-1]
                    log.info(
                        "New bar %s @ %s  O:%.5f H:%.5f L:%.5f C:%.5f",
                        sym_cfg.symbol, last_ts,
                        latest_bar["open"], latest_bar["high"],
                        latest_bar["low"], latest_bar["close"],
                    )
                    await self.bus.publish(sym_cfg.symbol, latest_bar)

                    if self.config.csv_dir:
                        path = os.path.join(self.config.csv_dir, f"{sym_cfg.symbol}.csv")
                        df.to_csv(path, index=False)

            except Exception as exc:
                log.warning("Poll failed for %s: %s — retrying in %.1fs",
                            sym_cfg.symbol, exc, backoff)
                backoff = min(backoff * 2, self.config.max_backoff_sec)

            await asyncio.sleep(backoff if backoff else self.config.poll_interval_sec)

    async def run(self) -> None:
        self.start()
        log.info("DataCollectorAgent starting for symbols: %s",
                  [s.symbol for s in self.config.symbols])
        tasks = [asyncio.create_task(self._poll_symbol(s)) for s in self.config.symbols]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.stop()
            raise

    def latest(self, symbol: str) -> Optional[pd.Series]:
        df = self.store.get(symbol)
        return None if df is None else df.iloc[-1]

    def history(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.store.get(symbol)


# --------------------------------------------------------------------------
# Example standalone usage
# --------------------------------------------------------------------------
def _example_subscriber(symbol: str, bar: pd.Series) -> None:
    """Stand-in for a future signal/strategy agent."""
    log.info(">> [strategy-agent-stub] received %s bar closing at %.5f", symbol, bar["close"])


async def main() -> None:
    config = AgentConfig(
        symbols=[
            SymbolConfig(symbol="EURUSD", exchange="FX_IDC", interval="1", n_bars=200),
            SymbolConfig(symbol="GBPUSD", exchange="FX_IDC", interval="1", n_bars=200),
        ],
        poll_interval_sec=5.0,
        csv_dir="./data",
    )
    agent = DataCollectorAgent(config)
    agent.bus.subscribe(_example_subscriber)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down.")
