"""
journal.py

Trade journal for the forex agent system.

One JSON file per trade in  <dir>/trades/<trade_id>.json  (restart-safe: open trades are
re-loaded on startup) and a human-readable  <dir>/journal.md  that is regenerated from those
files whenever a trade closes or gets reviewed.

The journal only RECORDS. It never places orders or changes any trading setting.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

log = logging.getLogger("journal")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.2f}" if abs(x) >= 100 else f"{x:.5f}"


def _as_list(x) -> List[str]:
    if not x:
        return []
    return [str(i) for i in x] if isinstance(x, (list, tuple)) else [str(x)]


class TradeJournal:
    def __init__(self, directory: str = "journal"):
        self.dir = Path(directory)
        self.trades_dir = self.dir / "trades"
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        # ticket -> trade_id for trades that are currently open
        self._open_by_ticket: dict[int, str] = {
            rec["ticket"]: rec["trade_id"] for rec in self.open_trades()
        }

    # ------------------------------------------------------------------ storage
    def _path(self, trade_id: str) -> Path:
        return self.trades_dir / f"{trade_id}.json"

    def _write(self, rec: dict) -> None:
        path = self._path(rec["trade_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)          # atomic: a crash can't leave a half-written record

    def get(self, trade_id: str) -> dict:
        return json.loads(self._path(trade_id).read_text(encoding="utf-8"))

    def all_trades(self) -> List[dict]:
        out = []
        for path in self.trades_dir.glob("*.json"):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                log.warning("Skipping unreadable journal file %s", path.name)
        return sorted(out, key=lambda r: r.get("opened_at", ""))

    def open_trades(self) -> List[dict]:
        return [t for t in self.all_trades() if t.get("status") == "open"]

    def unreviewed_closed(self) -> List[dict]:
        return [t for t in self.all_trades() if t.get("status") == "closed" and not t.get("review")]

    # ------------------------------------------------------------------ lifecycle
    def open_trade(
        self, *, symbol: str, direction: str, ticket: int, volume: float,
        signal_price: float, fill_price: Optional[float], stop_loss: float,
        take_profit: Optional[float], risk_amount: float, equity: float,
        context: dict, strategy_rules: str, mode: str, bar_time: Optional[str] = None,
    ) -> str:
        opened = _now()
        trade_id = f"{opened:%Y%m%d-%H%M%S}-{symbol}-{abs(ticket)}"
        rec = {
            "trade_id": trade_id, "status": "open", "mode": mode,   # "dry-run" | "demo" | "real"
            "symbol": symbol, "direction": direction, "ticket": ticket, "volume": volume,
            "opened_at": opened.isoformat(timespec="seconds"), "bar_time": bar_time,
            "signal_price": signal_price, "fill_price": fill_price,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "risk_amount": risk_amount, "equity_at_entry": equity,
            "strategy_rules": strategy_rules, "signal_context": context,
            "closed_at": None, "close_reason": None, "exit_price": None, "net_pnl": None,
            "price_move": None, "duration_minutes": None, "mfe": None, "mae": None,
            "review": None, "review_model": None, "reviewed_at": None,
        }
        self._write(rec)
        self._open_by_ticket[ticket] = trade_id
        log.info("Journal: opened %s", trade_id)
        return trade_id

    def close_trade(
        self, ticket: int, *, close_reason: str, exit_price: Optional[float] = None,
        net_pnl: Optional[float] = None, history: Optional[pd.DataFrame] = None,
    ) -> Optional[dict]:
        """Returns the closed record, or None if this ticket was never journaled."""
        trade_id = self._open_by_ticket.pop(ticket, None)
        if trade_id is None:
            return None
        rec = self.get(trade_id)
        closed = _now()
        entry = rec["fill_price"] or rec["signal_price"]
        sign = 1 if rec["direction"] == "buy" else -1
        rec.update(
            status="closed", closed_at=closed.isoformat(timespec="seconds"),
            close_reason=close_reason, exit_price=exit_price, net_pnl=net_pnl,
            price_move=round((exit_price - entry) * sign, 5) if exit_price is not None else None,
            duration_minutes=round(
                (closed - datetime.fromisoformat(rec["opened_at"])).total_seconds() / 60, 1),
        )
        rec["mfe"], rec["mae"] = self._excursions(rec, history)
        self._write(rec)
        log.info("Journal: closed %s (%s)", trade_id, close_reason)
        return rec

    def abandon(self, trade_id: str, reason: str) -> None:
        """Mark a trade that can never be closed normally (e.g. a dry-run trade left open at exit)."""
        rec = self.get(trade_id)
        rec.update(status="abandoned", close_reason=reason)
        self._write(rec)
        if self._open_by_ticket.get(rec["ticket"]) == trade_id:
            del self._open_by_ticket[rec["ticket"]]

    def attach_review(self, trade_id: str, review: dict, model: str) -> None:
        rec = self.get(trade_id)
        rec.update(review=review, review_model=model, reviewed_at=_now().isoformat(timespec="seconds"))
        self._write(rec)

    @staticmethod
    def _excursions(rec: dict, history: Optional[pd.DataFrame]):
        """Approximate max favorable / adverse price excursion from 1-minute candle highs/lows."""
        if history is None or not rec.get("bar_time"):
            return None, None
        try:
            since = history[history["datetime"] >= pd.Timestamp(rec["bar_time"])]
            if since.empty:
                return None, None
            entry = rec["fill_price"] or rec["signal_price"]
            hi, lo = float(since["high"].max()), float(since["low"].min())
            if rec["direction"] == "buy":
                return round(max(hi - entry, 0), 5), round(max(entry - lo, 0), 5)
            return round(max(entry - lo, 0), 5), round(max(hi - entry, 0), 5)
        except Exception:
            return None, None

    # ------------------------------------------------------------------ markdown
    @staticmethod
    def _render(rec: dict) -> str:
        sign = "BUY" if rec["direction"] == "buy" else "SELL"
        status = rec["status"]
        if status == "closed":
            pnl = rec.get("net_pnl")
            result = (f"{pnl:+,.2f}" if pnl is not None else
                      (f"{rec['price_move']:+,.2f} price" if rec.get("price_move") is not None else "result n/a"))
            head = f"{sign} {rec['symbol']} — {result} ({rec.get('close_reason')})"
        elif status == "open":
            head = f"{sign} {rec['symbol']} — OPEN"
        else:
            head = f"{sign} {rec['symbol']} — abandoned ({rec.get('close_reason')})"
        lines = [f"## {rec['opened_at'].replace('T', ' ')[:16]} UTC — {head}  `[{rec.get('mode')}]`", ""]
        entry = rec.get("fill_price") or rec.get("signal_price")
        lines.append(f"- Entry {_fmt(entry)} | SL {_fmt(rec.get('stop_loss'))} | TP {_fmt(rec.get('take_profit'))} "
                     f"| volume {rec.get('volume')} | risk {_fmt(rec.get('risk_amount'))}")
        if status == "closed":
            lines.append(f"- Exit {_fmt(rec.get('exit_price'))} | held {rec.get('duration_minutes')} min | "
                         f"max favorable {_fmt(rec.get('mfe'))} | max adverse {_fmt(rec.get('mae'))} (approx.)")
        ctx = rec.get("signal_context") or {}
        if ctx.get("signal"):
            lines.append(f"- Signal: {ctx['signal']} at price {_fmt(ctx.get('price'))}, "
                         f"price minus trend EMA {_fmt(ctx.get('price_minus_trend_ema'))}")
        rv = rec.get("review")
        if rv:
            lines += ["", f"**Review ({rec.get('review_model')}):** {rv.get('outcome_summary', '')}"]
            if rv.get("followed_rules"):
                lines.append(f"- Followed its rules: {rv['followed_rules']}. {rv.get('rules_notes', '')}".rstrip())
            for label, key in (("Went well", "what_went_well"), ("Concerns", "concerns"),
                               ("Ideas to test (untested)", "ideas_to_test")):
                items = _as_list(rv.get(key))
                if items:
                    lines.append(f"- {label}: " + "; ".join(items))
            if rv.get("tags"):
                lines.append("- Tags: " + ", ".join(_as_list(rv["tags"])))
            if rv.get("data_limits"):
                lines.append(f"- Limits: {rv['data_limits']}")
        elif status == "closed":
            lines += ["", "_Not reviewed yet._"]
        return "\n".join(lines)

    def write_markdown(self) -> Path:
        trades = sorted(self.all_trades(), key=lambda r: r.get("opened_at", ""), reverse=True)
        body = "\n\n---\n\n".join(self._render(t) for t in trades) or "_No trades yet._"
        path = self.dir / "journal.md"
        path.write_text("# Trade journal\n\nNewest first. Reviews are written by an LLM and are "
                        "advisory notes only.\n\n" + body + "\n", encoding="utf-8")
        return path
