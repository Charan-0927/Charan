"""
trade_reviewer_agent.py

Agentic AI Forex Trading System — Trade Reviewer Agent (LLM)
------------------------------------------------------------------
After a trade closes, asks Claude for a short, structured post-trade review and returns it
as a dict. ADVISORY ONLY: this agent has no access to orders, risk limits or settings, and
nothing it returns is ever used to trade. The review is stored in the journal for a human.

All numbers (R-multiples, excursions, durations) are computed here in plain code and handed
to the model; the model is told not to do arithmetic of its own.

review() is blocking (a network call): the orchestrator runs it in a worker thread.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

log = logging.getLogger("trade_reviewer_agent")

DEFAULT_MODEL = "claude-sonnet-5-5"

REVIEW_SYSTEM_PROMPT = """You are the post-trade reviewer for an automated trading bot. You review ONE \
closed trade at a time and write a short, factual review for the bot's human owner.

Ground rules:
- Use ONLY the data in the trade record. Never invent news, indicators, prices or events. If something \
cannot be judged from the data, say so in "data_limits".
- Do not predict future prices and do not give financial advice.
- Judge two things separately: (1) did the bot follow ITS OWN stated rules at entry (the rules are in \
"strategy_rules" and the entry facts are in "signal_context"), and (2) how the trade turned out. A rule-following \
trade that loses is not a mistake, and a lucky win is not proof of skill. One trade is mostly noise: do not \
over-explain a single outcome.
- All numbers you need are already computed in "derived" (result_in_R, max_favorable_in_R, ...). Quote them; \
do not recompute them.
- If "mode" is "dry-run", the exit is simulated (approximate price, no spread or fees): say so.
- You cannot place orders or change settings. "ideas_to_test" are hypotheses for the human to check on a \
demo account or in a backtest; word them as untested hypotheses, never as instructions.

Reply with ONE JSON object and nothing else (no markdown fences), with exactly these keys:
{
  "followed_rules": "yes" | "no" | "unclear",
  "rules_notes": "one sentence on the entry-rule check",
  "outcome_summary": "2-3 plain sentences: what happened and how it ended",
  "what_went_well": ["at most 3 short items"],
  "concerns": ["at most 3 short items, only real ones"],
  "tags": ["short_snake_case_tags, e.g. stopped_out_fast, near_miss_tp, trend_continuation, choppy_entry"],
  "ideas_to_test": ["at most 2 untested hypotheses"],
  "data_limits": "what you could not judge from this data"
}"""

SUMMARY_SYSTEM_PROMPT = """You summarize the performance statistics of an automated trading bot for its human \
owner. The statistics were computed by code from the trade journal: quote them, never recompute them. Always \
state the sample size and say plainly when it is too small for conclusions. Note recurring tags from the \
per-trade reviews, and finish with at most 3 untested hypotheses to check on a demo account or in a backtest. \
No financial advice and no price predictions. Write short markdown."""

_TRADE_KEYS = (
    "mode", "symbol", "direction", "opened_at", "closed_at", "signal_price", "fill_price",
    "stop_loss", "take_profit", "volume", "risk_amount", "equity_at_entry", "strategy_rules",
    "signal_context", "close_reason", "exit_price", "net_pnl", "price_move",
    "duration_minutes", "mfe", "mae",
)


def derive_metrics(t: dict) -> dict:
    """Numbers computed in code so the model never has to do arithmetic."""
    entry = t.get("fill_price") or t.get("signal_price")
    d: dict = {}
    sl_dist = abs(entry - t["stop_loss"]) if entry and t.get("stop_loss") else None
    tp_dist = abs(t["take_profit"] - entry) if entry and t.get("take_profit") else None
    if sl_dist:
        d["planned_stop_distance"] = round(sl_dist, 5)
    if tp_dist:
        d["planned_target_distance"] = round(tp_dist, 5)
    if sl_dist and tp_dist:
        d["planned_reward_to_risk"] = round(tp_dist / sl_dist, 2)
    if sl_dist and t.get("price_move") is not None:
        d["result_in_R"] = round(t["price_move"] / sl_dist, 2)
    if sl_dist and t.get("mfe") is not None:
        d["max_favorable_in_R"] = round(t["mfe"] / sl_dist, 2)
    if sl_dist and t.get("mae") is not None:
        d["max_adverse_in_R"] = round(t["mae"] / sl_dist, 2)
    if t.get("fill_price") and t.get("signal_price"):
        d["fill_minus_signal_price"] = round(t["fill_price"] - t["signal_price"], 5)
    if t.get("duration_minutes") is not None:
        d["duration_minutes"] = t["duration_minutes"]
    return d


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Model didn't return usable JSON: keep its words rather than losing the review.
    return {"outcome_summary": text.strip()[:1500], "parse_error": True}


class TradeReviewerAgent:
    def __init__(self, model: str = DEFAULT_MODEL, client=None, max_tokens: int = 1000):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:       # pragma: no cover
                raise ImportError("The 'anthropic' package is not installed. Run: pip install anthropic") from exc
            client = anthropic.Anthropic(timeout=60.0)    # reads ANTHROPIC_API_KEY from the environment
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    def _ask(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def review(self, trade: dict) -> dict:
        """Blocking. Raises on API errors; the caller logs and retries later."""
        view = {k: trade.get(k) for k in _TRADE_KEYS}
        view["derived"] = derive_metrics(trade)
        user = "Review this closed trade.\n\n<trade>\n" + json.dumps(view, indent=2, default=str) + "\n</trade>"
        return _parse_json(self._ask(REVIEW_SYSTEM_PROMPT, user))

    def summarize_period(self, stats: dict, reviewed: list) -> str:
        """Narrative over code-computed stats plus the per-trade review one-liners."""
        brief = [{"trade": t["trade_id"], "close_reason": t.get("close_reason"),
                  "summary": (t.get("review") or {}).get("outcome_summary"),
                  "tags": (t.get("review") or {}).get("tags")} for t in reviewed[-30:]]
        user = ("Statistics (computed by code):\n" + json.dumps(stats, indent=2, default=str) +
                "\n\nRecent per-trade reviews:\n" + json.dumps(brief, indent=2, default=str))
        return self._ask(SUMMARY_SYSTEM_PROMPT, user)
