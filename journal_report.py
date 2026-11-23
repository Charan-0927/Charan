"""
journal_report.py

Summarizes the trade journal. All numbers are computed here in plain code; the optional
--llm flag asks Claude for a short narrative ON TOP of those numbers (it never recomputes them).

Usage (venv active, from the project folder):
    python journal_report.py                 # stats printed + written to journal/summary.md
    python journal_report.py --llm           # same, plus an LLM narrative (needs ANTHROPIC_API_KEY)
    python journal_report.py --mode demo     # only demo trades (modes: dry-run, demo, real)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import List

from dotenv import load_dotenv

from journal import TradeJournal


def r_multiple(t: dict):
    """Result in units of the planned risk (price move / stop distance); None if unknown."""
    entry = t.get("fill_price") or t.get("signal_price")
    if t.get("price_move") is None or not entry or not t.get("stop_loss"):
        return None
    stop = abs(entry - t["stop_loss"])
    return t["price_move"] / stop if stop else None


def _bucket(rs: List[float]) -> dict:
    return {"trades": len(rs), "avg_R": round(sum(rs) / len(rs), 2),
            "win_rate_pct": round(100 * sum(r > 0 for r in rs) / len(rs), 1)}


def compute_stats(trades: List[dict]) -> dict:
    closed = [t for t in trades if t.get("status") == "closed"]
    rows = [(t, r_multiple(t)) for t in closed]
    rows = [(t, r) for t, r in rows if r is not None]
    stats: dict = {"closed_trades": len(closed), "trades_with_known_result": len(rows),
                   "open_trades": sum(t.get("status") == "open" for t in trades)}
    if not rows:
        return stats
    rs = [r for _, r in rows]
    wins, losses = [r for r in rs if r > 0], [r for r in rs if r <= 0]
    streak = best = 0
    for r in rs:
        streak = streak + 1 if r <= 0 else 0
        best = max(best, streak)
    stats.update(
        win_rate_pct=round(100 * len(wins) / len(rs), 1),
        avg_win_R=round(sum(wins) / len(wins), 2) if wins else None,
        avg_loss_R=round(sum(losses) / len(losses), 2) if losses else None,
        expectancy_R=round(sum(rs) / len(rs), 3),
        total_R=round(sum(rs), 2),
        profit_factor=round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) else None,
        longest_losing_streak=best,
    )
    pnls = [t["net_pnl"] for t, _ in rows if t.get("net_pnl") is not None]
    if pnls:
        stats["net_pnl_total"] = round(sum(pnls), 2)
        stats["net_pnl_known_for_trades"] = len(pnls)

    def group(key):
        buckets = defaultdict(list)
        for t, r in rows:
            buckets[key(t)].append(r)
        return {k: _bucket(v) for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}

    stats["by_direction"] = group(lambda t: t["direction"])
    stats["by_close_reason"] = group(lambda t: t.get("close_reason") or "unknown")
    stats["by_hour_opened_utc"] = group(
        lambda t: f"{datetime.fromisoformat(t['opened_at']).hour:02d}:00")
    stats["sample_note"] = ("Fewer than 30 trades: treat all of this as anecdotal, not evidence."
                            if len(rows) < 30 else "Sample is large enough to look at, not to trust blindly.")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the trade journal")
    parser.add_argument("--dir", default=os.getenv("JOURNAL_DIR") or "journal")
    parser.add_argument("--mode", choices=["dry-run", "demo", "real"], help="only trades from this mode")
    parser.add_argument("--llm", action="store_true", help="add an LLM narrative (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()
    load_dotenv()

    journal = TradeJournal(args.dir)
    trades = [t for t in journal.all_trades() if t.get("status") != "abandoned"]
    if args.mode:
        trades = [t for t in trades if t.get("mode") == args.mode]
    if not trades:
        print("No trades in the journal yet.")
        return

    stats = compute_stats(trades)
    text = "## Statistics (computed by code)\n\n```json\n" + json.dumps(stats, indent=2) + "\n```\n"

    if args.llm:
        from trade_reviewer_agent import TradeReviewerAgent
        if not (os.getenv("ANTHROPIC_API_KEY") or "").strip():
            print("--llm needs ANTHROPIC_API_KEY in .env; skipping the narrative.")
        else:
            reviewer = TradeReviewerAgent(model=(os.getenv("REVIEW_MODEL") or "claude-sonnet-5-5").strip())
            reviewed = [t for t in trades if t.get("review")]
            text += f"\n## Narrative ({reviewer.model}, advisory only)\n\n" + reviewer.summarize_period(stats, reviewed) + "\n"

    out = journal.dir / "summary.md"
    out.write_text("# Journal summary\n\n" + text, encoding="utf-8")
    print(text)
    print(f"(saved to {out})")


if __name__ == "__main__":
    main()
