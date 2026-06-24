from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from config import (
    DEFAULT_COVERAGE_LOOKBACK_HOURS,
    DEFAULT_MIN_WEIGHTED_SIGNAL_COVERAGE,
    DEFAULT_REFRESH_INTERVAL_HOURS,
)
from database import fetch_articles, get_refresh_state, update_refresh_state
from news_collector import collect_from_presets


@dataclass
class RefreshDecision:
    should_run: bool
    reason: str
    weighted_coverage: float
    last_run_at: str | None
    hours_since_last_run: float | None


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.replace(tzinfo=None)
    return parsed


def _article_weight(row: dict[str, Any]) -> float:
    signal_strength = row.get("signal_strength")
    try:
        if signal_strength is not None:
            return max(float(signal_strength), 0.0)
    except (TypeError, ValueError):
        pass

    try:
        score = abs(float(row.get("final_signal_score") or row.get("final_impact_score") or 0))
        source_weight = float(row.get("source_weight") or 0.4)
        novelty = float(row.get("novelty_score") or 1)
    except (TypeError, ValueError):
        return 0.0
    return max(score * source_weight * novelty, 0.0)


def weighted_signal_coverage(ticker: str, lookback_hours: int = DEFAULT_COVERAGE_LOOKBACK_HOURS) -> float:
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    coverage = 0.0
    for row in fetch_articles(ticker, include_duplicates=True):
        record = dict(row)
        if int(record.get("is_duplicate") or 0):
            continue
        created_at = _parse_datetime(record.get("published_at") or record.get("created_at"))
        if created_at is None or created_at < cutoff:
            continue
        coverage += _article_weight(record)
    return round(coverage, 3)


def evaluate_refresh_policy(
    ticker: str,
    *,
    min_weighted_coverage: float = DEFAULT_MIN_WEIGHTED_SIGNAL_COVERAGE,
    refresh_interval_hours: int = DEFAULT_REFRESH_INTERVAL_HOURS,
    lookback_hours: int = DEFAULT_COVERAGE_LOOKBACK_HOURS,
) -> RefreshDecision:
    ticker = ticker.upper().strip()
    coverage = weighted_signal_coverage(ticker, lookback_hours)
    state = get_refresh_state(ticker)
    last_run_at = state["last_run_at"] if state else None
    last_run_dt = _parse_datetime(last_run_at)
    hours_since_last_run = None
    if last_run_dt is not None:
        hours_since_last_run = round((datetime.utcnow() - last_run_dt).total_seconds() / 3600, 2)

    reasons: list[str] = []
    if coverage < min_weighted_coverage:
        reasons.append(f"coverage {coverage:.2f} below threshold {min_weighted_coverage:.2f}")
    if last_run_dt is None:
        reasons.append("no previous refresh")
    elif datetime.utcnow() - last_run_dt >= timedelta(hours=refresh_interval_hours):
        reasons.append(f"{hours_since_last_run:.2f} hours since last refresh")

    return RefreshDecision(
        should_run=bool(reasons),
        reason="; ".join(reasons) if reasons else "coverage and time threshold satisfied",
        weighted_coverage=coverage,
        last_run_at=last_run_at,
        hours_since_last_run=hours_since_last_run,
    )


def collect_if_due(
    ticker: str,
    company_name: str,
    selected_feeds: list[str] | None,
    *,
    per_feed_limit: int = 3,
    min_weighted_coverage: float = DEFAULT_MIN_WEIGHTED_SIGNAL_COVERAGE,
    refresh_interval_hours: int = DEFAULT_REFRESH_INTERVAL_HOURS,
    lookback_hours: int = DEFAULT_COVERAGE_LOOKBACK_HOURS,
    force: bool = False,
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    decision = evaluate_refresh_policy(
        ticker,
        min_weighted_coverage=min_weighted_coverage,
        refresh_interval_hours=refresh_interval_hours,
        lookback_hours=lookback_hours,
    )
    if not force and not decision.should_run:
        update_refresh_state(
            ticker,
            company_name,
            last_run_at=decision.last_run_at,
            last_status="skipped",
            last_reason=decision.reason,
            last_saved=0,
            last_weighted_coverage=decision.weighted_coverage,
        )
        return {"status": "skipped", "decision": decision, "results": []}

    results = collect_from_presets(ticker, company_name, selected_feeds, per_feed_limit)
    saved_count = sum(1 for result in results if result.get("status") == "saved")
    updated_coverage = weighted_signal_coverage(ticker, lookback_hours)
    run_time = datetime.utcnow().isoformat(timespec="seconds")
    update_refresh_state(
        ticker,
        company_name,
        last_run_at=run_time,
        last_status="forced" if force else "collected",
        last_reason="forced refresh" if force else decision.reason,
        last_saved=saved_count,
        last_weighted_coverage=updated_coverage,
    )
    return {
        "status": "forced" if force else "collected",
        "decision": decision,
        "results": results,
        "saved_count": saved_count,
        "updated_weighted_coverage": updated_coverage,
        "last_run_at": run_time,
    }
