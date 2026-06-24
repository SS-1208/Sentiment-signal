from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from math import isnan, sqrt
from statistics import median, stdev
from typing import Any


SIGNAL_BUCKET_ORDER = [
    "strongly negative",
    "mildly negative",
    "neutral",
    "mildly positive",
    "strongly positive",
]

MIN_RESEARCH_SAMPLE_SIZE = 30


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(str(value)[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def time_decay_factor(published_at: Any, now: datetime | None = None) -> float:
    timestamp = _parse_datetime(published_at)
    if timestamp is None:
        return 0.6
    current = now or datetime.utcnow()
    age_hours = max(0.0, (current - timestamp).total_seconds() / 3600)
    if age_hours <= 6:
        return 1.0
    if age_hours <= 24:
        return 0.6
    if age_hours <= 72:
        return 0.3
    if age_hours <= 168:
        return 0.1
    return 0.0


def rows_to_records(rows: list[Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and isnan(value):
        return False
    return True


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    unique = [row for row in records if not row.get("is_duplicate")]

    def avg(field: str, rows: list[dict[str, Any]] = unique) -> float:
        values = [float(row[field]) for row in rows if _is_present(row.get(field))]
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "total_articles": len(records),
        "positive_articles": sum(1 for row in unique if row.get("sentiment_direction") == "positive"),
        "negative_articles": sum(1 for row in unique if row.get("sentiment_direction") == "negative"),
        "neutral_articles": sum(1 for row in unique if row.get("sentiment_direction") == "neutral"),
        "average_materiality": avg("materiality"),
        "average_confidence": avg("confidence"),
        "average_impact_score": avg("final_impact_score"),
        "average_final_signal_score": avg("final_signal_score"),
        "duplicate_articles": sum(1 for row in records if row.get("is_duplicate")),
        "bullish_articles": sum(1 for row in unique if row.get("market_signal") == "bullish"),
        "bearish_articles": sum(1 for row in unique if row.get("market_signal") == "bearish"),
        "mixed_articles": sum(1 for row in unique if row.get("market_signal") == "mixed"),
        "irrelevant_articles": sum(1 for row in unique if row.get("market_signal") == "irrelevant"),
    }


def validation_metrics(records: list[dict[str, Any]], return_field: str = "return_5d") -> dict[str, float]:
    unique = [row for row in records if not row.get("is_duplicate")]

    def avg_return(rows: list[dict[str, Any]], field: str = return_field) -> float:
        values = [float(row[field]) for row in rows if _is_present(row.get(field))]
        return round(sum(values) / len(values), 3) if values else 0.0

    high_materiality = [row for row in unique if float(row.get("materiality") or 0) >= 7]
    low_materiality = [row for row in unique if float(row.get("materiality") or 0) < 7]
    return {
        "avg_return_after_positive_news": avg_return([row for row in unique if row.get("sentiment_direction") == "positive"]),
        "avg_return_after_negative_news": avg_return([row for row in unique if row.get("sentiment_direction") == "negative"]),
        "avg_return_after_high_materiality_news": avg_return(high_materiality),
        "avg_return_after_low_materiality_news": avg_return(low_materiality),
    }


def default_return_field(records: list[dict[str, Any]], horizon: str = "5d") -> str:
    excess_field = f"excess_return_{horizon}"
    raw_field = f"return_{horizon}"
    return excess_field if any(_is_present(row.get(excess_field)) for row in records) else raw_field


def signal_bucket(final_impact_score: Any) -> str:
    score = float(final_impact_score or 0)
    if score >= 7:
        return "strongly positive"
    if score >= 1:
        return "mildly positive"
    if score <= -7:
        return "strongly negative"
    if score <= -1:
        return "mildly negative"
    return "neutral"


def add_research_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in records:
        copy = dict(row)
        final_signal = copy.get("final_signal_score")
        copy["final_signal_score"] = float(final_signal) if _is_present(final_signal) else float(copy.get("final_impact_score") or 0)
        copy["signal_bucket"] = signal_bucket(copy.get("final_signal_score"))
        copy["materiality_group"] = "high materiality" if float(copy.get("materiality") or 0) >= 7 else "low materiality"
        copy["source_quality"] = "high-quality source" if float(copy.get("source_weight") or 0) >= 1.2 else "low-quality source"
        copy["event_type"] = copy.get("event_type") or "other"
        copy["published_at"] = copy.get("published_at") or copy.get("published_date") or copy.get("created_at") or ""
        copy["expectation_surprise"] = copy.get("expectation_surprise") or "neutral"
        copy["novelty_score"] = float(copy.get("novelty_score") or 1)
        copy["signal_strength"] = float(copy.get("signal_strength") or abs(copy["final_signal_score"]))
        decay = time_decay_factor(copy.get("published_at") or copy.get("published_date"))
        copy["time_decay_factor"] = decay
        copy["decayed_signal_score"] = round(copy["final_signal_score"] * decay, 4)
        copy["decayed_signal_strength"] = round(copy["signal_strength"] * decay, 4)
        copy["article_tone"] = copy.get("article_tone") or copy.get("sentiment_direction") or "neutral"
        copy["market_signal"] = copy.get("market_signal") or "neutral"
        copy["ticker_relevance"] = copy.get("ticker_relevance") or "low"
        copy["contradiction_flag"] = int(copy.get("contradiction_flag") or 0)
        copy["market_cap_bucket"] = copy.get("market_cap_bucket") or "Unknown"
        copy["sector"] = copy.get("sector") or "Unknown"
        copy["sector_etf"] = copy.get("sector_etf") or "Unknown"
        copy["data_quality_score"] = float(copy.get("data_quality_score") or 100)
        enriched.append(copy)
    return enriched


def filter_duplicates(records: list[dict[str, Any]], include_duplicates: bool = False) -> list[dict[str, Any]]:
    if include_duplicates:
        return list(records)
    return [row for row in records if not int(row.get("is_duplicate") or 0)]


def _score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("final_signal_score") or row.get("final_impact_score") or 0)
    except (TypeError, ValueError):
        return 0.0


def _chronological_key(row: dict[str, Any]) -> tuple[str, int]:
    timestamp = row.get("published_at") or row.get("published_date") or row.get("created_at") or ""
    try:
        row_id = int(row.get("id") or 0)
    except (TypeError, ValueError):
        row_id = 0
    return str(timestamp), row_id


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def data_readiness_report(records: list[dict[str, Any]], return_field: str = "return_5d") -> dict[str, Any]:
    unique = filter_duplicates(records)
    with_returns = [row for row in unique if _is_present(row.get(return_field))]
    timestamps = sorted(str(row.get("published_at") or row.get("published_date") or "") for row in unique if row.get("published_at") or row.get("published_date"))
    return {
        "non_duplicate_articles": len(unique),
        "articles_with_selected_returns": len(with_returns),
        "return_coverage_pct": round(len(with_returns) / len(unique) * 100, 2) if unique else 0,
        "first_article": timestamps[0] if timestamps else "N/A",
        "latest_article": timestamps[-1] if timestamps else "N/A",
        "sample_warning": "low sample" if len(with_returns) < MIN_RESEARCH_SAMPLE_SIZE else "ok",
    }


def return_summary(records: list[dict[str, Any]], group_field: str, return_prefix: str = "return") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row.get(group_field, "unknown")), []).append(row)

    def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
        return [float(row[field]) for row in rows if _is_present(row.get(field))]

    def avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    summaries: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        return_1d = numeric_values(rows, f"{return_prefix}_1d")
        return_5d = numeric_values(rows, f"{return_prefix}_5d")
        return_20d = numeric_values(rows, f"{return_prefix}_20d")
        win_base = return_5d
        summaries.append(
            {
                group_field: group,
                "observations": len(rows),
                "avg_1d_return": avg(return_1d),
                "avg_5d_return": avg(return_5d),
                "avg_20d_return": avg(return_20d),
                "median_1d_return": round(median(return_1d), 4) if return_1d else None,
                "median_5d_return": round(median(return_5d), 4) if return_5d else None,
                "median_20d_return": round(median(return_20d), 4) if return_20d else None,
                "win_rate_5d": round(sum(1 for value in win_base if value > 0) / len(win_base) * 100, 2) if win_base else None,
                "sample_warning": "low sample" if len(return_5d) < MIN_RESEARCH_SAMPLE_SIZE else "ok",
            }
        )

    if group_field == "signal_bucket":
        order = {bucket: index for index, bucket in enumerate(SIGNAL_BUCKET_ORDER)}
        summaries.sort(key=lambda row: order.get(row[group_field], 999))
    else:
        summaries.sort(key=lambda row: str(row[group_field]))
    return summaries


def simulate_research_strategy(records: list[dict[str, Any]], return_field: str = "return_5d") -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    cumulative_return = 0.0
    candidates = sorted(
        [row for row in records if _is_present(row.get(return_field))],
        key=lambda row: (str(row.get("published_date", "")), int(row.get("id") or 0)),
    )
    for row in candidates:
        score = float(row.get("final_signal_score") or row.get("final_impact_score") or 0)
        if score >= 7:
            action = "buy"
            strategy_return = float(row[return_field])
        elif score <= -7:
            action = "short_or_avoid"
            strategy_return = -float(row[return_field])
        else:
            continue
        cumulative_return += strategy_return
        trades.append(
            {
                "published_date": row.get("published_date"),
                "ticker": row.get("ticker"),
                "title": row.get("title"),
                "action": action,
                "final_signal_score": score,
                "hold_days": 5,
                "trade_return": round(strategy_return, 4),
                "cumulative_return": round(cumulative_return, 4),
            }
        )
    return trades


def walk_forward_backtest(
    records: list[dict[str, Any]],
    return_field: str = "return_5d",
    *,
    min_train_observations: int = 30,
    threshold_quantile: float = 0.8,
    minimum_threshold: float = 7.0,
) -> dict[str, Any]:
    candidates = sorted(
        [
            row
            for row in filter_duplicates(records)
            if _is_present(row.get(return_field)) and _score(row) != 0
        ],
        key=_chronological_key,
    )
    trades: list[dict[str, Any]] = []
    cumulative_return = 0.0
    for index, row in enumerate(candidates):
        prior = candidates[:index]
        if len(prior) < min_train_observations:
            continue
        historical_scores = [abs(_score(item)) for item in prior if abs(_score(item)) > 0]
        learned_threshold = max(minimum_threshold, _percentile(historical_scores, threshold_quantile))
        score = _score(row)
        if score >= learned_threshold:
            action = "buy"
            strategy_return = float(row[return_field])
        elif score <= -learned_threshold:
            action = "short_or_avoid"
            strategy_return = -float(row[return_field])
        else:
            continue
        cumulative_return += strategy_return
        trades.append(
            {
                "published_at": row.get("published_at") or row.get("published_date"),
                "ticker": row.get("ticker"),
                "title": row.get("title"),
                "action": action,
                "learned_threshold": round(learned_threshold, 4),
                "final_signal_score": round(score, 4),
                "trade_return": round(strategy_return, 4),
                "cumulative_return": round(cumulative_return, 4),
                "training_observations_available": len(prior),
            }
        )

    returns = [float(row["trade_return"]) for row in trades]
    wins = [value for value in returns if value > 0]
    summary = {
        "eligible_observations": len(candidates),
        "minimum_training_observations": min_train_observations,
        "walk_forward_trades": len(trades),
        "average_trade_return": round(sum(returns) / len(returns), 4) if returns else None,
        "win_rate": round(len(wins) / len(returns) * 100, 2) if returns else None,
        "cumulative_return": round(cumulative_return, 4) if returns else None,
        "sample_warning": "low sample" if len(trades) < MIN_RESEARCH_SAMPLE_SIZE else "ok",
    }
    return {"summary": summary, "trades": trades}


def holdout_validation(records: list[dict[str, Any]], return_field: str = "return_5d", holdout_fraction: float = 0.3) -> list[dict[str, Any]]:
    candidates = sorted(
        [
            row
            for row in filter_duplicates(records)
            if _is_present(row.get(return_field)) and _score(row) != 0
        ],
        key=_chronological_key,
    )
    if len(candidates) < 4:
        return [
            {
                "split": "insufficient data",
                "observations": len(candidates),
                "threshold": None,
                "evaluated_signals": 0,
                "accuracy": None,
                "average_strategy_return": None,
                "sample_warning": "insufficient data",
            }
        ]

    split_index = max(1, min(len(candidates) - 1, int(len(candidates) * (1 - holdout_fraction))))
    train = candidates[:split_index]
    holdout = candidates[split_index:]
    threshold = max(1.0, _percentile([abs(_score(row)) for row in train], 0.6))

    def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        evaluated = [row for row in rows if abs(_score(row)) >= threshold and float(row[return_field]) != 0]
        correct = [
            row
            for row in evaluated
            if (_score(row) > 0 and float(row[return_field]) > 0) or (_score(row) < 0 and float(row[return_field]) < 0)
        ]
        strategy_returns = [float(row[return_field]) if _score(row) > 0 else -float(row[return_field]) for row in evaluated]
        return {
            "split": name,
            "observations": len(rows),
            "threshold": round(threshold, 4),
            "evaluated_signals": len(evaluated),
            "accuracy": round(len(correct) / len(evaluated) * 100, 2) if evaluated else None,
            "average_strategy_return": round(sum(strategy_returns) / len(strategy_returns), 4) if strategy_returns else None,
            "sample_warning": "low sample" if len(evaluated) < MIN_RESEARCH_SAMPLE_SIZE else "ok",
        }

    return [summarize("train", train), summarize("holdout", holdout)]


def confidence_calibration(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    output: list[dict[str, Any]] = []
    for low, high in buckets:
        rows = [
            row
            for row in records
            if row.get("prediction_correct") is not None
            and _is_present(row.get("prediction_correct"))
            and low <= float(row.get("confidence") or 0) < (high if high < 100 else 101)
        ]
        actual = sum(int(row["prediction_correct"]) for row in rows) / len(rows) * 100 if rows else None
        output.append(
            {
                "confidence_bucket": f"{low}-{high}",
                "predicted_confidence_midpoint": (low + high) / 2,
                "actual_success_rate": round(actual, 2) if actual is not None else None,
                "observations": len(rows),
            }
        )
    return output


def _return_values(rows: list[dict[str, Any]], return_field: str) -> list[float]:
    return [float(row[return_field]) for row in rows if _is_present(row.get(return_field))]


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _win_rate(values: list[float]) -> float | None:
    return round(sum(1 for value in values if value > 0) / len(values) * 100, 2) if values else None


def _t_test(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    if len({round(value, 12) for value in values}) <= 1:
        return None, None
    try:
        from scipy import stats

        result = stats.ttest_1samp(values, 0.0)
        return round(float(result.statistic), 4), round(float(result.pvalue), 6)
    except Exception:
        std_dev = stdev(values)
        if std_dev == 0:
            return None, None
        t_stat = (sum(values) / len(values)) / (std_dev / sqrt(len(values)))
        return round(t_stat, 4), None


def _signed_values(rows: list[dict[str, Any]], field: str, direction_fn) -> list[float]:
    values: list[float] = []
    for row in rows:
        if not _is_present(row.get(field)):
            continue
        direction = direction_fn(row)
        if direction == 0:
            continue
        values.append(float(row[field]) * direction)
    return values


def _baseline_row(name: str, rows: list[dict[str, Any]], direction_fn=lambda row: 1) -> dict[str, Any]:
    fields = [
        "return_1d",
        "return_5d",
        "return_20d",
        "excess_return_1d",
        "excess_return_5d",
        "excess_return_20d",
    ]
    metrics: dict[str, Any] = {"baseline": name, "sample_size": 0}
    primary = _signed_values(rows, "return_5d", direction_fn)
    metrics["sample_size"] = len(primary)
    for field in fields:
        values = _signed_values(rows, field, direction_fn)
        metrics[f"average_{field}"] = _mean(values)
    metrics["win_rate"] = _win_rate(primary)
    t_stat, p_value = _t_test(primary)
    metrics["t_statistic"] = t_stat
    metrics["p_value"] = p_value
    return metrics


def _stable_random_direction(row: dict[str, Any]) -> int:
    seed = f"{row.get('ticker')}|{row.get('id')}|{row.get('title')}".encode("utf-8")
    return 1 if int(sha256(seed).hexdigest(), 16) % 2 == 0 else -1


def baseline_comparison(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = sorted(filter_duplicates(records), key=_chronological_key)
    high_source = [row for row in unique if float(row.get("source_weight") or 0) >= 1.2]
    high_materiality = [row for row in unique if float(row.get("materiality") or 0) >= 7]
    sentiment_signal = [row for row in unique if abs(_score(row)) >= 7]
    bullish_signal = [row for row in unique if row.get("market_signal") == "bullish"]
    positive_tone = [row for row in unique if row.get("article_tone") == "positive"]

    prior_by_ticker: dict[str, float] = {}
    momentum_rows: list[dict[str, Any]] = []
    momentum_direction: dict[int, int] = {}
    for row in unique:
        ticker = str(row.get("ticker") or "")
        prior = prior_by_ticker.get(ticker)
        if prior is not None:
            momentum_rows.append(row)
            momentum_direction[id(row)] = 1 if prior > 0 else -1
        if _is_present(row.get("return_1d")):
            prior_by_ticker[ticker] = float(row["return_1d"])

    return [
        _baseline_row("SentimentSignal final_signal_score", sentiment_signal, lambda row: 1 if _score(row) > 0 else -1),
        _baseline_row("random signal baseline", unique, _stable_random_direction),
        _baseline_row("buy all collected news", unique),
        _baseline_row("buy only positive article_tone", positive_tone),
        _baseline_row("buy only bullish market_signal", bullish_signal),
        _baseline_row("buy only high source_weight articles", high_source),
        _baseline_row("buy only high materiality articles", high_materiality),
        _baseline_row("momentum proxy", momentum_rows, lambda row: momentum_direction.get(id(row), 0)),
    ]


def baseline_interpretation(rows: list[dict[str, Any]]) -> str:
    signal = next((row for row in rows if row["baseline"] == "SentimentSignal final_signal_score"), None)
    if not signal or not _is_present(signal.get("average_excess_return_5d")):
        return "Not enough return data to determine whether SentimentSignal outperforms simple baselines."
    signal_return = float(signal["average_excess_return_5d"])
    comparable = [
        row for row in rows
        if row is not signal and _is_present(row.get("average_excess_return_5d")) and int(row.get("sample_size") or 0) >= 2
    ]
    if not comparable:
        return "SentimentSignal has return data, but baselines do not yet have enough comparable observations."
    best_baseline = max(float(row["average_excess_return_5d"]) for row in comparable)
    if signal_return > best_baseline:
        return "SentimentSignal currently outperforms the simple baselines on average 5-day excess return, but check sample size and p-values before trusting it."
    return "SentimentSignal does not currently outperform the best simple baseline on average 5-day excess return."


def performance_by_group(records: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in filter_duplicates(records):
        grouped.setdefault(str(row.get(group_field) or "Unknown"), []).append(row)
    output: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        excess_5d = _return_values(rows, "excess_return_5d")
        t_stat, p_value = _t_test(excess_5d)
        row = {
            group_field: group,
            "count": len(rows),
            "average_return_1d": _mean(_return_values(rows, "return_1d")),
            "average_return_5d": _mean(_return_values(rows, "return_5d")),
            "average_return_20d": _mean(_return_values(rows, "return_20d")),
            "average_excess_return_1d": _mean(_return_values(rows, "excess_return_1d")),
            "average_excess_return_5d": _mean(excess_5d),
            "average_excess_return_20d": _mean(_return_values(rows, "excess_return_20d")),
            "average_sector_excess_return_1d": _mean(_return_values(rows, "sector_excess_return_1d")),
            "average_sector_excess_return_5d": _mean(_return_values(rows, "sector_excess_return_5d")),
            "average_sector_excess_return_20d": _mean(_return_values(rows, "sector_excess_return_20d")),
            "win_rate": _win_rate(excess_5d),
            "average_final_signal_score": _mean([_score(row) for row in rows]),
            "average_signal_strength": _mean([float(row.get("signal_strength") or 0) for row in rows]),
            "t_statistic": t_stat,
            "p_value": p_value,
            "sample_warning": "low sample" if len(excess_5d) < MIN_RESEARCH_SAMPLE_SIZE else "ok",
        }
        meaningful = row["average_excess_return_5d"] is not None and abs(float(row["average_excess_return_5d"])) >= 1.0
        row["highlight"] = bool(len(excess_5d) >= MIN_RESEARCH_SAMPLE_SIZE and meaningful and p_value is not None and p_value < 0.05)
        output.append(row)
    return sorted(output, key=lambda row: (row["highlight"] is not True, str(row[group_field])))


def duplicate_cluster_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        group = str(row.get("duplicate_group") or f"story-{row.get('id')}")
        clusters.setdefault(group, []).append(row)
    output: list[dict[str, Any]] = []
    for group, rows in clusters.items():
        timestamps = sorted(str(row.get("published_at") or row.get("published_date") or row.get("created_at") or "") for row in rows)
        sources = sorted({str(row.get("source") or "Unknown") for row in rows})
        source_weights = [float(row.get("source_weight") or 0) for row in rows]
        strengths = [float(row.get("signal_strength") or 0) for row in rows]
        scores = [_score(row) for row in rows]
        number_of_sources = len(sources)
        total_signal_strength = sum(strengths)
        story_virality_score = round(total_signal_strength * max(1, number_of_sources) * (1 + max(0, len(rows) - 1) * 0.15), 4)
        output.append(
            {
                "duplicate_group": group,
                "first_seen_at": timestamps[0] if timestamps else "",
                "latest_seen_at": timestamps[-1] if timestamps else "",
                "articles_in_cluster": len(rows),
                "number_of_sources": number_of_sources,
                "source_list": ", ".join(sources),
                "strongest_source_weight": round(max(source_weights), 3) if source_weights else 0,
                "average_signal_score": _mean(scores),
                "total_signal_strength": round(total_signal_strength, 4),
                "story_virality_score": story_virality_score,
            }
        )
    return sorted(output, key=lambda row: row["story_virality_score"], reverse=True)


def validation_guardrails(records: list[dict[str, Any]], return_field: str = "excess_return_5d") -> list[dict[str, Any]]:
    unique = filter_duplicates(records)
    warnings: list[dict[str, Any]] = []
    values = _return_values(unique, return_field)
    if len(values) < MIN_RESEARCH_SAMPLE_SIZE:
        warnings.append({"severity": "high", "warning": f"Sample size is below 30 for {return_field}.", "detail": f"n={len(values)}"})

    tests = statistical_tests(unique, "signal_bucket", return_field)
    if tests and not any(row.get("p_value") is not None and float(row["p_value"]) < 0.05 for row in tests):
        warnings.append({"severity": "medium", "warning": "No signal bucket has a strong p-value.", "detail": "Treat observed patterns as exploratory."})

    holdout = holdout_validation(unique, return_field)
    if len(holdout) == 2:
        train = holdout[0].get("average_strategy_return")
        test = holdout[1].get("average_strategy_return")
        if train is not None and test is not None and float(train) > 0 and float(test) <= 0:
            warnings.append({"severity": "high", "warning": "Performance weakens or disappears in holdout.", "detail": "Train average return is positive but holdout is not."})

    walk = walk_forward_backtest(unique, return_field)
    if walk["summary"].get("average_trade_return") is None or (walk["summary"].get("average_trade_return") or 0) <= 0:
        warnings.append({"severity": "high", "warning": "Walk-forward validation does not currently show positive average trade return.", "detail": "Avoid trusting static backtest results."})

    def concentration(field: str, label: str) -> None:
        counts: dict[str, int] = {}
        for row in unique:
            counts[str(row.get(field) or "Unknown")] = counts.get(str(row.get(field) or "Unknown"), 0) + 1
        if unique and counts:
            winner, count = max(counts.items(), key=lambda item: item[1])
            share = count / len(unique) * 100
            if share >= 50:
                warnings.append({"severity": "medium", "warning": f"One {label} dominates the sample.", "detail": f"{winner}: {share:.1f}% of non-duplicate articles"})

    concentration("ticker", "ticker")
    concentration("event_type", "event type")
    concentration("source", "source")

    raw_rows = [row for row in unique if _is_present(row.get("return_5d"))]
    spy_rows = [row for row in unique if _is_present(row.get("excess_return_5d"))]
    sector_rows = [row for row in unique if _is_present(row.get("sector_excess_return_5d"))]
    raw_avg = _mean([float(row["return_5d"]) for row in raw_rows])
    spy_avg = _mean([float(row["excess_return_5d"]) for row in spy_rows])
    sector_avg = _mean([float(row["sector_excess_return_5d"]) for row in sector_rows])
    if raw_avg is not None and raw_avg > 0 and spy_avg is not None and spy_avg <= 0:
        warnings.append({"severity": "high", "warning": "Raw return performance disappears after SPY adjustment.", "detail": "The apparent signal may be broad market drift."})
    if spy_avg is not None and spy_avg > 0 and sector_avg is not None and sector_avg <= 0:
        warnings.append({"severity": "high", "warning": "SPY-adjusted performance disappears after sector adjustment.", "detail": "The apparent signal may be sector movement."})

    if not warnings:
        warnings.append({"severity": "info", "warning": "No major validation guardrail triggered.", "detail": "Continue monitoring sample size and out-of-sample behavior."})
    return warnings


def statistical_tests(records: list[dict[str, Any]], group_field: str, return_field: str = "return_5d") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row.get(group_field, "unknown")), []).append(row)

    try:
        from scipy import stats
    except Exception:
        stats = None

    results: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        values = _return_values(rows, return_field)
        sample_size = len(values)
        mean_return = sum(values) / sample_size if values else None
        median_return = median(values) if values else None
        std_dev = stdev(values) if sample_size >= 2 else None
        t_stat = None
        p_value = None
        ci_low = None
        ci_high = None

        if sample_size >= 2 and std_dev is not None and std_dev > 0:
            if stats is not None:
                test = stats.ttest_1samp(values, 0.0)
                t_stat = float(test.statistic)
                p_value = float(test.pvalue)
                margin = float(stats.t.ppf(0.975, sample_size - 1)) * std_dev / sqrt(sample_size)
            else:
                t_stat = mean_return / (std_dev / sqrt(sample_size)) if mean_return is not None else None
                margin = 1.96 * std_dev / sqrt(sample_size)
            if mean_return is not None:
                ci_low = mean_return - margin
                ci_high = mean_return + margin

        results.append(
            {
                group_field: group,
                "sample_size": sample_size,
                "mean_return": round(mean_return, 4) if mean_return is not None else None,
                "median_return": round(median_return, 4) if median_return is not None else None,
                "standard_deviation": round(std_dev, 4) if std_dev is not None else None,
                "t_statistic": round(t_stat, 4) if t_stat is not None else None,
                "p_value": round(p_value, 6) if p_value is not None else None,
                "confidence_interval_low": round(ci_low, 4) if ci_low is not None else None,
                "confidence_interval_high": round(ci_high, 4) if ci_high is not None else None,
                "sample_warning": (
                    "insufficient data"
                    if sample_size < 2
                    else "low sample"
                    if sample_size < MIN_RESEARCH_SAMPLE_SIZE
                    else "ok"
                ),
                "interpretation": (
                    "statistically significant at 5%"
                    if p_value is not None and p_value < 0.05
                    else "not statistically significant"
                    if p_value is not None
                    else "not enough data"
                ),
            }
        )

    if group_field == "signal_bucket":
        order = {bucket: index for index, bucket in enumerate(SIGNAL_BUCKET_ORDER)}
        results.sort(key=lambda row: order.get(row[group_field], 999))
    else:
        results.sort(key=lambda row: str(row[group_field]))
    return results


def multiple_testing_report(records: list[dict[str, Any]], group_fields: list[str], return_field: str = "return_5d") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_field in group_fields:
        for result in statistical_tests(records, group_field, return_field):
            p_value = result.get("p_value")
            rows.append(
                {
                    "test_family": group_field,
                    "group_value": result.get(group_field),
                    "sample_size": result.get("sample_size"),
                    "mean_return": result.get("mean_return"),
                    "p_value": p_value,
                    "sample_warning": result.get("sample_warning"),
                    "raw_interpretation": result.get("interpretation"),
                    "adjusted_p_value": None,
                    "significant_after_fdr_5pct": False,
                }
            )
    indexed = [(index, row) for index, row in enumerate(rows) if row["p_value"] is not None]
    if not indexed:
        return rows
    indexed.sort(key=lambda item: float(item[1]["p_value"]))
    total = len(indexed)
    adjusted: dict[int, float] = {}
    running_min = 1.0
    for rank_from_end, (index, row) in enumerate(reversed(indexed), start=1):
        rank = total - rank_from_end + 1
        value = min(1.0, float(row["p_value"]) * total / rank)
        running_min = min(running_min, value)
        adjusted[index] = running_min
    for index, value in adjusted.items():
        rows[index]["adjusted_p_value"] = round(value, 6)
        rows[index]["significant_after_fdr_5pct"] = value < 0.05
    rows.sort(key=lambda row: (row["adjusted_p_value"] is None, row["adjusted_p_value"] or 999, str(row["test_family"])))
    return rows


def event_study(records: list[dict[str, Any]], return_prefix: str = "return") -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in return_summary(records, "event_type", return_prefix):
        output.append(
            {
                "event_type": row["event_type"],
                "observations": row["observations"],
                "day_1_avg_return": row["avg_1d_return"],
                "day_5_avg_return": row["avg_5d_return"],
                "day_20_avg_return": row["avg_20d_return"],
                "sample_warning": row["sample_warning"],
            }
        )
    return output


def confusion_matrix(records: list[dict[str, Any]], return_field: str = "return_5d") -> dict[str, Any]:
    evaluated = [
        row
        for row in records
        if row.get("market_signal") in {"bullish", "bearish"} and _is_present(row.get(return_field)) and float(row[return_field]) != 0
    ]
    tp = sum(1 for row in evaluated if row["market_signal"] == "bullish" and float(row[return_field]) > 0)
    fp = sum(1 for row in evaluated if row["market_signal"] == "bullish" and float(row[return_field]) < 0)
    tn = sum(1 for row in evaluated if row["market_signal"] == "bearish" and float(row[return_field]) < 0)
    fn = sum(1 for row in evaluated if row["market_signal"] == "bearish" and float(row[return_field]) > 0)
    total = len(evaluated)

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator * 100, 2) if denominator else None

    return {
        "matrix": [
            {"predicted": "bullish", "actual_positive": tp, "actual_negative": fp},
            {"predicted": "bearish", "actual_positive": fn, "actual_negative": tn},
        ],
        "summary": {
            "evaluated_predictions": total,
            "accuracy": ratio(tp + tn, total),
            "bullish_precision": ratio(tp, tp + fp),
            "bearish_precision": ratio(tn, tn + fn),
            "sample_warning": "low sample" if total < MIN_RESEARCH_SAMPLE_SIZE else "ok",
        },
    }


def audit_columns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "id",
        "ticker",
        "published_at",
        "published_date",
        "source",
        "title",
        "market_cap_bucket",
        "sector",
        "sector_etf",
        "market_signal",
        "article_tone",
        "ticker_relevance",
        "event_type",
        "expectation_surprise",
        "contradiction_flag",
        "final_signal_score",
        "decayed_signal_score",
        "signal_strength",
        "decayed_signal_strength",
        "data_quality_score",
        "data_quality_notes",
        "positive_evidence",
        "negative_evidence",
        "market_reaction_evidence",
        "uncertainty_evidence",
        "relevance_evidence",
        "reasoning",
    ]
    return [{field: row.get(field) for field in fields} for row in records]
