from __future__ import annotations

from typing import Any


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "None", "nan"}


def article_data_quality(record: dict[str, Any]) -> dict[str, Any]:
    score = 100
    notes: list[str] = []

    body = str(record.get("body") or "").strip()
    title = str(record.get("title") or "").strip()
    title_lower = title.lower()
    body_status = str(record.get("body_extraction_status") or "").strip()
    if not body:
        score -= 15
        notes.append("missing body")
    elif body.lstrip().startswith("<") or "news.google.com/rss/articles" in body:
        score -= 25
        notes.append("html/rss wrapper body")
    elif body_status in {"google_news_title_only", "title_only"} or body == title:
        score -= 25
        notes.append("title-only body")
    elif body_status == "rss_summary" or len(body) < 120:
        score -= 15
        notes.append("article extraction weak or summary-only")

    if not _present(record.get("published_at")):
        score -= 10
        notes.append("missing published_at")

    if "stock price" in title_lower and "in real time" in title_lower:
        score -= 35
        notes.append("quote/watchlist page")
    elif any(
        phrase in title_lower
        for phrase in (
            "lightning round",
            "biggest moves",
            "stocks making the biggest moves",
            "analyst calls",
            "morning squawk",
            "final trades",
        )
    ):
        score -= 15
        notes.append("market chatter or roundup article")

    try:
        source_weight = float(record.get("source_weight") or 0.4)
    except (TypeError, ValueError):
        source_weight = 0.4
    if source_weight < 0.8:
        score -= 10
        notes.append("weak or unknown source")

    relevance = str(record.get("ticker_relevance") or "low")
    if relevance == "related":
        score -= 8
        notes.append("secondary ticker relevance")
    elif relevance in {"low", "irrelevant"}:
        score -= 20
        notes.append("low ticker relevance")

    if int(record.get("is_duplicate") or 0):
        score -= 20
        notes.append("duplicate story")

    try:
        confidence = float(record.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0
    if confidence < 55 or str(record.get("market_signal") or "neutral") in {"mixed", "neutral", "irrelevant"}:
        score -= 10
        notes.append("uncertain classification")

    if not any(_present(record.get(field)) for field in ("return_1d", "return_5d", "return_20d")):
        score -= 10
        notes.append("return data unavailable")

    existing_notes = str(record.get("data_quality_notes") or "").strip()
    if existing_notes:
        for note in existing_notes.split(" | "):
            if note and note not in notes:
                notes.append(note)

    return {
        "data_quality_score": max(0, min(100, score)),
        "data_quality_notes": " | ".join(notes),
    }
