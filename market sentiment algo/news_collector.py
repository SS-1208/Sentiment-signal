from __future__ import annotations

import html
import logging
import re
import socket
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

from classifier import analyze_article, calculate_signal_strength
from config import NEWS_FEED_PRESETS, RELEVANCE_WEIGHTS, SURPRISE_MULTIPLIERS
from database import fetch_existing_for_dedupe, insert_article, update_returns
from deduplication import find_duplicate
from market_data import calculate_future_returns


logger = logging.getLogger(__name__)


def _entry_date(entry: Any) -> str:
    timestamp = _entry_timestamp(entry)
    if timestamp:
        return timestamp[:10]
    return date.today().isoformat()


def _entry_timestamp(entry: Any) -> str:
    raw = entry.get("published") or entry.get("updated") or ""
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).isoformat(timespec="seconds")
    except Exception:
        return ""


def _within_date_window(published_date: str, published_from: str | None, published_to: str | None) -> bool:
    try:
        article_date = datetime.strptime(published_date[:10], "%Y-%m-%d").date()
    except Exception:
        return True
    if published_from:
        try:
            if article_date < datetime.strptime(published_from[:10], "%Y-%m-%d").date():
                return False
        except Exception:
            pass
    if published_to:
        try:
            if article_date >= datetime.strptime(published_to[:10], "%Y-%m-%d").date():
                return False
        except Exception:
            pass
    return True


def _is_google_news_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host.endswith("news.google.com")


def _download_article(url: str) -> str:
    if not url:
        return ""
    if _is_google_news_url(url):
        return ""
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        return trafilatura.extract(downloaded) or ""
    except Exception as exc:
        logger.info("Article extraction failed for %s: %s", url, exc)
        return ""


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _is_non_article_entry(title: str, source: str = "") -> bool:
    text = f"{title} {source}".lower()
    if "stock price" in text and "in real time" in text:
        return True
    if "real-time quote" in text or "real time quote" in text:
        return True
    if "watchlist" in text and "stock price" in text:
        return True
    return False


def _body_from_entry(url: str, summary: str, title: str) -> tuple[str, str]:
    if _is_google_news_url(url):
        return title, "google_news_title_only"

    downloaded = _download_article(url)
    if downloaded:
        return downloaded, "full_text"

    clean_summary = _strip_html(summary)
    if clean_summary and clean_summary.lower() != title.lower():
        return clean_summary, "rss_summary"
    return title, "title_only"


def _recalculate_analysis_scores(analysis: dict[str, Any]) -> None:
    try:
        sentiment_score = float(analysis.get("sentiment_score") or 0)
        materiality = int(analysis.get("materiality") or 1)
        confidence = float(analysis.get("confidence") or 0)
        source_weight = float(analysis.get("source_weight") or 0.4)
        market_signal_score = float(analysis.get("market_signal_score") or 0)
        novelty_score = int(analysis.get("novelty_score") or 1)
    except (TypeError, ValueError):
        return
    surprise = str(analysis.get("expectation_surprise") or "neutral")
    relevance = str(analysis.get("ticker_relevance") or "low")
    surprise_multiplier = SURPRISE_MULTIPLIERS.get(surprise, 1.0)
    relevance_weight = RELEVANCE_WEIGHTS.get(relevance, 0.0)
    final_impact_score = sentiment_score * materiality * (confidence / 100) * source_weight
    final_signal_score = market_signal_score * materiality * (confidence / 100) * source_weight * novelty_score * surprise_multiplier * relevance_weight
    analysis["final_impact_score"] = round(max(-10.0, min(10.0, final_impact_score)), 4)
    analysis["final_signal_score"] = round(max(-10.0, min(10.0, final_signal_score)), 4)
    analysis["signal_strength"] = calculate_signal_strength(
        analysis["final_signal_score"],
        novelty_score,
        source_weight,
        surprise,
    )


def _cap_final_signal_score(analysis: dict[str, Any], cap: float) -> None:
    try:
        final_signal_score = float(analysis.get("final_signal_score") or 0)
        novelty_score = int(analysis.get("novelty_score") or 1)
        source_weight = float(analysis.get("source_weight") or 0.4)
    except (TypeError, ValueError):
        return
    if abs(final_signal_score) <= cap:
        return
    analysis["final_signal_score"] = round(cap if final_signal_score > 0 else -cap, 4)
    analysis["signal_strength"] = calculate_signal_strength(
        analysis["final_signal_score"],
        novelty_score,
        source_weight,
        str(analysis.get("expectation_surprise") or "neutral"),
    )


def _apply_body_extraction_adjustment(analysis: dict[str, Any], body_status: str) -> dict[str, Any]:
    adjusted = dict(analysis)
    if body_status in {"google_news_title_only", "title_only"}:
        adjusted["confidence"] = round(max(25.0, float(adjusted.get("confidence") or 0) * 0.72), 1)
        adjusted["materiality"] = min(int(adjusted.get("materiality") or 1), 6)
        adjusted["novelty_score"] = min(int(adjusted.get("novelty_score") or 1), 5)
        adjusted["reasoning"] = f"{adjusted.get('reasoning') or ''} | title-only evidence capped confidence, materiality, and novelty".strip(" |")
        _recalculate_analysis_scores(adjusted)
        _cap_final_signal_score(adjusted, 6.0)
    elif body_status == "rss_summary":
        adjusted["confidence"] = round(max(25.0, float(adjusted.get("confidence") or 0) * 0.88), 1)
        adjusted["novelty_score"] = min(int(adjusted.get("novelty_score") or 1), 7)
        adjusted["reasoning"] = f"{adjusted.get('reasoning') or ''} | summary-only evidence reduced confidence".strip(" |")
        _recalculate_analysis_scores(adjusted)
        _cap_final_signal_score(adjusted, 8.0)
    return adjusted


def _matches_company(title: str, body: str, ticker: str, company_name: str) -> bool:
    haystack = f"{title} {body[:1000]}".lower()
    ticker_hit = ticker.lower() in haystack
    company_hit = bool(company_name and company_name.lower() in haystack)
    return ticker_hit or company_hit


def _feed_url(template: str, ticker: str, company_name: str) -> str:
    company_query = company_name.replace(" ", "%20") if company_name else ticker
    return template.format(ticker=ticker, company=company_query)


def collect_from_rss(
    ticker: str,
    company_name: str,
    rss_url: str,
    limit: int = 10,
    source_override: str | None = None,
    feed_name: str | None = None,
    ticker_filter: bool = False,
    published_from: str | None = None,
    published_to: str | None = None,
    max_entries_to_scan: int | None = None,
    fetch_returns: bool = True,
    include_sector_returns: bool = True,
    feed_timeout_seconds: int | None = 15,
    dedupe_candidates: list[Any] | None = None,
    entry_lag_days: int = 0,
) -> list[dict[str, Any]]:
    try:
        import feedparser
    except Exception as exc:
        logger.info("feedparser unavailable; RSS collection skipped: %s", exc)
        return [{"status": "error", "message": "feedparser is unavailable"}]

    old_timeout = socket.getdefaulttimeout()
    try:
        if feed_timeout_seconds:
            socket.setdefaulttimeout(feed_timeout_seconds)
        try:
            feed = feedparser.parse(rss_url)
        except Exception as exc:
            logger.info("RSS feed failed for %s: %s", rss_url, exc)
            return [{"status": "error", "message": f"RSS feed failed: {exc}"}]
    finally:
        socket.setdefaulttimeout(old_timeout)

    if getattr(feed, "bozo", False):
        logger.info("RSS parser warning for %s: %s", rss_url, getattr(feed, "bozo_exception", "unknown"))

    results: list[dict[str, Any]] = []
    entries = list(getattr(feed, "entries", []))
    scanned = 0
    for entry in entries:
        scanned += 1
        if max_entries_to_scan is not None and scanned > max_entries_to_scan:
            break
        if len([result for result in results if result.get("status") == "saved"]) >= limit:
            break
        title = entry.get("title", "").strip()
        url = entry.get("link", "").strip()
        summary = entry.get("summary", "").strip()
        published_at = _entry_timestamp(entry)
        published_date = _entry_date(entry)
        source = source_override or entry.get("source", {}).get("title", "") or feed.feed.get("title", "RSS")

        if not title:
            results.append({"status": "skipped", "message": "Entry skipped because it had no title"})
            continue

        if _is_non_article_entry(title, source):
            results.append({"status": "skipped", "source": feed_name or source, "title": title, "published_date": published_date, "message": "Skipped because entry is a quote/watchlist page"})
            continue

        if not _within_date_window(published_date, published_from, published_to):
            results.append({"status": "skipped", "source": feed_name or source, "title": title, "published_date": published_date, "message": "Skipped because article date was outside requested window"})
            continue

        body, body_status = _body_from_entry(url, summary, title)

        if ticker_filter and not _matches_company(title, body, ticker, company_name):
            results.append({"status": "skipped", "source": feed_name or source, "title": title, "message": "Skipped because ticker/company was not detected"})
            continue

        analysis = analyze_article(ticker, title, body, source, company_name)
        analysis = _apply_body_extraction_adjustment(analysis, body_status)
        candidates = dedupe_candidates if dedupe_candidates is not None else fetch_existing_for_dedupe(ticker)
        duplicate = find_duplicate(title, body, candidates)
        if duplicate.get("is_duplicate"):
            old_novelty = max(float(analysis.get("novelty_score") or 1), 1.0)
            analysis["novelty_score"] = min(int(analysis.get("novelty_score") or 1), 2)
            analysis["final_signal_score"] = round(float(analysis.get("final_signal_score") or 0) * (analysis["novelty_score"] / old_novelty), 4)
            analysis["signal_strength"] = abs(analysis["final_signal_score"])
        article = {
            "ticker": ticker,
            "company_name": company_name,
            "title": title,
            "body": body,
            "body_extraction_status": body_status,
            "source": source,
            "url": url,
            "published_at": published_at or published_date,
            "published_date": published_date,
            **analysis,
            **duplicate,
        }
        article_id = insert_article(article)
        returns = (
            calculate_future_returns(ticker, published_date, include_sector=include_sector_returns, entry_lag_days=entry_lag_days)
            if fetch_returns
            else {}
        )
        update_returns(article_id, returns)
        if dedupe_candidates is not None:
            dedupe_candidates.append(
                {
                    "id": article_id,
                    "ticker": ticker,
                    "title": title,
                    "body": body,
                    "duplicate_group": article.get("duplicate_group") or duplicate.get("duplicate_group") or f"story-{article_id}",
                }
            )
        results.append(
            {
                "status": "saved",
                "feed": feed_name or source,
                "source": source,
                "id": article_id,
                "title": title,
                "body": body,
                "published_at": published_at or published_date,
                "body_extraction_status": body_status,
                **analysis,
                **duplicate,
                **returns,
            }
        )

    if not results:
        results.append({"status": "empty", "message": "No RSS entries found"})
    return results


def available_feed_names() -> list[str]:
    return [feed["name"] for feed in NEWS_FEED_PRESETS]


def collect_from_presets(ticker: str, company_name: str, selected_feeds: list[str] | None = None, per_feed_limit: int = 5) -> list[dict[str, Any]]:
    selected = set(selected_feeds or available_feed_names())
    results: list[dict[str, Any]] = []
    for preset in NEWS_FEED_PRESETS:
        if preset["name"] not in selected:
            continue
        url = _feed_url(preset["url_template"], ticker, company_name)
        feed_results = collect_from_rss(
            ticker=ticker,
            company_name=company_name,
            rss_url=url,
            limit=per_feed_limit,
            source_override=preset["source"],
            feed_name=preset["name"],
            ticker_filter=not preset["ticker_specific"],
        )
        for result in feed_results:
            result.setdefault("feed", preset["name"])
            result.setdefault("source", preset["source"])
        results.extend(feed_results)
    return results or [{"status": "empty", "message": "No feeds selected or no entries found"}]
