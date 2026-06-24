from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from config import COMPANY_NAME_BY_TICKER, TICKER_UNIVERSES
from data_quality import article_data_quality
from database import connect, init_db
from news_collector import collect_from_rss


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


HISTORICAL_SOURCES = {
    "Reuters": "reuters.com",
    "Bloomberg": "bloomberg.com",
    "Financial Times": "ft.com",
    "Wall Street Journal": "wsj.com",
    "CNBC": "cnbc.com",
    "Yahoo Finance": "finance.yahoo.com",
    "Seeking Alpha": "seekingalpha.com",
    "GlobeNewswire": "globenewswire.com",
    "Business Wire": "businesswire.com",
    "PR Newswire": "prnewswire.com",
    "Accesswire": "accesswire.com",
    "Fierce Biotech": "fiercebiotech.com",
    "SpaceNews": "spacenews.com",
    "Defense News": "defensenews.com",
}

HISTORICAL_SOURCE_SETS = {
    "major": ["Reuters", "CNBC", "Seeking Alpha"],
    "smallcap": [
        "Reuters",
        "CNBC",
        "Yahoo Finance",
        "GlobeNewswire",
        "Business Wire",
        "PR Newswire",
        "Accesswire",
        "Seeking Alpha",
    ],
    "biotech": [
        "Reuters",
        "Yahoo Finance",
        "GlobeNewswire",
        "Business Wire",
        "PR Newswire",
        "Accesswire",
        "Fierce Biotech",
    ],
    "space_defense": [
        "Reuters",
        "CNBC",
        "Yahoo Finance",
        "GlobeNewswire",
        "Business Wire",
        "PR Newswire",
        "SpaceNews",
        "Defense News",
    ],
    "all": list(HISTORICAL_SOURCES.keys()),
}


@dataclass
class HistoricalWindow:
    index: int
    start: date
    end: date


def infer_tickers() -> list[dict[str, str]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT ticker,
                   COALESCE(NULLIF(MAX(company_name), ''), '') AS company_name,
                   COUNT(*) AS article_count
            FROM analyses
            WHERE ticker IS NOT NULL AND ticker != ''
            GROUP BY ticker
            ORDER BY article_count DESC, ticker
            """
        ).fetchall()
    return [{"ticker": row["ticker"], "company_name": COMPANY_NAME_BY_TICKER.get(row["ticker"], row["company_name"] or "")} for row in rows]


def _unique_tickers(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        ticker = str(value or "").upper().strip()
        if ticker and ticker not in seen:
            seen.add(ticker)
            output.append(ticker)
    return output


def ticker_rows_from_symbols(symbols: list[str]) -> list[dict[str, str]]:
    return [{"ticker": ticker, "company_name": COMPANY_NAME_BY_TICKER.get(ticker, "")} for ticker in _unique_tickers(symbols)]


def _unique_names(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        name = str(value or "").strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            output.append(name)
    return output


def resolve_ticker_universe(args: argparse.Namespace) -> list[dict[str, str]]:
    requested_symbols: list[str] = []
    if args.universe:
        requested_symbols.extend(TICKER_UNIVERSES[args.universe])
    if args.tickers:
        requested_symbols.extend(args.tickers)
    if requested_symbols:
        return ticker_rows_from_symbols(requested_symbols)
    return infer_tickers()


def resolve_historical_sources(args: argparse.Namespace) -> list[str]:
    if args.source_set:
        sources = list(HISTORICAL_SOURCE_SETS[args.source_set])
        if args.sources:
            sources.extend(args.sources)
        return _unique_names(sources)
    if args.sources:
        return _unique_names(args.sources)
    return list(HISTORICAL_SOURCE_SETS["major"])


def build_windows(months_back: int, windows: int, window_days: int, min_forward_days: int) -> list[HistoricalWindow]:
    latest_end = date.today() - timedelta(days=min_forward_days)
    earliest_start = latest_end - timedelta(days=months_back * 30)
    if windows <= 1:
        return [HistoricalWindow(1, earliest_start, earliest_start + timedelta(days=window_days))]
    span_days = max(1, (latest_end - earliest_start).days - window_days)
    step = max(1, span_days // (windows - 1))
    output: list[HistoricalWindow] = []
    for index in range(windows):
        start = earliest_start + timedelta(days=index * step)
        end = min(start + timedelta(days=window_days), latest_end)
        output.append(HistoricalWindow(index + 1, start, end))
    return output


def historical_google_news_url(ticker: str, company_name: str, source: str, start: date, end: date) -> str:
    site = HISTORICAL_SOURCES[source]
    terms = [f'"{company_name}"'] if company_name else []
    terms.extend([f'"${ticker}"', f'"{ticker} stock"'])
    query_terms = " OR ".join(terms)
    query = f"site:{site} ({query_terms}) after:{start.isoformat()} before:{end.isoformat()}"
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"


def _directional_match(score: Any, return_value: Any, threshold: float, return_threshold: float = 0.0) -> bool | None:
    if return_value is None:
        return None
    try:
        signal = float(score or 0)
        realized = float(return_value)
    except (TypeError, ValueError):
        return None
    if abs(realized) < abs(float(return_threshold or 0.0)):
        return None
    if signal >= threshold:
        return realized > return_threshold
    if signal <= -threshold:
        return realized < -return_threshold
    return None


def _issue_notes(result: dict[str, Any], horizon: str, threshold: float, return_threshold: float = 0.0) -> str:
    notes: list[str] = []
    if result.get("status") != "saved":
        return str(result.get("message") or result.get("status") or "not saved")
    if int(result.get("is_duplicate") or 0):
        notes.append("duplicate excluded from independent evidence")
    if abs(float(result.get("final_signal_score") or 0)) < threshold:
        notes.append("signal below directional threshold")
    if result.get(f"return_{horizon}") is None:
        notes.append(f"return_{horizon} unavailable")
    if result.get(f"excess_return_{horizon}") is None:
        notes.append(f"excess_return_{horizon} unavailable")
    quality = article_data_quality(result)
    if quality["data_quality_score"] < 70:
        notes.append(f"low data quality: {quality['data_quality_notes']}")
    raw_match = _directional_match(result.get("final_signal_score"), result.get(f"return_{horizon}"), threshold, return_threshold)
    excess_match = _directional_match(result.get("final_signal_score"), result.get(f"excess_return_{horizon}"), threshold, return_threshold)
    if raw_match is False:
        notes.append(f"raw {horizon} return contradicted signal thesis")
    if excess_match is False:
        notes.append(f"excess {horizon} return contradicted signal thesis")
    return " | ".join(dict.fromkeys(note for note in notes if note))


def _article_record(run_id: str, window: HistoricalWindow, ticker: str, source: str, result: dict[str, Any], horizon: str, threshold: float, return_threshold: float = 0.0) -> dict[str, Any]:
    quality = article_data_quality(result)
    return {
        "run_id": run_id,
        "window_index": window.index,
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
        "ticker": ticker,
        "source_query": source,
        "status": result.get("status"),
        "article_id": result.get("id"),
        "published_at": result.get("published_at"),
        "title": result.get("title"),
        "source": result.get("source"),
        "body_extraction_status": result.get("body_extraction_status"),
        "market_signal": result.get("market_signal"),
        "event_type": result.get("event_type"),
        "expectation_surprise": result.get("expectation_surprise"),
        "final_signal_score": result.get("final_signal_score"),
        "signal_strength": result.get("signal_strength"),
        f"return_{horizon}": result.get(f"return_{horizon}"),
        f"excess_return_{horizon}": result.get(f"excess_return_{horizon}"),
        "raw_thesis_match": _directional_match(result.get("final_signal_score"), result.get(f"return_{horizon}"), threshold, return_threshold),
        "excess_thesis_match": _directional_match(result.get("final_signal_score"), result.get(f"excess_return_{horizon}"), threshold, return_threshold),
        "is_duplicate": int(result.get("is_duplicate") or 0),
        "data_quality_score": quality["data_quality_score"],
        "data_quality_notes": quality["data_quality_notes"],
        "what_went_wrong": _issue_notes(result, horizon, threshold, return_threshold),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_pct_from_matches(values: list[bool]) -> str:
    if not values:
        return "N/A"
    return f"{sum(1 for value in values if value) / len(values) * 100:.2f}%"


def write_markdown_report(
    path: Path,
    run_id: str,
    args: argparse.Namespace,
    tickers: list[dict[str, str]],
    sources: list[str],
    article_rows: list[dict[str, Any]],
    source_error_counts: dict[str, int],
) -> None:
    saved = [row for row in article_rows if row.get("status") == "saved"]
    independent = [row for row in saved if not int(row.get("is_duplicate") or 0)]
    raw_matches = [row["raw_thesis_match"] for row in independent if row.get("raw_thesis_match") is not None]
    excess_matches = [row["excess_thesis_match"] for row in independent if row.get("excess_thesis_match") is not None]
    status_counts = Counter(str(row.get("status") or "unknown") for row in article_rows)
    source_status_counts = Counter((str(row.get("source_query") or "unknown"), str(row.get("status") or "unknown")) for row in article_rows)
    issue_counts = Counter(str(row.get("what_went_wrong") or "no issue") for row in article_rows if row.get("what_went_wrong"))
    enough_raw = len(raw_matches) >= args.min_directional_observations
    enough_excess = len(excess_matches) >= args.min_directional_observations
    verdict = "conclusive enough for review" if enough_raw or enough_excess else "inconclusive: insufficient directional observations"

    lines = [
        f"# Historical Backtest Report: {run_id}",
        "",
        f"- Verdict: **{verdict}**",
        f"- Universe: {', '.join(row['ticker'] for row in tickers)}",
        f"- Sources: {', '.join(sources)}",
        f"- Windows: {args.windows} x {args.window_days} days over {args.months_back} months",
        f"- Horizon: {args.horizon}",
        f"- Entry lag: {args.entry_lag_days} trading day(s)",
        f"- Directional return noise band: +/-{args.return_threshold_pct}%",
        f"- Minimum directional observations for interpretation: {args.min_directional_observations}",
        "",
        "## Summary",
        "",
        f"- Feed rows recorded: {len(article_rows)}",
        f"- Saved articles: {len(saved)}",
        f"- Independent saved articles: {len(independent)}",
        f"- Raw directional observations: {len(raw_matches)}",
        f"- Raw thesis match rate: {_format_pct_from_matches(raw_matches)}",
        f"- Excess directional observations: {len(excess_matches)}",
        f"- Excess thesis match rate: {_format_pct_from_matches(excess_matches)}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Source Reliability", ""])
    for source in sources:
        saved_count = source_status_counts.get((source, "saved"), 0)
        errors = source_error_counts.get(source, 0)
        skipped = source_status_counts.get((source, "skipped"), 0)
        empty = source_status_counts.get((source, "empty"), 0)
        lines.append(f"- {source}: saved={saved_count}, errors={errors}, skipped={skipped}, empty={empty}")

    lines.extend(["", "## Top Issues", ""])
    for issue, count in issue_counts.most_common(12):
        lines.append(f"- {count}: {issue}")

    lines.extend(["", "## Saved Articles", ""])
    if saved:
        for row in saved[:25]:
            lines.append(
                f"- {row.get('ticker')} | {row.get('source_query')} | score={row.get('final_signal_score')} | "
                f"return_{args.horizon}={row.get(f'return_{args.horizon}')} | {row.get('title')}"
            )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Do not treat this run as evidence of predictive power unless it has enough independent directional observations. "
            "Low sample size, failed feeds, title-only bodies, and missing return data are data-quality failures, not model results.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_windows(article_rows: list[dict[str, Any]], horizon: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in article_rows:
        grouped[(int(row["window_index"]), str(row["ticker"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (window_index, ticker), rows in sorted(grouped.items()):
        saved = [row for row in rows if row.get("status") == "saved"]
        independent = [row for row in saved if not int(row.get("is_duplicate") or 0)]
        raw_matches = [row["raw_thesis_match"] for row in independent if row.get("raw_thesis_match") is not None]
        excess_matches = [row["excess_thesis_match"] for row in independent if row.get("excess_thesis_match") is not None]
        raw_returns = [float(row[f"return_{horizon}"]) for row in independent if row.get(f"return_{horizon}") not in {None, ""}]
        excess_returns = [float(row[f"excess_return_{horizon}"]) for row in independent if row.get(f"excess_return_{horizon}") not in {None, ""}]
        issues = [row["what_went_wrong"] for row in rows if row.get("what_went_wrong")]
        summaries.append(
            {
                "window_index": window_index,
                "ticker": ticker,
                "window_start": rows[0]["window_start"],
                "window_end": rows[0]["window_end"],
                "feed_queries": len({row["source_query"] for row in rows}),
                "rss_rows_recorded": len(rows),
                "queries_attempted": len(rows),
                "articles_saved": len(saved),
                "independent_articles": len(independent),
                "directional_raw_observations": len(raw_matches),
                "raw_match_rate": round(sum(1 for value in raw_matches if value) / len(raw_matches) * 100, 2) if raw_matches else None,
                "directional_excess_observations": len(excess_matches),
                "excess_match_rate": round(sum(1 for value in excess_matches if value) / len(excess_matches) * 100, 2) if excess_matches else None,
                f"average_return_{horizon}": round(sum(raw_returns) / len(raw_returns), 4) if raw_returns else None,
                f"average_excess_return_{horizon}": round(sum(excess_returns) / len(excess_returns), 4) if excess_returns else None,
                "what_went_wrong": " || ".join(issues[:8]),
            }
        )
    return summaries


def run_historical_backtest(args: argparse.Namespace) -> dict[str, Path]:
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tickers = resolve_ticker_universe(args)
    windows = build_windows(args.months_back, args.windows, args.window_days, args.min_forward_days)
    sources = resolve_historical_sources(args)
    bad_sources = [source for source in sources if source not in HISTORICAL_SOURCES]
    if bad_sources:
        raise SystemExit(f"Unknown historical sources: {', '.join(bad_sources)}")
    if not tickers:
        raise SystemExit("No tickers found in the database.")

    logger.info("run_id=%s tickers=%s windows=%s sources=%s", run_id, [row["ticker"] for row in tickers], len(windows), sources)
    article_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    dedupe_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_error_counts: dict[str, int] = defaultdict(int)
    for window in windows:
        for ticker_info in tickers:
            ticker = ticker_info["ticker"].upper()
            company_name = ticker_info.get("company_name", "")
            for source in sources:
                if args.max_source_errors and source_error_counts[source] >= args.max_source_errors:
                    results = [{"status": "skipped", "message": f"Skipped because {source} reached source failure cap ({args.max_source_errors})", "source": source}]
                else:
                    url = historical_google_news_url(ticker, company_name, source, window.start, window.end)
                    try:
                        results = collect_from_rss(
                            ticker=ticker,
                            company_name=company_name,
                            rss_url=url,
                            limit=args.per_source_limit,
                            source_override=source,
                            feed_name=f"Historical {source}",
                            ticker_filter=True,
                            published_from=window.start.isoformat(),
                            published_to=window.end.isoformat(),
                            max_entries_to_scan=args.max_entries_per_query,
                            include_sector_returns=args.include_sector_returns,
                            feed_timeout_seconds=args.feed_timeout_seconds,
                            dedupe_candidates=dedupe_by_ticker[ticker] if args.dedupe_scope == "run" else None,
                            entry_lag_days=args.entry_lag_days,
                        )
                    except Exception as exc:
                        results = [{"status": "error", "message": f"collector crashed: {exc}", "source": source}]
                if not results:
                    results = [{"status": "empty", "message": "collector returned no rows", "source": source}]
                if any(result.get("status") == "error" for result in results):
                    source_error_counts[source] += 1
                    if args.max_source_errors and source_error_counts[source] == args.max_source_errors:
                        logger.info("source failure cap reached for %s after %s errors", source, args.max_source_errors)
                for result in results:
                    record = _article_record(run_id, window, ticker, source, result, args.horizon, args.signal_threshold, args.return_threshold_pct)
                    article_rows.append(record)
                    if record["what_went_wrong"]:
                        issue_rows.append(record)

    window_rows = summarize_windows(article_rows, args.horizon)
    output_dir = Path(args.output_dir)
    paths = {
        "articles": output_dir / f"historical_backtest_articles_{run_id}.csv",
        "windows": output_dir / f"historical_backtest_windows_{run_id}.csv",
        "issues": output_dir / f"historical_backtest_issues_{run_id}.csv",
        "report": output_dir / f"historical_backtest_report_{run_id}.md",
    }
    _write_csv(paths["articles"], article_rows)
    _write_csv(paths["windows"], window_rows)
    _write_csv(paths["issues"], issue_rows)
    write_markdown_report(paths["report"], run_id, args, tickers, sources, article_rows, source_error_counts)

    saved = sum(1 for row in article_rows if row.get("status") == "saved")
    raw_matches = [row["raw_thesis_match"] for row in article_rows if row.get("raw_thesis_match") is not None and not int(row.get("is_duplicate") or 0)]
    excess_matches = [row["excess_thesis_match"] for row in article_rows if row.get("excess_thesis_match") is not None and not int(row.get("is_duplicate") or 0)]
    logger.info("saved_articles=%s raw_obs=%s raw_match_rate=%s excess_obs=%s excess_match_rate=%s issues=%s", saved, len(raw_matches), _rate(raw_matches), len(excess_matches), _rate(excess_matches), len(issue_rows))
    logger.info("articles=%s", paths["articles"])
    logger.info("windows=%s", paths["windows"])
    logger.info("issues=%s", paths["issues"])
    logger.info("report=%s", paths["report"])
    if not args.keep_inserted:
        deleted = cleanup_inserted_articles(article_rows)
        logger.info("cleanup_deleted_inserted_articles=%s", deleted)
    return paths


def cleanup_inserted_articles(article_rows: list[dict[str, Any]]) -> int:
    ids = [int(row["article_id"]) for row in article_rows if row.get("status") == "saved" and row.get("article_id")]
    if not ids:
        return 0
    placeholders = ", ".join("?" for _ in ids)
    init_db()
    with connect() as conn:
        conn.execute(f"DELETE FROM analyses WHERE id IN ({placeholders})", ids)
        deleted = conn.total_changes
        conn.commit()
    return int(deleted)


def _rate(values: list[bool]) -> str:
    if not values:
        return "N/A"
    return f"{sum(1 for value in values if value) / len(values) * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical SentimentSignal article collection and thesis-match backtest.")
    parser.add_argument("--months-back", type=int, default=6, help="Historical span to sample, ending before the forward-return buffer.")
    parser.add_argument("--windows", type=int, default=20, help="Number of historical windows to test.")
    parser.add_argument("--window-days", type=int, default=7, help="Days in each historical article-search window.")
    parser.add_argument("--min-forward-days", type=int, default=35, help="Buffer before today so 20 trading day returns are more likely available.")
    parser.add_argument("--per-source-limit", type=int, default=1, help="Maximum saved articles per source per ticker/window.")
    parser.add_argument("--max-entries-per-query", type=int, default=25, help="Maximum RSS entries to scan for each ticker/window/source query.")
    parser.add_argument("--feed-timeout-seconds", type=int, default=10, help="Socket timeout for each RSS request.")
    parser.add_argument("--max-source-errors", type=int, default=8, help="Skip a source for the rest of the run after this many feed errors. Use 0 to disable.")
    parser.add_argument("--include-sector-returns", action="store_true", help="Also fetch yfinance profile and sector ETF returns. Slower and more rate-limit prone.")
    parser.add_argument("--sources", nargs="*", default=None, help=f"Historical sources. Options: {', '.join(HISTORICAL_SOURCES)}")
    parser.add_argument("--source-set", choices=sorted(HISTORICAL_SOURCE_SETS), default=None, help="Preset source mix. Use smallcap for PR wires and niche sources.")
    parser.add_argument("--universe", choices=sorted(TICKER_UNIVERSES), default=None, help="Named ticker universe to test.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker list. Works even if tickers are not already in SQLite.")
    parser.add_argument("--horizon", choices=["1d", "5d", "20d"], default="20d", help="Return horizon used for thesis-match reporting.")
    parser.add_argument("--signal-threshold", type=float, default=1.0, help="Minimum absolute final_signal_score treated as a directional thesis.")
    parser.add_argument("--return-threshold-pct", type=float, default=0.25, help="Ignore realized returns smaller than this absolute percentage when judging thesis direction.")
    parser.add_argument("--min-directional-observations", type=int, default=30, help="Minimum non-duplicate directional observations needed before a run is treated as interpretable.")
    parser.add_argument("--entry-lag-days", type=int, default=1, help="Trading-day close offset used as the entry base for historical validation.")
    parser.add_argument("--dedupe-scope", choices=["run", "database"], default="run", help="Run-scoped dedupe avoids historical tests being affected by old app rows.")
    parser.add_argument("--keep-inserted", action="store_true", help="Keep articles inserted during the historical run in the main SQLite database.")
    parser.add_argument("--output-dir", default="backtest_runs", help="Directory for CSV reports.")
    return parser.parse_args()


if __name__ == "__main__":
    run_historical_backtest(parse_args())
