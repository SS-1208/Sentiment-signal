from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from config import DATABASE_PATH
from data_quality import article_data_quality


REQUIRED_COLUMNS: dict[str, str] = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "ticker": "TEXT NOT NULL",
    "company_name": "TEXT DEFAULT ''",
    "title": "TEXT NOT NULL",
    "body": "TEXT NOT NULL",
    "body_extraction_status": "TEXT DEFAULT ''",
    "source": "TEXT DEFAULT ''",
    "url": "TEXT DEFAULT ''",
    "published_at": "TEXT DEFAULT ''",
    "published_date": "TEXT NOT NULL",
    "sentiment_direction": "TEXT DEFAULT 'neutral'",
    "sentiment_score": "REAL DEFAULT 0",
    "confidence": "REAL DEFAULT 0",
    "materiality": "REAL DEFAULT 1",
    "source_weight": "REAL DEFAULT 0.4",
    "affected_driver": "TEXT DEFAULT 'other'",
    "time_horizon": "TEXT DEFAULT 'short-term'",
    "final_impact_score": "REAL DEFAULT 0",
    "event_type": "TEXT DEFAULT 'other'",
    "expectation_surprise": "TEXT DEFAULT 'neutral'",
    "novelty_score": "REAL DEFAULT 1",
    "prediction_correct": "INTEGER",
    "signal_strength": "REAL DEFAULT 0",
    "article_tone": "TEXT DEFAULT 'neutral'",
    "market_signal": "TEXT DEFAULT 'neutral'",
    "market_signal_score": "REAL DEFAULT 0",
    "positive_evidence": "TEXT DEFAULT ''",
    "negative_evidence": "TEXT DEFAULT ''",
    "market_reaction_evidence": "TEXT DEFAULT ''",
    "uncertainty_evidence": "TEXT DEFAULT ''",
    "contradiction_flag": "INTEGER DEFAULT 0",
    "ticker_relevance": "TEXT DEFAULT 'low'",
    "relevance_evidence": "TEXT DEFAULT ''",
    "final_signal_score": "REAL DEFAULT 0",
    "decayed_signal_score": "REAL DEFAULT 0",
    "decayed_signal_strength": "REAL DEFAULT 0",
    "reasoning": "TEXT DEFAULT ''",
    "is_duplicate": "INTEGER DEFAULT 0",
    "duplicate_group": "TEXT DEFAULT ''",
    "return_1d": "REAL",
    "return_5d": "REAL",
    "return_20d": "REAL",
    "benchmark_ticker": "TEXT DEFAULT 'SPY'",
    "market_data_ticker": "TEXT DEFAULT ''",
    "benchmark_return_1d": "REAL",
    "benchmark_return_5d": "REAL",
    "benchmark_return_20d": "REAL",
    "excess_return_1d": "REAL",
    "excess_return_5d": "REAL",
    "excess_return_20d": "REAL",
    "market_cap": "REAL",
    "market_cap_bucket": "TEXT DEFAULT 'Unknown'",
    "sector": "TEXT DEFAULT 'Unknown'",
    "sector_etf": "TEXT DEFAULT 'Unknown'",
    "sector_return_1d": "REAL",
    "sector_return_5d": "REAL",
    "sector_return_20d": "REAL",
    "sector_excess_return_1d": "REAL",
    "sector_excess_return_5d": "REAL",
    "sector_excess_return_20d": "REAL",
    "data_quality_score": "REAL DEFAULT 100",
    "data_quality_notes": "TEXT DEFAULT ''",
    "created_at": "TEXT NOT NULL",
}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        columns = ", ".join(f"{name} {definition}" for name, definition in REQUIRED_COLUMNS.items())
        conn.execute(f"CREATE TABLE IF NOT EXISTS analyses ({columns})")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refresh_state (
                ticker TEXT PRIMARY KEY,
                company_name TEXT DEFAULT '',
                last_run_at TEXT,
                last_status TEXT DEFAULT '',
                last_reason TEXT DEFAULT '',
                last_saved INTEGER DEFAULT 0,
                last_weighted_coverage REAL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
        for name, definition in REQUIRED_COLUMNS.items():
            if name not in existing and name != "id":
                conn.execute(f"ALTER TABLE analyses ADD COLUMN {name} {definition}")
        _backfill_legacy_rows(conn)
        conn.commit()


def _backfill_legacy_rows(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
    if "sentiment" in columns:
        conn.execute(
            """
            UPDATE analyses
            SET sentiment_score = CASE
                    WHEN sentiment > 55 THEN 1
                    WHEN sentiment < 45 THEN -1
                    ELSE 0
                END,
                sentiment_direction = CASE
                    WHEN sentiment > 55 THEN 'positive'
                    WHEN sentiment < 45 THEN 'negative'
                    ELSE 'neutral'
                END
            WHERE sentiment_direction IS NULL OR sentiment_direction = ''
            """
        )
    if "key_reasons" in columns:
        conn.execute(
            """
            UPDATE analyses
            SET reasoning = COALESCE(NULLIF(reasoning, ''), key_reasons)
            WHERE key_reasons IS NOT NULL
            """
        )
    conn.execute("UPDATE analyses SET event_type = COALESCE(NULLIF(event_type, ''), 'other')")
    conn.execute("UPDATE analyses SET published_at = COALESCE(NULLIF(published_at, ''), published_date)")
    conn.execute("UPDATE analyses SET expectation_surprise = COALESCE(NULLIF(expectation_surprise, ''), 'neutral')")
    conn.execute("UPDATE analyses SET novelty_score = COALESCE(novelty_score, 1)")
    conn.execute("UPDATE analyses SET article_tone = COALESCE(NULLIF(article_tone, ''), sentiment_direction, 'neutral')")
    conn.execute("UPDATE analyses SET market_signal = COALESCE(NULLIF(market_signal, ''), 'neutral')")
    conn.execute("UPDATE analyses SET ticker_relevance = COALESCE(NULLIF(ticker_relevance, ''), 'low')")
    conn.execute("UPDATE analyses SET market_cap_bucket = COALESCE(NULLIF(market_cap_bucket, ''), 'Unknown')")
    conn.execute("UPDATE analyses SET sector = COALESCE(NULLIF(sector, ''), 'Unknown')")
    conn.execute("UPDATE analyses SET sector_etf = COALESCE(NULLIF(sector_etf, ''), 'Unknown')")
    conn.execute("UPDATE analyses SET data_quality_score = COALESCE(data_quality_score, 100)")
    conn.execute(
        """
        UPDATE analyses
        SET signal_strength = ABS(COALESCE(final_impact_score, 0)) * COALESCE(novelty_score, 1) * COALESCE(source_weight, 0.4)
        WHERE signal_strength IS NULL OR signal_strength = 0
        """
    )


def insert_article(article: dict[str, Any]) -> int:
    init_db()
    payload = {
        "ticker": article.get("ticker", "").upper().strip(),
        "company_name": article.get("company_name", "").strip(),
        "title": article.get("title", "").strip(),
        "body": article.get("body", "").strip(),
        "body_extraction_status": article.get("body_extraction_status", ""),
        "source": article.get("source", "").strip(),
        "url": article.get("url", "").strip(),
        "published_at": article.get("published_at") or article.get("published_date", ""),
        "published_date": article.get("published_date", ""),
        "sentiment_direction": article.get("sentiment_direction", "neutral"),
        "sentiment_score": article.get("sentiment_score", 0),
        "confidence": article.get("confidence", 0),
        "materiality": article.get("materiality", 1),
        "source_weight": article.get("source_weight", 0.4),
        "affected_driver": article.get("affected_driver", "other"),
        "time_horizon": article.get("time_horizon", "short-term"),
        "final_impact_score": article.get("final_impact_score", 0),
        "event_type": article.get("event_type", "other"),
        "expectation_surprise": article.get("expectation_surprise", "neutral"),
        "novelty_score": article.get("novelty_score", 1),
        "prediction_correct": article.get("prediction_correct"),
        "signal_strength": article.get("signal_strength", 0),
        "article_tone": article.get("article_tone", "neutral"),
        "market_signal": article.get("market_signal", "neutral"),
        "market_signal_score": article.get("market_signal_score", 0),
        "positive_evidence": article.get("positive_evidence", ""),
        "negative_evidence": article.get("negative_evidence", ""),
        "market_reaction_evidence": article.get("market_reaction_evidence", ""),
        "uncertainty_evidence": article.get("uncertainty_evidence", ""),
        "contradiction_flag": int(bool(article.get("contradiction_flag", False))),
        "ticker_relevance": article.get("ticker_relevance", "low"),
        "relevance_evidence": article.get("relevance_evidence", ""),
        "final_signal_score": article.get("final_signal_score", 0),
        "decayed_signal_score": article.get("decayed_signal_score", article.get("final_signal_score", 0)),
        "decayed_signal_strength": article.get("decayed_signal_strength", article.get("signal_strength", 0)),
        "reasoning": article.get("reasoning", ""),
        "is_duplicate": int(bool(article.get("is_duplicate", False))),
        "duplicate_group": article.get("duplicate_group", ""),
        "return_1d": article.get("return_1d"),
        "return_5d": article.get("return_5d"),
        "return_20d": article.get("return_20d"),
        "benchmark_ticker": article.get("benchmark_ticker", "SPY"),
        "market_data_ticker": article.get("market_data_ticker", ""),
        "benchmark_return_1d": article.get("benchmark_return_1d"),
        "benchmark_return_5d": article.get("benchmark_return_5d"),
        "benchmark_return_20d": article.get("benchmark_return_20d"),
        "excess_return_1d": article.get("excess_return_1d"),
        "excess_return_5d": article.get("excess_return_5d"),
        "excess_return_20d": article.get("excess_return_20d"),
        "market_cap": article.get("market_cap"),
        "market_cap_bucket": article.get("market_cap_bucket", "Unknown"),
        "sector": article.get("sector", "Unknown"),
        "sector_etf": article.get("sector_etf", "Unknown"),
        "sector_return_1d": article.get("sector_return_1d"),
        "sector_return_5d": article.get("sector_return_5d"),
        "sector_return_20d": article.get("sector_return_20d"),
        "sector_excess_return_1d": article.get("sector_excess_return_1d"),
        "sector_excess_return_5d": article.get("sector_excess_return_5d"),
        "sector_excess_return_20d": article.get("sector_excess_return_20d"),
        "data_quality_score": article.get("data_quality_score", 100),
        "data_quality_notes": article.get("data_quality_notes", ""),
        "created_at": article.get("created_at") or datetime.utcnow().isoformat(timespec="seconds"),
    }
    with connect() as conn:
        table_columns = {row["name"] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
    if "sentiment" in table_columns:
        payload["sentiment"] = float(article.get("sentiment_score", 0) or 0)
    if "key_reasons" in table_columns:
        payload["key_reasons"] = article.get("reasoning", "")
    columns = list(payload.keys())
    placeholders = ", ".join("?" for _ in columns)
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO analyses ({', '.join(columns)}) VALUES ({placeholders})",
            [payload[column] for column in columns],
        )
        article_id = int(cur.lastrowid)
        if not payload["duplicate_group"]:
            conn.execute("UPDATE analyses SET duplicate_group = ? WHERE id = ?", (f"story-{article_id}", article_id))
        conn.commit()
        return article_id


def update_returns(article_id: int, returns: dict[str, Any]) -> None:
    init_db()
    with connect() as conn:
        article = conn.execute(
            "SELECT * FROM analyses WHERE id = ?",
            (article_id,),
        ).fetchone()
        prediction_correct = None
        quality = {"data_quality_score": 100, "data_quality_notes": returns.get("data_quality_notes", "")}
        if article is not None:
            quality_record = {**dict(article), **returns}
            quality = article_data_quality(quality_record)
            benchmark_return = returns.get("excess_return_5d")
            if benchmark_return is None:
                benchmark_return = returns.get("return_5d")
            if benchmark_return is None:
                benchmark_return = returns.get("excess_return_1d")
            if benchmark_return is None:
                benchmark_return = returns.get("return_1d")
            if benchmark_return is None:
                benchmark_return = returns.get("excess_return_20d")
            if benchmark_return is None:
                benchmark_return = returns.get("return_20d")
            market_signal = article["market_signal"] if "market_signal" in article.keys() else None
            if benchmark_return is not None and market_signal in {"bullish", "bearish"}:
                prediction_correct = int((market_signal == "bullish" and benchmark_return > 0) or (market_signal == "bearish" and benchmark_return < 0))
            else:
                direction = article["sentiment_direction"]
                if benchmark_return is not None and direction in {"positive", "negative"}:
                    prediction_correct = int((direction == "positive" and benchmark_return > 0) or (direction == "negative" and benchmark_return < 0))
        conn.execute(
            """
            UPDATE analyses
            SET return_1d = ?,
                return_5d = ?,
                return_20d = ?,
                benchmark_ticker = ?,
                market_data_ticker = ?,
                benchmark_return_1d = ?,
                benchmark_return_5d = ?,
                benchmark_return_20d = ?,
                excess_return_1d = ?,
                excess_return_5d = ?,
                excess_return_20d = ?,
                market_cap = ?,
                market_cap_bucket = ?,
                sector = ?,
                sector_etf = ?,
                sector_return_1d = ?,
                sector_return_5d = ?,
                sector_return_20d = ?,
                sector_excess_return_1d = ?,
                sector_excess_return_5d = ?,
                sector_excess_return_20d = ?,
                data_quality_score = ?,
                data_quality_notes = ?,
                prediction_correct = ?
            WHERE id = ?
            """,
            (
                returns.get("return_1d"),
                returns.get("return_5d"),
                returns.get("return_20d"),
                returns.get("benchmark_ticker", "SPY"),
                returns.get("market_data_ticker", ""),
                returns.get("benchmark_return_1d"),
                returns.get("benchmark_return_5d"),
                returns.get("benchmark_return_20d"),
                returns.get("excess_return_1d"),
                returns.get("excess_return_5d"),
                returns.get("excess_return_20d"),
                returns.get("market_cap"),
                returns.get("market_cap_bucket", "Unknown"),
                returns.get("sector", "Unknown"),
                returns.get("sector_etf", "Unknown"),
                returns.get("sector_return_1d"),
                returns.get("sector_return_5d"),
                returns.get("sector_return_20d"),
                returns.get("sector_excess_return_1d"),
                returns.get("sector_excess_return_5d"),
                returns.get("sector_excess_return_20d"),
                quality.get("data_quality_score", 100),
                quality.get("data_quality_notes", returns.get("data_quality_notes", "")),
                prediction_correct,
                article_id,
            ),
        )
        conn.commit()


def update_analysis(article_id: int, analysis: dict[str, Any]) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            """
            UPDATE analyses
            SET sentiment_direction = ?,
                sentiment_score = ?,
                confidence = ?,
                materiality = ?,
                source_weight = ?,
                affected_driver = ?,
                time_horizon = ?,
                final_impact_score = ?,
                event_type = ?,
                expectation_surprise = ?,
                novelty_score = ?,
                signal_strength = ?,
                article_tone = ?,
                market_signal = ?,
                market_signal_score = ?,
                positive_evidence = ?,
                negative_evidence = ?,
                market_reaction_evidence = ?,
                uncertainty_evidence = ?,
                contradiction_flag = ?,
                ticker_relevance = ?,
                relevance_evidence = ?,
                final_signal_score = ?,
                reasoning = ?
            WHERE id = ?
            """,
            (
                analysis.get("sentiment_direction", "neutral"),
                analysis.get("sentiment_score", 0),
                analysis.get("confidence", 0),
                analysis.get("materiality", 1),
                analysis.get("source_weight", 0.4),
                analysis.get("affected_driver", "other"),
                analysis.get("time_horizon", "short-term"),
                analysis.get("final_impact_score", 0),
                analysis.get("event_type", "other"),
                analysis.get("expectation_surprise", "neutral"),
                analysis.get("novelty_score", 1),
                analysis.get("signal_strength", 0),
                analysis.get("article_tone", "neutral"),
                analysis.get("market_signal", "neutral"),
                analysis.get("market_signal_score", 0),
                analysis.get("positive_evidence", ""),
                analysis.get("negative_evidence", ""),
                analysis.get("market_reaction_evidence", ""),
                analysis.get("uncertainty_evidence", ""),
                int(bool(analysis.get("contradiction_flag", False))),
                analysis.get("ticker_relevance", "low"),
                analysis.get("relevance_evidence", ""),
                analysis.get("final_signal_score", 0),
                analysis.get("reasoning", ""),
                article_id,
            ),
        )
        conn.commit()


def fetch_articles(ticker: str | None = None, include_duplicates: bool = True) -> list[sqlite3.Row]:
    init_db()
    clauses: list[str] = []
    params: list[Any] = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper().strip())
    if not include_duplicates:
        clauses.append("COALESCE(is_duplicate, 0) = 0")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect() as conn:
        return conn.execute(
            f"""
            SELECT *
            FROM analyses
            {where}
            ORDER BY COALESCE(NULLIF(published_at, ''), published_date) DESC, created_at DESC
            """,
            params,
        ).fetchall()


def fetch_existing_for_dedupe(ticker: str) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT id, ticker, title, body, duplicate_group
            FROM analyses
            WHERE ticker = ?
            ORDER BY created_at DESC
            """,
            (ticker.upper().strip(),),
        ).fetchall()


def get_refresh_state(ticker: str) -> sqlite3.Row | None:
    init_db()
    with connect() as conn:
        return conn.execute(
            """
            SELECT *
            FROM refresh_state
            WHERE ticker = ?
            """,
            (ticker.upper().strip(),),
        ).fetchone()


def update_refresh_state(
    ticker: str,
    company_name: str,
    *,
    last_run_at: str | None,
    last_status: str,
    last_reason: str,
    last_saved: int,
    last_weighted_coverage: float,
) -> None:
    init_db()
    now = datetime.utcnow().isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO refresh_state (
                ticker,
                company_name,
                last_run_at,
                last_status,
                last_reason,
                last_saved,
                last_weighted_coverage,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name = excluded.company_name,
                last_run_at = excluded.last_run_at,
                last_status = excluded.last_status,
                last_reason = excluded.last_reason,
                last_saved = excluded.last_saved,
                last_weighted_coverage = excluded.last_weighted_coverage,
                updated_at = excluded.updated_at
            """,
            (
                ticker.upper().strip(),
                company_name.strip(),
                last_run_at,
                last_status,
                last_reason,
                int(last_saved),
                float(last_weighted_coverage),
                now,
            ),
        )
        conn.commit()
