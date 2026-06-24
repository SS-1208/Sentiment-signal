from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import classifier
import dashboard
import data_quality
import database
import market_data
import news_collector
import refresh_policy
import historical_backtest


class ClassifierTests(unittest.TestCase):
    def analyze(self, ticker: str, title: str, body: str = "") -> dict:
        with patch.object(classifier, "_finbert_sentiment", return_value=None):
            return classifier.analyze_article(ticker, title, body or title, "Reuters", "")

    def test_guidance_beat_is_bullish_and_capped(self) -> None:
        result = self.analyze("AAPL", "Apple raises guidance after earnings beat")
        self.assertEqual(result["event_type"], "guidance")
        self.assertEqual(result["market_signal"], "bullish")
        self.assertEqual(result["final_signal_score"], 10.0)
        self.assertGreater(result["signal_strength"], result["final_signal_score"])

    def test_investigation_is_bearish(self) -> None:
        result = self.analyze("TSLA", "Tesla faces SEC investigation after fraud allegations")
        self.assertEqual(result["event_type"], "investigation")
        self.assertEqual(result["market_signal"], "bearish")
        self.assertEqual(result["final_signal_score"], -10.0)

    def test_generic_product_launch_is_not_automatically_bullish(self) -> None:
        result = self.analyze("MSFT", "Microsoft announces new product update")
        self.assertEqual(result["event_type"], "product_launch")
        self.assertEqual(result["market_signal"], "neutral")
        self.assertEqual(result["final_signal_score"], 0.0)

    def test_wrongful_gains_is_not_bullish(self) -> None:
        result = self.analyze("MSFT", "Musk seeks up to $134 billion from OpenAI and Microsoft in wrongful gains")
        self.assertNotEqual(result["market_signal"], "bullish")

    def test_competitor_stock_drop_not_scored_against_target(self) -> None:
        result = self.analyze("GOOGLE", "Figma's stock drops 12% in two days after Google releases design product", "Figma's stock drops after Google releases a product.")
        self.assertNotEqual(result["market_signal"], "bearish")

    def test_favorable_legal_context_is_not_bearish(self) -> None:
        result = self.analyze("GOOGLE", "Google fends off news publishers' antitrust lawsuit over online search")
        self.assertNotEqual(result["market_signal"], "bearish")

    def test_smallcap_launch_date_is_operational_signal(self) -> None:
        result = self.analyze("ASTS", "AST SpaceMobile announces BlueBird 6 launch date")
        self.assertEqual(result["event_type"], "product_launch")
        self.assertEqual(result["market_signal"], "bullish")

    def test_contract_award_language_is_bullish(self) -> None:
        result = self.analyze("RKLB", "Rocket Lab selected by JAXA after contract award")
        self.assertEqual(result["event_type"], "contract_win")
        self.assertEqual(result["market_signal"], "bullish")

    def test_retrospective_price_surge_is_not_bullish_news(self) -> None:
        result = self.analyze("ASTS", "AST SpaceMobile: Reassessing valuation after a powerful 1-year share price surge")
        self.assertNotEqual(result["market_signal"], "bullish")


class DashboardValidationTests(unittest.TestCase):
    def make_records(self, count: int = 45) -> list[dict]:
        records = []
        for index in range(count):
            positive = index % 2 == 0
            score = 8.0 if positive else -8.0
            realized = 2.0 if positive else -2.0
            records.append(
                {
                    "id": index + 1,
                    "ticker": "TST",
                    "title": f"Story {index}",
                    "published_at": f"2024-01-{(index % 28) + 1:02d}T09:30:00",
                    "published_date": f"2024-01-{(index % 28) + 1:02d}",
                    "final_signal_score": score,
                    "final_impact_score": score,
                    "signal_strength": abs(score),
                    "market_signal": "bullish" if positive else "bearish",
                    "event_type": "guidance",
                    "expectation_surprise": "positive" if positive else "negative",
                    "materiality": 8,
                    "source_weight": 1.5,
                    "source": "Reuters",
                    "is_duplicate": 0,
                    "return_1d": realized / 2,
                    "return_5d": realized,
                    "return_20d": realized * 2,
                    "excess_return_1d": realized / 3,
                    "excess_return_5d": realized,
                    "excess_return_20d": realized * 1.5,
                    "sector_excess_return_1d": realized / 4,
                    "sector_excess_return_5d": realized / 2,
                    "sector_excess_return_20d": realized,
                    "market_cap_bucket": "Large-cap" if index % 3 else "Small-cap",
                    "sector": "Technology",
                    "sector_etf": "XLK",
                    "article_tone": "positive" if positive else "negative",
                    "ticker_relevance": "direct",
                    "data_quality_score": 95,
                }
            )
        return records

    def test_holdout_validation_has_train_and_holdout_rows(self) -> None:
        rows = dashboard.holdout_validation(self.make_records(), "return_5d")
        self.assertEqual([row["split"] for row in rows], ["train", "holdout"])
        self.assertGreater(rows[1]["evaluated_signals"], 0)
        self.assertEqual(rows[1]["accuracy"], 100.0)

    def test_walk_forward_uses_prior_training_window(self) -> None:
        result = dashboard.walk_forward_backtest(self.make_records(), "return_5d", min_train_observations=10)
        self.assertGreater(result["summary"]["walk_forward_trades"], 0)
        self.assertGreater(result["summary"]["average_trade_return"], 0)
        first_trade = result["trades"][0]
        self.assertGreaterEqual(first_trade["training_observations_available"], 10)

    def test_multiple_testing_report_adds_adjusted_p_values(self) -> None:
        rows = dashboard.multiple_testing_report(self.make_records(), ["market_signal", "event_type"], "return_5d")
        self.assertTrue(rows)
        self.assertIn("adjusted_p_value", rows[0])
        self.assertIn("significant_after_fdr_5pct", rows[0])

    def test_time_decay_factor(self) -> None:
        now = datetime(2026, 6, 23, 12, 0, 0)
        self.assertEqual(dashboard.time_decay_factor(now - timedelta(hours=3), now), 1.0)
        self.assertEqual(dashboard.time_decay_factor(now - timedelta(hours=12), now), 0.6)
        self.assertEqual(dashboard.time_decay_factor(now - timedelta(days=2), now), 0.3)
        self.assertEqual(dashboard.time_decay_factor(now - timedelta(days=5), now), 0.1)
        self.assertEqual(dashboard.time_decay_factor(now - timedelta(days=9), now), 0.0)

    def test_baseline_comparison(self) -> None:
        rows = dashboard.baseline_comparison(self.make_records())
        names = {row["baseline"] for row in rows}
        self.assertIn("SentimentSignal final_signal_score", names)
        self.assertIn("buy all collected news", names)
        self.assertTrue(any(row["sample_size"] > 0 for row in rows))

    def test_event_type_grouping(self) -> None:
        rows = dashboard.performance_by_group(self.make_records(), "event_type")
        self.assertEqual(rows[0]["event_type"], "guidance")
        self.assertIn("average_excess_return_5d", rows[0])
        self.assertIn("highlight", rows[0])

    def test_market_cap_grouping(self) -> None:
        rows = dashboard.performance_by_group(self.make_records(), "market_cap_bucket")
        groups = {row["market_cap_bucket"] for row in rows}
        self.assertIn("Large-cap", groups)
        self.assertIn("Small-cap", groups)

    def test_duplicate_cluster_analysis(self) -> None:
        records = self.make_records(3)
        for row in records:
            row["duplicate_group"] = "story-1"
            row["source"] = f"Source {row['id']}"
        clusters = dashboard.duplicate_cluster_analysis(records)
        self.assertEqual(clusters[0]["number_of_sources"], 3)
        self.assertGreater(clusters[0]["story_virality_score"], clusters[0]["total_signal_strength"])

    def test_validation_guardrails(self) -> None:
        warnings = dashboard.validation_guardrails(self.make_records(5), "excess_return_5d")
        self.assertTrue(any("Sample size" in row["warning"] for row in warnings))


class MarketDataAndQualityTests(unittest.TestCase):
    def test_market_data_ticker_alias(self) -> None:
        self.assertEqual(market_data.market_symbol_for_data("GOOGLE"), "GOOG")
        self.assertEqual(market_data.market_symbol_for_data("msft"), "MSFT")

    def test_return_calculation_from_closes(self) -> None:
        returns = market_data._returns_from_closes([100, 101, 102, 103, 104, 105, 106], "return")
        self.assertEqual(returns["return_1d"], 1.0)
        self.assertEqual(returns["return_5d"], 5.0)
        self.assertIsNone(returns["return_20d"])
        lagged = market_data._returns_from_closes([100, 101, 102, 103, 104, 105, 106], "return", entry_lag_days=1)
        self.assertEqual(lagged["return_1d"], 0.9901)

    def test_google_news_wrapper_is_not_extracted_directly(self) -> None:
        url = "https://news.google.com/rss/articles/example?oc=5"
        self.assertEqual(news_collector._download_article(url), "")

    def test_google_news_body_is_title_only(self) -> None:
        body, status = news_collector._body_from_entry(
            "https://news.google.com/rss/articles/example?oc=5",
            '<a href="https://news.google.com/rss/articles/example">Apple shares rise</a>',
            "Apple shares rise",
        )
        self.assertEqual(body, "Apple shares rise")
        self.assertEqual(status, "google_news_title_only")

    def test_title_only_body_reduces_quality(self) -> None:
        quality = data_quality.article_data_quality(
            {
                "title": "Apple shares rise",
                "body": "Apple shares rise",
                "body_extraction_status": "google_news_title_only",
                "published_at": "2026-01-01T08:00:00+00:00",
                "source_weight": 1.5,
                "ticker_relevance": "direct",
                "is_duplicate": 0,
                "confidence": 80,
                "market_signal": "bullish",
                "return_5d": 1.0,
            }
        )
        self.assertLess(quality["data_quality_score"], 90)

    def test_title_only_analysis_caps_signal_strength(self) -> None:
        with patch.object(classifier, "_finbert_sentiment", return_value=None):
            analysis = classifier.analyze_article("ASTS", "AST SpaceMobile announces BlueBird 6 launch date", "AST SpaceMobile announces BlueBird 6 launch date", "Yahoo Finance", "")
        adjusted = news_collector._apply_body_extraction_adjustment(analysis, "google_news_title_only")
        self.assertLess(adjusted["confidence"], analysis["confidence"])
        self.assertLessEqual(adjusted["materiality"], 6)
        self.assertLessEqual(adjusted["novelty_score"], 5)
        self.assertLess(adjusted["final_signal_score"], analysis["final_signal_score"])
        self.assertLessEqual(abs(adjusted["final_signal_score"]), 6.0)
        self.assertIn("title-only evidence", adjusted["reasoning"])

    def test_quote_pages_are_identified_as_non_articles(self) -> None:
        self.assertTrue(news_collector._is_non_article_entry("Check out IONQ's stock price (1IONQ-IT) in real time", "CNBC"))
        self.assertFalse(news_collector._is_non_article_entry("IonQ announces quantum networking contract win", "Business Wire"))

    def test_smallcap_sources_are_available_as_feed_presets(self) -> None:
        names = set(news_collector.available_feed_names())
        self.assertIn("GlobeNewswire via Google News", names)
        self.assertIn("Business Wire via Google News", names)
        self.assertIn("PR Newswire via Google News", names)
        self.assertIn("Accesswire via Google News", names)

    def test_market_chatter_reduces_quality(self) -> None:
        quality = data_quality.article_data_quality(
            {
                "title": "Lightning Round: Rocket Lab is a winner, says Jim Cramer",
                "body": "Rocket Lab is a winner, says Jim Cramer.",
                "body_extraction_status": "title_only",
                "published_at": "2026-01-01T08:00:00+00:00",
                "source_weight": 1.2,
                "ticker_relevance": "direct",
                "is_duplicate": 0,
                "confidence": 70,
                "market_signal": "bullish",
                "return_5d": 1.0,
            }
        )
        self.assertLess(quality["data_quality_score"], 70)
        self.assertIn("market chatter", quality["data_quality_notes"])

    def test_market_cap_buckets(self) -> None:
        self.assertEqual(market_data.market_cap_bucket(300_000_000_000), "Mega-cap")
        self.assertEqual(market_data.market_cap_bucket(20_000_000_000), "Large-cap")
        self.assertEqual(market_data.market_cap_bucket(5_000_000_000), "Mid-cap")
        self.assertEqual(market_data.market_cap_bucket(500_000_000), "Small-cap")
        self.assertEqual(market_data.market_cap_bucket(50_000_000), "Micro-cap")
        self.assertEqual(market_data.market_cap_bucket(None), "Unknown")

    def test_sector_etf_mapping(self) -> None:
        self.assertEqual(market_data.sector_etf_for_sector("Technology"), "XLK")
        self.assertEqual(market_data.sector_etf_for_sector("Financial Services"), "XLF")
        self.assertEqual(market_data.sector_etf_for_sector("Not A Sector"), "Unknown")

    def test_data_quality_score(self) -> None:
        quality = data_quality.article_data_quality(
            {
                "title": "Tiny story",
                "body": "",
                "published_at": "",
                "source_weight": 0.4,
                "ticker_relevance": "low",
                "is_duplicate": 1,
                "confidence": 40,
                "market_signal": "mixed",
                "return_5d": None,
            }
        )
        self.assertLess(quality["data_quality_score"], 50)
        self.assertIn("missing body", quality["data_quality_notes"])


class HistoricalBacktestTests(unittest.TestCase):
    def test_builds_requested_number_of_windows(self) -> None:
        windows = historical_backtest.build_windows(months_back=6, windows=20, window_days=7, min_forward_days=35)
        self.assertEqual(len(windows), 20)
        self.assertLess(windows[0].start, windows[-1].start)

    def test_requested_tickers_do_not_need_existing_database_rows(self) -> None:
        rows = historical_backtest.resolve_ticker_universe(Namespace(universe=None, tickers=["RXRX", "AI"]))
        self.assertEqual([row["ticker"] for row in rows], ["RXRX", "AI"])
        self.assertEqual(rows[0]["company_name"], "Recursion Pharmaceuticals")
        self.assertEqual(rows[1]["company_name"], "C3.ai")

    def test_ambiguous_ticker_query_uses_company_and_dollar_symbol(self) -> None:
        url = historical_backtest.historical_google_news_url(
            "AI",
            "C3.ai",
            "Reuters",
            datetime(2026, 1, 1).date(),
            datetime(2026, 1, 8).date(),
        )
        self.assertIn("%22C3.ai%22", url)
        self.assertIn("%22%24AI%22", url)
        self.assertNotIn("+AI+stock+", url)

    def test_directional_match(self) -> None:
        self.assertTrue(historical_backtest._directional_match(2, 1.5, 1))
        self.assertTrue(historical_backtest._directional_match(-2, -1.5, 1))
        self.assertFalse(historical_backtest._directional_match(2, -1.5, 1))
        self.assertIsNone(historical_backtest._directional_match(0.2, 1.5, 1))
        self.assertIsNone(historical_backtest._directional_match(2, 0.1, 1, return_threshold=0.25))

    def test_skipped_backtest_issue_is_not_padded_with_return_failures(self) -> None:
        message = historical_backtest._issue_notes(
            {"status": "skipped", "message": "Skipped because ticker/company was not detected"},
            "20d",
            1,
        )
        self.assertEqual(message, "Skipped because ticker/company was not detected")

    def test_smallcap_source_set_includes_press_release_sources(self) -> None:
        sources = historical_backtest.resolve_historical_sources(Namespace(source_set="smallcap", sources=None))
        self.assertIn("GlobeNewswire", sources)
        self.assertIn("Business Wire", sources)
        self.assertIn("PR Newswire", sources)
        self.assertIn("Yahoo Finance", sources)

    def test_source_set_can_be_extended_with_explicit_sources(self) -> None:
        sources = historical_backtest.resolve_historical_sources(Namespace(source_set="biotech", sources=["CNBC", "Reuters"]))
        self.assertIn("Fierce Biotech", sources)
        self.assertIn("CNBC", sources)
        self.assertEqual(sources.count("Reuters"), 1)


class RefreshPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old_path = database.DATABASE_PATH
        database.DATABASE_PATH = Path(self.tmp.name) / "test.sqlite3"
        database.init_db()

    def tearDown(self) -> None:
        database.DATABASE_PATH = self.old_path
        self.tmp.cleanup()

    def test_weighted_coverage_excludes_duplicates_and_old_rows(self) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds")
        today = now[:10]
        database.insert_article(
            {
                "ticker": "TST",
                "title": "Test raises guidance",
                "body": "Test raises guidance",
                "source": "Reuters",
                "published_date": today,
                "published_at": now,
                "final_signal_score": 8,
                "signal_strength": 12,
                "is_duplicate": 0,
            }
        )
        database.insert_article(
            {
                "ticker": "TST",
                "title": "Duplicate copy",
                "body": "Duplicate copy",
                "source": "Reuters",
                "published_date": today,
                "published_at": now,
                "final_signal_score": 8,
                "signal_strength": 99,
                "is_duplicate": 1,
            }
        )
        coverage = refresh_policy.weighted_signal_coverage("TST", lookback_hours=24 * 365)
        self.assertEqual(coverage, 12.0)

    def test_policy_runs_when_coverage_is_low(self) -> None:
        decision = refresh_policy.evaluate_refresh_policy("TST", min_weighted_coverage=10, refresh_interval_hours=6, lookback_hours=24)
        self.assertTrue(decision.should_run)
        self.assertIn("coverage", decision.reason)


if __name__ == "__main__":
    unittest.main()
