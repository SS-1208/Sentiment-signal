from __future__ import annotations

import argparse
import logging
import time

from config import (
    DEFAULT_COVERAGE_LOOKBACK_HOURS,
    DEFAULT_MIN_WEIGHTED_SIGNAL_COVERAGE,
    DEFAULT_REFRESH_INTERVAL_HOURS,
)
from news_collector import available_feed_names
from refresh_policy import collect_if_due


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SentimentSignal coverage/time-based news refresh.")
    parser.add_argument("--ticker", required=True, help="Ticker to monitor, for example AAPL.")
    parser.add_argument("--company", default="", help="Company name used for relevance filtering.")
    parser.add_argument(
        "--feeds",
        nargs="*",
        default=None,
        help="Preset feed names to use. Defaults to the first eight weighted presets.",
    )
    parser.add_argument("--per-feed-limit", type=int, default=3, help="Maximum saved articles per source.")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_WEIGHTED_SIGNAL_COVERAGE)
    parser.add_argument("--refresh-hours", type=int, default=DEFAULT_REFRESH_INTERVAL_HOURS)
    parser.add_argument("--lookback-hours", type=int, default=DEFAULT_COVERAGE_LOOKBACK_HOURS)
    parser.add_argument("--force", action="store_true", help="Run collection regardless of policy decision.")
    parser.add_argument("--loop", action="store_true", help="Keep running and check repeatedly.")
    parser.add_argument("--sleep-minutes", type=float, default=15.0, help="Minutes to sleep between loop checks.")
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> dict:
    feeds = args.feeds
    if feeds is None:
        feeds = available_feed_names()[:8]
    outcome = collect_if_due(
        args.ticker,
        args.company,
        feeds,
        per_feed_limit=max(1, args.per_feed_limit),
        min_weighted_coverage=args.min_coverage,
        refresh_interval_hours=args.refresh_hours,
        lookback_hours=args.lookback_hours,
        force=args.force,
    )
    decision = outcome["decision"]
    logger.info(
        "ticker=%s status=%s saved=%s coverage_before=%.3f reason=%s",
        args.ticker.upper(),
        outcome["status"],
        outcome.get("saved_count", 0),
        decision.weighted_coverage,
        decision.reason,
    )
    if "updated_weighted_coverage" in outcome:
        logger.info("ticker=%s updated_coverage=%.3f", args.ticker.upper(), outcome["updated_weighted_coverage"])
    return outcome


def main() -> None:
    args = parse_args()
    if not args.loop:
        run_once(args)
        return

    sleep_seconds = max(60, int(args.sleep_minutes * 60))
    logger.info("starting refresh loop for %s; checking every %.2f minutes", args.ticker.upper(), sleep_seconds / 60)
    while True:
        run_once(args)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
