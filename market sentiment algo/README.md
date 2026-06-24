# SentimentSignal

SentimentSignal is a Streamlit MVP for collecting stock-related news, classifying financial sentiment, detecting duplicate stories, and comparing sentiment against future stock returns.

## Features

- Monitor page for a simple ticker-first signal view, coverage status, latest stories, strongest signals, and refresh controls.
- Manual article entry for ticker, company name, title, body, source, URL, date, and publication time.
- Rule-based financial news classifier with optional FinBERT sentiment blending.
- Source weighting for Reuters, Bloomberg, Financial Times, Wall Street Journal, CNBC, Yahoo Finance, Seeking Alpha, small-cap PR wires/niche outlets, Reddit, and unknown sources.
- Duplicate story detection with sentence-transformers when available and RapidFuzz/difflib fallback.
- RSS collector using feedparser and trafilatura.
- Ticker-first source collector that checks weighted preset feeds for Yahoo Finance, Reuters, CNBC, small-cap PR wires, niche outlets, Bloomberg, Financial Times, Wall Street Journal, Seeking Alpha, and Reddit.
- Timed and coverage-based refresh policy that reruns collection when weighted signal coverage is too low or the configured refresh window has elapsed.
- Streamlit dashboard with aggregate metrics, story rankings, trend chart, ticker breakdown, and duplicate list.
- Market validation using Yahoo chart data with yfinance fallback to calculate 1 day, 5 day, and 20 day future returns.
- Phase 5 research validation with signal buckets, grouped return analysis, materiality/source comparisons, experimental backtesting, charts, and CSV export.
- Phase 6 signal quality upgrade with event classification, expectation surprise, novelty scoring, confidence calibration, statistical significance tests, event studies, signal ranking, and a Research Lab.
- Hardening layer with train/holdout validation, walk-forward validation, multiple-testing correction, explicit data-readiness metrics, publication timestamps, command-line refresh runner, and regression tests.
- Baseline comparisons against random signals, buy-all-news, positive-tone news, bullish market-signal news, high-source-weight news, high-materiality news, and a simple momentum proxy.
- Time-decayed monitor scoring so older articles remain in research history but contribute less to the current ticker signal.
- Optional market-cap buckets and sector-adjusted returns using mapped sector ETFs when yfinance profile data is available.
- Duplicate cluster analysis that measures source spread and story virality without counting duplicates as independent validation evidence.
- Per-article data quality scoring with explicit notes for weak source data, title-only evidence, missing body/timestamp, duplicates, low relevance, uncertainty, and missing returns.

## Launch

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The app stores data in `sentiment.sqlite3` in the project directory.

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

Run a non-Streamlit refresh check from the terminal:

```bash
python3 refresh_runner.py --ticker AAPL --company "Apple" --per-feed-limit 3
```

Run it repeatedly as a lightweight monitor:

```bash
python3 refresh_runner.py --ticker AAPL --company "Apple" --loop --sleep-minutes 15
```

Run a historical collection/backtest across the tickers already stored in SQLite:

```bash
python3 historical_backtest.py --windows 20 --months-back 6 --window-days 7 --min-forward-days 35 --per-source-limit 1 --sources Reuters CNBC --horizon 20d
```

Run the recommended small-cap starter universe with the small-cap source mix:

```bash
python3 historical_backtest.py --universe starter10 --source-set smallcap --windows 10 --months-back 6 --window-days 7 --min-forward-days 35 --per-source-limit 1 --horizon 20d
```

This samples historical article windows, runs them through the same classifier/storage/return pipeline, and writes CSV and Markdown reports under `backtest_runs/`:

- `historical_backtest_articles_<run_id>.csv`: article-level saved/skipped rows, scores, returns, thesis-match flags, and `what_went_wrong`.
- `historical_backtest_windows_<run_id>.csv`: ticker/window-level coverage and match-rate summaries.
- `historical_backtest_issues_<run_id>.csv`: rows with collection, quality, duplicate, missing-return, or thesis-mismatch issues.
- `historical_backtest_report_<run_id>.md`: human-readable verdict, source reliability, top issues, and saved article summary.

Historical runs use run-scoped duplicate detection by default and clean up inserted test articles from the main SQLite table after export. Pass `--keep-inserted` only when you intentionally want historical test rows to remain in the app database. Historical validation also defaults to a one-trading-day entry lag and ignores realized returns smaller than `0.25%` when judging directional match. The runner marks a report as inconclusive when it does not reach the minimum directional-observation count, and `--max-source-errors` prevents repeatedly failing feeds from stalling the full run.

Use this as a research audit, not as proof of a profitable strategy. Small samples, source gaps, neutral classifications, duplicate stories, and broad market moves can all dominate the result.

## News Collection

The News Collector page supports two modes:

- Timed / coverage-based refresh: enter a ticker, choose weighted sources, set a minimum weighted coverage threshold, and set the refresh interval.
- Weighted source collection: enter a ticker once, choose the preset sources to check, and the app collects/analyzes articles across those feeds.
- Custom RSS: paste any RSS URL manually when you want to test a specific feed.

Preset sources include Yahoo Finance ticker RSS, Google News RSS filters for Reuters/CNBC/GlobeNewswire/Business Wire/PR Newswire/Accesswire/Seeking Alpha/Bloomberg/FT/WSJ/Fierce Biotech/SpaceNews/Defense News, Reddit ticker search, and broad CNBC/Seeking Alpha feeds. Generic feeds are filtered by ticker/company text before saving. Google News RSS article links are treated as wrappers; when full extraction is unavailable, the app stores them as title-only evidence, caps confidence/materiality/novelty/final signal strength, and lowers the data-quality score instead of pretending the RSS HTML snippet is a full article body.

Refresh policy is based on recent weighted signal coverage, not article count:

```text
coverage = sum(signal_strength for non-duplicate articles collected inside the lookback window)
```

Because `signal_strength` already reflects impact magnitude, novelty, source weight, surprise, and relevance, this avoids treating low-impact duplicate or weak-source articles as enough evidence. A refresh runs when either:

- Weighted coverage is below the chosen threshold.
- The configured number of hours has passed since the last refresh.

Use "Run If Due" for normal operation and "Force Refresh Now" when you want to manually override the policy.

The command-line `refresh_runner.py` uses the same policy and can be scheduled with cron, launchd, or any process supervisor. Streamlit is still the UI, but refresh logic no longer has to live only inside button clicks.

## Monitor Page

The Monitor page is the simplest ticker-first workflow. It shows:

- Current time-decayed `Research Signal`
- Raw weighted signal
- Weighted coverage
- Most recent article time
- Data quality warning
- Refresh recommendation
- Strongest current bullish story
- Strongest current bearish story
- Latest story cards and an audit table

The Monitor intentionally uses research wording such as `Research Signal`, `Strong Bullish`, `Bullish`, `Neutral`, `Bearish`, and `Strong Bearish`. These labels describe the research signal only. They are not trading instructions.

Time decay defaults:

```text
0-6 hours: 1.00
6-24 hours: 0.60
1-3 days: 0.30
3-7 days: 0.10
7+ days: 0.00
```

The app calculates:

```text
decayed_signal_score = final_signal_score * time_decay_factor
decayed_signal_strength = signal_strength * time_decay_factor
```

Old articles are not deleted. They stay available for validation and research, but their current monitor influence decays.

## Phase 5: Statistical Validation and Backtesting

The validation page groups analysed articles into impact-score buckets:

- Strongly positive: `final_impact_score >= 7`
- Mildly positive: `1` to `6.99`
- Neutral: `-0.99` to `0.99`
- Mildly negative: `-6.99` to `-1`
- Strongly negative: `<= -7`

For each bucket, the app calculates:

- Average 1-day, 5-day, and 20-day returns
- Median 1-day, 5-day, and 20-day returns
- 5-day win rate
- Number of observations

The page also compares:

- High materiality articles: `materiality >= 7`
- Low materiality articles: `materiality < 7`
- High-quality sources: `source_weight >= 1.2`
- Lower-quality sources: `source_weight < 1.2`

Duplicate articles are excluded from validation statistics by default. Use the duplicate toggle only to compare how repeated syndicated stories affect the signal.

## Experimental Strategy Simulation

The app includes a research-only 5-day strategy simulation:

- Buy when `final_impact_score >= 7`
- Short or avoid when `final_impact_score <= -7`
- Hold for 5 trading days

It reports average trade return, cumulative simulated return, and trade count. This is not live trading and does not connect to a broker.

## How To Interpret The Validation Page

The validation page is designed to test whether sentiment has predictive value, not to assume it does. A useful signal would generally show stronger future returns after strongly positive stories than after neutral or strongly negative stories, especially when materiality and source quality are high.

The Market Validation page now includes `Baseline Comparison`, which compares SentimentSignal against:

- Random signal baseline
- Buy all collected news
- Buy only positive `article_tone`
- Buy only bullish `market_signal`
- Buy only high `source_weight` articles
- Buy only high `materiality` articles
- Momentum proxy when prior same-ticker article return data exists

Each baseline reports sample size, average raw returns, average SPY-adjusted excess returns, win rate, t-statistic, and p-value. The app also displays a plain-English interpretation asking whether SentimentSignal currently outperforms simple baselines.

Small sample sizes can be misleading. A bucket with only a few articles can show extreme returns by chance. Treat early results as directional research, not proof.

Avoid overfitting. If you repeatedly tune thresholds until historical results look good, the signal may fail on new data. Keep thresholds simple, test on out-of-sample periods, and compare against baselines.

This project is not financial advice. It is a research tool for evaluating whether structured news sentiment has measurable predictive value.

## Hardening Against Common Critiques

The app now includes specific defenses against the most common objections:

- Small sample sizes: validation pages show sample warnings and data-readiness metrics.
- Overfitting: train/holdout validation learns a threshold on earlier observations and tests later observations separately.
- Backtests lying: walk-forward validation only uses prior observations to choose the signal threshold for each later trade.
- Multiple testing: the validation page reports Benjamini-Hochberg adjusted p-values across common research groups.
- Weak baselines: the validation page compares the signal against simple non-model baselines.
- Market drift: the app keeps raw returns, SPY-adjusted excess returns, and sector-adjusted excess returns.
- Duplicate story inflation: duplicate articles are excluded from validation while duplicate clusters measure story spread and virality.
- Current signal staleness: Monitor uses time-decayed signal scores.
- Data quality opacity: each article receives a 0-100 data quality score and notes.
- RSS/app-only refresh: `refresh_runner.py` can run outside Streamlit and use the same time/coverage policy.
- Weak retail workflow: the Monitor page gives a ticker-first view before the user enters Research Lab.
- Missing timestamp precision: articles now store `published_at` when feeds provide it, while retaining `published_date` for daily return calculations.
- Blank returns: market-data failures and incomplete return horizons are recorded in `data_quality_notes`.

## Phase 6: Signal Quality Upgrade

Phase 6 improves the signal model beyond simple positive-news/negative-news assumptions. Each article now receives:

- `event_type`: earnings, guidance, analyst upgrade/downgrade, management change, acquisition, merger, buyback, dividend, product launch, contract win, regulation, lawsuit, investigation, macro, financing, bankruptcy, or other.
- `expectation_surprise`: strongly positive, positive, neutral, negative, or strongly negative.
- `novelty_score`: 1-10 estimate of how much genuinely new information the article contains.
- `article_tone`: the literal tone of the article language.
- `market_signal`: bullish, bearish, mixed, neutral, or irrelevant for the ticker.
- `ticker_relevance`: direct, related, sector, low, or irrelevant.
- `positive_evidence`, `negative_evidence`, and `market_reaction_evidence`: explainable evidence used by the market-signal classifier.
- `contradiction_flag`: marks headlines such as "beats expectations but shares fall."
- `final_signal_score`: the primary market-signal score used for research and ranking.
- `prediction_correct`: calibration label for non-neutral articles once future return data exists.
- `signal_strength`: ranking score based on impact magnitude, novelty, source weight, and surprise.

The newer market-signal model separates article tone from market interpretation. It prioritizes market-reaction phrases, expectation surprise, event-specific playbooks, contradiction handling, ticker relevance, source quality, novelty, and confidence.

Event-specific playbooks add priors and materiality floors for categories such as bankruptcy, investigations, lawsuits, guidance, earnings, analyst actions, buybacks, dividends, contract wins, and financing. Generic product announcements remain neutral unless there is positive surprise or market reaction evidence.

Final signal score is calculated from:

```text
market_signal_direction * materiality * confidence * source_weight * novelty_score * surprise_multiplier * relevance_weight
```

Surprise multipliers:

- Strongly positive: `1.5`
- Positive: `1.2`
- Neutral: `1.0`
- Negative: `1.2`
- Strongly negative: `1.5`

The Research Lab page lets you filter by ticker, event type, date range, source, materiality, surprise score, and signal strength. It displays strongest signals, event-study returns, surprise analysis, novelty analysis, statistical tests, confidence calibration, and CSV export.

Statistical testing uses scipy when available. For each group it reports sample size, mean return, median return, standard deviation, t-statistic, p-value, and confidence interval. Small samples can still be misleading even when a p-value is shown.

## Rigorous Evaluation Layer

The validation tools now include:

- Raw stock returns and benchmark-adjusted excess returns versus `SPY`.
- Sector-adjusted returns when the company sector maps to a sector ETF.
- Return-basis selector for raw 5-day return, SPY-adjusted 5-day return, or sector-adjusted 5-day return.
- Data-readiness metrics showing non-duplicate articles, return coverage, sample warnings, and dataset range.
- Confusion matrix for bullish/bearish market-signal predictions versus realized future returns.
- Accuracy, bullish precision, bearish precision, and evaluated prediction count.
- Sample-size warnings when groups have fewer than 30 observations.
- Baseline comparison table and interpretation.
- Performance by event type.
- Performance by market-cap bucket.
- Duplicate story cluster analysis.
- Validation guardrails for sample size, weak p-values, holdout/walk-forward instability, ticker/source/event concentration, and signal disappearance after SPY or sector adjustment.
- Train/holdout validation for simple out-of-sample testing.
- Walk-forward validation that learns thresholds only from prior observations.
- Multiple-testing correction for statistical-test p-values.
- Signal audit tables showing the evidence behind every score.
- Refresh action to backfill raw, benchmark, and excess returns after articles are collected.

The benchmark adjustment is intentionally simple:

```text
excess_return = stock_future_return - SPY_future_return
```

This is not a full factor model, but it is a better research target than raw return because it separates company-specific movement from broad market drift.

Sector adjustment uses this mapping when yfinance sector data is available:

```text
Technology -> XLK
Financial Services -> XLF
Energy -> XLE
Healthcare -> XLV
Consumer Defensive -> XLP
Consumer Cyclical -> XLY
Industrials -> XLI
Utilities -> XLU
Real Estate -> XLRE
Communication Services -> XLC
Basic Materials -> XLB
```

Market-cap buckets:

```text
Mega-cap: >= $200B
Large-cap: >= $10B
Mid-cap: >= $2B
Small-cap: >= $300M
Micro-cap: > $0
Unknown: unavailable or invalid market cap
```

## Duplicate Clusters and Data Quality

Duplicate detection still prevents syndicated copies from inflating validation statistics. The app also analyses duplicate groups as story clusters:

- `first_seen_at`
- `latest_seen_at`
- `number_of_sources`
- `source_list`
- `strongest_source_weight`
- `average_signal_score`
- `total_signal_strength`
- `story_virality_score`

This separates two ideas:

- Duplicates should not count as independent proof in validation.
- A story spreading across multiple sources can still matter for visibility.

Each article also receives `data_quality_score` from 0 to 100 and text notes. Penalties include missing body, missing timestamp, weak source, low ticker relevance, duplicate status, extraction weakness, uncertain classification, and unavailable return data.

## Reliability

The app is designed to keep running when optional services fail:

- FinBERT unavailable: rule-based sentiment is used.
- sentence-transformers unavailable: RapidFuzz or Python difflib duplicate detection is used.
- RSS feed fails: the collector returns an error row instead of crashing.
- Article extraction fails: RSS summary/title is used.
- Title-only evidence is available: confidence, materiality, novelty, and final signal strength are capped before storage.
- yfinance fails: return fields remain blank.

## Remaining Limitations

This is now a stronger MVP, not an institutional-grade terminal. The main remaining gaps are:

- RSS and Google News feeds can be delayed, incomplete, rate-limited, or blocked by redirects.
- Free historical collection can fail even when the classifier is working; treat low-coverage backtests as data-backend failures, not predictive evidence.
- Yahoo chart data and yfinance are useful for MVP validation but not institutional market data.
- Publication time is only as precise as the feed provides.
- The classifier is explainable and tested, but still rule-heavy unless FinBERT is available.
- SQLite is appropriate for a local MVP; production multi-user deployment should move to Postgres or another server database.
- Sector adjustment depends on yfinance profile data and ETF return availability.
- Baseline and walk-forward tests need larger datasets before conclusions are reliable.
- The app does not perform live trading, does not connect to a broker, and does not claim profitability.
