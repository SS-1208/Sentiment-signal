# Historical Backtest Report - 20260624T141703Z

Command:

```bash
python3 historical_backtest.py --windows 20 --months-back 6 --window-days 7 --min-forward-days 35 --per-source-limit 1 --max-entries-per-query 15 --feed-timeout-seconds 8 --sources Reuters CNBC --horizon 20d
```

## Method Fixes Active
- Google News RSS wrappers are stored as title-only evidence, not fake HTML article bodies.
- Historical rows use run-scoped duplicate detection by default.
- Inserted historical rows are cleaned from the main SQLite database after CSV export.
- Returns use a one-trading-day entry lag by default.
- Realized moves below 0.25% are ignored for directional thesis matching.
- Known false positives were patched: wrongful-gains language, competitor stock moves, and favorable legal outcomes.

## Files
- articles: `/Users/sahibsidhu/Documents/market sentiment algo/backtest_runs/historical_backtest_articles_20260624T141703Z.csv`
- windows: `/Users/sahibsidhu/Documents/market sentiment algo/backtest_runs/historical_backtest_windows_20260624T141703Z.csv`
- issues: `/Users/sahibsidhu/Documents/market sentiment algo/backtest_runs/historical_backtest_issues_20260624T141703Z.csv`

## Headline Result
- Tickers tested: AAPL, CRSP, GOOGLE, MSFT, NVDA, QCOM
- Historical windows: 20
- Article rows scanned/recorded: 1100
- Saved articles: 133
- Independent saved articles after duplicate exclusion: 131
- Directional 20-day raw-return observations: 27
- Raw thesis match rate: 59.26%
- Directional 20-day SPY-excess observations: 31
- Excess thesis match rate: 54.84%
- Body extraction statuses: {'google_news_title_only': 133}

Interpretation: the stricter method is more honest but still not enough evidence for a predictive claim. Every saved article in this run is title-only because Google News RSS did not provide original article bodies.

## By Ticker
| Ticker | Saved | Independent | Directional obs | Raw match | Avg 20d return |
|---|---:|---:|---:|---:|---:|
| AAPL | 39 | 38 | 8 | 62.50% | 1.915% |
| CRSP | 0 | 0 | 0 | N/A | N/A |
| GOOGLE | 40 | 40 | 13 | 61.54% | 3.178% |
| MSFT | 37 | 36 | 5 | 60.00% | -1.567% |
| NVDA | 1 | 1 | 0 | N/A | 6.345% |
| QCOM | 16 | 16 | 1 | 0.00% | 10.589% |

## By Source
| Source | Saved | Independent | Directional obs | Raw match |
|---|---:|---:|---:|---:|
| CNBC | 70 | 68 | 11 | 45.45% |
| Reuters | 63 | 63 | 16 | 68.75% |

## By Signal Bucket
| Bucket | Independent articles | Directional obs | Raw match | Avg 20d return |
|---|---:|---:|---:|---:|
| neutral | 100 | 0 | N/A | 2.555% |
| strongly positive | 16 | 13 | 53.85% | 5.928% |
| strongly negative | 13 | 12 | 66.67% | -1.751% |
| mildly positive | 1 | 1 | 0.00% | -1.443% |
| mildly negative | 1 | 1 | 100.00% | -6.813% |

## What Went Wrong
Collection and feed misses:
- 802: Skipped because ticker/company was not detected
- 129: Skipped because article date was outside requested window
- 36: No RSS entries found

Saved article/model issues:
- 105: low data quality: title-only body
- 105: uncertain classification
- 105: return base uses 1 trading-day entry lag
- 105: sector/profile context skipped for faster historical validation
- 102: signal below directional threshold
- 29: market data ticker normalized to GOOG
- 14: excess 20d return contradicted signal thesis
- 11: raw 20d return contradicted signal thesis
- 2: duplicate excluded from independent evidence
- 2: duplicate story

## Conclusion
The repaired method reduces false positives and prevents the backtest from contaminating the app database. The remaining bottleneck is source quality: title-only Google News RSS evidence is too thin for institutional-grade validation. The next step is a proper article source or publisher URL resolution so event/entity classification can use full text.
