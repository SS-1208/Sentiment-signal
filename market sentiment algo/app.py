from __future__ import annotations

import logging
import html
from datetime import date, datetime, time

try:
    import pandas as pd
    import streamlit as st
except Exception as exc:  # pragma: no cover - launch-time dependency guard
    raise SystemExit(
        "Streamlit and pandas are required to launch the UI. Install them with: "
        "python3 -m pip install -r requirements.txt"
    ) from exc

try:
    import altair as alt
except Exception:  # pragma: no cover - charts degrade to Streamlit defaults
    alt = None

from classifier import analyze_article
from dashboard import (
    add_research_fields,
    aggregate_metrics,
    audit_columns,
    baseline_comparison,
    baseline_interpretation,
    confidence_calibration,
    confusion_matrix,
    data_readiness_report,
    duplicate_cluster_analysis,
    default_return_field,
    event_study,
    filter_duplicates,
    holdout_validation,
    multiple_testing_report,
    performance_by_group,
    return_summary,
    rows_to_records,
    simulate_research_strategy,
    statistical_tests,
    validation_guardrails,
    validation_metrics,
    walk_forward_backtest,
)
from database import fetch_articles, fetch_existing_for_dedupe, init_db, insert_article, update_analysis, update_returns
from deduplication import find_duplicate
from market_data import calculate_future_returns
from news_collector import available_feed_names, collect_from_presets, collect_from_rss
from refresh_policy import collect_if_due, evaluate_refresh_policy, weighted_signal_coverage


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


POSITIVE = "#00C853"
NEGATIVE = "#FF5252"
NEUTRAL = "#9E9E9E"
BG = "#0E1117"
CARD = "#161B22"
TEXT = "#FFFFFF"
SECONDARY = "#B0B3B8"


def inject_terminal_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {BG};
            --card: {CARD};
            --positive: {POSITIVE};
            --negative: {NEGATIVE};
            --neutral: {NEUTRAL};
            --text: {TEXT};
            --secondary: {SECONDARY};
        }}
        html {{
            color-scheme: dark;
        }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: var(--bg);
            color: var(--text);
        }}
        [data-testid="stSidebar"] {{
            background: #0A0D12;
            border-right: 1px solid #30363D;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp [data-testid="stMarkdownContainer"], .stApp label,
        .stApp [data-testid="stWidgetLabel"] {{
            color: var(--text);
        }}
        .stCaption, [data-testid="stCaptionContainer"], .stMarkdown small,
        .stApp [data-testid="stCaptionContainer"] * {{
            color: var(--secondary);
        }}
        div[data-testid="stForm"] {{
            background: var(--card);
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 14px;
        }}
        div[data-testid="stExpander"] {{
            background: rgba(22, 27, 34, .72);
            border: 1px solid #30363D;
            border-radius: 8px;
        }}
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            border: 1px solid #30363D;
            border-radius: 8px;
            overflow: hidden;
        }}
        div[data-testid="stMetric"] {{
            background: var(--card);
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 12px;
        }}
        div[data-testid="stRadio"] > label {{
            display: none;
        }}
        div[role="radiogroup"] {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 8px 0 16px;
        }}
        div[role="radiogroup"] label {{
            min-height: 38px;
            border: 1px solid #30363D;
            border-radius: 8px;
            background: #111720;
            padding: 8px 12px;
            margin: 0;
            cursor: pointer;
        }}
        div[role="radiogroup"] label > div:first-child {{
            display: none;
        }}
        div[role="radiogroup"] label:has(input:checked) {{
            border-color: rgba(0, 200, 83, .72);
            background: rgba(0, 200, 83, .14);
            box-shadow: inset 0 -2px 0 var(--positive);
        }}
        div[role="radiogroup"] label p {{
            color: var(--text);
            font-weight: 750;
            white-space: nowrap;
        }}
        button, [data-testid="stDownloadButton"] button {{
            border-radius: 8px !important;
        }}
        .terminal-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 12px;
            margin: 12px 0 18px;
        }}
        .terminal-card {{
            background: var(--card);
            border: 1px solid #30363D;
            border-left: 5px solid var(--neutral);
            border-radius: 8px;
            padding: 14px 15px;
            min-height: 94px;
        }}
        .terminal-card.positive {{
            border-color: rgba(0, 200, 83, .35);
            border-left-color: var(--positive);
            background: rgba(0, 200, 83, .08);
        }}
        .terminal-card.negative {{
            border-color: rgba(255, 82, 82, .38);
            border-left-color: var(--negative);
            background: rgba(255, 82, 82, .08);
        }}
        .terminal-card.neutral {{
            border-left-color: var(--neutral);
            background: rgba(158, 158, 158, .08);
        }}
        .metric-label {{
            color: var(--secondary);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
        }}
        .metric-value {{
            color: var(--text);
            font-size: 28px;
            font-weight: 850;
            margin-top: 7px;
        }}
        .metric-note {{
            color: var(--secondary);
            font-size: 12px;
            margin-top: 4px;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border-radius: 999px;
            padding: 4px 9px;
            font-weight: 800;
            font-size: 12px;
            border: 1px solid #30363D;
            white-space: nowrap;
        }}
        .badge.positive {{
            color: var(--positive);
            background: rgba(0, 200, 83, .12);
            border-color: rgba(0, 200, 83, .45);
        }}
        .badge.negative {{
            color: var(--negative);
            background: rgba(255, 82, 82, .12);
            border-color: rgba(255, 82, 82, .45);
        }}
        .badge.neutral {{
            color: var(--neutral);
            background: rgba(158, 158, 158, .12);
            border-color: rgba(158, 158, 158, .35);
        }}
        .badge-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }}
        .story-card {{
            background: var(--card);
            border: 1px solid #30363D;
            border-left: 6px solid var(--neutral);
            border-radius: 8px;
            padding: 15px 16px;
            margin: 10px 0;
        }}
        .story-card.positive {{ border-left-color: var(--positive); }}
        .story-card.negative {{ border-left-color: var(--negative); }}
        .story-card.neutral {{ border-left-color: var(--neutral); }}
        .story-title {{
            color: var(--text);
            font-size: 16px;
            font-weight: 850;
            line-height: 1.35;
            margin-bottom: 8px;
        }}
        .story-meta {{
            color: var(--secondary);
            font-size: 12px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 10px;
        }}
        .story-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 8px;
            margin: 12px 0;
        }}
        .story-stat {{
            border: 1px solid #30363D;
            border-radius: 6px;
            padding: 8px;
            background: #0F141B;
        }}
        .story-stat span {{
            display: block;
            color: var(--secondary);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .story-stat strong {{
            display: block;
            color: var(--text);
            margin-top: 3px;
        }}
        .reasoning {{
            color: var(--secondary);
            font-size: 13px;
            line-height: 1.45;
            margin-top: 8px;
        }}
        .positive-text {{ color: var(--positive) !important; }}
        .negative-text {{ color: var(--negative) !important; }}
        .neutral-text {{ color: var(--neutral) !important; }}
        .block-container {{
            padding-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def polarity_class(value: object) -> str:
    if value is None:
        return "neutral"
    text_value = str(value).lower()
    if any(word in text_value for word in ["bullish", "positive", "buy", "strongly_positive"]):
        return "positive"
    if any(word in text_value for word in ["bearish", "negative", "avoid", "sell", "strongly_negative"]):
        return "negative"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "neutral"


def signed_fmt(value: object, suffix: str = "", precision: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.{precision}f}{suffix}"


def signal_badge(label: object) -> str:
    raw = "neutral" if label is None or pd.isna(label) else str(label)
    cls = polarity_class(raw)
    display = raw.replace("_", " ").title()
    return f'<span class="badge {cls}"><span class="badge-dot"></span>{html.escape(display)}</span>'


def metric_cards(cards: list[dict[str, object]]) -> None:
    html_cards = ['<div class="terminal-grid">']
    for card in cards:
        cls = card.get("polarity") or polarity_class(card.get("value"))
        html_cards.append(
            f'<div class="terminal-card {cls}">'
            f'<div class="metric-label">{html.escape(str(card.get("label", "")))}</div>'
            f'<div class="metric-value">{html.escape(str(card.get("value", "")))}</div>'
            f'<div class="metric-note">{html.escape(str(card.get("note", "")))}</div>'
            f'</div>'
        )
    html_cards.append("</div>")
    st.markdown("".join(html_cards), unsafe_allow_html=True)


def story_card(row: dict[str, object], compact: bool = False) -> str:
    cls = polarity_class(row.get("market_signal") or row.get("final_signal_score"))
    title = html.escape(str(row.get("title") or "Untitled"))
    source = html.escape(str(row.get("source") or "Unknown"))
    date_value = html.escape(str(row.get("published_date") or "Unknown date"))
    impact = signed_fmt(row.get("final_signal_score"), precision=2)
    materiality = signed_fmt(row.get("materiality"), precision=0)
    return_5d = signed_fmt(row.get("excess_return_5d") if row.get("excess_return_5d") is not None else row.get("return_5d"), "%", 2)
    reasoning = html.escape(str(row.get("reasoning") or row.get("positive_evidence") or row.get("negative_evidence") or "No reasoning captured."))
    if compact and len(reasoning) > 260:
        reasoning = reasoning[:260] + "..."
    return (
        f'<div class="story-card {cls}">'
        f'<div class="story-title">{title}</div>'
        f'<div class="story-meta"><span>{source}</span><span>{date_value}</span>{signal_badge(row.get("market_signal"))}{signal_badge(row.get("expectation_surprise"))}</div>'
        f'<div class="story-stats">'
        f'<div class="story-stat"><span>Signal</span><strong class="{polarity_class(row.get("final_signal_score"))}-text">{impact}</strong></div>'
        f'<div class="story-stat"><span>Materiality</span><strong>{materiality}</strong></div>'
        f'<div class="story-stat"><span>5D Return</span><strong class="{polarity_class(row.get("excess_return_5d") if row.get("excess_return_5d") is not None else row.get("return_5d"))}-text">{return_5d}</strong></div>'
        f'<div class="story-stat"><span>Relevance</span><strong>{html.escape(str(row.get("ticker_relevance") or "unknown")).title()}</strong></div>'
        f'</div><div class="reasoning">{reasoning}</div></div>'
    )


def story_cards(records: list[dict[str, object]], limit: int = 6, compact: bool = True) -> None:
    cards = [story_card(row, compact=compact) for row in records[:limit]]
    st.markdown("".join(cards), unsafe_allow_html=True)


def styled_dataframe(df: pd.DataFrame):
    if df.empty:
        return df

    def style_cell(value):
        cls = polarity_class(value)
        if cls == "positive":
            return f"color: {POSITIVE}; font-weight: 700;"
        if cls == "negative":
            return f"color: {NEGATIVE}; font-weight: 700;"
        return f"color: {NEUTRAL};"

    numeric_or_signal = [
        column
        for column in df.columns
        if any(token in column for token in ["return", "score", "impact", "signal", "surprise", "sentiment"])
    ]
    try:
        styler = df.style.map(style_cell, subset=numeric_or_signal)
        if "signal_strength" in df.columns and df["signal_strength"].notna().any():
            q90 = df["signal_strength"].quantile(0.9)
            q10 = df["signal_strength"].quantile(0.1)

            def signal_strength_style(value):
                if pd.isna(value):
                    return f"color: {NEUTRAL};"
                if value >= q90 and value > 0:
                    return f"background-color: rgba(0, 200, 83, .32); color: {TEXT}; font-weight: 850;"
                if value <= q10:
                    return f"background-color: rgba(255, 82, 82, .24); color: {TEXT}; font-weight: 850;"
                return ""

            styler = styler.map(signal_strength_style, subset=["signal_strength"])
        return styler
    except Exception:
        return df


def colored_bar_chart(df: pd.DataFrame, x: str, y: str, title: str | None = None) -> None:
    if df.empty or alt is None or x not in df or y not in df:
        if x in df and y in df:
            st.bar_chart(df.set_index(x)[[y]])
        return
    chart_data = df[[x, y]].dropna().copy()
    chart_data[y] = pd.to_numeric(chart_data[y], errors="coerce")
    chart_data = chart_data.dropna(subset=[x, y])
    if chart_data.empty:
        return
    chart_data["polarity"] = chart_data[y].apply(lambda value: "positive" if value > 0 else "negative" if value < 0 else "neutral")
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", sort=None, title=None),
            y=alt.Y(f"{y}:Q", title=y.replace("_", " ").title()),
            color=alt.Color(
                "polarity:N",
                scale=alt.Scale(domain=["positive", "negative", "neutral"], range=[POSITIVE, NEGATIVE, NEUTRAL]),
                legend=None,
            ),
            tooltip=[x, y],
        )
        .properties(height=280, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def line_chart_if_data(df: pd.DataFrame, index_col: str, value_col: str) -> bool:
    if df.empty or index_col not in df or value_col not in df:
        return False
    chart_data = df[[index_col, value_col]].dropna().copy()
    chart_data[value_col] = pd.to_numeric(chart_data[value_col], errors="coerce")
    chart_data = chart_data.dropna(subset=[index_col, value_col])
    if chart_data.empty:
        return False
    st.line_chart(chart_data.set_index(index_col)[[value_col]])
    return True


def cumulative_return_chart(df: pd.DataFrame) -> None:
    if df.empty or "published_date" not in df or "cumulative_return" not in df:
        return
    if alt is None:
        st.line_chart(df.set_index("published_date")[["cumulative_return"]])
        return
    chart_data = df[["published_date", "cumulative_return"]].dropna().copy()
    if chart_data.empty:
        return
    chart_data["polarity"] = chart_data["cumulative_return"].apply(lambda value: "positive" if value > 0 else "negative" if value < 0 else "neutral")
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("published_date:N", title=None),
            y=alt.Y("cumulative_return:Q", title="Cumulative Return"),
            color=alt.Color(
                "polarity:N",
                scale=alt.Scale(domain=["positive", "negative", "neutral"], range=[POSITIVE, NEGATIVE, NEUTRAL]),
                legend=None,
            ),
            tooltip=["published_date", "cumulative_return", "polarity"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def save_article(article: dict) -> int:
    analysis = analyze_article(article["ticker"], article["title"], article["body"], article["source"], article.get("company_name", ""))
    duplicate = find_duplicate(article["title"], article["body"], fetch_existing_for_dedupe(article["ticker"]))
    if duplicate.get("is_duplicate"):
        old_novelty = max(float(analysis.get("novelty_score") or 1), 1.0)
        analysis["novelty_score"] = min(int(analysis.get("novelty_score") or 1), 2)
        analysis["final_signal_score"] = round(float(analysis.get("final_signal_score") or 0) * (analysis["novelty_score"] / old_novelty), 4)
        analysis["signal_strength"] = abs(analysis["final_signal_score"])
    payload = {**article, **analysis, **duplicate}
    article_id = insert_article(payload)
    returns = calculate_future_returns(article["ticker"], article["published_date"])
    update_returns(article_id, returns)
    return article_id


def reanalyze_stored_articles() -> int:
    rows = fetch_articles(include_duplicates=True)
    for row in rows:
        analysis = analyze_article(row["ticker"], row["title"], row["body"], row["source"], row["company_name"])
        if row["is_duplicate"]:
            old_novelty = max(float(analysis.get("novelty_score") or 1), 1.0)
            analysis["novelty_score"] = min(int(analysis.get("novelty_score") or 1), 2)
            analysis["final_signal_score"] = round(float(analysis.get("final_signal_score") or 0) * (analysis["novelty_score"] / old_novelty), 4)
            analysis["signal_strength"] = abs(analysis["final_signal_score"])
        update_analysis(row["id"], analysis)
    return len(rows)


def refresh_market_returns() -> int:
    rows = fetch_articles(include_duplicates=True)
    updated = 0
    for row in rows:
        returns = calculate_future_returns(row["ticker"], row["published_date"])
        update_returns(row["id"], returns)
        updated += 1
    return updated


def research_signal_label(score: float) -> str:
    if score >= 7:
        return "Strong Bullish"
    if score >= 1:
        return "Bullish"
    if score <= -7:
        return "Strong Bearish"
    if score <= -1:
        return "Bearish"
    return "Neutral"


def render_monitor() -> None:
    st.subheader("Signal Monitor")
    st.caption("Ticker-first view for current signal state. Experimental research only; not financial advice.")
    with st.form("monitor-load-form"):
        col1, col2 = st.columns([1, 1.5])
        ticker = col1.text_input("Ticker", placeholder="AAPL", key="monitor-ticker").upper().strip()
        company_name = col2.text_input("Company name", placeholder="Apple Inc.", key="monitor-company")
        st.form_submit_button("Load Monitor")

    feed_names = available_feed_names()
    with st.expander("Refresh controls"):
        selected_feeds = st.multiselect("Sources", feed_names, default=feed_names[:8], key="monitor-feeds")
        col1, col2, col3 = st.columns(3)
        min_coverage = col1.number_input("Minimum weighted coverage", min_value=0.0, value=25.0, step=5.0, key="monitor-min-coverage")
        refresh_hours = col2.number_input("Refresh after hours", min_value=1, value=6, step=1, key="monitor-refresh-hours")
        lookback_hours = col3.number_input("Coverage lookback hours", min_value=1, value=24, step=1, key="monitor-lookback-hours")
        col1, col2 = st.columns(2)
        run_due = col1.button("Run Refresh If Due")
        force_run = col2.button("Force Refresh")

    if not ticker:
        st.info("Enter a ticker to see the monitor.")
        return

    if run_due or force_run:
        with st.spinner("Checking weighted sources and updating the monitor dataset..."):
            outcome = collect_if_due(
                ticker,
                company_name,
                selected_feeds,
                per_feed_limit=3,
                min_weighted_coverage=min_coverage,
                refresh_interval_hours=int(refresh_hours),
                lookback_hours=int(lookback_hours),
                force=force_run,
            )
        if outcome["status"] == "skipped":
            st.info("Refresh skipped because coverage/time thresholds were already satisfied.")
        else:
            st.success(f"Saved {outcome.get('saved_count', 0)} new articles.")

    rows = fetch_articles(ticker, include_duplicates=True)
    records = add_research_fields(rows_to_records(rows))
    decision = evaluate_refresh_policy(
        ticker,
        min_weighted_coverage=min_coverage,
        refresh_interval_hours=int(refresh_hours),
        lookback_hours=int(lookback_hours),
    )
    readiness = data_readiness_report(records, default_return_field(records, "5d") if records else "return_5d")
    unique_records = filter_duplicates(records)

    if not records:
        metric_cards(
            [
                {"label": "Weighted Coverage", "value": signed_fmt(decision.weighted_coverage), "polarity": "negative"},
                {"label": "Policy", "value": "Run", "polarity": "positive"},
                {"label": "Stored Articles", "value": 0, "polarity": "neutral"},
            ]
        )
        st.info("No stored articles for this ticker yet. Run a refresh to collect weighted source data.")
        return

    df = pd.DataFrame(unique_records)
    if "published_at" not in df:
        df["published_at"] = df.get("published_date", "")
    weights = df["signal_strength"].fillna(0).astype(float).abs()
    scores = df["final_signal_score"].fillna(0).astype(float)
    decayed_weights = df["decayed_signal_strength"].fillna(0).astype(float).abs()
    decayed_scores = df["decayed_signal_score"].fillna(0).astype(float)
    net_signal = round((scores * weights).sum() / weights.sum(), 3) if weights.sum() else round(scores.mean(), 3)
    decayed_signal = round((decayed_scores * decayed_weights).sum() / decayed_weights.sum(), 3) if decayed_weights.sum() else 0.0
    latest_df = df.sort_values(["published_at", "created_at"], ascending=False)
    strongest = df.sort_values("signal_strength", ascending=False)
    bullish = df[df["final_signal_score"].fillna(0).astype(float) > 0].sort_values("final_signal_score", ascending=False)
    bearish = df[df["final_signal_score"].fillna(0).astype(float) < 0].sort_values("final_signal_score", ascending=True)
    latest_article_time = latest_df.iloc[0]["published_at"] if not latest_df.empty else "N/A"
    avg_quality = round(df["data_quality_score"].fillna(100).astype(float).mean(), 1) if "data_quality_score" in df else 100
    low_quality_count = int((df["data_quality_score"].fillna(100).astype(float) < 70).sum()) if "data_quality_score" in df else 0
    quality_warning = "Review weak data" if avg_quality < 75 or low_quality_count else "No major warning"
    refresh_recommendation = "Refresh recommended" if decision.should_run else "Refresh not due"
    signal_polarity = polarity_class(decayed_signal)

    metric_cards(
        [
            {"label": "Research Signal", "value": f"{research_signal_label(decayed_signal)} {signed_fmt(decayed_signal)}", "polarity": signal_polarity},
            {"label": "Raw Signal", "value": f"{research_signal_label(net_signal)} {signed_fmt(net_signal)}", "polarity": polarity_class(net_signal)},
            {"label": "Weighted Coverage", "value": signed_fmt(decision.weighted_coverage), "polarity": "positive" if decision.weighted_coverage >= min_coverage else "negative"},
            {"label": "Latest Article", "value": latest_article_time, "polarity": "neutral"},
            {"label": "Data Quality", "value": f"{avg_quality}/100", "note": quality_warning, "polarity": "negative" if avg_quality < 75 else "positive"},
            {"label": "Refresh", "value": refresh_recommendation, "polarity": "positive" if decision.should_run else "neutral"},
        ]
    )
    st.caption(f"Refresh reason: {decision.reason}")
    st.caption("Research Signal is time-decayed. Older articles remain available for research but have less current monitor influence.")
    if avg_quality < 75 or low_quality_count:
        st.warning(f"Data quality warning: {low_quality_count} current article(s) have quality score below 70.")

    st.markdown("#### Strongest Current Bullish Story")
    story_cards(bullish.to_dict("records"), limit=1)
    st.markdown("#### Strongest Current Bearish Story")
    story_cards(bearish.to_dict("records"), limit=1)

    st.markdown("#### Strongest Current Signals")
    story_cards(strongest.to_dict("records"), limit=5)
    st.markdown("#### Latest Stories")
    story_cards(latest_df.to_dict("records"), limit=6)

    st.markdown("#### Monitor Audit")
    st.dataframe(
        styled_dataframe(
            latest_df[
                [
                    "published_at",
                    "source",
                    "title",
                    "market_signal",
                    "event_type",
                    "expectation_surprise",
                    "final_signal_score",
                    "decayed_signal_score",
                    "signal_strength",
                    "data_quality_score",
                    "ticker_relevance",
                ]
            ].head(25)
        ),
        use_container_width=True,
    )


def render_manual_entry() -> None:
    st.subheader("Manual Article Entry")
    with st.form("manual-entry", clear_on_submit=False):
        col1, col2, col3, col4 = st.columns([1, 1.3, 1, 1])
        ticker = col1.text_input("Ticker", placeholder="AAPL").upper().strip()
        company_name = col2.text_input("Company name", placeholder="Apple Inc.")
        published_date = col3.date_input("Article date", value=date.today())
        published_time = col4.time_input("Article time", value=time(9, 30))
        title = st.text_input("Article title", placeholder="Company raises guidance after earnings beat")
        body = st.text_area("Article body", height=220)
        source = st.text_input("Source", placeholder="Reuters")
        url = st.text_input("URL", placeholder="https://...")
        submitted = st.form_submit_button("Analyze and Save")

    if submitted:
        if not ticker or not title or not body:
            st.error("Ticker, title, and body are required.")
            return
        published_at = datetime.combine(published_date, published_time).isoformat(timespec="minutes")
        article_id = save_article(
            {
                "ticker": ticker,
                "company_name": company_name,
                "title": title,
                "body": body,
                "source": source,
                "url": url,
                "published_at": published_at,
                "published_date": published_date.isoformat(),
            }
        )
        st.success(f"Saved analysis #{article_id}.")


def render_news_collector() -> None:
    st.subheader("News Collector")
    st.caption("Enter a ticker once and collect articles across weighted source feeds.")

    feed_names = available_feed_names()
    with st.expander("Timed / Coverage-Based Refresh Policy", expanded=True):
        st.caption(
            "Refresh runs when recent weighted signal coverage is too low or when the configured time window has elapsed."
        )
        with st.form("refresh-policy"):
            col1, col2 = st.columns(2)
            policy_ticker = col1.text_input("Policy ticker", placeholder="MSFT").upper().strip()
            policy_company = col2.text_input("Policy company name", placeholder="Microsoft")
            policy_feeds = st.multiselect("Weighted sources", feed_names, default=feed_names[:8], key="policy-feeds")
            col1, col2, col3 = st.columns(3)
            min_coverage = col1.number_input("Minimum weighted coverage", min_value=0.0, value=25.0, step=5.0)
            refresh_hours = col2.number_input("Refresh after hours", min_value=1, value=6, step=1)
            lookback_hours = col3.number_input("Coverage lookback hours", min_value=1, value=24, step=1)
            per_feed_limit = st.slider("Max saved articles per source for policy runs", min_value=1, max_value=10, value=3)
            auto_check = st.checkbox("Auto-check once when this page loads")
            col1, col2 = st.columns(2)
            run_due = col1.form_submit_button("Run If Due")
            force_run = col2.form_submit_button("Force Refresh Now")

        if policy_ticker:
            decision = evaluate_refresh_policy(
                policy_ticker,
                min_weighted_coverage=min_coverage,
                refresh_interval_hours=int(refresh_hours),
                lookback_hours=int(lookback_hours),
            )
            metric_cards(
                [
                    {"label": "Weighted Coverage", "value": signed_fmt(decision.weighted_coverage), "polarity": "positive" if decision.weighted_coverage >= min_coverage else "negative"},
                    {"label": "Coverage Threshold", "value": signed_fmt(min_coverage), "polarity": "neutral"},
                    {"label": "Last Refresh", "value": decision.last_run_at or "Never", "polarity": "neutral"},
                    {"label": "Policy Decision", "value": "Run" if decision.should_run else "Wait", "polarity": "positive" if decision.should_run else "neutral"},
                ]
            )
            st.caption(f"Reason: {decision.reason}")

            auto_key = f"auto-refresh-policy:{policy_ticker}:{min_coverage}:{refresh_hours}:{lookback_hours}"
            should_auto_run = auto_check and st.session_state.get(auto_key) != "checked"
            if run_due or force_run or should_auto_run:
                st.session_state[auto_key] = "checked"
                with st.spinner("Applying refresh policy across weighted sources..."):
                    outcome = collect_if_due(
                        policy_ticker,
                        policy_company,
                        policy_feeds,
                        per_feed_limit=per_feed_limit,
                        min_weighted_coverage=min_coverage,
                        refresh_interval_hours=int(refresh_hours),
                        lookback_hours=int(lookback_hours),
                        force=force_run,
                    )
                if outcome["status"] == "skipped":
                    st.info("Refresh skipped: coverage and time policy did not require a run.")
                else:
                    st.success(
                        f"Refresh complete. Saved {outcome.get('saved_count', 0)} articles. "
                        f"Updated weighted coverage: {signed_fmt(outcome.get('updated_weighted_coverage'))}."
                    )
                    st.dataframe(pd.DataFrame(outcome.get("results", [])), use_container_width=True)

    with st.form("preset-collector"):
        col1, col2 = st.columns(2)
        ticker = col1.text_input("Ticker", placeholder="MSFT").upper().strip()
        company_name = col2.text_input("Company name", placeholder="Microsoft")
        selected_feeds = st.multiselect("Sources to check", feed_names, default=feed_names[:8])
        per_feed_limit = st.slider("Max saved articles per source", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("Collect From Weighted Sources")

    if submitted:
        if not ticker:
            st.error("Ticker is required.")
            return
        with st.spinner("Checking source feeds, extracting articles, scoring signals, and storing results..."):
            results = collect_from_presets(ticker, company_name, selected_feeds, per_feed_limit)
        st.dataframe(pd.DataFrame(results), use_container_width=True)

    with st.expander("Custom RSS feed"):
        with st.form("rss-collector"):
            col1, col2 = st.columns(2)
            custom_ticker = col1.text_input("Custom ticker", placeholder="MSFT").upper().strip()
            custom_company = col2.text_input("Custom company name", placeholder="Microsoft")
            rss_url = st.text_input("RSS feed URL", placeholder="https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US")
            source_name = st.text_input("Source override", placeholder="Yahoo Finance")
            limit = st.slider("Max articles", min_value=1, max_value=25, value=10)
            custom_submitted = st.form_submit_button("Collect Custom RSS")

        if custom_submitted:
            if not custom_ticker or not rss_url:
                st.error("Ticker and RSS feed URL are required.")
                return
            with st.spinner("Collecting and analyzing custom feed..."):
                results = collect_from_rss(custom_ticker, custom_company, rss_url, limit, source_override=source_name or None)
            st.dataframe(pd.DataFrame(results), use_container_width=True)


def render_dashboard() -> None:
    st.subheader("Sentiment Dashboard")
    if st.button("Re-analyze Stored Articles"):
        updated = reanalyze_stored_articles()
        st.success(f"Re-analyzed {updated} stored articles with the latest classifier rules.")
    tickers = sorted({row["ticker"] for row in fetch_articles() if row["ticker"]})
    selected = st.selectbox("Ticker filter", ["All"] + tickers)
    rows = fetch_articles(None if selected == "All" else selected)
    records = add_research_fields(rows_to_records(rows))
    if not records:
        st.info("No articles saved yet.")
        return

    event_types = sorted({row.get("event_type") or "other" for row in records})
    selected_event = st.selectbox("Event type filter", ["All"] + event_types)
    if selected_event != "All":
        records = [row for row in records if (row.get("event_type") or "other") == selected_event]
    if not records:
        st.info("No articles match the selected filters.")
        return

    metrics = aggregate_metrics(records)
    metric_cards(
        [
            {"label": "Total Articles", "value": metrics["total_articles"], "polarity": "neutral"},
            {"label": "Bullish Signals", "value": metrics["bullish_articles"], "polarity": "positive"},
            {"label": "Bearish Signals", "value": metrics["bearish_articles"], "polarity": "negative"},
            {"label": "Mixed Signals", "value": metrics["mixed_articles"], "polarity": "neutral"},
            {"label": "Average Materiality", "value": metrics["average_materiality"], "polarity": "neutral"},
            {"label": "Average Confidence", "value": metrics["average_confidence"], "polarity": "neutral"},
            {"label": "Average Final Signal", "value": signed_fmt(metrics["average_final_signal_score"]), "polarity": polarity_class(metrics["average_final_signal_score"])},
            {"label": "Duplicate Stories", "value": metrics["duplicate_articles"], "polarity": "neutral"},
        ]
    )

    df = pd.DataFrame(records)
    unique_df = df[df["is_duplicate"].fillna(0).astype(int) == 0].copy()
    if not unique_df.empty:
        trend = unique_df.groupby("published_date", as_index=False)["final_signal_score"].mean()
        line_chart_if_data(trend, "published_date", "final_signal_score")

    st.markdown("#### Strongest Bullish Signals")
    story_cards(unique_df.sort_values("final_signal_score", ascending=False).to_dict("records"), limit=5)
    st.markdown("#### Strongest Bearish Signals")
    story_cards(unique_df.sort_values("final_signal_score", ascending=True).to_dict("records"), limit=5)
    st.markdown("#### Strongest Signals")
    st.dataframe(styled_dataframe(unique_df.sort_values("signal_strength", ascending=False).head(10)), use_container_width=True)
    st.markdown("#### Mixed / Contradictory Stories")
    mixed = unique_df[(unique_df["market_signal"] == "mixed") | (unique_df["contradiction_flag"].fillna(0).astype(int) == 1)]
    st.dataframe(styled_dataframe(mixed.sort_values("signal_strength", ascending=False).head(10)), use_container_width=True)
    st.markdown("#### Low Relevance / Irrelevant Stories")
    low_relevance = unique_df[unique_df["ticker_relevance"].isin(["low", "irrelevant"])]
    st.dataframe(styled_dataframe(low_relevance.sort_values("signal_strength", ascending=False).head(10)), use_container_width=True)
    st.markdown("#### Highest Materiality Stories")
    st.dataframe(styled_dataframe(unique_df.sort_values("materiality", ascending=False).head(5)), use_container_width=True)
    st.markdown("#### Latest Stories")
    story_cards(df.sort_values(["published_date", "created_at"], ascending=False).to_dict("records"), limit=8)

    duplicates = df[df["is_duplicate"].fillna(0).astype(int) == 1]
    st.markdown("#### Duplicate Stories")
    st.dataframe(styled_dataframe(duplicates), use_container_width=True)

    st.markdown("#### Ticker Breakdown")
    ticker_breakdown = unique_df.groupby("ticker").agg(
        articles=("id", "count"),
        avg_materiality=("materiality", "mean"),
        avg_confidence=("confidence", "mean"),
        avg_impact=("final_impact_score", "mean"),
        avg_final_signal=("final_signal_score", "mean"),
        avg_signal_strength=("signal_strength", "mean"),
    )
    st.dataframe(styled_dataframe(ticker_breakdown.round(3)), use_container_width=True)


def render_market_validation() -> None:
    st.subheader("Market Validation & Backtesting")
    st.caption(
        "Experimental research only. This page tests whether the news signal has predictive value; "
        "it is not financial advice and does not place trades."
    )

    include_duplicates = st.toggle(
        "Include duplicate articles in validation statistics",
        value=False,
        help="Duplicates are excluded by default so syndicated copies do not inflate the signal.",
    )
    if st.button("Refresh Raw and Benchmark Returns"):
        with st.spinner("Refreshing stock, SPY benchmark, and excess returns..."):
            updated = refresh_market_returns()
        st.success(f"Refreshed market returns for {updated} articles.")
    rows = fetch_articles(include_duplicates=True)
    records = filter_duplicates(add_research_fields(rows_to_records(rows)), include_duplicates)
    if not records:
        st.info("No articles available for validation with the current duplicate setting.")
        return

    default_field = default_return_field(records, "5d")
    return_field = st.selectbox(
        "Validation return basis",
        ["excess_return_5d", "sector_excess_return_5d", "return_5d"],
        index=0 if default_field == "excess_return_5d" else 2,
        help="SPY excess return subtracts SPY. Sector excess subtracts the mapped sector ETF when available. Raw return is the stock's own future return.",
    )
    return_prefix = "sector_excess_return" if return_field.startswith("sector_excess") else "excess_return" if return_field.startswith("excess") else "return"

    df = pd.DataFrame(records)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Analysed Articles CSV",
        data=csv,
        file_name="sentimentsignal_analysed_articles.csv",
        mime="text/csv",
    )

    readiness = data_readiness_report(records, return_field)
    st.markdown("#### Data Readiness")
    metric_cards(
        [
            {"label": "Non-Duplicate Articles", "value": readiness["non_duplicate_articles"], "polarity": "neutral"},
            {"label": "Articles With Returns", "value": readiness["articles_with_selected_returns"], "polarity": "neutral"},
            {"label": "Return Coverage", "value": f"{readiness['return_coverage_pct']}%", "polarity": "positive" if readiness["return_coverage_pct"] >= 80 else "neutral"},
            {"label": "Sample Warning", "value": readiness["sample_warning"].title(), "polarity": "negative" if readiness["sample_warning"] != "ok" else "positive"},
        ]
    )
    st.caption(f"Dataset range: {readiness['first_article']} to {readiness['latest_article']}")

    metrics = validation_metrics(records, return_field)
    metric_cards(
        [
            {"label": "Avg Return After Positive News", "value": signed_fmt(metrics["avg_return_after_positive_news"], "%"), "polarity": polarity_class(metrics["avg_return_after_positive_news"])},
            {"label": "Avg Return After Negative News", "value": signed_fmt(metrics["avg_return_after_negative_news"], "%"), "polarity": polarity_class(metrics["avg_return_after_negative_news"])},
            {"label": "Avg Return After High Materiality", "value": signed_fmt(metrics["avg_return_after_high_materiality_news"], "%"), "polarity": polarity_class(metrics["avg_return_after_high_materiality_news"])},
            {"label": "Avg Return After Low Materiality", "value": signed_fmt(metrics["avg_return_after_low_materiality_news"], "%"), "polarity": polarity_class(metrics["avg_return_after_low_materiality_news"])},
        ]
    )

    cm = confusion_matrix(records, return_field)
    st.markdown("#### Confusion Matrix")
    metric_cards(
        [
            {"label": "Evaluated Predictions", "value": cm["summary"]["evaluated_predictions"], "polarity": "neutral"},
            {"label": "Accuracy", "value": signed_fmt(cm["summary"]["accuracy"] or 0, "%"), "polarity": "neutral"},
            {"label": "Bullish Precision", "value": signed_fmt(cm["summary"]["bullish_precision"] or 0, "%"), "polarity": "positive"},
            {"label": "Bearish Precision", "value": signed_fmt(cm["summary"]["bearish_precision"] or 0, "%"), "polarity": "negative"},
        ]
    )
    if cm["summary"]["sample_warning"] != "ok":
        st.warning("Low sample size: treat confusion-matrix results as directional only.")
    st.dataframe(styled_dataframe(pd.DataFrame(cm["matrix"])), use_container_width=True)

    st.markdown("#### Validation Guardrails")
    guardrails = validation_guardrails(records, return_field)
    st.dataframe(styled_dataframe(pd.DataFrame(guardrails)), use_container_width=True)
    for warning in guardrails:
        if warning["severity"] == "high":
            st.warning(f"{warning['warning']} {warning['detail']}")

    bucket_summary = pd.DataFrame(return_summary(records, "signal_bucket", return_prefix))
    materiality_summary = pd.DataFrame(return_summary(records, "materiality_group", return_prefix))
    source_summary = pd.DataFrame(return_summary(records, "source_quality", return_prefix))
    event_summary = pd.DataFrame(event_study(records, return_prefix))
    surprise_summary = pd.DataFrame(return_summary(records, "expectation_surprise", return_prefix))

    st.markdown("#### Signal Buckets")
    st.dataframe(styled_dataframe(bucket_summary), use_container_width=True)

    if not bucket_summary.empty:
        bucket_chart = bucket_summary.set_index("signal_bucket")
        st.markdown("#### Average Return by Sentiment Bucket")
        colored_bar_chart(bucket_summary, "signal_bucket", "avg_5d_return", "Average 5D Return By Signal Bucket")
        st.markdown("#### 5-Day Win Rate by Sentiment Bucket")
        colored_bar_chart(bucket_summary, "signal_bucket", "win_rate_5d", "5D Win Rate By Signal Bucket")

    st.markdown("#### High vs Low Materiality")
    st.dataframe(styled_dataframe(materiality_summary), use_container_width=True)
    st.markdown(
        "High materiality means `materiality >= 7`. Compare average and median future returns here "
        "to see whether more important stories show stronger post-news patterns."
    )

    st.markdown("#### Source Quality Analysis")
    st.dataframe(styled_dataframe(source_summary), use_container_width=True)
    st.markdown("High-quality sources use `source_weight >= 1.2`; lower-quality sources use `< 1.2`.")

    st.markdown("#### Event Study")
    st.dataframe(styled_dataframe(event_summary), use_container_width=True)

    st.markdown("#### Surprise Score Analysis")
    st.dataframe(styled_dataframe(surprise_summary), use_container_width=True)

    st.markdown("#### Baseline Comparison")
    baseline_df = pd.DataFrame(baseline_comparison(records))
    st.dataframe(styled_dataframe(baseline_df), use_container_width=True)
    st.info(baseline_interpretation(baseline_df.to_dict("records")))

    st.markdown("#### Performance by Event Type")
    event_performance = pd.DataFrame(performance_by_group(records, "event_type"))
    st.dataframe(styled_dataframe(event_performance), use_container_width=True)
    if not event_performance.empty and event_performance["sample_warning"].eq("low sample").any():
        st.warning("Some event types have low sample sizes. Treat event-specific results as exploratory.")

    st.markdown("#### Performance by Market Cap")
    market_cap_performance = pd.DataFrame(performance_by_group(records, "market_cap_bucket"))
    st.dataframe(styled_dataframe(market_cap_performance), use_container_width=True)
    st.caption("This tests whether signals behave differently for Mega-cap, Large-cap, Mid-cap, Small-cap, Micro-cap, or Unknown market-cap groups.")

    st.markdown("#### Duplicate Story Cluster Analysis")
    cluster_df = pd.DataFrame(duplicate_cluster_analysis(add_research_fields(rows_to_records(rows))))
    st.dataframe(styled_dataframe(cluster_df.head(50)), use_container_width=True)
    st.caption("Duplicates are not counted as independent validation evidence, but cluster spread contributes to story visibility and virality.")

    st.markdown("#### Confidence Calibration")
    calibration = pd.DataFrame(confidence_calibration(records))
    st.dataframe(styled_dataframe(calibration), use_container_width=True)
    if not calibration.empty:
        st.bar_chart(calibration.set_index("confidence_bucket")[["predicted_confidence_midpoint", "actual_success_rate"]])

    st.markdown("#### Statistical Significance Tests")
    stat_tabs = st.tabs(["Signal Buckets", "Market Signal", "Event Types", "Surprise", "Materiality"])
    with stat_tabs[0]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(records, "signal_bucket", return_field))), use_container_width=True)
    with stat_tabs[1]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(records, "market_signal", return_field))), use_container_width=True)
    with stat_tabs[2]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(records, "event_type", return_field))), use_container_width=True)
    with stat_tabs[3]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(records, "expectation_surprise", return_field))), use_container_width=True)
    with stat_tabs[4]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(records, "materiality_group", return_field))), use_container_width=True)

    st.markdown("#### Multiple-Testing Control")
    st.caption("Benjamini-Hochberg FDR adjustment across common validation groups. Raw p-values can look better than they are when many groups are tested.")
    multi_tests = multiple_testing_report(
        records,
        ["signal_bucket", "market_signal", "event_type", "expectation_surprise", "materiality_group", "source_quality"],
        return_field,
    )
    st.dataframe(styled_dataframe(pd.DataFrame(multi_tests)), use_container_width=True)

    st.markdown("#### Train / Holdout Validation")
    st.caption("The first 70% of chronological observations learns a simple signal threshold; the final 30% tests it out-of-sample.")
    holdout_df = pd.DataFrame(holdout_validation(records, return_field))
    st.dataframe(styled_dataframe(holdout_df), use_container_width=True)

    st.markdown("#### Walk-Forward Validation")
    st.caption("Each trade uses only prior observations to learn the threshold, then tests the next unseen articles.")
    walk_forward = walk_forward_backtest(records, return_field)
    metric_cards(
        [
            {"label": "Eligible Observations", "value": walk_forward["summary"]["eligible_observations"], "polarity": "neutral"},
            {"label": "Walk-Forward Trades", "value": walk_forward["summary"]["walk_forward_trades"], "polarity": "neutral"},
            {"label": "Avg Trade Return", "value": signed_fmt(walk_forward["summary"]["average_trade_return"], "%"), "polarity": polarity_class(walk_forward["summary"]["average_trade_return"])},
            {"label": "Win Rate", "value": signed_fmt(walk_forward["summary"]["win_rate"], "%"), "polarity": "neutral"},
        ]
    )
    if walk_forward["summary"]["sample_warning"] != "ok":
        st.warning("Walk-forward sample is small. Treat it as a method check, not proof of alpha.")
    if walk_forward["trades"]:
        cumulative_return_chart(pd.DataFrame(walk_forward["trades"]).rename(columns={"published_at": "published_date"}))
        st.dataframe(styled_dataframe(pd.DataFrame(walk_forward["trades"])), use_container_width=True)
    else:
        st.info("No walk-forward trades yet. The app needs enough historical return-labelled observations before testing unseen signals.")

    trades = simulate_research_strategy(records, return_field)
    trades_df = pd.DataFrame(trades)
    st.markdown("#### Experimental 5-Day Strategy Simulation")
    st.warning(
        "Research-only backtest: buy when final signal score is >= 7, short or avoid when <= -7, "
        "and measure the next 5 trading days. This is not financial advice."
    )
    if trades:
        metric_cards(
            [
                {"label": "Trades", "value": len(trades), "polarity": "neutral"},
                {"label": "Average Trade Return", "value": signed_fmt(trades_df["trade_return"].mean(), "%"), "polarity": polarity_class(trades_df["trade_return"].mean())},
                {"label": "Cumulative Simulated Return", "value": signed_fmt(trades_df["cumulative_return"].iloc[-1], "%"), "polarity": polarity_class(trades_df["cumulative_return"].iloc[-1])},
            ]
        )
        cumulative_return_chart(trades_df)
        st.dataframe(styled_dataframe(trades_df), use_container_width=True)
    else:
        metric_cards(
            [
                {"label": "Trades", "value": 0, "polarity": "neutral"},
                {"label": "Average Trade Return", "value": "0.00%", "polarity": "neutral"},
                {"label": "Cumulative Simulated Return", "value": "0.00%", "polarity": "neutral"},
            ]
        )
        st.info("No trades yet. The simulation needs articles with 5-day returns and final signal scores >= 7 or <= -7.")

    st.markdown("#### Article Count Over Time")
    count_over_time = df.groupby("published_date", as_index=False)["id"].count().rename(columns={"id": "article_count"})
    if not line_chart_if_data(count_over_time, "published_date", "article_count"):
        st.caption("Not enough dated articles for this chart yet.")

    st.markdown("#### Average Impact Score Over Time")
    impact_over_time = df.groupby("published_date", as_index=False)["final_signal_score"].mean()
    if not line_chart_if_data(impact_over_time, "published_date", "final_signal_score"):
        st.caption("Not enough scored articles for this chart yet.")

    st.markdown("#### Validation Dataset")
    st.dataframe(styled_dataframe(df), use_container_width=True)

    st.markdown("#### Signal Audit")
    st.dataframe(styled_dataframe(pd.DataFrame(audit_columns(records)).sort_values("signal_strength", ascending=False)), use_container_width=True)


def render_research_lab() -> None:
    st.subheader("Research Lab")
    st.caption("Research terminal for testing event type, surprise, novelty, materiality, source quality, and signal strength.")

    rows = fetch_articles(include_duplicates=True)
    records = add_research_fields(rows_to_records(rows))
    records = filter_duplicates(records, include_duplicates=False)
    if not records:
        st.info("No non-duplicate articles available for research.")
        return

    df = pd.DataFrame(records)
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        tickers = sorted(df["ticker"].dropna().unique().tolist())
        event_types = sorted(df["event_type"].dropna().unique().tolist())
        sources = sorted(df["source"].dropna().unique().tolist())
        surprises = sorted(df["expectation_surprise"].dropna().unique().tolist())
        market_signals = sorted(df["market_signal"].dropna().unique().tolist())
        relevance_options = sorted(df["ticker_relevance"].dropna().unique().tolist())
        selected_tickers = col1.multiselect("Ticker", tickers)
        selected_events = col2.multiselect("Event type", event_types)
        selected_sources = col3.multiselect("Source", sources)
        selected_surprises = col1.multiselect("Expectation surprise", surprises)
        selected_market_signals = col2.multiselect("Market signal", market_signals)
        selected_relevance = col3.multiselect("Ticker relevance", relevance_options)
        min_materiality = col2.slider("Minimum materiality", 1, 10, 1)
        min_signal_strength = col3.number_input("Minimum signal strength", min_value=0.0, value=0.0, step=1.0)
        contradictions_only = col1.checkbox("Contradictions only")
        valid_dates = df["published_date"].dropna()
        if not valid_dates.empty:
            start_date = col1.date_input("Start date", valid_dates.min().date())
            end_date = col2.date_input("End date", valid_dates.max().date())
        else:
            start_date = end_date = None

    filtered = df.copy()
    if selected_tickers:
        filtered = filtered[filtered["ticker"].isin(selected_tickers)]
    if selected_events:
        filtered = filtered[filtered["event_type"].isin(selected_events)]
    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]
    if selected_surprises:
        filtered = filtered[filtered["expectation_surprise"].isin(selected_surprises)]
    if selected_market_signals:
        filtered = filtered[filtered["market_signal"].isin(selected_market_signals)]
    if selected_relevance:
        filtered = filtered[filtered["ticker_relevance"].isin(selected_relevance)]
    if contradictions_only:
        filtered = filtered[filtered["contradiction_flag"].fillna(0).astype(int) == 1]
    filtered = filtered[filtered["materiality"].fillna(0).astype(float) >= min_materiality]
    filtered = filtered[filtered["signal_strength"].fillna(0).astype(float) >= min_signal_strength]
    if start_date and end_date:
        filtered = filtered[
            (filtered["published_date"] >= pd.Timestamp(start_date))
            & (filtered["published_date"] <= pd.Timestamp(end_date))
        ]

    if filtered.empty:
        st.info("No articles match the selected research filters.")
        return

    filtered_records = filtered.to_dict("records")
    default_field = default_return_field(filtered_records, "5d")
    return_field = st.selectbox(
        "Research return basis",
        ["excess_return_5d", "return_5d"],
        index=0 if default_field == "excess_return_5d" else 1,
        help="Use excess return when benchmark data exists; otherwise raw return is available.",
    )
    return_prefix = "excess_return" if return_field.startswith("excess") else "return"
    st.download_button(
        "Export Research Dataset CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="sentimentsignal_research_lab.csv",
        mime="text/csv",
    )

    avg_return = filtered[return_field].dropna().mean() if return_field in filtered and filtered[return_field].notna().any() else 0
    avg_final_signal = filtered["final_signal_score"].mean()
    metric_cards(
        [
            {"label": "Articles", "value": len(filtered), "polarity": "neutral"},
            {"label": "Avg Signal Strength", "value": signed_fmt(filtered["signal_strength"].mean()), "polarity": "neutral"},
            {"label": "Avg Novelty", "value": signed_fmt(filtered["novelty_score"].mean()), "polarity": "neutral"},
            {"label": "Avg 5D Return", "value": signed_fmt(avg_return, "%"), "polarity": polarity_class(avg_return)},
            {"label": "Avg Final Signal", "value": signed_fmt(avg_final_signal), "polarity": polarity_class(avg_final_signal)},
        ]
    )

    cm = confusion_matrix(filtered_records, return_field)
    st.markdown("#### Confusion Matrix")
    st.dataframe(styled_dataframe(pd.DataFrame(cm["matrix"])), use_container_width=True)
    st.json(cm["summary"])

    st.markdown("#### Strongest Signals")
    story_cards(filtered.sort_values("signal_strength", ascending=False).to_dict("records"), limit=8)
    st.dataframe(styled_dataframe(filtered.sort_values("signal_strength", ascending=False)), use_container_width=True)

    st.markdown("#### Event Study")
    event_df = pd.DataFrame(event_study(filtered_records, return_prefix))
    st.dataframe(styled_dataframe(event_df), use_container_width=True)
    if not event_df.empty:
        colored_bar_chart(event_df, "event_type", "day_5_avg_return", "Event Study: Average 5D Return")

    st.markdown("#### Surprise vs Returns")
    surprise_df = pd.DataFrame(return_summary(filtered_records, "expectation_surprise", return_prefix))
    st.dataframe(styled_dataframe(surprise_df), use_container_width=True)
    if not surprise_df.empty:
        colored_bar_chart(surprise_df, "expectation_surprise", "avg_5d_return", "Surprise: Average 5D Return")

    st.markdown("#### Novelty vs 5-Day Return")
    novelty_df = filtered.groupby("novelty_score", as_index=False)["return_5d"].mean()
    if not line_chart_if_data(novelty_df, "novelty_score", "return_5d"):
        st.caption("Not enough 5-day return data for this chart yet.")

    st.markdown("#### Statistical Tests")
    tabs = st.tabs(["Market Signal", "Event Types", "Surprise", "Signal Buckets", "Materiality", "Source Quality", "Ticker Relevance"])
    with tabs[0]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(filtered_records, "market_signal", return_field))), use_container_width=True)
    with tabs[1]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(filtered_records, "event_type", return_field))), use_container_width=True)
    with tabs[2]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(filtered_records, "expectation_surprise", return_field))), use_container_width=True)
    with tabs[3]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(filtered_records, "signal_bucket", return_field))), use_container_width=True)
    with tabs[4]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(filtered_records, "materiality_group", return_field))), use_container_width=True)
    with tabs[5]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(filtered_records, "source_quality", return_field))), use_container_width=True)
    with tabs[6]:
        st.dataframe(styled_dataframe(pd.DataFrame(statistical_tests(filtered_records, "ticker_relevance", return_field))), use_container_width=True)

    st.markdown("#### Confidence Calibration")
    calibration_df = pd.DataFrame(confidence_calibration(filtered_records))
    st.dataframe(styled_dataframe(calibration_df), use_container_width=True)

    st.markdown("#### Signal Audit")
    st.dataframe(styled_dataframe(pd.DataFrame(audit_columns(filtered_records)).sort_values("signal_strength", ascending=False)), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="SentimentSignal", layout="wide", initial_sidebar_state="collapsed")
    inject_terminal_theme()
    init_db()
    st.title("SentimentSignal")
    st.caption("Financial news intelligence for testing whether sentiment has predictive value.")

    pages = ["Monitor", "Manual Entry", "News Collector", "Dashboard", "Market Validation", "Research Lab"]
    page = st.radio(
        "Navigation",
        pages,
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Monitor":
        render_monitor()
    elif page == "Manual Entry":
        render_manual_entry()
    elif page == "News Collector":
        render_news_collector()
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Market Validation":
        render_market_validation()
    else:
        render_research_lab()


if __name__ == "__main__":
    main()
