from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from config import (
    CONTRADICTION_TERMS,
    DRIVER_KEYWORDS,
    EVENT_TYPE_KEYWORDS,
    EVENT_TYPE_PRIORITY,
    EVENT_SIGNAL_PRIORS,
    MARKET_MOVING_TERMS,
    MARKET_REACTION_NEGATIVE,
    MARKET_REACTION_POSITIVE,
    NEGATIVE_KEYWORDS,
    NEUTRAL_KEYWORDS,
    POSITIVE_KEYWORDS,
    RELEVANCE_WEIGHTS,
    SOURCE_WEIGHTS,
    SURPRISE_MULTIPLIERS,
    TICKER_ALIASES,
    UNKNOWN_SOURCE_WEIGHT,
    UNCERTAINTY_TERMS,
)


logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _source_weight(source: str) -> float:
    key = _normalize(source)
    for source_name, weight in SOURCE_WEIGHTS.items():
        if source_name in key:
            return weight
    return UNKNOWN_SOURCE_WEIGHT


def _phrase_hits(text: str, phrases: list[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase in text]


def _favorable_legal_hits(text: str) -> list[str]:
    phrases = [
        "lawsuit dismissed",
        "fends off",
        "dismisses complaint",
        "court dismisses",
        "wins lawsuit",
        "prevails in lawsuit",
        "case dismissed",
        "settles favorably",
    ]
    return _phrase_hits(text, phrases)


def _negated_bad_news_hits(text: str) -> list[str]:
    phrases = [
        "denies report",
        "denied report",
        "rebuts claim",
        "rebuts claims",
        "rejects allegation",
        "rejects allegations",
        "denies allegations",
    ]
    return _phrase_hits(text, phrases)


def _weighted_hits(title: str, body: str, phrases: list[str]) -> tuple[list[str], float]:
    title_text = _normalize(title)
    body_text = _normalize(body)
    hits: list[str] = []
    weight = 0.0
    for phrase in phrases:
        in_title = phrase in title_text
        in_body = phrase in body_text
        if in_title or in_body:
            hits.append(phrase)
            weight += 1.6 if in_title else 1.0
    return hits, weight


def _retrospective_price_move_hits(text: str, hits: list[str]) -> list[str]:
    price_move_terms = {
        "surge",
        "surges",
        "soar",
        "soars",
        "soaring",
        "rally",
        "rallies",
        "jumps",
        "strong year",
    }
    output: list[str] = []
    for hit in hits:
        if hit not in price_move_terms:
            continue
        patterns = [
            rf"(valuation|reassessing valuation|assessing valuation).{{0,100}}{re.escape(hit)}",
            rf"(after|following|since).{{0,100}}(share price|stock|year|run).{{0,60}}{re.escape(hit)}",
            rf"(after|following|since).{{0,100}}{re.escape(hit)}",
            rf"{re.escape(hit)}.{{0,100}}(over the past|year to date|since last earnings|following a strong year)",
        ]
        if any(re.search(pattern, text) for pattern in patterns):
            output.append(hit)
    return output


def _direction_from_score(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def _driver(text: str) -> tuple[str, list[str]]:
    counts: dict[str, int] = {}
    hits: list[str] = []
    for driver, keywords in DRIVER_KEYWORDS.items():
        matched = [keyword for keyword in keywords if keyword in text]
        if matched:
            counts[driver] = len(matched)
            hits.extend(f"{driver}: {keyword}" for keyword in matched[:2])
    if not counts:
        return "other", []
    return max(counts, key=counts.get), hits[:5]


def _time_horizon(text: str, driver: str) -> str:
    if any(term in text for term in ["lawsuit", "investigation", "sec probe", "regulatory risk", "acquisition", "contract"]):
        return "long-term"
    if driver in {"earnings", "revenue", "margins", "capital allocation"}:
        return "medium-term"
    return "short-term"


def _event_type(text: str) -> tuple[str, list[str]]:
    matched: dict[str, list[str]] = {}
    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword in text]
        if hits:
            matched[event_type] = hits
    if not matched:
        return "other", []
    for event_type in EVENT_TYPE_PRIORITY:
        if event_type in matched:
            return event_type, matched[event_type][:4]
    event_type = max(matched, key=lambda key: len(matched[key]))
    return event_type, matched[event_type][:4]


def _expectation_surprise(text: str) -> tuple[str, list[str]]:
    favorable_legal = _favorable_legal_hits(text)
    if favorable_legal:
        return "positive", favorable_legal

    strongly_positive = ["beats expectations", "raises guidance", "lawsuit dismissed"]
    positive = [
        "record revenue",
        "profit growth",
        "margin expansion",
        "approval",
        "contract win",
        "contract award",
        "awarded contract",
        "task order",
        "selected by",
        "jaxa win",
        "launch date",
        "successful launch",
        "milestone payment",
        "narrowed losses",
        "breakeven",
        "break-even",
        "expands manufacturing",
        "manufacturing footprint",
        "accelerates production",
        "acquisition completed",
        "upgrade",
        "buy rating",
    ]
    strongly_negative = ["misses expectations", "cuts guidance", "profit warning", "bankruptcy", "fraud", "sec probe"]
    negative = ["revenue decline", "margin compression", "downgrade", "lawsuit", "investigation", "regulatory risk", "debt concern"]

    strong_pos_hits = _phrase_hits(text, strongly_positive)
    pos_hits = _phrase_hits(text, positive)
    strong_neg_hits = _phrase_hits(text, strongly_negative)
    neg_hits = _phrase_hits(text, negative)

    if strong_pos_hits and not strong_neg_hits:
        return "strongly_positive", strong_pos_hits
    if strong_neg_hits and not strong_pos_hits:
        return "strongly_negative", strong_neg_hits
    if len(pos_hits) > len(neg_hits):
        return "positive", pos_hits
    if len(neg_hits) > len(pos_hits):
        return "negative", neg_hits
    return "neutral", []


def _novelty_score(text: str, event_type: str, source_weight: float) -> int:
    score = 4
    if event_type in {"bankruptcy", "investigation", "acquisition", "merger", "guidance", "contract_win"}:
        score += 3
    if any(term in text for term in ["breaking", "exclusive", "first report", "first reported", "newly disclosed"]):
        score += 3
    if any(term in text for term in ["commentary", "opinion", "recap", "repeats", "previously reported"]):
        score -= 3
    if source_weight >= 1.4:
        score += 1
    return max(1, min(10, score))


def calculate_signal_strength(final_impact_score: float, novelty_score: int, source_weight: float, expectation_surprise: str) -> float:
    surprise_multiplier = SURPRISE_MULTIPLIERS.get(expectation_surprise, 1.0)
    return round(abs(float(final_impact_score or 0)) * int(novelty_score or 1) * float(source_weight or 0.4) * surprise_multiplier, 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _company_terms(ticker: str, company_name: str = "") -> list[str]:
    terms = [ticker.lower()]
    terms.extend(TICKER_ALIASES.get(ticker.upper(), []))
    if company_name:
        normalized = _normalize(company_name)
        terms.append(normalized)
        first_word = normalized.split(" ")[0]
        if len(first_word) > 2:
            terms.append(first_word)
    return sorted({term for term in terms if term})


def _contains_term(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _targeted_market_hits(title: str, body: str, phrases: list[str], ticker: str, company_name: str) -> list[str]:
    title_text = _normalize(title)
    body_text = _normalize(body)
    terms = _company_terms(ticker, company_name)
    hits: list[str] = []
    for phrase in phrases:
        if phrase not in f"{title_text} {body_text}":
            continue
        matched = False
        for text in (title_text, body_text):
            start = 0
            while True:
                index = text.find(phrase, start)
                if index < 0:
                    break
                before = text[max(0, index - 80):index]
                after = text[index + len(phrase):index + len(phrase) + 45]
                subject_window = text[max(0, index - 35):index + len(phrase) + 35]
                if _contains_term(before, terms) or f"shares of " in before and _contains_term(subject_window, terms):
                    matched = True
                    break
                if phrase.startswith("shares ") and _contains_term(after, terms):
                    matched = True
                    break
                start = index + len(phrase)
            if matched:
                break
        if matched:
            hits.append(phrase)
    return hits


def _ticker_relevance(ticker: str, company_name: str, title: str, body: str, event_type: str) -> tuple[str, list[str]]:
    title_text = _normalize(title)
    body_text = _normalize(body)
    terms = _company_terms(ticker, company_name)
    title_hits = [term for term in terms if term in title_text]
    body_hits = [term for term in terms if term in body_text]
    if title_hits:
        return "direct", [f"title: {', '.join(title_hits[:3])}"]
    if body_hits:
        return "related", [f"body: {', '.join(body_hits[:3])}"]
    if event_type in {"macro", "regulation"}:
        return "sector", ["sector/macro event without direct company mention"]
    if event_type != "other":
        return "low", ["event detected but company/ticker not clearly central"]
    return "irrelevant", ["no clear ticker/company relevance found"]


def _article_tone(positive_weight: float, negative_weight: float) -> str:
    if positive_weight and negative_weight:
        if abs(positive_weight - negative_weight) <= 0.8:
            return "mixed"
        return "positive" if positive_weight > negative_weight else "negative"
    if positive_weight:
        return "positive"
    if negative_weight:
        return "negative"
    return "neutral"


def _market_signal_from_score(score: float, positive_score: float, negative_score: float) -> str:
    if positive_score >= 1.2 and negative_score >= 1.2 and abs(score) < 2.0:
        return "mixed"
    if score >= 1.0:
        return "bullish"
    if score <= -1.0:
        return "bearish"
    if positive_score or negative_score:
        return "mixed"
    return "neutral"


def _market_signal_analysis(
    ticker: str,
    company_name: str,
    title: str,
    body: str,
    event_type: str,
    expectation_surprise: str,
    materiality: int,
    confidence: float,
    source_weight: float,
    novelty_score: int,
    positive_hits: list[str],
    negative_hits: list[str],
) -> dict[str, Any]:
    title_text = _normalize(title)
    text = _normalize(f"{title}. {body}")
    favorable_legal = _favorable_legal_hits(text)
    negated_bad_news = _negated_bad_news_hits(text)
    market_positive = _targeted_market_hits(title, body, MARKET_REACTION_POSITIVE, ticker, company_name)
    market_negative = _targeted_market_hits(title, body, MARKET_REACTION_NEGATIVE, ticker, company_name)
    positive_hits = [hit for hit in positive_hits if hit not in MARKET_REACTION_POSITIVE or hit in market_positive]
    negative_hits = [hit for hit in negative_hits if hit not in MARKET_REACTION_NEGATIVE or hit in market_negative]
    uncertainty = _phrase_hits(text, UNCERTAINTY_TERMS)
    contradiction_hits = [term.strip() for term in CONTRADICTION_TERMS if term in f" {text} "]
    positive_evidence = list(dict.fromkeys(positive_hits + market_positive))
    if favorable_legal:
        positive_evidence.extend(favorable_legal)
        negative_hits = [hit for hit in negative_hits if hit not in {"lawsuit", "antitrust", "court"}]
    if negated_bad_news:
        positive_evidence.extend(negated_bad_news)
        negative_score_adjustment = min(1.5, len(negated_bad_news) * 0.8)
    else:
        negative_score_adjustment = 0.0
    negative_evidence = list(dict.fromkeys(negative_hits + market_negative))

    positive_score = len(positive_hits) * 1.0 + len(market_positive) * 2.7
    negative_score = max(0.0, len(negative_hits) * 1.0 + len(market_negative) * 2.7 - negative_score_adjustment)
    if favorable_legal:
        positive_score += 2.0

    if expectation_surprise == "strongly_positive":
        positive_score += 2.2
        positive_evidence.append("strong positive surprise")
    elif expectation_surprise == "positive":
        positive_score += 1.2
        positive_evidence.append("positive surprise")
    elif expectation_surprise == "strongly_negative":
        negative_score += 2.2
        negative_evidence.append("strong negative surprise")
    elif expectation_surprise == "negative":
        negative_score += 1.2
        negative_evidence.append("negative surprise")

    event_prior = EVENT_SIGNAL_PRIORS.get(event_type, {})
    prior_direction = event_prior.get("direction")
    prior_score = float(event_prior.get("score") or 0)
    if prior_direction == "bullish":
        positive_score += prior_score
        positive_evidence.append(f"event prior: {event_prior.get('note')}")
    elif prior_direction == "bearish":
        if event_type == "lawsuit" and favorable_legal:
            positive_score += min(prior_score, 1.5)
            positive_evidence.append("favorable legal context overrides generic lawsuit prior")
        else:
            negative_score += prior_score
            negative_evidence.append(f"event prior: {event_prior.get('note')}")
    elif prior_direction == "surprise":
        if expectation_surprise in {"strongly_positive", "positive"}:
            positive_score += prior_score
            positive_evidence.append(f"event-surprise prior: {event_prior.get('note')}")
        elif expectation_surprise in {"strongly_negative", "negative"}:
            negative_score += prior_score
            negative_evidence.append(f"event-surprise prior: {event_prior.get('note')}")

    if event_type == "guidance":
        if any(term in text for term in ["raises guidance", "raised guidance", "raise guidance"]):
            positive_score += 2.5
            positive_evidence.append("guidance raised")
        if any(term in text for term in ["cuts guidance", "cut guidance"]):
            negative_score += 2.8
            negative_evidence.append("guidance cut")
    elif event_type == "earnings":
        if any(term in text for term in ["beats expectations", "beat expectations", "beats estimates", "earnings beat"]):
            positive_score += 1.8
            positive_evidence.append("earnings beat")
        if any(term in text for term in ["misses expectations", "missed expectations", "misses estimates"]):
            negative_score += 1.9
            negative_evidence.append("earnings miss")
    elif event_type == "analyst_upgrade":
        positive_score += 2.0
        positive_evidence.append("analyst upgrade")
    elif event_type == "analyst_downgrade":
        negative_score += 2.0
        negative_evidence.append("analyst downgrade")
    elif event_type in {"lawsuit", "investigation", "bankruptcy"}:
        if event_type == "lawsuit" and favorable_legal:
            positive_score += 2.0
            positive_evidence.append("favorable legal outcome")
        else:
            negative_score += 2.3
            negative_evidence.append(event_type)
    elif event_type == "regulation":
        if any(term in text for term in ["approval", "approved", "cleared"]):
            positive_score += 1.7
            positive_evidence.append("regulatory approval")
        else:
            negative_score += 1.5
            negative_evidence.append("regulatory risk")
    elif event_type in {"acquisition", "merger"}:
        if any(term in text for term in ["completed", "approved", "deal completed"]):
            positive_score += 2.0
            positive_evidence.append("deal completion")
        elif any(term in text for term in ["nears", "in talks", "reportedly"]):
            positive_score += 1.0
            positive_evidence.append("potential strategic deal")
    elif event_type == "management_change":
        if any(term in text for term in ["resigns", "steps down", "departs", "leaves", "quit", "exits"]):
            negative_score += 2.0
            negative_evidence.append("management/talent departure")
        if any(term in text for term in ["appointed", "hires", "named"]):
            positive_score += 1.2
            positive_evidence.append("management appointment")
    elif event_type in {"contract_win", "share_buyback", "dividend"}:
        positive_score += 1.5
        positive_evidence.append(event_type)
    elif event_type == "product_launch":
        if expectation_surprise in {"strongly_positive", "positive"} or market_positive:
            positive_score += 1.0
            positive_evidence.append("product launch with positive surprise or market reaction")

    if contradiction_hits:
        if any(term in title_text for term in MARKET_REACTION_NEGATIVE):
            negative_score += 1.7
            negative_evidence.append("contradiction resolved by negative market reaction")
        elif any(term in title_text for term in MARKET_REACTION_POSITIVE):
            positive_score += 1.7
            positive_evidence.append("contradiction resolved by positive market reaction")
        elif " but " in f" {title_text} " or " however " in f" {title_text} ":
            tail = re.split(r"\bbut\b|\bhowever\b", title_text, maxsplit=1)[-1]
            tail_positive = _phrase_hits(tail, POSITIVE_KEYWORDS + MARKET_REACTION_POSITIVE)
            tail_negative = _phrase_hits(tail, NEGATIVE_KEYWORDS + MARKET_REACTION_NEGATIVE)
            positive_score += len(tail_positive) * 1.0
            negative_score += len(tail_negative) * 1.0

    relevance, relevance_evidence = _ticker_relevance(ticker, company_name, title, body, event_type)
    relevance_weight = RELEVANCE_WEIGHTS.get(relevance, 0.0)
    article_tone = _article_tone(len(positive_hits), len(negative_hits))
    score_delta = positive_score - negative_score
    market_signal = _market_signal_from_score(score_delta, positive_score, negative_score)
    if relevance == "irrelevant":
        market_signal = "irrelevant"
    elif relevance == "low" and market_signal in {"bullish", "bearish"}:
        market_signal = "mixed"
    elif relevance == "related" and market_signal in {"bullish", "bearish"}:
        market_signal = "mixed"

    direction_map = {"bullish": 1.0, "bearish": -1.0, "mixed": 0.25 if score_delta >= 0 else -0.25, "neutral": 0.0, "irrelevant": 0.0}
    market_signal_score = direction_map[market_signal]
    if uncertainty and market_signal in {"bullish", "bearish"}:
        confidence = max(35.0, confidence - min(18, len(uncertainty) * 5))

    surprise_multiplier = SURPRISE_MULTIPLIERS.get(expectation_surprise, 1.0)
    impact_magnitude = max(1, min(10, materiality))
    raw_final_signal_score = (
        market_signal_score
        * impact_magnitude
        * (confidence / 100)
        * source_weight
        * novelty_score
        * surprise_multiplier
        * relevance_weight
    )
    final_signal_score = round(_clamp(raw_final_signal_score, -10.0, 10.0), 4)

    return {
        "article_tone": article_tone,
        "market_signal": market_signal,
        "market_signal_score": round(market_signal_score, 4),
        "positive_evidence": " | ".join(dict.fromkeys(positive_evidence)),
        "negative_evidence": " | ".join(dict.fromkeys(negative_evidence)),
        "market_reaction_evidence": " | ".join(dict.fromkeys(market_positive + market_negative)),
        "uncertainty_evidence": " | ".join(dict.fromkeys(uncertainty)),
        "contradiction_flag": int(bool(contradiction_hits)),
        "ticker_relevance": relevance,
        "relevance_evidence": " | ".join(relevance_evidence),
        "final_signal_score": final_signal_score,
        "signal_strength": round(abs(raw_final_signal_score), 4),
        "adjusted_confidence": round(confidence, 1),
    }


def _rule_sentiment(title: str, body: str) -> dict[str, Any]:
    text = _normalize(f"{title}. {body}")
    positive_hits, positive_weight = _weighted_hits(title, body, POSITIVE_KEYWORDS)
    negative_hits, negative_weight = _weighted_hits(title, body, NEGATIVE_KEYWORDS)
    neutral_hits = _phrase_hits(text, NEUTRAL_KEYWORDS)
    retrospective_hits = _retrospective_price_move_hits(text, positive_hits)
    if retrospective_hits:
        title_text = _normalize(title)
        body_text = _normalize(body)
        positive_hits = [hit for hit in positive_hits if hit not in retrospective_hits]
        positive_weight = sum(1.6 if hit in title_text else 1.0 for hit in positive_hits if hit in title_text or hit in body_text)
    if _favorable_legal_hits(text):
        negative_hits = [hit for hit in negative_hits if hit not in {"lawsuit", "antitrust"}]
        title_text = _normalize(title)
        body_text = _normalize(body)
        negative_weight = sum(1.6 if hit in title_text else 1.0 for hit in negative_hits if hit in title_text or hit in body_text)

    raw = positive_weight - negative_weight
    if raw > 0:
        score = min(1.0, 0.22 + raw * 0.18)
    elif raw < 0:
        score = max(-1.0, -0.22 + raw * 0.18)
    else:
        score = 0.0

    direction = _direction_from_score(score)
    directional_signal_count = len(positive_hits) + len(negative_hits)
    signal_count = directional_signal_count + len(neutral_hits)
    confidence = min(95, 42 + directional_signal_count * 12 + abs(raw) * 6)
    if direction == "neutral" and neutral_hits:
        confidence = max(confidence, 60)

    material_terms = _phrase_hits(text, MARKET_MOVING_TERMS)
    materiality = 2 + min(5, signal_count)
    if material_terms:
        materiality += 2
    if "guidance" in text or "earnings" in text or "revenue" in text:
        materiality += 1
    materiality = max(1, min(10, materiality))

    reasons: list[str] = []
    if positive_hits:
        reasons.append(f"positive rule hits: {', '.join(positive_hits)}")
    if negative_hits:
        reasons.append(f"negative rule hits: {', '.join(negative_hits)}")
    if neutral_hits and not positive_hits and not negative_hits:
        reasons.append(f"neutral rule hits: {', '.join(neutral_hits)}")
    if retrospective_hits:
        reasons.append(f"ignored retrospective price-move terms: {', '.join(retrospective_hits)}")
    if material_terms:
        reasons.append(f"market-moving terms: {', '.join(material_terms)}")
    if not reasons:
        reasons.append("no strong rule-based sentiment phrases found")

    return {
        "score": score,
        "direction": direction,
        "confidence": confidence,
        "materiality": materiality,
        "reasons": reasons,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "neutral_hits": neutral_hits,
    }


@lru_cache(maxsize=1)
def _finbert_pipeline():
    try:
        from transformers import pipeline

        return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    except Exception as exc:
        logger.info("FinBERT unavailable, using rule-only classifier: %s", exc)
        return None


def _finbert_sentiment(title: str, body: str) -> dict[str, Any] | None:
    pipe = _finbert_pipeline()
    if pipe is None:
        return None

    text = f"{title}. {body[:2500]}"
    try:
        output = pipe(text)
    except Exception as exc:
        logger.info("FinBERT inference failed, using rule-only classifier: %s", exc)
        return None

    scores = output[0] if output and isinstance(output[0], list) else output
    by_label = {item["label"].lower(): float(item["score"]) for item in scores}
    positive = by_label.get("positive", 0.0)
    negative = by_label.get("negative", 0.0)
    neutral = by_label.get("neutral", 0.0)
    score = positive - negative
    confidence = max(positive, negative, neutral) * 100
    return {
        "score": score,
        "direction": _direction_from_score(score),
        "confidence": confidence,
        "raw": by_label,
    }


def analyze_article(ticker: str, title: str, body: str, source: str, company_name: str = "") -> dict[str, Any]:
    normalized_text = _normalize(f"{title}. {body}")
    rule = _rule_sentiment(title, body)
    finbert = _finbert_sentiment(title, body)

    if finbert:
        sentiment_score = (rule["score"] * 0.4) + (finbert["score"] * 0.6)
        confidence = round((rule["confidence"] * 0.4) + (finbert["confidence"] * 0.6), 1)
        model_reason = f"FinBERT signal included: {finbert['raw']}"
    else:
        sentiment_score = rule["score"]
        confidence = round(float(rule["confidence"]), 1)
        model_reason = "FinBERT unavailable; rule-only score used"

    direction = _direction_from_score(sentiment_score)
    affected_driver, driver_hits = _driver(normalized_text)
    time_horizon = _time_horizon(normalized_text, affected_driver)
    source_weight = _source_weight(source)
    event_type, event_hits = _event_type(normalized_text)
    expectation_surprise, surprise_hits = _expectation_surprise(normalized_text)
    materiality = int(rule["materiality"])
    if event_type in {"bankruptcy", "investigation", "guidance", "earnings", "acquisition", "merger"}:
        materiality = min(10, materiality + 1)
    event_prior = EVENT_SIGNAL_PRIORS.get(event_type, {})
    if event_prior.get("materiality_floor"):
        materiality = max(materiality, int(event_prior["materiality_floor"]))
    final_impact_score = round(sentiment_score * materiality * (confidence / 100) * source_weight, 4)
    novelty_score = _novelty_score(normalized_text, event_type, source_weight)
    market = _market_signal_analysis(
        ticker=ticker,
        company_name=company_name,
        title=title,
        body=body,
        event_type=event_type,
        expectation_surprise=expectation_surprise,
        materiality=materiality,
        confidence=confidence,
        source_weight=source_weight,
        novelty_score=novelty_score,
        positive_hits=rule["positive_hits"],
        negative_hits=rule["negative_hits"],
    )
    confidence = market["adjusted_confidence"]
    signal_strength = market["signal_strength"]

    reasoning_parts = list(rule["reasons"])
    if driver_hits:
        reasoning_parts.append(f"driver evidence: {', '.join(driver_hits)}")
    if event_hits:
        reasoning_parts.append(f"event type {event_type}: {', '.join(event_hits)}")
    if event_prior.get("note"):
        reasoning_parts.append(f"event playbook: {event_prior['note']}")
    if surprise_hits:
        reasoning_parts.append(f"expectation surprise {expectation_surprise}: {', '.join(surprise_hits)}")
    reasoning_parts.append(f"source weight {source_weight} applied for {source or 'unknown source'}")
    reasoning_parts.append(f"novelty score estimated at {novelty_score}/10")
    reasoning_parts.append(f"market signal {market['market_signal']} with {market['ticker_relevance']} relevance")
    if market["market_reaction_evidence"]:
        reasoning_parts.append(f"market reaction evidence: {market['market_reaction_evidence']}")
    if market["contradiction_flag"]:
        reasoning_parts.append("contradiction language detected")
    reasoning_parts.append(model_reason)

    return {
        "sentiment_direction": direction,
        "sentiment_score": round(sentiment_score, 4),
        "confidence": confidence,
        "materiality": materiality,
        "source_weight": source_weight,
        "affected_driver": affected_driver,
        "time_horizon": time_horizon,
        "final_impact_score": final_impact_score,
        "event_type": event_type,
        "expectation_surprise": expectation_surprise,
        "novelty_score": novelty_score,
        "signal_strength": signal_strength,
        "article_tone": market["article_tone"],
        "market_signal": market["market_signal"],
        "market_signal_score": market["market_signal_score"],
        "positive_evidence": market["positive_evidence"],
        "negative_evidence": market["negative_evidence"],
        "market_reaction_evidence": market["market_reaction_evidence"],
        "uncertainty_evidence": market["uncertainty_evidence"],
        "contradiction_flag": market["contradiction_flag"],
        "ticker_relevance": market["ticker_relevance"],
        "relevance_evidence": market["relevance_evidence"],
        "final_signal_score": market["final_signal_score"],
        "reasoning": " | ".join(reasoning_parts),
    }
