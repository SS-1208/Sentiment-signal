from __future__ import annotations

import difflib
import logging
from functools import lru_cache
from typing import Any

from config import DUPLICATE_SIMILARITY_THRESHOLD


logger = logging.getLogger(__name__)


def _summary_text(title: str, body: str) -> str:
    return f"{title.strip()} {body.strip()[:600]}".strip()


@lru_cache(maxsize=1)
def _embedding_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        logger.info("sentence-transformers unavailable, using text-similarity fallback: %s", exc)
        return None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _embedding_similarity(left: str, right: str) -> float | None:
    model = _embedding_model()
    if model is None:
        return None
    try:
        vectors = model.encode([left, right], normalize_embeddings=True)
        return float(_cosine_similarity(vectors[0], vectors[1]))
    except Exception as exc:
        logger.info("Embedding similarity failed, using text fallback: %s", exc)
        return None


def _text_similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz import fuzz

        return fuzz.token_set_ratio(left, right) / 100
    except Exception:
        return difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()


def find_duplicate(title: str, body: str, existing_articles: list[Any]) -> dict[str, Any]:
    candidate = _summary_text(title, body)
    best = {"is_duplicate": False, "duplicate_group": "", "matched_id": None, "similarity": 0.0, "method": "none"}

    for article in existing_articles:
        article_title = article["title"] if hasattr(article, "keys") else article.get("title", "")
        article_body = article["body"] if hasattr(article, "keys") else article.get("body", "")
        comparison = _summary_text(article_title, article_body)

        similarity = _embedding_similarity(candidate, comparison)
        method = "sentence-transformers"
        if similarity is None:
            similarity = max(_text_similarity(title, article_title), _text_similarity(candidate, comparison))
            method = "rapidfuzz/difflib"

        if similarity > best["similarity"]:
            article_id = article["id"] if hasattr(article, "keys") else article.get("id")
            duplicate_group = article["duplicate_group"] if hasattr(article, "keys") else article.get("duplicate_group", "")
            best = {
                "is_duplicate": similarity >= DUPLICATE_SIMILARITY_THRESHOLD,
                "duplicate_group": duplicate_group or f"story-{article_id}",
                "matched_id": article_id,
                "similarity": round(float(similarity), 4),
                "method": method,
            }

    return best
