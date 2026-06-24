from __future__ import annotations

import logging
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from config import MARKET_CAP_BUCKETS, SECTOR_ETFS, YFINANCE_TICKER_ALIASES


logger = logging.getLogger(__name__)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value[:10], "%Y-%m-%d")


def market_symbol_for_data(ticker: str) -> str:
    clean = str(ticker or "").upper().strip()
    return YFINANCE_TICKER_ALIASES.get(clean, clean)


def _blank_returns(prefix: str = "return") -> dict[str, float | None]:
    return {
        f"{prefix}_1d": None,
        f"{prefix}_5d": None,
        f"{prefix}_20d": None,
    }


def _returns_from_closes(closes: list[float], prefix: str, entry_lag_days: int = 0) -> dict[str, float | None]:
    base_index = max(0, int(entry_lag_days or 0))
    if len(closes) <= base_index + 1:
        return _blank_returns(prefix)
    base = float(closes[base_index])

    def horizon_return(days: int) -> float | None:
        target_index = base_index + days
        if len(closes) <= target_index or base == 0:
            return None
        return round(((float(closes[target_index]) / base) - 1) * 100, 4)

    return {
        f"{prefix}_1d": horizon_return(1),
        f"{prefix}_5d": horizon_return(5),
        f"{prefix}_20d": horizon_return(20),
    }


def market_cap_bucket(market_cap: int | float | None) -> str:
    if market_cap is None:
        return "Unknown"
    try:
        value = float(market_cap)
    except (TypeError, ValueError):
        return "Unknown"
    if value <= 0:
        return "Unknown"
    for bucket, floor in MARKET_CAP_BUCKETS:
        if value >= floor:
            return bucket
    return "Unknown"


def sector_etf_for_sector(sector: str | None) -> str:
    if not sector:
        return "Unknown"
    return SECTOR_ETFS.get(str(sector).strip().lower(), "Unknown")


@lru_cache(maxsize=256)
def _ticker_profile(ticker: str) -> dict[str, Any]:
    ticker = market_symbol_for_data(ticker)
    try:
        import yfinance as yf
    except Exception as exc:
        logger.info("yfinance unavailable; profile left unknown: %s", exc)
        return {"market_cap": None, "market_cap_bucket": "Unknown", "sector": "Unknown", "sector_etf": "Unknown"}
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception as exc:
        logger.info("Failed to load yfinance profile for %s: %s", ticker, exc)
        return {"market_cap": None, "market_cap_bucket": "Unknown", "sector": "Unknown", "sector_etf": "Unknown"}

    market_cap = info.get("marketCap") if isinstance(info, dict) else None
    sector = info.get("sector") if isinstance(info, dict) else None
    bucket = market_cap_bucket(market_cap)
    sector_name = sector or "Unknown"
    sector_etf = sector_etf_for_sector(sector_name)
    return {
        "market_cap": market_cap,
        "market_cap_bucket": bucket,
        "sector": sector_name,
        "sector_etf": sector_etf,
    }


@lru_cache(maxsize=4096)
def _download_yahoo_chart_returns(ticker: str, published_date: str, prefix: str = "return", entry_lag_days: int = 0) -> dict[str, float | None]:
    ticker = market_symbol_for_data(ticker)
    try:
        start = _parse_date(published_date)
        end = start + timedelta(days=45)
        period1 = int(start.replace(tzinfo=timezone.utc).timestamp())
        period2 = int(end.replace(tzinfo=timezone.utc).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            return _blank_returns(prefix)
        adjusted = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
        quoted = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        closes = [float(value) for value in (adjusted or quoted) if value is not None]
        return _returns_from_closes(closes, prefix, entry_lag_days)
    except Exception as exc:
        logger.info("Yahoo chart fallback failed for %s: %s", ticker, exc)
        return _blank_returns(prefix)


@lru_cache(maxsize=4096)
def _download_yfinance_returns(ticker: str, published_date: str, prefix: str = "return", entry_lag_days: int = 0) -> dict[str, float | None]:
    ticker = market_symbol_for_data(ticker)
    try:
        import yfinance as yf
    except Exception as exc:
        logger.info("yfinance unavailable; returns left blank: %s", exc)
        return _blank_returns(prefix)

    try:
        start = _parse_date(published_date)
        end = start + timedelta(days=45)
        history = yf.download(ticker, start=start.date().isoformat(), end=end.date().isoformat(), progress=False, auto_adjust=True)
        if history is None or history.empty or "Close" not in history:
            return _blank_returns(prefix)

        closes = history["Close"].dropna()
        if hasattr(closes, "columns"):
            closes = closes.iloc[:, 0]
        return _returns_from_closes([float(value) for value in closes.tolist()], prefix, entry_lag_days)
    except Exception as exc:
        logger.info("Failed to calculate future returns for %s: %s", ticker, exc)
        return _blank_returns(prefix)


def _has_any_return(values: dict[str, float | None], prefix: str) -> bool:
    return any(values.get(f"{prefix}_{horizon}") is not None for horizon in ("1d", "5d", "20d"))


def _download_future_returns(ticker: str, published_date: str, prefix: str = "return", entry_lag_days: int = 0) -> dict[str, float | None]:
    chart_returns = _download_yahoo_chart_returns(ticker, published_date, prefix, entry_lag_days)
    if _has_any_return(chart_returns, prefix):
        return chart_returns
    return _download_yfinance_returns(ticker, published_date, prefix, entry_lag_days)


def calculate_future_returns(
    ticker: str,
    published_date: str,
    benchmark_ticker: str = "SPY",
    include_sector: bool = True,
    entry_lag_days: int = 0,
) -> dict[str, float | None | str]:
    market_ticker = market_symbol_for_data(ticker)
    stock_returns = _download_future_returns(ticker, published_date, "return", entry_lag_days)
    benchmark_returns = _download_future_returns(benchmark_ticker, published_date, "benchmark_return", entry_lag_days)
    profile = _ticker_profile(ticker) if include_sector else {
        "market_cap": None,
        "market_cap_bucket": "Unknown",
        "sector": "Unknown",
        "sector_etf": "Unknown",
    }
    sector_etf = str(profile.get("sector_etf") or "Unknown") if include_sector else "Unknown"
    sector_returns = (
        _download_future_returns(sector_etf, published_date, "sector_return", entry_lag_days)
        if include_sector and sector_etf != "Unknown"
        else _blank_returns("sector_return")
    )
    notes: list[str] = []
    if market_ticker != str(ticker or "").upper().strip():
        notes.append(f"market data ticker normalized to {market_ticker}")
    if entry_lag_days:
        notes.append(f"return base uses {entry_lag_days} trading-day entry lag")
    try:
        days_since_publication = (datetime.utcnow() - _parse_date(published_date)).days
        if days_since_publication < 1:
            notes.append("1-day return horizon is not complete yet")
        if days_since_publication < 5:
            notes.append("5-day return horizon is not complete yet")
        if days_since_publication < 20:
            notes.append("20-day return horizon is not complete yet")
    except Exception:
        notes.append("published date could not be parsed for horizon readiness")
    if all(stock_returns.get(f"return_{horizon}") is None for horizon in ("1d", "5d", "20d")):
        notes.append("stock return data unavailable from yfinance")
    if all(benchmark_returns.get(f"benchmark_return_{horizon}") is None for horizon in ("1d", "5d", "20d")):
        notes.append(f"benchmark return data unavailable for {benchmark_ticker}")
    if not include_sector:
        notes.append("sector/profile context skipped for faster historical validation")
    elif sector_etf == "Unknown":
        notes.append("sector ETF unavailable")
    elif all(sector_returns.get(f"sector_return_{horizon}") is None for horizon in ("1d", "5d", "20d")):
        notes.append(f"sector ETF return data unavailable for {sector_etf}")
    output: dict[str, float | None | str] = {
        **stock_returns,
        **benchmark_returns,
        **sector_returns,
        "benchmark_ticker": benchmark_ticker,
        "market_data_ticker": market_ticker,
        "market_cap": profile.get("market_cap"),
        "market_cap_bucket": profile.get("market_cap_bucket", "Unknown"),
        "sector": profile.get("sector", "Unknown"),
        "sector_etf": sector_etf,
        "data_quality_notes": " | ".join(notes),
    }
    for horizon in ("1d", "5d", "20d"):
        stock_value = output.get(f"return_{horizon}")
        benchmark_value = output.get(f"benchmark_return_{horizon}")
        sector_value = output.get(f"sector_return_{horizon}")
        output[f"excess_return_{horizon}"] = (
            round(float(stock_value) - float(benchmark_value), 4)
            if stock_value is not None and benchmark_value is not None
            else None
        )
        output[f"sector_excess_return_{horizon}"] = (
            round(float(stock_value) - float(sector_value), 4)
            if stock_value is not None and sector_value is not None
            else None
        )
    return output
