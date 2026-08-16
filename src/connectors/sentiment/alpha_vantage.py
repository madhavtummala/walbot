from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.core.config import get_config
from src.data.universe import resolve_project_path

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_time_published(value: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp


def fetch_news_sentiment(
    api_key: str,
    symbol: str,
    time_from: datetime,
    limit: int,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Fetch Alpha Vantage news sentiment for one ticker symbol."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "time_from": time_from.strftime("%Y%m%dT%H%M"),
        "sort": "LATEST",
        "limit": max(1, min(limit, 1000)),
        "apikey": api_key,
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    if "Error Message" in payload:
        raise RuntimeError(f"Alpha Vantage error for {symbol}: {payload['Error Message']}")
    if "Note" in payload:
        raise RuntimeError(f"Alpha Vantage rate limit for {symbol}: {payload['Note']}")
    if "Information" in payload and "feed" not in payload:
        raise RuntimeError(f"Alpha Vantage response for {symbol}: {payload['Information']}")
    return payload


def _ticker_sentiment_for_article(article: dict[str, Any], symbol: str) -> tuple[float, float]:
    symbol = symbol.upper()
    for item in article.get("ticker_sentiment", []) or []:
        if str(item.get("ticker", "")).upper() == symbol:
            sentiment = _safe_float(item.get("ticker_sentiment_score"))
            relevance = _safe_float(item.get("relevance_score"), 1.0)
            return sentiment, max(relevance, 0.0)
    return _safe_float(article.get("overall_sentiment_score")), 1.0


def normalize_news_sentiment(symbol: str, payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize Alpha Vantage NEWS_SENTIMENT output into the bot's social CSV shape."""
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for article in payload.get("feed", []) or []:
        timestamp = _parse_time_published(str(article.get("time_published", "")))
        if pd.isna(timestamp):
            continue

        sentiment, relevance = _ticker_sentiment_for_article(article, symbol)
        rows.append(
            {
                "timestamp": timestamp.floor("D"),
                "symbol": symbol.upper(),
                "mentions": 1.0,
                "weighted_sentiment": sentiment * relevance,
                "relevance": relevance,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "mentions", "sentiment", "social_score"])

    raw_df = pd.DataFrame(rows)
    grouped = (
        raw_df.groupby(["timestamp", "symbol"], as_index=False)
        .agg({"mentions": "sum", "weighted_sentiment": "sum", "relevance": "sum"})
        .sort_values(["timestamp", "symbol"])
    )
    grouped["sentiment"] = grouped.apply(
        lambda row: row["weighted_sentiment"] / row["relevance"] if row["relevance"] > 0 else 0.0,
        axis=1,
    )
    grouped["social_score"] = grouped["sentiment"].clip(-1.0, 1.0)
    return grouped[["timestamp", "symbol", "mentions", "sentiment", "social_score"]]


def collect_alpha_vantage_news(
    api_key: str,
    symbols: list[str],
    lookback_days: int,
    limit: int,
    max_symbols: int,
    request_delay_seconds: float,
) -> pd.DataFrame:
    """Collect and aggregate Alpha Vantage news sentiment for a symbol list."""
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY is required.")

    symbols_to_fetch = [symbol.upper() for symbol in symbols[:max_symbols]]
    time_from = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    frames: list[pd.DataFrame] = []

    for index, symbol in enumerate(symbols_to_fetch):
        logger.info("Fetching Alpha Vantage news sentiment for %s", symbol)
        payload = fetch_news_sentiment(api_key, symbol, time_from, limit)
        frames.append(normalize_news_sentiment(symbol, payload))
        if request_delay_seconds > 0 and index < len(symbols_to_fetch) - 1:
            time.sleep(request_delay_seconds)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "mentions", "sentiment", "social_score"])
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def write_social_trends_csv(df: pd.DataFrame, output_path: str) -> Path:
    """Write social trend rows in the format consumed by src.social.load_social_trends_csv."""
    csv_path = resolve_project_path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = df.copy()
    if not output_df.empty:
        output_df["timestamp"] = pd.to_datetime(output_df["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    output_df.to_csv(csv_path, index=False)
    return csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build social_trends.csv from Alpha Vantage NEWS_SENTIMENT.")
    parser.add_argument("--symbols", help="Comma-separated symbols. Defaults to configured universe.")
    parser.add_argument("--output", help="Output CSV path. Defaults to ALPHA_VANTAGE_NEWS_CSV.")
    parser.add_argument("--lookback-days", type=int, help="News lookback window in calendar days.")
    parser.add_argument("--limit", type=int, help="Alpha Vantage articles per symbol.")
    parser.add_argument("--max-symbols", type=int, help="Maximum symbols to request.")
    parser.add_argument("--delay", type=float, help="Delay between requests in seconds.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    config = get_config()
    args = _parse_args()

    symbols = (
        [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
        if args.symbols
        else config.symbols
    )
    output = args.output or config.alpha_vantage_news_csv
    lookback_days = args.lookback_days or config.alpha_vantage_news_lookback_days
    limit = args.limit or config.alpha_vantage_news_limit
    max_symbols = args.max_symbols or config.alpha_vantage_max_symbols
    delay = config.alpha_vantage_request_delay_seconds if args.delay is None else args.delay

    df = collect_alpha_vantage_news(
        config.alpha_vantage_api_key,
        symbols,
        lookback_days=lookback_days,
        limit=limit,
        max_symbols=max_symbols,
        request_delay_seconds=delay,
    )
    csv_path = write_social_trends_csv(df, output)
    logger.info("Wrote %s social trend rows for %s symbols to %s", len(df), min(len(symbols), max_symbols), csv_path)


if __name__ == "__main__":
    main()
