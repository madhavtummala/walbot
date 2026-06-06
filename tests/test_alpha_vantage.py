from __future__ import annotations

from src.connectors.sentiment.alpha_vantage import normalize_news_sentiment


def test_normalize_news_sentiment_aggregates_daily_rows() -> None:
    payload = {
        "feed": [
            {
                "time_published": "20260516T120000",
                "overall_sentiment_score": "0.10",
                "ticker_sentiment": [
                    {"ticker": "SPY", "ticker_sentiment_score": "0.25", "relevance_score": "0.80"}
                ],
            },
            {
                "time_published": "20260516T150000",
                "overall_sentiment_score": "-0.20",
                "ticker_sentiment": [
                    {"ticker": "SPY", "ticker_sentiment_score": "0.50", "relevance_score": "0.20"}
                ],
            },
            {
                "time_published": "20260517T090000",
                "overall_sentiment_score": "-0.30",
                "ticker_sentiment": [],
            },
        ]
    }

    df = normalize_news_sentiment("SPY", payload)

    assert list(df["symbol"]) == ["SPY", "SPY"]
    assert list(df["mentions"]) == [2.0, 1.0]
    assert round(float(df["sentiment"].iloc[0]), 6) == 0.3
    assert round(float(df["social_score"].iloc[1]), 6) == -0.3
