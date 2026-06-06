from __future__ import annotations

import pandas as pd

from src.data.social import load_social_trends_csv


def test_load_social_trends_csv_normalizes_vendor_columns(tmp_path) -> None:
    csv_path = tmp_path / "social.csv"
    pd.DataFrame(
        {
            "ticker": ["aaa", "AAA", "bbb"],
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "mention_count": [10, 5, 7],
            "sentiment_score": [0.2, 0.6, -0.1],
        }
    ).to_csv(csv_path, index=False)

    trends = load_social_trends_csv(str(csv_path), ["AAA"])

    assert list(trends.keys()) == ["AAA"]
    assert trends["AAA"]["mentions"].iloc[0] == 15
    assert round(trends["AAA"]["sentiment"].iloc[0], 6) == 0.4
