from __future__ import annotations

import pandas as pd
import numpy as np

def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD, Signal line, and Histogram."""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }, index=df.index)

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def momentum(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Compute simple momentum (percentage change)."""
    return df['close'].pct_change(periods=lookback)

def z_score(df: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling Z-score."""
    return (df - df.rolling(window=window).mean()) / df.rolling(window=window).std()
