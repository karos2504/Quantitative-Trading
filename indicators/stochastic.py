"""Stochastic Oscillator (%K, %D) indicator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def calculate_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    """
    Calculate Stochastic Oscillator (%K and %D).

    %K = SMA( (Close - Lowest Low) / (Highest High - Lowest Low) , smooth_k )
    %D = SMA(%K, d_period)

    Args:
        df: OHLCV DataFrame with 'High', 'Low', 'Close' columns.
        k_period: Lookback for highest-high / lowest-low (default 14).
        d_period: Smoothing period for %D signal line (default 3).
        smooth_k: Smoothing period for fast %K (default 3).

    Returns:
        pd.DataFrame: Original DataFrame with 'Stoch_K', 'Stoch_D' columns added.
    """
    df = df.copy()

    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()

    range_ = (high_max - low_min).replace(0, float('nan'))
    fast_k = (df['Close'] - low_min) / range_ * 100

    # Smoothed %K
    df['Stoch_K'] = fast_k.rolling(window=smooth_k).mean()
    # %D (signal line)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()

    return df


if __name__ == '__main__':
    from data_ingestion.data import fetch_ohlcv_data

    data = fetch_ohlcv_data(['AAPL'], period='1mo', interval='1h')
    df = calculate_stochastic(data['AAPL'])
    print(df[['Close', 'Stoch_K', 'Stoch_D']].tail(10))
