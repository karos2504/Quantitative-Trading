"""Average Directional Index (ADX) indicator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from indicators.atr import calculate_atr


def calculate_adx(df, period=20):
    """
    Calculate Average Directional Index (ADX) using Wilder's smoothing.

    Args:
        df: OHLCV DataFrame with 'High', 'Low', 'Close' columns.
        period: Lookback period (default 20).

    Returns:
        pd.DataFrame: Original DataFrame with 'ADX', '+DI', '-DI' columns added.
    """
    df = calculate_atr(df, period)

    up_move = df['High'].diff()
    down_move = -df['Low'].diff()  # negate so falling lows are positive

    positive_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    negative_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smooth +DM and -DM first, then divide by smoothed ATR (standard Wilder's)
    smooth_pdm = pd.Series(positive_dm.ravel(), index=df.index).ewm(
        alpha=1 / period, min_periods=period
    ).mean()
    smooth_ndm = pd.Series(negative_dm.ravel(), index=df.index).ewm(
        alpha=1 / period, min_periods=period
    ).mean()

    positive_di = 100 * smooth_pdm / df['ATR']
    negative_di = 100 * smooth_ndm / df['ATR']

    dx = 100 * abs(positive_di - negative_di) / (positive_di + negative_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    df['+DI'] = positive_di
    df['-DI'] = negative_di
    df['ADX'] = adx
    return df


if __name__ == '__main__':
    from utils.data import fetch_ohlcv_data

    tickers = ['AAPL', 'MSFT', 'GOOG']
    data = fetch_ohlcv_data(tickers, period='1mo', interval='5m')
    for ticker in data:
        data[ticker] = calculate_adx(data[ticker])
    print(data['AAPL'].tail())
