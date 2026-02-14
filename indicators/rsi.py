"""Relative Strength Index (RSI) indicator."""

import numpy as np
import pandas as pd


def calculate_rsi(df, period=14):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: OHLCV DataFrame with 'Adj Close' column.
        period: Lookback period (default 14).

    Returns:
        pd.DataFrame: Original DataFrame with 'RSI' column added.
    """
    df = df.copy()
    delta = df['Adj Close'].diff().squeeze()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_series = pd.Series(gain, index=df.index)
    loss_series = pd.Series(loss, index=df.index)

    avg_gain = gain_series.ewm(span=period, min_periods=period).mean()
    avg_loss = loss_series.ewm(span=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


if __name__ == '__main__':
    from utils.data import fetch_ohlcv_data

    tickers = ['AAPL', 'MSFT', 'GOOG']
    data = fetch_ohlcv_data(tickers, period='1mo', interval='5m')
    for ticker in data:
        data[ticker] = calculate_rsi(data[ticker])
    print(data['AAPL'].tail())
