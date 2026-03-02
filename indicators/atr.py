"""Average True Range (ATR) indicator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR).

    Args:
        df: OHLCV DataFrame with 'High', 'Low', 'Close' columns.
        period: Lookback period for ATR smoothing (default 14).

    Returns:
        pd.DataFrame: Original DataFrame with 'ATR' column added.
    """
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    df['ATR'] = tr.ewm(span=period, adjust=False).mean()
    return df


if __name__ == '__main__':
    from utils.data import fetch_ohlcv_data

    tickers = ['AAPL', 'MSFT', 'GOOG']
    data = fetch_ohlcv_data(tickers, period='1mo', interval='5m')
    for ticker in data:
        data[ticker] = calculate_atr(data[ticker])
    print(data['AAPL'][['Close', 'ATR']].tail())
