"""Average Directional Index (ADX) indicator."""

import numpy as np
from indicators.atr import calculate_atr


def calculate_adx(df, period=20):
    """
    Calculate Average Directional Index (ADX).

    Args:
        df: OHLCV DataFrame with 'High', 'Low', 'Close' columns.
        period: Lookback period (default 20).

    Returns:
        pd.DataFrame: Original DataFrame with 'ADX' column added.
    """
    df = calculate_atr(df, period)

    up_move = df['High'].diff()
    down_move = df['Low'].diff()

    positive_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    negative_dm = np.where((up_move < down_move) & (down_move > 0), down_move, 0)

    positive_di = 100 * (positive_dm / df['ATR']).ewm(span=period, min_periods=period).mean()
    negative_di = 100 * (negative_dm / df['ATR']).ewm(span=period, min_periods=period).mean()

    adx = 100 * abs((positive_di - negative_di) / (positive_di + negative_di)).ewm(
        span=period, min_periods=period
    ).mean()
    df['ADX'] = adx.iloc[:, 0]
    return df


if __name__ == '__main__':
    from utils.data import fetch_ohlcv_data

    tickers = ['AAPL', 'MSFT', 'GOOG']
    data = fetch_ohlcv_data(tickers, period='1mo', interval='5m')
    for ticker in data:
        data[ticker] = calculate_adx(data[ticker])
    print(data['AAPL'].tail())
