"""Renko chart conversion utilities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from indicators.atr import calculate_atr


def convert_to_renko(df, atr_period=120):
    """
    Convert OHLC data into Renko bricks using ATR-based brick size.

    This is a pure-Python implementation that does not depend on the
    `stocktrends` library.

    Args:
        df: OHLCV DataFrame with 'High', 'Low', 'Close', 'ATR' columns.
            Must have a DatetimeIndex.
        atr_period: Period for ATR calculation (default 120).

    Returns:
        pd.DataFrame: Renko DataFrame with 'date', 'close', 'uptrend', 'bar_num'.
    """
    df_atr = calculate_atr(df, period=atr_period)

    try:
        brick_size = max(0.5, round(float(df_atr['ATR'].iloc[-1]), 0))
    except (IndexError, KeyError):
        brick_size = 0.5

    prices = df['Close'].to_numpy().ravel()
    dates = df.index.to_numpy()

    renko_dates, renko_closes, renko_uptrend = [], [], []
    current_close = float(prices[0])

    for i in range(1, len(prices)):
        change = float(prices[i]) - current_close
        if abs(change) >= brick_size:
            num_bricks = int(change / brick_size)
            trend = num_bricks > 0
            for _ in range(abs(num_bricks)):
                current_close += np.sign(num_bricks) * brick_size
                renko_dates.append(dates[i])
                renko_closes.append(current_close)
                renko_uptrend.append(trend)

    renko_df = pd.DataFrame({
        'date': renko_dates,
        'close': renko_closes,
        'uptrend': renko_uptrend,
    })

    if renko_df.empty:
        return pd.DataFrame(columns=['date', 'uptrend', 'bar_num'])

    # Calculate bar_num (consecutive bricks in same direction)
    renko_df['bar_num'] = np.where(renko_df['uptrend'], 1, -1)
    for i in range(1, len(renko_df)):
        same_trend = (renko_df['bar_num'].iloc[i] > 0) == (renko_df['bar_num'].iloc[i - 1] > 0)
        if same_trend:
            renko_df.iloc[i, renko_df.columns.get_loc('bar_num')] += renko_df['bar_num'].iloc[i - 1]

    renko_df.drop_duplicates(subset='date', keep='last', inplace=True)
    renko_df = renko_df[['date', 'bar_num']]
    return renko_df


if __name__ == '__main__':
    from data_ingestion.data import fetch_ohlcv_data

    tickers = ['AAPL']
    data = fetch_ohlcv_data(tickers, period='1mo', interval='5m')
    for ticker in data:
        renko = convert_to_renko(data[ticker])
        print(f"\nRenko data for {ticker}:")
        print(renko.tail(10))
