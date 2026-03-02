"""Bollinger Bands indicator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def calculate_bollinger_bands(df, window=20, std_dev=2):
    """
    Calculate Bollinger Bands (Middle, Upper, Lower, Width).

    Args:
        df: OHLCV DataFrame with 'Adj Close' column.
        window: Simple moving average window (default 20).
        std_dev: Number of standard deviations for bands (default 2).

    Returns:
        pd.DataFrame: DataFrame with 'BB_Middle', 'BB_Upper', 'BB_Lower',
                       'BB_Width' columns added.
    """
    df = df.copy()
    sma = df['Adj Close'].rolling(window=window).mean()
    std = df['Adj Close'].rolling(window=window).std()

    df['BB_Middle'] = sma
    df['BB_Upper'] = sma + (std_dev * std)
    df['BB_Lower'] = sma - (std_dev * std)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    return df


if __name__ == '__main__':
    from utils.data import fetch_ohlcv_data

    tickers = ['AAPL', 'MSFT', 'GOOG']
    data = fetch_ohlcv_data(tickers, period='1mo', interval='5m')
    for ticker in data:
        data[ticker] = calculate_bollinger_bands(data[ticker])
    print(data['AAPL'][['Adj Close', 'BB_Upper', 'BB_Lower']].tail())
