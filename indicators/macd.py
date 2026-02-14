"""Moving Average Convergence Divergence (MACD) indicator."""


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD, Signal line, and Histogram.

    Args:
        df: OHLCV DataFrame with 'Adj Close' column.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        pd.DataFrame: Original DataFrame with 'MACD', 'Signal', 'Hist' columns added.
    """
    df = df.copy()
    ema_fast = df['Adj Close'].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df['Adj Close'].ewm(span=slow, min_periods=slow).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()
    hist = macd - signal_line

    df['MACD'] = macd
    df['Signal'] = signal_line
    df['Hist'] = hist
    return df


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.data import fetch_ohlcv_data

    tickers = ['AAPL', 'MSFT', 'GOOG']
    data = fetch_ohlcv_data(tickers, period='1mo', interval='15m')

    for ticker in data:
        data[ticker] = calculate_macd(data[ticker])
        print(f"\n{ticker} Latest Data:")
        print(data[ticker][['Adj Close', 'MACD', 'Signal', 'Hist']].tail())

        plt.figure(figsize=(12, 6))
        plt.plot(data[ticker].index, data[ticker]['MACD'], label='MACD', color='blue')
        plt.plot(data[ticker].index, data[ticker]['Signal'], label='Signal', color='red')
        plt.bar(data[ticker].index, data[ticker]['Hist'], label='Histogram', color='gray', alpha=0.4)
        plt.title(f"{ticker} MACD (15m Interval)")
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
