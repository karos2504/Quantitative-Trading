"""Slope (linear regression angle) indicator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import statsmodels.api as sm


def calculate_slope(series, n):
    """
    Calculate the slope (angle in degrees) of a rolling linear regression
    on a normalized price series over `n` consecutive points.

    Args:
        series: pd.Series of values (e.g. prices or indicator values).
        n: Number of consecutive points for the regression window.

    Returns:
        np.ndarray: Array of slope angles in degrees, same length as input.
    """
    values = series.values if hasattr(series, 'values') else np.array(series)
    slopes = [0.0] * (n - 1)

    for i in range(n, len(values) + 1):
        y = values[i - n:i]
        x = np.arange(n, dtype=float)

        y_range = y.max() - y.min()
        x_range = x.max() - x.min()

        y_scaled = (y - y.min()) / y_range if y_range != 0 else np.zeros(n)
        x_scaled = (x - x.min()) / x_range if x_range != 0 else np.zeros(n)

        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])

    return np.rad2deg(np.arctan(np.array(slopes)))


if __name__ == '__main__':
    import datetime as dt
    import matplotlib.pyplot as plt
    from utils.data import fetch_ohlcv_data

    start = dt.datetime.today() - dt.timedelta(days=365)
    end = dt.datetime.today()
    data = fetch_ohlcv_data(['AAPL'], start=start, end=end)

    df = data['AAPL'][['Adj Close']].copy()
    df['Slope'] = calculate_slope(df['Adj Close'], n=5)

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Slope'], label='Slope Angle', color='b')
    plt.title('Slope of Stock Price (Angle over Time)')
    plt.xlabel('Date')
    plt.ylabel('Slope (Degrees)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
