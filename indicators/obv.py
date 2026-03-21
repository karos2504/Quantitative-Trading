"""On-Balance Volume (OBV) indicator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def calculate_obv(df):
    """
    Calculate On-Balance Volume (OBV).

    Args:
        df: OHLCV DataFrame with 'Adj Close' and 'Volume' columns.

    Returns:
        pd.DataFrame: Original DataFrame with 'OBV' column added.
    """
    df = df.copy()
    price_col = 'Close' if 'Close' in df.columns else 'Adj Close'
    returns = df[price_col].pct_change()

    direction = np.where(returns > 0, 1, np.where(returns < 0, -1, 0))
    direction[0] = 0

    vol_adj = df['Volume'] * direction
    df['OBV'] = vol_adj.cumsum()
    return df


if __name__ == '__main__':
    import datetime
    from data_ingestion.data import fetch_ohlcv_data

    start = datetime.date.today() - datetime.timedelta(days=365)
    end = datetime.date.today()
    data = fetch_ohlcv_data(['AAPL'], start=start, end=end)
    result = calculate_obv(data['AAPL'])
    print(result[['Adj Close', 'Volume', 'OBV']].tail())
