"""Volume-Weighted Average Price (VWAP) indicator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def calculate_vwap(df, session_col=None):
    """
    Calculate VWAP with upper/lower bands (±1 std dev from VWAP).

    For intraday data the VWAP resets each trading session (day).
    For daily data the VWAP is computed as a cumulative running value.

    Args:
        df: OHLCV DataFrame with 'High', 'Low', 'Close', 'Volume' columns.
        session_col: Optional column name identifying sessions.
                     If None, sessions are derived from the date portion of the index.

    Returns:
        pd.DataFrame: Original DataFrame with 'VWAP', 'VWAP_Upper', 'VWAP_Lower' added.
    """
    df = df.copy()

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    tp_vol = typical_price * df['Volume']

    # Determine session boundaries
    if session_col and session_col in df.columns:
        session = df[session_col]
    elif hasattr(df.index, 'date'):
        session = pd.Series(df.index.date, index=df.index)
    else:
        # Fallback: treat entire dataset as one session
        session = pd.Series(0, index=df.index)

    cum_tp_vol = tp_vol.groupby(session).cumsum()
    cum_vol = df['Volume'].groupby(session).cumsum().replace(0, np.nan)

    df['VWAP'] = cum_tp_vol / cum_vol

    # Standard deviation bands
    cum_tp2_vol = (typical_price ** 2 * df['Volume']).groupby(session).cumsum()
    variance = (cum_tp2_vol / cum_vol) - (df['VWAP'] ** 2)
    variance = variance.clip(lower=0)
    std = np.sqrt(variance)

    df['VWAP_Upper'] = df['VWAP'] + std
    df['VWAP_Lower'] = df['VWAP'] - std

    return df


if __name__ == '__main__':
    from data_ingestion.data import fetch_ohlcv_data

    data = fetch_ohlcv_data(['AAPL'], period='5d', interval='1h')
    df = calculate_vwap(data['AAPL'])
    print(df[['Close', 'VWAP', 'VWAP_Upper', 'VWAP_Lower']].tail(10))
