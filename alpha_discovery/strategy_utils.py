"""Shared utility functions for trading strategies."""

import pandas as pd
import numpy as np

def align_indicator_data(df: pd.DataFrame, indicator_df: pd.DataFrame, merge_col: str, fill_method: str = 'ffill') -> pd.DataFrame:
    """
    Safely merge an indicator DataFrame (like Renko or daily metrics) into an intraday OHLCV DataFrame.
    """
    df = df.copy()
    
    # Ensure timezone naive datetime
    if 'Date' not in df.columns:
        df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not indicator_df.empty:
        if 'date' in indicator_df.columns:
            indicator_df['date'] = pd.to_datetime(indicator_df['date']).dt.tz_localize(None)
            indicator_df.rename(columns={'date': 'Date'}, inplace=True)
            
        merged = df.merge(indicator_df[['Date', merge_col]], how='outer', on='Date')
    else:
        merged = df.copy()
        merged[merge_col] = np.nan

    if fill_method == 'ffill':
        merged[merge_col] = merged[merge_col].ffill()
        
    return merged

def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize OHLCV columns and set Date index."""
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            if col == 'Close' and 'Adj Close' in df.columns:
                df['Close'] = df['Close']
            elif col == 'Open' and 'Close' in df.columns:
                df['Open'] = df['Close']

    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
        df.index.name = None
        
    return df
