"""
Normalization Layer
Enforces a unified schema across all disjoint data vendors (OHLCV, Fundamentals, NLP Sentiment)
before persisting to the centralized Parquet cache.
"""
import pandas as pd
import numpy as np

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantees the existence and type-safety of standard OHLCV columns.
    """
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    # Force float32 for memory efficiency on massive historical sets
    for col in required_cols:
        df[col] = df[col].astype(np.float32)
        
    return df[required_cols]

def normalize_alternative_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Transforms specific vendor data into standardized signal streams.
    """
    pass
