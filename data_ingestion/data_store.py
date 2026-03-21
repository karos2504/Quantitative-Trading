"""
Centralized Data Store for Quantitative Research Pipeline

Handles fetching, storing, and retrieving OHLCV data to ensure
point-in-time correctness and eliminate redundant API calls.
Supports Parquet storage for high-performance reading/writing.
"""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _get_filename(ticker: str, interval: str) -> Path:
    return DATA_DIR / f"{ticker}_{interval}.parquet"

def update_universe_data(tickers: list, interval: str = "1h", period: str = "730d", start=None, end=None, force: bool = False):
    """
    Downloads and caches data for a list of tickers.
    If data already exists and force is False, skips downloading.
    """
    updated_count = 0
    print(f"--- Updating {interval} Data Store ---")
    
    for ticker in tickers:
        file_path = _get_filename(ticker, interval)
        
        if file_path.exists() and not force:
            continue
            
        try:
            if start is not None and end is not None:
                data = yf.download(ticker, start=start, end=end, interval=interval,
                                   progress=False, auto_adjust=True)
            else:
                data = yf.download(ticker, period=period, interval=interval,
                                   progress=False, auto_adjust=True)
                
            if data.empty:
                print(f"  ⚠️  {ticker}: No data returned.")
                continue

            # Flatten MultiIndex columns if yfinance returns them
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            if 'Adj Close' in data.columns and 'Close' not in data.columns:
                data['Close'] = data['Close']
            elif 'Close' in data.columns and 'Adj Close' not in data.columns:
                data['Close'] = data['Close']

            # Make sure we have the standard columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required_cols if c not in data.columns]
            if missing:
                print(f"  ⚠️  {ticker}: Missing columns {missing}")
                continue
                
            data = data[required_cols].dropna()
            
            # Save to Parquet
            data.to_parquet(file_path, engine="pyarrow")
            updated_count += 1
            print(f"  ✅ {ticker}: Saved {len(data)} rows to {file_path.name}")
            
        except Exception as e:
            print(f"  ❌ {ticker}: Failed to fetch/save - {e}")
            
    if updated_count == 0:
        print("  All requested data is already up to date.")
    else:
        print(f"  Updated {updated_count} files.")

def load_universe_data(tickers: list, interval: str = "1h") -> dict:
    """
    Loads requested tickers from the local Parquet data store.
    Returns a dictionary mapping ticker -> DataFrame.
    """
    data_dict = {}
    missing = []
    
    for ticker in tickers:
        file_path = _get_filename(ticker, interval)
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path, engine="pyarrow")
                data_dict[ticker] = df
            except Exception as e:
                print(f"  ❌ {ticker}: Error reading {file_path.name} - {e}")
                missing.append(ticker)
        else:
            missing.append(ticker)
            
    if missing:
        print(f"  ⚠️ Missing {interval} data for {len(missing)} tickers. Triggering update...")
        update_universe_data(missing, interval=interval)
        
        # Retry loading after update
        for ticker in missing:
            file_path = _get_filename(ticker, interval)
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path, engine="pyarrow")
                    data_dict[ticker] = df
                except Exception:
                    pass
                    
    return data_dict

def clear_data_store(interval: str = None):
    """
    Removes cached data files. If interval is provided, only removes those files.
    """
    count = 0
    pattern = f"*_{interval}.parquet" if interval else "*.parquet"
    for file_path in DATA_DIR.glob(pattern):
        file_path.unlink()
        count += 1
    print(f"Cleared {count} files from Data Store.")
