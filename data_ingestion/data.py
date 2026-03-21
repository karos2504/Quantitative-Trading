"""Shared data fetching utilities for the Quantitative Trading project."""

import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Union
import datetime


def fetch_ohlcv_data(tickers: List[str], period: Optional[str] = None, interval: str = '1d', 
                     start: Optional[Union[str, datetime.datetime]] = None, 
                     end: Optional[Union[str, datetime.datetime]] = None,
                     auto_adjust: bool = False, progress: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of tickers from Yahoo Finance.

    Supports both period-based ('7mo', '1y') and date-range (start/end) fetching.
    Returns a dict mapping ticker → DataFrame.

    Args:
        tickers: List of ticker symbols.
        period: Time period string (e.g. '7mo', '1y'). Ignored if start/end given.
        interval: Data interval (e.g. '1d', '5m', '1h').
        start: Start date (datetime or string). Used with end.
        end: End date (datetime or string). Used with start.
        auto_adjust: Whether to use auto-adjusted OHLC (default False).
        progress: Show download progress bar (default False).

    Returns:
        dict[str, pd.DataFrame]: Ticker → OHLCV DataFrame.
    """
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            kwargs = {
                'interval': interval,
                'auto_adjust': auto_adjust,
                'progress': progress,
            }
            if start and end:
                kwargs['start'] = start
                kwargs['end'] = end
            elif period:
                kwargs['period'] = period

            df = yf.download(ticker, **kwargs).dropna()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")
    return data


def fetch_financial_data(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch financial statements (balance sheet, income statement, cash flow)
    and key stats for a single ticker.

    Returns:
        dict with keys: 'balance_sheet', 'income_statement', 'cash_flow', 'key_stats'.
        Returns None if fetch fails.
    """
    try:
        stock = yf.Ticker(ticker)
        return {
            'balance_sheet': stock.balance_sheet,
            'income_statement': stock.financials,
            'cash_flow': stock.cashflow,
            'key_stats': stock.info,
        }
    except Exception as e:
        print(f"Could not fetch financial data for {ticker}: {e}")
        return None
