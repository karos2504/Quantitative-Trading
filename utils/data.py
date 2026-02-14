"""Shared data fetching utilities for the Quantitative Trading project."""

import yfinance as yf


def fetch_ohlcv_data(tickers, period=None, interval='1d', start=None, end=None,
                     auto_adjust=False, progress=False):
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
    data = {}
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
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")
    return data


def fetch_financial_data(ticker):
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
