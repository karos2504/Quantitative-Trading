"""Unified KPI (Key Performance Indicator) functions for trading strategies."""

import numpy as np
import pandas as pd


def _to_scalar(val):
    """Extract a Python float from potential Series/numpy scalar."""
    if hasattr(val, 'item'):
        return val.item()
    if isinstance(val, pd.Series):
        return float(val.squeeze())
    return float(val)


def cagr_from_prices(df, periods_per_year, price_col='Adj Close'):
    """
    Compound Annual Growth Rate from a price DataFrame.

    Args:
        df: DataFrame with a price column.
        periods_per_year: Number of data periods in a year (e.g. 252 for daily).
        price_col: Column name for prices (default 'Adj Close').

    Returns:
        float: CAGR value.
    """
    prices = df[price_col]
    start = _to_scalar(prices.iloc[0])
    end = _to_scalar(prices.iloc[-1])
    years = len(df) / periods_per_year
    return (end / start) ** (1 / years) - 1


def cagr_from_returns(returns, periods_per_year):
    """
    Compound Annual Growth Rate from a returns Series.

    Args:
        returns: pd.Series of periodic returns.
        periods_per_year: Number of data periods in a year.

    Returns:
        float: CAGR value.
    """
    cumulative = _to_scalar((1 + returns).prod())
    years = len(returns) / periods_per_year
    if years == 0:
        return 0.0
    return cumulative ** (1 / years) - 1


def volatility(returns, periods_per_year):
    """
    Annualized volatility from a returns Series.

    Args:
        returns: pd.Series of periodic returns.
        periods_per_year: Number of data periods in a year.

    Returns:
        float: Annualized volatility.
    """
    return _to_scalar(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns, risk_free_rate, periods_per_year):
    """
    Annualized Sharpe ratio.

    Args:
        returns: pd.Series of periodic returns.
        risk_free_rate: Annual risk-free rate (e.g. 0.025 for 2.5%).
        periods_per_year: Number of data periods in a year.

    Returns:
        float: Sharpe ratio, or NaN if volatility is zero.
    """
    cagr = cagr_from_returns(returns, periods_per_year)
    vol = volatility(returns, periods_per_year)
    return (cagr - risk_free_rate) / vol if vol != 0 else np.nan


def sortino_ratio(returns, risk_free_rate, periods_per_year):
    """
    Annualized Sortino ratio (penalizes only downside volatility).

    Args:
        returns: pd.Series of periodic returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of data periods in a year.

    Returns:
        float: Sortino ratio.
    """
    cagr = cagr_from_returns(returns, periods_per_year)
    downside = returns[returns < 0]
    downside_std = _to_scalar(downside.std() * np.sqrt(periods_per_year))
    if downside_std == 0:
        return np.nan
    return (cagr - risk_free_rate) / downside_std


def max_drawdown(returns):
    """
    Maximum drawdown from a returns Series.

    Args:
        returns: pd.Series of periodic returns.

    Returns:
        float: Maximum drawdown (negative value).
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return _to_scalar(drawdown.min())


def max_drawdown_from_prices(df, price_col='Adj Close'):
    """
    Maximum drawdown from a price DataFrame.

    Args:
        df: DataFrame with a price column.
        price_col: Column name for prices (default 'Adj Close').

    Returns:
        float: Maximum drawdown (negative value).
    """
    prices = df[price_col]
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return _to_scalar(drawdown.min())


def calmar_ratio(returns, periods_per_year):
    """
    Calmar ratio: CAGR / |Max Drawdown|.

    Args:
        returns: pd.Series of periodic returns.
        periods_per_year: Number of data periods in a year.

    Returns:
        float: Calmar ratio.
    """
    cagr = cagr_from_returns(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    return cagr / mdd if mdd != 0 else np.nan

