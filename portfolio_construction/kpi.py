"""Unified KPI (Key Performance Indicator) functions for trading strategies."""

import numpy as np
import pandas as pd
from typing import Union


def _to_scalar(val: Union[float, int, pd.Series, np.ndarray]) -> float:
    """Extract a Python float from potential Series/numpy scalar."""
    if hasattr(val, 'item'):
        return float(val.item())
    if isinstance(val, pd.Series):
        return float(val.squeeze())
    return float(val)


def cagr_from_prices(df: pd.DataFrame, periods_per_year: int, price_col: str = 'Close') -> float:
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


def cagr_from_returns(returns: pd.Series, periods_per_year: int) -> float:
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


def volatility(returns: pd.Series, periods_per_year: int) -> float:
    """
    Annualized volatility from a returns Series.

    Args:
        returns: pd.Series of periodic returns.
        periods_per_year: Number of data periods in a year.

    Returns:
        float: Annualized volatility.
    """
    return _to_scalar(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> float:
    """
    Annualized Sharpe ratio.

    SR = mean(excess_return) / std(excess_return) * sqrt(N)

    Args:
        returns: pd.Series of periodic returns.
        risk_free_rate: Annual risk-free rate (e.g. 0.025 for 2.5%).
        periods_per_year: Number of data periods in a year.

    Returns:
        float: Sharpe ratio, or NaN if volatility is zero.
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    mean_excess = _to_scalar(excess.mean())
    std_excess = _to_scalar(excess.std())
    if std_excess == 0:
        return np.nan
    return (mean_excess / std_excess) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> float:
    """
    Annualized Sortino ratio (penalizes only downside volatility).

    Sortino = annualized_mean_excess / annualized_downside_deviation
    Downside deviation = sqrt(mean(min(excess, 0)^2)) * sqrt(N)

    Args:
        returns: pd.Series of periodic returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of data periods in a year.

    Returns:
        float: Sortino ratio.
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    mean_excess = _to_scalar(excess.mean())
    downside = excess.clip(upper=0)
    downside_dev = _to_scalar(np.sqrt((downside ** 2).mean())) * np.sqrt(periods_per_year)
    if downside_dev == 0:
        return np.nan
    annualized_excess = mean_excess * periods_per_year
    return annualized_excess / downside_dev


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int) -> float:
    """
    Annualized Information Ratio: (Average Active Return) / (Tracking Error).
    """
    active_returns = returns - benchmark_returns.reindex(returns.index).fillna(0)
    std_active = _to_scalar(active_returns.std())
    if std_active == 0:
        return np.nan
    return (active_returns.mean() * periods_per_year) / (std_active * np.sqrt(periods_per_year))


def gain_pain_ratio(returns: pd.Series) -> float:
    """
    Gain/Pain Ratio: sum(positive returns) / |sum(negative returns)|.
    """
    pos = returns[returns > 0].sum()
    neg = abs(returns[returns < 0].sum())
    return _to_scalar(pos / neg) if neg != 0 else np.nan


def max_recovery_period(returns: pd.Series) -> int:
    """
    Calculates the maximum number of periods in a drawdown.
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    in_drawdown = (cumulative < peak).astype(int)
    
    # Calculate consecutive ones
    runs = []
    current_run = 0
    for val in in_drawdown:
        if val == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    
    return max(runs) if runs else 0


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown from a returns Series.

    Args:
        returns: pd.Series of periodic returns.

    Returns:
        float: Maximum drawdown (negative value).
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (peak - cumulative) / peak
    return _to_scalar(drawdown.max())


def max_drawdown_from_prices(df: pd.DataFrame, price_col: str = 'Close') -> float:
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
    drawdown = (peak - prices) / peak
    return _to_scalar(drawdown.max())


def calmar_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """
    Calmar ratio: CAGR / |Max Drawdown|.

    Args:
        returns: pd.Series of periodic returns.
        periods_per_year: Number of data periods in a year.

    Returns:
        float: Calmar ratio.
    """
    cagr = cagr_from_returns(returns, periods_per_year)
    mdd = max_drawdown(returns)
    return cagr / mdd if mdd != 0 else np.nan

