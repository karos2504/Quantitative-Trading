"""
Factor Zoo (Baseline Risk Drivers)
Provides standard Fama-French style factors (Momentum, Value, Volatility, Liquidity)
to be used as the basis for decorrelating and testing new alpha signals.
"""
import pandas as pd
import numpy as np

class FactorZoo:
    @staticmethod
    def compute_momentum(prices: pd.Series, lookback: int = 252) -> pd.Series:
        """Standard 12-month trailing price momentum."""
        return prices.pct_change(lookback)

    @staticmethod
    def compute_volatility(returns: pd.Series, lookback: int = 20) -> pd.Series:
        """Trailing realized volatility."""
        return returns.rolling(lookback).std()

    @staticmethod
    def compute_liquidity(volume: pd.Series, close: pd.Series, lookback: int = 20) -> pd.Series:
        """Trailing Average Daily Dollar Volume (ADV)."""
        dollar_volume = volume * close
        return dollar_volume.rolling(lookback).mean()
