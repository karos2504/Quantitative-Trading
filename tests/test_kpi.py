import pytest
import numpy as np
import pandas as pd
from portfolio_construction.kpi import cagr_from_returns, volatility, sharpe_ratio, sortino_ratio, max_drawdown

@pytest.fixture
def mock_returns():
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

def test_cagr_from_returns(mock_returns):
    cagr = cagr_from_returns(mock_returns, periods_per_year=252)
    assert isinstance(cagr, float)
    # 5 days
    # cumulative return = 1.01 * 0.98 * 1.03 * 0.99 * 1.02 = 1.0294
    # years = 5 / 252 = 0.0198
    # cagr approx > 0
    assert cagr > 0

def test_volatility(mock_returns):
    vol = volatility(mock_returns, periods_per_year=252)
    assert isinstance(vol, float)
    assert vol > 0

def test_sharpe_ratio(mock_returns):
    sharpe = sharpe_ratio(mock_returns, risk_free_rate=0.02, periods_per_year=252)
    assert isinstance(sharpe, float)

def test_sortino_ratio(mock_returns):
    sortino = sortino_ratio(mock_returns, risk_free_rate=0.02, periods_per_year=252)
    assert isinstance(sortino, float)

def test_max_drawdown():
    returns = pd.Series([0.1, -0.1, -0.1, 0.1])
    # cum: 1.1, 0.99, 0.891, 0.9801
    # peak: 1.1 -> 0.891 drawdown: (0.891 - 1.1) / 1.1 = -0.19
    dd = max_drawdown(returns)
    assert isinstance(dd, float)
    assert dd < 0

def test_empty_returns():
    empty_returns = pd.Series([], dtype=float)
    assert cagr_from_returns(empty_returns, 252) == 0.0
