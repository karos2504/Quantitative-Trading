import pytest
import numpy as np
import pandas as pd
from portfolio_construction import kpi

@pytest.fixture
def sample_returns():
    # Mix of positive and negative returns
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.015, 0.005, -0.005, 0.01, 0.015])

def test_gain_pain_ratio(sample_returns):
    ratio = kpi.gain_pain_ratio(sample_returns)
    pos = sample_returns[sample_returns > 0].sum()
    neg = abs(sample_returns[sample_returns < 0].sum())
    expected = pos / neg
    assert pytest.approx(ratio) == expected

def test_max_recovery_period():
    # Drawdown of 2 bars: 1.0 -> 0.9 -> 0.8 -> 1.1
    returns = pd.Series([-0.1, -0.11, 0.375])
    # cum: 0.9, 0.801, 1.101375
    # peak: 0.9, 0.9, 1.101375
    # in_dd: [0, 1, 0]
    recovery = kpi.max_recovery_period(returns)
    assert recovery == 1

def test_information_ratio(sample_returns):
    bench = pd.Series([0.005] * len(sample_returns), index=sample_returns.index)
    ir = kpi.information_ratio(sample_returns, bench, periods_per_year=12)
    active = sample_returns - bench
    expected = (active.mean() * 12) / (active.std() * np.sqrt(12))
    assert pytest.approx(ir) == expected

def test_sortino_standard(sample_returns):
    # Sharpe/Sortino logic check
    sr = kpi.sharpe_ratio(sample_returns, risk_free_rate=0.04, periods_per_year=12)
    sortino = kpi.sortino_ratio(sample_returns, risk_free_rate=0.04, periods_per_year=12)
    # Sortino should be >= Sharpe if there are positive returns (skew)
    assert sortino >= sr
