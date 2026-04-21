"""
Execution Modeling (Module 8)
=============================
Almgren-Chriss optimal execution trajectory and market impact modelling.
Manages the distribution of large rebalances over multiple trading sessions.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ExecutionTrajectory:
    asset: str
    total_delta: float    # Total units/weight to trade
    days_to_execute: int  # T
    schedule: List[float] # List of weight delta per day
    start_idx: int        # Backtest index when trade started

def plan_almgren_chriss_trajectory(
    total_delta: float,
    daily_vol: float,
    adv_usd: float,
    portfolio_value: float = 10_000_000,
    lambda_risk_aversion: float = 1e-6,
    max_days: int = 5
) -> List[float]:
    """
    Computes an optimal trading trajectory using a discrete-time Almgren-Chriss model.
    Balances market impact cost against price risk (variance).
    
    Returns a list of deltas to execute each day.
    """
    if np.isnan(total_delta) or abs(total_delta) < 1e-6:
        return []

    # Robustness checks for WFO/Segmented data
    if np.isnan(adv_usd) or adv_usd <= 0:
        adv_usd = 1e9
    if np.isnan(daily_vol) or daily_vol <= 0:
        daily_vol = 0.02

    # Estimate impact coefficients (standard institutional proxies)
    # Temporary impact η: how much 100% of ADV moves the price (usually ~100-500 bps)
    eta = 0.025 / adv_usd  # 2.5% impact per $1 of volume traded
    # Permanent impact γ: usually ~10% of temporary
    gamma = 0.1 * eta
    
    # Portfolio-normalized volatility
    sigma = daily_vol * portfolio_value
    
    # Calculate kappa (the liquidity-adjusted risk-aversion parameter)
    # kappa^2 approx lambda * sigma^2 / eta
    if eta > 0:
        kappa_sq = (lambda_risk_aversion * (sigma**2)) / eta
        kappa = np.sqrt(kappa_sq)
    else:
        kappa = 0.1  # Fallback
    
    # Determine execution horizon N
    # In a simple model, N is often fixed by the user or derived from ADV constraints.
    # We'll use a dynamic N based on 10% ADV limit, capped at max_days.
    trade_size_usd = abs(total_delta) * portfolio_value
    
    # Safe division and integer conversion to handle NaNs in rolling windows
    try:
        denom = (0.1 * adv_usd)
        ratio = trade_size_usd / denom if denom > 0 else 0
        
        if np.isnan(ratio) or np.isinf(ratio):
            n_days_adv_limit = 1
        else:
            n_days_adv_limit = int(np.ceil(ratio))
    except:
        n_days_adv_limit = 1
        
    N = max(1, min(max_days, n_days_adv_limit))
    
    if N == 1:
        return [total_delta]

    # Hyperbolic sine trajectory: x_k = sinh(kappa*(N-k)) / sinh(kappa*N) * X
    # x_k is quantity REMAINING at start of day k.
    # n_k is quantity traded ON day k = x_k - x_{k+1}.
    
    # To handle small kappa (linear trajectory limit)
    if kappa < 1e-4:
        return [total_delta / N] * N
        
    remaining = []
    for k in range(N + 1):
        rem = np.sinh(kappa * (N - k)) / np.sinh(kappa * N) * total_delta
        remaining.append(rem)
        
    schedule = []
    for k in range(N):
        step = remaining[k] - remaining[k+1]
        schedule.append(step)
        
    return schedule

def compute_slippage_cost(
    traded_usd: float,
    adv_usd: float,
    daily_vol: float,
) -> float:
    """
    Estimate the temporary market impact (slippage) cost for a trade.
    Cost = η * (n/ADV) * n  (simplified quadratic impact)
    """
    if adv_usd <= 0:
        return 0.0
        
    # Standard square-root law or linear impact
    # Using 50% of daily vol for a trade of 100% ADV as a proxy
    impact_bps = daily_vol * np.sqrt(abs(traded_usd) / adv_usd)
    return abs(traded_usd) * impact_bps
