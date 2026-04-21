"""
Quantitative Utilities (Module 7)
=================================
Advanced mathematical models for risk parity, geometric growth optimization,
and macro forecasting.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import List, Dict

from macro_rotation.config import logger


# ============================================================================
# 1. RISK PARITY OPTIMIZER
# ============================================================================
def optimize_risk_parity(
    cov_matrix: pd.DataFrame,
    target_total_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Find weights that equalize risk contribution from each asset.
    Minimizes the dispersion of Risk Contributions (RC).
    
    Target: sum(w) <= target_total_weight, w_i >= 0.
    """
    n = len(cov_matrix)
    if n == 0:
        return {}
    if n == 1:
        return {cov_matrix.columns[0]: target_total_weight}

    assets = cov_matrix.columns.tolist()
    sigma = cov_matrix.values

    # Objective function: sum of squared differences of risk contributions
    def objective(w):
        # Portfolio variance
        port_var = w.T @ sigma @ w
        # Marginal contribution to risk (MCR)
        mcr = (sigma @ w) / np.sqrt(port_var)
        # Risk contribution (RC)
        rc = w * mcr
        # Target RC is 1/N of total risk
        target_rc = np.sqrt(port_var) / n
        return np.sum(np.square(rc - target_rc))

    # Initial guess: equal weights
    w0 = np.ones(n) / n * target_total_weight
    
    # Constraints: sum(w) <= target_total_weight
    cons = ({'type': 'ineq', 'fun': lambda w: target_total_weight - np.sum(w)})
    # Bounds: w_i >= 0
    bounds = [(0, target_total_weight) for _ in range(n)]

    res = minimize(
        objective, 
        w0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        options={'ftol': 1e-10}
    )

    if not res.success:
        logger.warning(f"  ⚠️ Risk Parity optimization failed: {res.message}")
        # Fallback to equal weight
        return {a: 1.0/n * target_total_weight for a in assets}

    return {assets[i]: res.x[i] for i in range(n)}


# ============================================================================
# 2. KELLY CRITERION SIZING
# ============================================================================
def calculate_kelly_fraction(
    mu: float,           # Expected excess return (Daily)
    variance: float,     # Expected variance (Daily)
    risk_free_rate: float, # Daily risk free rate
    max_fraction: float = 0.5  # Default to Half-Kelly for safety
) -> float:
    """
    Continuous Kelly Criterion sizing formula: f* = (mu - r) / sigma^2.
    Determines the fraction of capital to allocate to maximize geometric growth.
    """
    if variance <= 1e-9:
        return 0.0
    
    # Raw Kelly fraction
    f_star = (mu - risk_free_rate) / variance
    
    # Apply safety cap and non-negativity
    return np.clip(f_star, 0, max_fraction)


def calculate_ewma_stats(
    returns: pd.Series,
    span: int = 60
) -> tuple[float, float]:
    """
    Calculate EWMA mean and variance for a returns series.
    Used for responsive Kelly sizing.
    """
    if len(returns) < 2:
        return 0.0, 0.0
        
    ewm = returns.ewm(span=span, adjust=True)
    mu = ewm.mean().iloc[-1]
    var = ewm.var().iloc[-1]
    
    return float(mu), float(var)


# ============================================================================
# 3. HOLT-WINTERS FORECASTING
# ============================================================================
def apply_holt_winters(
    series: pd.Series, 
    forecast_periods: int = 3,
    fit_window_years: int = 5
) -> float:
    """
    Apply Triple Exponential Smoothing (Holt-Winters) to forecast time series.
    Optimized for macro data by resampling to monthly frequency before fitting.
    """
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    
    # Pre-process: Resample to monthly (macro data frequency) to speed up fitting
    # Daily forward-filled macro data contains redundant information for HW.
    fit_window = fit_window_years * 252
    sub_series = series.tail(fit_window).dropna()
    
    if len(sub_series) < 30:
        return series.iloc[-1] if not series.empty else 0.0

    # Resample to Monthly Start to reduce data points by ~20x
    sub_series_monthly = sub_series.resample('MS').last().dropna()
    
    if len(sub_series_monthly) < 12: # Need at least a year of monthly data
        return series.iloc[-1]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            
            model = ExponentialSmoothing(
                sub_series_monthly,
                trend='add',
                seasonal=None,
                initialization_method="estimated"
            )
            res = model.fit(disp=False)
            forecast = res.forecast(forecast_periods)
            return float(forecast.iloc[-1])
    except Exception:
        return series.iloc[-1]
