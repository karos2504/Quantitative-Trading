"""
Advanced Statistical Validation Tools

Includes testing for stationarity (ADF) and transforming
non-stationary series while preserving memory via Fractional Differentiation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series: pd.Series, maxlag=None, regression='c', autolag='AIC') -> dict:
    """
    Runs the Augmented Dickey-Fuller (ADF) test for stationarity.
    Returns a dictionary of the test results.
    """
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return {'is_stationary': False, 'p_value': 1.0, 'error': 'Insufficient data'}
        
    try:
        adf_result = adfuller(series_clean, maxlag=maxlag, regression=regression, autolag=autolag)
        p_value = adf_result[1]
        return {
            'statistic': adf_result[0],
            'p_value': p_value,
            'lags_used': adf_result[2],
            'n_observations': adf_result[3],
            'critical_values': adf_result[4],
            'is_stationary': p_value < 0.05
        }
    except Exception as e:
        return {'is_stationary': False, 'p_value': 1.0, 'error': str(e)}

def _get_weights_ffd(d, size, threshold):
    """
    Helper function to generate weights for Fractional Differentiation.
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < threshold:
            break
        w.append(w_)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series: pd.Series, d: float, threshold=1e-4) -> pd.Series:
    """
    Fractional Differentiation (Fixed-Window).
    Transforms a non-stationary time series into a stationary one while
    preserving the maximum amount of memory (unlike integer differencing).
    
    Args:
        series: Pandas Series (e.g., Close prices).
        d: The fractional differencing value (usually between 0 and 1).
        threshold: Weight threshold to truncate the window and avoid data loss.
    """
    w = _get_weights_ffd(d, len(series), threshold)
    width = len(w) - 1
    
    df = series.to_frame('value')
    df['frac_diff'] = np.nan
    
    # We iterate over the series. It's unoptimized Python but perfectly robust for research size
    series_vals = df['value'].values
    frac_diff_vals = np.full(len(series_vals), np.nan)
    
    for i in range(width, len(series_vals)):
        window = series_vals[i - width: i + 1]
        if not np.isnan(window).any():
            frac_diff_vals[i] = np.dot(w.T, window)[0]
            
    return pd.Series(frac_diff_vals, index=series.index)

def find_min_d_for_stationarity(series: pd.Series, threshold=1e-4, p_val_limit=0.05) -> float:
    """
    Searches for the minimum fractional differencing value `d` in [0, 1]
    that makes the series stationary (p-value < p_val_limit).
    """
    for d in np.arange(0.0, 1.05, 0.05):
        try:
            diff_series = frac_diff_ffd(series, d, threshold=threshold)
            res = test_stationarity(diff_series.dropna())
            if res.get('is_stationary') and res.get('p_value', 1.0) < p_val_limit:
                return round(d, 2)
        except Exception:
            continue
    return 1.0 # Fallback to 1st derivative
