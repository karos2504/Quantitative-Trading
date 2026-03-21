import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def calculate_matrix_ols_slope(series: pd.Series, window: int = 40) -> pd.Series:
    """Calculates rolling OLS slope using the closed-form matrix inverse."""
    X = np.vstack([np.ones(window), np.arange(window)]).T
    P = np.linalg.inv(X.T @ X) @ X.T
    slope_weights = P[1]
    return series.rolling(window).apply(lambda y: np.dot(y, slope_weights), raw=True)

def calculate_hw_trend(series: pd.Series, window: int = 120, seasonal_periods: int = 24) -> pd.Series:
    """Rolling Holt-Winters trend extraction."""
    def fit_hw(y):
        try:
            model = ExponentialSmoothing(
                y, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=seasonal_periods, 
                initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
            return fit.trend[-1]
        except:
            return np.nan
    return series.rolling(window).apply(fit_hw, raw=True)

def renko_momentum(bar_series: pd.Series, halflife: int = 5) -> pd.Series:
    """Continuous Renko momentum: exponentially-weighted sum of bar_num deltas."""
    delta = bar_series.diff().fillna(0)
    alpha = 1 - np.exp(-np.log(2) / halflife)
    return delta.ewm(alpha=alpha, adjust=False).mean()
