"""
Hidden Markov Model (HMM) Regime Detection
Identifies latent market states (e.g., Bull, Bear, Chop) from returns and volatility.
"""

import numpy as np
import pandas as pd
import warnings
from hmmlearn.hmm import GaussianHMM

def fit_hmm_regimes(prices: pd.Series, n_components: int = 2) -> pd.Series:
    """
    Fits a Gaussian HMM to the price series returns and volatility.
    Returns a pandas Series of the same index with regime labels (integers 0 to n_components-1).
    """
    warnings.filterwarnings("ignore")
    
    # Calculate features
    returns = np.log(prices / prices.shift(1)).dropna()
    volatility = returns.rolling(window=10).std().dropna()
    
    # Align features
    aligned_rets, aligned_vol = returns.align(volatility, join='inner')
    if len(aligned_rets) < 20:
        raise ValueError("Insufficient data for HMM fitting.")
        
    X = np.column_stack([aligned_rets.values, aligned_vol.values])
    
    # Fit HMM
    hmm_model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
    hmm_model.fit(X)
    hidden_states = hmm_model.predict(X)
    
    # Create output series with matching index
    regime_series = pd.Series(index=prices.index, dtype=float)
    regime_series.loc[aligned_rets.index] = hidden_states
    
    # Forward fill gaps and fill initial warm-up period with the first known regime
    return regime_series.ffill().bfill().astype(int)
