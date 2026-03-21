"""
Signal Orthogonalization
Ensures newly discovered signals are statistically uncorrelated to the existing Factor Zoo
to maximize portfolio diversification and prevent factor crowding before deploying capital.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

def orthogonalize_signal(new_signal: pd.Series, factor_exposures: pd.DataFrame) -> pd.Series:
    """
    Purges known factor betas from a new signal using OLS regression residuals.
    
    Args:
        new_signal (pd.Series): The candidate alpha signal scores.
        factor_exposures (pd.DataFrame): Known factors (e.g. Momentum, Volatility) sharing index.
        
    Returns:
        pd.Series: The decorrelated (orthogonal) signal holding uniquely new information.
    """
    warnings.filterwarnings('ignore')
    
    df = pd.concat([new_signal.rename('signal'), factor_exposures], axis=1).dropna()
    if df.empty:
        return new_signal
        
    y = df['signal']
    X = df.drop(columns=['signal'])
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    
    # The orthogonal signal is the unexplained residual from the regression
    residuals = y - model.predict(X)
    
    # Realign index exactly as input to prevent length misalignment biases
    ortho_signal = pd.Series(index=new_signal.index, dtype=float)
    ortho_signal.loc[residuals.index] = residuals
    
    return ortho_signal.fillna(0)
