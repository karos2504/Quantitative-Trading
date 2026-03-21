"""
Convex Optimizer for Portfolio Construction
Solves Mean-Variance problems to optimally balance risk and return across factors or sub-strategies.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class ConvexOptimizer:
    @staticmethod
    def maximize_sharpe_ratio(expected_returns: pd.Series, cov_matrix: pd.DataFrame, 
                              risk_free_rate: float = 0.0, max_weight: float = 1.0) -> pd.Series:
        """
        Uses Sequential Least Squares Programming (SLSQP) to find the portfolio allocation
        vector that maximizes the ex-ante Sharpe Ratio.
        """
        n_assets = len(expected_returns)
        
        def objective(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Maximize Sharpe = Minimize Negative Sharpe
            return -(port_return - risk_free_rate) / max(port_vol, 1e-8)

        # Standard self-financing constraint (Fully invested / sum to 100%)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Bounds: Long-only up to max_weight
        bounds = tuple((0.0, max_weight) for _ in range(n_assets))
        
        init_guess = np.ones(n_assets) / n_assets
        
        res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")
            
        return pd.Series(res.x, index=expected_returns.index)

    @staticmethod
    def minimize_volatility(cov_matrix: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
        """
        Finds the Global Minimum Variance (GMV) portfolio weights.
        """
        n_assets = len(cov_matrix)
        
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0.0, max_weight) for _ in range(n_assets))
        init_guess = np.ones(n_assets) / n_assets
        
        res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(res.x, index=cov_matrix.index)
