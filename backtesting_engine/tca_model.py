"""
Transaction Cost Analysis (TCA) Model
Provides execution modeling for backtests to simulate Adverse Selection,
Limit Order queue positioning, and Almgren-Chriss market impact.
"""
import numpy as np
import pandas as pd

class TCAModel:
    def __init__(self, adv: float, daily_volatility: float, participation_rate: float = 0.05):
        """
        Initializes the execution physics simulator.
        adv: Average Daily Volume of the security.
        daily_volatility: The realized standard deviation of returns.
        participation_rate: The trader's target execution pace.
        """
        self.adv = adv
        self.daily_vol = daily_volatility
        self.participation_rate = participation_rate

    def estimate_market_impact(self, trade_size: float) -> float:
        """
        Almgren-Chriss simplified market impact model.
        Returns the expected frictional cost (slippage) as a percentage of price.
        """
        if self.adv <= 0:
            return 0.0
        # Temporary impact roughly proportional to sqrt(trade_size / ADV)
        impact = self.daily_vol * np.sqrt(abs(trade_size) / self.adv) * self.participation_rate
        return float(impact)

    def estimate_limit_order_fill_prob(self, limit_price: float, current_price: float, side: str) -> float:
        """
        Estimates the probability of a limit order being filled based on adverse selection 
        distance from the mid-price.
        """
        distance_bps = abs(limit_price - current_price) / current_price * 10000
        
        # Exponential decay: further away from mid, exponentially less likely to fill
        # Calibration depends on volatility, defaulting to a generic decay constant.
        decay_constant = 0.8 
        
        prob = np.exp(-decay_constant * distance_bps)
        return min(max(prob, 0.0), 1.0)
