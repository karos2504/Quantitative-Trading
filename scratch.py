import pandas as pd
import numpy as np

# Test imports
try:
    from macro_rotation.config import SystemConfig, CONFIG
    from macro_rotation.data_loader import load_all_data
    from macro_rotation.portfolios import CryptoGoldRotation, CoreAssetMacroRotation
    from macro_rotation.backtester import run_backtest
    from strategies.rebalance_portfolio import CONFIG as REBAL_CONFIG, UNIVERSE, build_prices, compute_returns
except Exception as e:
    print(e)
