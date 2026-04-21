import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker:UserWarning'

# System paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 1. Macro Rotation Imports
from macro_rotation.config import SystemConfig, CONFIG as MACRO_CONFIG
from macro_rotation.data_loader import load_all_data
from macro_rotation.portfolios import CryptoGoldRotation, CoreAssetMacroRotation
from macro_rotation.backtester import run_backtest

# 2. Rebalance Portfolio Imports
from strategies.rebalance_portfolio import run_strategy as run_rebalance_strategy
from strategies.rebalance_portfolio import compute_12_1, UNIVERSE, START_DATE as REBAL_START, END_DATE as REBAL_END
from strategies.rebalance_portfolio import _load_market_data

# 3. Renko Imports
from strategies.renko_macd import RenkoMACDStrategy, DEFAULT_PARAMS as RENKO_PARAMS, _precompute_indicators as renko_precompute
from data_ingestion.data_store import load_universe_data
from backtesting import Backtest

# Globals
REPORTS_DIR = PROJECT_ROOT / "macro_rotation" / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def extract_macro_rotation_returns(data: dict) -> dict:
    """Run event-driven backtests and extract daily returns."""
    print("  [1/4] Running Macro Rotation (Crypto & Gold)...")
    res_crypto = run_backtest(
        CryptoGoldRotation(),
        data["prices"], data["fred"], data["proxy_prices"], data["sector_prices"], MACRO_CONFIG
    )
    
    print("  [2/4] Running Macro Rotation (Core Assets & VN-Index)...")
    res_core = run_backtest(
        CoreAssetMacroRotation(),
        data["prices"], data["fred"], data["proxy_prices"], data["sector_prices"], MACRO_CONFIG
    )
    
    ret_crypto = res_crypto["equity_curve"].pct_change().dropna()
    ret_core = res_core["equity_curve"].pct_change().dropna()
    
    # Convert index to UTC timezone-naive datetime directly
    ret_crypto.index = pd.to_datetime(ret_crypto.index, utc=True).tz_localize(None)
    ret_core.index = pd.to_datetime(ret_core.index, utc=True).tz_localize(None)
    
    return {"Crypto Rotation": ret_crypto, "Core Rotation": ret_core}

def extract_rebalance_portfolio() -> pd.Series:
    """Run the institutional adaptive engine and extract monthly returns mapped to daily."""
    print("  [3/4] Running Rebalance Portfolio (Institutional Adaptive)...")
    spy_price, spy_rets, prices, returns, pit_engine = _load_market_data(MACRO_CONFIG, REBAL_START, REBAL_END)
    mom_12_1 = compute_12_1(prices)
    strat_returns, _, _ = run_rebalance_strategy(prices, mom_12_1, UNIVERSE, pit_engine=pit_engine)
    
    # Needs to be daily interpolated or just use monthly points
    strat_returns.index = pd.to_datetime(strat_returns.index.to_timestamp(), utc=True).tz_localize(None)
    return strat_returns

def extract_renko_returns() -> pd.Series:
    """Run Renko MACD on BTC and downsample to daily returns."""
    print("  [4/4] Running Intraday Renko MACD (BTC)...")
    try:
        # Fetch 12h or 1h BTC data
        intraday = load_universe_data(["BTC"], interval='1h')
        if "BTC" not in intraday or intraday["BTC"].empty:
            return pd.Series(dtype=float)
            
        df = intraday["BTC"]
        proc = renko_precompute(df)
        
        # Run Backtest with default params
        bt = Backtest(proc, RenkoMACDStrategy, cash=100000, commission=0.001)
        res = bt.run(**RENKO_PARAMS)
        
        # Extract equity curve and resample to daily
        eq = res._equity_curve['Equity']
        
        # Make index timezone naive for joining
        eq.index = pd.to_datetime(eq.index, utc=True).tz_localize(None)
        daily_eq = eq.resample('D').last().ffill()
        daily_ret = daily_eq.pct_change().dropna()
        return daily_ret
    except Exception as e:
        print(f"Error running Renko MACD: {e}")
        return pd.Series(dtype=float)

def main():
    print("=" * 60)
    print("  Multi-Strategy Correlation & Ensemble Engine")
    print("=" * 60)
    
    # Load Macro Data
    print("\n[Loading Global Data...]")
    macro_data = load_all_data(MACRO_CONFIG)
    
    # 1. Gather all return streams
    returns_db = {}
    
    macro_rets = extract_macro_rotation_returns(macro_data)
    returns_db.update(macro_rets)
    
    rebal_rets = extract_rebalance_portfolio()
    returns_db["Adaptive Equities"] = rebal_rets
    
    renko_rets = extract_renko_returns()
    if not renko_rets.empty:
        returns_db["Intraday Renko (BTC)"] = renko_rets

    # 2. Build Unified Temporal Dataframe
    print("\n[Aligning Temporal DataFrames...]")
    # Resample all to daily to ensure common ground, then forward fill missing
    df_list = []
    for name, s in returns_db.items():
        s.name = name
        daily_s = s.resample('D').sum() # Sum of log returns technically, pct_change sum is fine for scale
        df_list.append(daily_s)
        
    master_df = pd.concat(df_list, axis=1).dropna()
    
    print(f"Master Temporal Matrix: {len(master_df)} identical trading days.")
    
    if master_df.empty:
        print("No intersecting dates found.")
        return

    # 3. Correlation Matrix
    corr_matrix = master_df.corr(method='pearson')
    print("\n--- Pearson Correlation Matrix ---")
    print(corr_matrix.round(3))
    
    # Visual Output
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".2f")
    plt.title("Multi-Strategy Return Correlation")
    plt.tight_layout()
    corr_path = REPORTS_DIR / "strategy_correlation_heatmap.png"
    plt.savefig(corr_path)
    print(f"\n✅ Heatmap saved to {corr_path}")
    
    # 4. Combination Logic
    print("\n[Ensemble Logic Matrix (> 0.7 Combine, < 0.7 Orthogonal Allocation)]")
    processed = set()
    combinations = []
    strats = list(corr_matrix.columns)
    
    for i in range(len(strats)):
        for j in range(i+1, len(strats)):
            s1, s2 = strats[i], strats[j]
            corr = corr_matrix.loc[s1, s2]
            
            if corr > 0.70:
                print(f"  [COMBINE]  {s1} ↔ {s2}  (corr: {corr:.3f})")
                combinations.append((s1, s2))
                processed.update([s1, s2])
            else:
                print(f"  [SEPARATE] {s1} ↔ {s2}  (corr: {corr:.3f})")
                
    for s in strats:
        if s not in processed:
            print(f"  [ORTHOGONAL KEEPER] {s} allocated natively (No high correlations).")

if __name__ == "__main__":
    main()
