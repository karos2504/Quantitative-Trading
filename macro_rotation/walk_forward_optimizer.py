"""
Walk-Forward Optimizer (Module 8)
==================================
Institutional WFO engine with Deflated Sharpe Ratio (DSR) gatekeeping.
Prevents over-fitting by statistically penalizing trial volume.
"""

import logging
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import norm
from dataclasses import dataclass, field

from macro_rotation.config import SystemConfig, logger
from macro_rotation.backtester import run_backtest_with_data, precompute_backtest_data
from macro_rotation.portfolios import AbstractPortfolio, NaiveBitcoinGoldPortfolio

@dataclass
class WFOResult:
    equity_curve: pd.Series
    oos_returns: pd.Series
    param_history: pd.DataFrame
    metrics: dict

def calculate_deflated_sharpe(
    trial_sharpes: list[float],
    winning_sharpe: float,
    ann_factor: int = 365
) -> float:
    """
    Calculates the Deflated Sharpe Ratio (DSR) to penalize Multiple Testing Bias.
    Based on Lopez de Prado methodology.
    """
    N = len(trial_sharpes)
    if N <= 1:
        return winning_sharpe
        
    variance_of_sharpes = np.var(trial_sharpes)
    if variance_of_sharpes == 0:
        return 0.0
    
    # Euler-Mascheroni constant approximation for Expected Maximum Sharpe
    gamma = 0.5772
    expected_max_sr = np.sqrt(variance_of_sharpes) * (
        (1 - gamma) * norm.ppf(1 - 1/N) + gamma * norm.ppf(1 - 1/(N * np.e))
    )
    
    # The Deflated Sharpe isolates the true signal above the expected noise
    dsr = winning_sharpe - expected_max_sr
    return dsr

def generate_walk_forward_splits(
    index: pd.DatetimeIndex,
    train_days: int = 730,
    test_days: int = 180
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generates rolling In-Sample (IS) and Out-of-Sample (OOS) date bounds.
    Returns: list of (train_start, train_end, test_start, test_end)
    """
    splits = []
    current_start = index[0]
    
    while True:
        train_end = current_start + pd.Timedelta(days=train_days)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days)
        
        if test_end > index[-1]:
            # Add final partial window if we have enough test data (e.g. > 30 days)
            if (index[-1] - test_start).days > 30:
                splits.append((current_start, train_end, test_start, index[-1]))
            break
            
        splits.append((current_start, train_end, test_start, test_end))
        
        # Roll forward by the test duration
        current_start = current_start + pd.Timedelta(days=test_days)
        
    return splits

def run_wfo_session(
    portfolio: AbstractPortfolio,
    data: dict,
    param_grid: dict,
    config: SystemConfig,
    train_days: int = 730,
    test_days: int = 180
) -> WFOResult:
    """
    Executes a full Walk-Forward Optimization session.
    1. Generates IS/OOS splits.
    2. Grid-searches IS for best Standard Sharpe.
    3. Validates winner with DSR gatekeeper.
    4. Runs OOS (optimized or naive fallback).
    5. Stitches results.
    """
    prices = data["prices"]
    splits = generate_walk_forward_splits(prices.index, train_days, test_days)
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    logger.info(f"\n🚀 Starting Institutional WFO Engine")
    logger.info(f"   Splits: {len(splits)} | Trials per split: {len(param_combinations)}")
    
    final_oos_rets = []
    param_history = []
    total_oos_dates = []
    
    # Cache full datasets for slicing efficiency
    fred_df = data["fred"]
    proxy_prices = data["proxy_prices"]
    sector_prices = data["sector_prices"]
    volumes = data["volumes"]

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(splits):
        logger.info(f"\n--- WINDOW {i+1}/{len(splits)}: IS ({tr_start.date()} to {tr_end.date()}) ---")
        
        # Slice Train Data
        is_prices = prices.loc[tr_start:tr_end]
        is_fred = fred_df.loc[:tr_end] # fred data can be longer for warmup
        is_proxy = proxy_prices.loc[:tr_end]
        is_sector = sector_prices.loc[:tr_end]
        is_volumes = volumes.loc[tr_start:tr_end]
        
        # Precompute deterministic IS data ONLY ONCE per fold
        is_precomputed = precompute_backtest_data(
            is_prices, is_fred, is_proxy, is_sector, config
        )
        
        trial_sharpes = []
        trial_results = []
        
        # 1. IS GRID SEARCH
        for j, params in enumerate(param_combinations):
            # logger.info(f"   Trial {j+1}/{len(param_combinations)}: {params}")
            trial_config = config.clone_with(**params)
            
            res = run_backtest_with_data(
                portfolio, is_prices, is_volumes, is_precomputed, trial_config
            )
            
            sr = res["metrics"]["Sharpe"]
            trial_sharpes.append(sr)
            trial_results.append((params, sr))
        
        # 2. SELECT WINNER & COMPUTE DSR
        best_trial = max(trial_results, key=lambda x: x[1])
        best_params, winner_sr = best_trial
        
        dsr = calculate_deflated_sharpe(trial_sharpes, winner_sr)
        logger.info(f"   🏆 IS Winner: SR={winner_sr:.3f}, DSR={dsr:.3f}")
        
        # 3. RUN OOS (Optimized vs Naive Fallback)
        oos_prices = prices.loc[te_start:te_end]
        oos_volumes = volumes.loc[te_start:te_end]
        
        # We need a slightly larger warm-up for OOS to ensure indicators are fresh? 
        # Actually, our precomputed data should handle the full historical range leading up to the test window.
        # So we slice the full precomputed data for the OOS range.
        
        # Precompute full dataset if not done yet
        if not hasattr(run_wfo_session, "full_precomputed"):
            logger.info("   📥 Pre-computing full dataset for OOS efficiency...")
            run_wfo_session.full_precomputed = precompute_backtest_data(
                prices, fred_df, proxy_prices, sector_prices, config
            )
            
        # Helper to slice precomputed data for OOS window
        def slice_precomputed(full_pre, start, end):
            sliced = {}
            for k, v in full_pre.items():
                if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                    sliced[k] = v.loc[start:end]
                elif isinstance(v, dict): # indicator_dfs
                    sliced[k] = {name: df.loc[start:end] for name, df in v.items()}
            return sliced

        oos_pre = slice_precomputed(run_wfo_session.full_precomputed, te_start, te_end)
        
        if dsr > 0:
            logger.info(f"   ✅ DSR Gate Passed. Running OOS with optimized params: {best_params}")
            oos_config = config.clone_with(**best_params)
            oos_portfolio = portfolio
            execution_type = "OPTIMIZED"
        else:
            logger.info(f"   🛑 DSR Gate Failed. Falling back to NAIVE baseline for OOS.")
            oos_config = config
            oos_portfolio = NaiveBitcoinGoldPortfolio()
            execution_type = "NAIVE_FALLBACK"

        # Note: run_backtest_with_data needs full warmup history if indicators aren't ffilled?
        # Actually, indicator_dfs are precomputed for ALL dates. 
        # The backtester loop will skip warmup based on date index 0->n.
        # To avoid re-precomputing, we run the loop ON THE OOS SLICE but ensure state is handled?
        # WFO stitching usually treats each window as a fresh start (realistic for strategy deployment).
        
        oos_res = run_backtest_with_data(
            oos_portfolio, oos_prices, oos_volumes, oos_pre, oos_config, warmup_days=0
        )
        
        final_oos_rets.append(oos_res["returns"])
        total_oos_dates.extend(oos_prices.index.tolist())
        
        param_history.append({
            "test_start": te_start,
            "test_end": te_end,
            "is_sharpe": winner_sr,
            "is_dsr": dsr,
            "type": execution_type,
            **best_params
        })

    # 4. STITCH RESULTS
    full_oos_returns = pd.concat(final_oos_rets)
    full_equity = config.initial_capital * (1 + full_oos_returns).cumprod()
    
    history_df = pd.DataFrame(param_history)
    
    # Calculate OOS Metrics
    total_ret = full_equity.iloc[-1] / config.initial_capital - 1
    n_years = len(full_oos_returns) / 365
    cagr = (1+total_ret)**(1/n_years)-1 if n_years > 0 else 0
    ann_vol = full_oos_returns.std() * np.sqrt(365)
    sharpe = (full_oos_returns.mean() * 365 - config.cash_yield_apy) / ann_vol if ann_vol > 0 else 0
    max_dd = (full_equity / full_equity.cummax() - 1).min()
    
    metrics = {
        "OOS Total Return (%)": round(total_ret * 100, 2),
        "OOS CAGR (%)": round(cagr * 100, 2),
        "OOS Max Drawdown (%)": round(max_dd * 100, 2),
        "OOS Sharpe": round(sharpe, 3),
        "OOS Annual Vol (%)": round(ann_vol * 100, 2),
        "DSR Pass Rate (%)": round(len(history_df[history_df["is_dsr"] > 0]) / len(history_df) * 100, 2)
    }

    logger.info(f"\n📊 FINAL WFO RESULTS")
    logger.info(f"  {'='*50}")
    for k, v in metrics.items():
        logger.info(f"     {k:<25} {v:>12}")

    return WFOResult(
        equity_curve=full_equity,
        oos_returns=full_oos_returns,
        param_history=history_df,
        metrics=metrics
    )
