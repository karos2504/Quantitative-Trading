import pandas as pd
import numpy as np
import inspect
from backtesting import Backtest
from utils.backtesting import VBTBacktester

def extract_best_params(opt_stats, default_params):
    """
    Extract the winning parameter set from backtesting.py optimization results
    dynamically using the keys of default_params.
    """
    strat = opt_stats._strategy
    params = {}
    for k, v in default_params.items():
        val = getattr(strat, k)
        # Cast to the same type as default
        params[k] = type(v)(val)
    return params

def run_strategy_pipeline(
    strategy_name: str,
    ohlcv_data: dict,
    strategy_class,
    default_params: dict,
    param_grid: dict,
    precompute_fn,
    vbt_signal_fn,
    cash=100_000,
    commission=0.001,
    freq='1h'
):
    """
    Shared pipeline for intraday strategies:
      1. Optimize parameters
      2. Final clean backtest
      3. Summary tables
      4. VBT advanced analysis (monte carlo, stress testing, etc.)
      5. ML signal enhancement (if available)
    """
    print("=" * 70)
    print(f"  {strategy_name} — Advanced Backtest")
    print("=" * 70)

    tickers = list(ohlcv_data.keys())
    if not tickers:
        print("No data provided. Exiting.")
        return

    all_stats   = {}
    best_params = {}

    for ticker in tickers:
        print(f"\n{'─' * 60}")
        print(f"📊  {ticker} — Step 1: Optimizing …")

        try:
            df = precompute_fn(ohlcv_data[ticker])
            if len(df) < 50:
                print(f"  ⚠️  Skipping {ticker}: insufficient data")
                continue

            bt = Backtest(df, strategy_class,
                          cash=cash, commission=commission,
                          exclusive_orders=True, finalize_trades=True)

            params = default_params.copy()
            try:
                opt_stats = bt.optimize(
                    **param_grid,
                    maximize='Sharpe Ratio',
                    max_tries=81 if len(param_grid) > 3 else 27,
                    return_heatmap=False,
                )

                if opt_stats['# Trades'] >= 10:
                    params = extract_best_params(opt_stats, default_params)
                    print(f"  ✅ Optimization complete. Best params: {params}")
                else:
                    print(f"  ⚠️  Only {opt_stats['# Trades']} trades found — falling back to defaults")

            except Exception as opt_err:
                print(f"  ⚠️  Optimization failed ({opt_err}) — using defaults")

            best_params[ticker] = params

            # ── Final backtest using best (or default) params
            print(f"📊  {ticker} — Step 2: Final backtest with best params …")
            final_stats = bt.run(**params)

            if final_stats['# Trades'] < 10:
                print(f"  ⚠️  {ticker}: only {final_stats['# Trades']} trades — interpret cautiously")

            stats = {
                'Return [%]':       final_stats['Return [%]'],
                'Sharpe Ratio':     final_stats['Sharpe Ratio'],
                'Max Drawdown [%]': final_stats['Max. Drawdown [%]'],
                '# Trades':         final_stats['# Trades'],
                'Win Rate [%]':     final_stats['Win Rate [%]'],
            }
            stats.update(params)
            all_stats[ticker] = stats

            print(
                f"  ✅ Return: {final_stats['Return [%]']:.2f}%  "
                f"Sharpe: {final_stats['Sharpe Ratio']:.2f}  "
                f"Max DD: {final_stats['Max. Drawdown [%]']:.2f}%  "
                f"Trades: {final_stats['# Trades']}"
            )

        except Exception as e:
            print(f"  ❌ Error for {ticker}: {e}")

    # 3. Summary tables
    if all_stats:
        print("\n" + "=" * 70)
        print("  FINAL BACKTEST RESULTS (using optimized parameters per ticker)")
        print("=" * 70)
        print(pd.DataFrame(all_stats).T.to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n" + "─" * 70)
        print("  Best Parameters Per Ticker")
        print("─" * 70)
        print(pd.DataFrame(best_params).T.to_string(float_format=lambda x: f"{x:.2f}"))

    # 4. VBT advanced analysis with best params
    for ticker in tickers:
        if ticker not in all_stats:
            continue
        print(f"\n{'=' * 70}")
        print(f"  Advanced VBT Analysis: {ticker}  (best params)")
        print(f"{'=' * 70}")

        try:
            df = precompute_fn(ohlcv_data[ticker])
            p  = best_params[ticker]
            
            sig = inspect.signature(vbt_signal_fn)
            valid_p = {k: v for k, v in p.items() if k in sig.parameters}
            
            entries, exits = vbt_signal_fn(df, **valid_p)

            bt_vbt = VBTBacktester(
                close      = df['Close'],
                entries    = entries,
                exits      = exits,
                freq       = freq,
                init_cash  = cash,
                commission = commission,
            )
            bt_vbt.full_analysis(n_mc=500, n_wf_splits=4, n_trials=len(tickers))

        except Exception as e:
            print(f"  ❌ VBT error: {e}")
        
        # Explicitly shut down loky after each ticker to prevent semaphore buildup
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except:
            pass

    # 5. ML/DL/RL signal enhancement (uses best params for signal gen)
    # try:
    #     from utils.ml_signals import run_ml_comparison
    #     for ticker in tickers:
    #         if ticker not in all_stats:
    #             continue
    #         try:
    #             df = precompute_fn(ohlcv_data[ticker])
    #             p  = best_params[ticker]
                
    #             sig = inspect.signature(vbt_signal_fn)
    #             valid_p = {k: v for k, v in p.items() if k in sig.parameters}
                
    #             entries, exits = vbt_signal_fn(df, **valid_p)
    #             run_ml_comparison(df, entries, exits, ticker, freq=freq)
    #         except Exception as e:
    #             print(f"  ❌ ML error for {ticker}: {e}")
    # except ImportError:
    #     print("\n  ⚠️  ML libraries not available. Skipping ML signal enhancement.")
